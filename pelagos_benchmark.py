#!/usr/bin/env python3
# (C) Copyright IBM 2026
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Relay-BP Detailed Decoder Script

This script provides detailed analysis of Relay-BP and Plain-BP decoding performance,
measuring iteration counts and per-cycle error rates to match the paper's methodology.
It supports both Rust and Triton backends with comprehensive benchmarking capabilities.

The script implements the exact methodology from the Relay-BP paper:
- X-axis: Average BP iteration count (not wall time)
- Y-axis: Per-cycle logical error rate (not per-shot)
- Parameters: S = solutions sought, R = max relay legs

Key features:
- Detailed iteration counting for both Relay-BP and Plain-BP
- Deterministic reproducibility with configurable seeding
- Support for multiple backends (Rust/Triton) and data types
- Comprehensive timing analysis (decoder-only and end-to-end)
- Multiple output formats (simple, JSON, CSV)
"""

import os
import sys

# Add the development source to the path to use the unified interface
# This MUST be done before any other imports that might import relay_bp
script_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(script_dir, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import argparse
import logging
import math
import random
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch

sys.path.append(str(Path(__file__).parent / "tests"))
import hashlib
import json

import relay_bp
from pelagos.stim.sinter.check_matrices import CheckMatrices
from pelagos.utils import (
    describe_check_matrices,
    format_batch_per_device_from_decoder,
    format_batch_per_worker,
    load_explicit_gammas,
    load_gamma_dist_by_error,
)
from testdata import filter_detectors_by_basis, get_test_circuit

# Optional Triton adapter imports (GPU backend).
try:
    from pelagos.adapter import (
        ObservableDecoderRunner as TritonObservableRunner,
    )
    from pelagos.adapter import (
        RelayDecoder as TritonRelayDecoder,
    )

    _HAS_TRITON = True
except ImportError:
    TritonRelayDecoder = None
    TritonObservableRunner = None
    _HAS_TRITON = False
import pandas as pd

log = logging.getLogger(__name__)


def setup_logging(level=logging.INFO):
    for h in log.handlers:
        log.removeHandler(h)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M"
    )
    handler.setFormatter(formatter)
    log.addHandler(handler)
    log.setLevel(level)
    log_components = [
        "pelagos.utils",
    ]
    for c in log_components:
        logger = logging.getLogger(c)
        logger.setLevel(level)
        logger.addHandler(handler)


def _select_backend(backend: str, data_type: str = "float32", require: str | None = None):
    """Select appropriate decoder classes for the chosen backend and data type.

    This function provides a unified interface for selecting decoder implementations
    across different backends (Rust/Triton) and data types ("float16", "float32", "float64").

    Args:
        backend: "rust" or "triton"
        data_type: "float16", "float32", or "float64"
        require: "relay", "plain", or None (which decoder types to require)

    Returns:
        Tuple of (RelayDecoder, ObservableDecoderRunner, MinSumBPDecoder)
        MinSumBPDecoder is None for Triton backend

    Raises:
        ValueError: If requested data_type/backend combination is unavailable
    """
    # Use the already imported relay_bp module
    _rb = relay_bp

    if backend == "triton":
        if data_type not in ("float16", "float32", "float64"):
            raise ValueError(
                f"Triton supports data_type in {{'float16','float32','float64'}}, got {data_type!r}"
            )
        if not _HAS_TRITON:
            raise ImportError("Triton backend requested but pelagos.adapter is unavailable")

        relay_cls = TritonRelayDecoder
        return relay_cls, TritonObservableRunner, None

    elif backend == "rust":
        if data_type not in ("float32", "float64"):
            raise ValueError(
                f"Rust backend supports only data_type in {{'float32','float64'}}, got {data_type!r}"
            )

        relay_cls = None
        minsum_cls = None

        if require in (None, "relay"):
            relay_cls = getattr(_rb, "RelayDecoderF32" if data_type == "float32" else "RelayDecoderF64")
        if require in (None, "plain"):
            minsum_cls = getattr(_rb, "MinSumBPDecoderF32" if data_type == "float32" else "MinSumBPDecoderF64")

        from relay_bp import ObservableDecoderRunner as _ObservableDecoderRunner

        return relay_cls, _ObservableDecoderRunner, minsum_cls
    else:
        raise ValueError(f"Unknown backend: {backend}. Available: {_rb.get_available_backends()}")


def _format_time_from_us(us: float) -> str:
    """Format time in microseconds to a compact multi-unit string including ns."""
    ns = us * 1e3
    ms = us / 1e3
    s = us / 1e6
    return f"{ns:.1f} ns | {us:.3f} µs | {ms:.3f} ms | {s:.6f} s"


def _compute_decoder_timing_metrics(
    total_decode_ns: int,
    total_decode_shots: int,
    total_iterations: float,
    total_legs: Optional[float] = None,
) -> dict:
    """Derive the standard decoder-timing metrics from raw counters.

    Args:
        total_decode_ns: cumulative wall-time spent inside ``decoder.decode()``,
            in nanoseconds.
        total_decode_shots: number of shots that contributed to ``total_decode_ns``.
        total_iterations: cumulative iteration count across shots.
        total_legs: cumulative leg-runs across shots; pass ``None`` (or 0) to
            skip the per-leg derivation.

    Returns:
        dict of metrics to ``results.update(...)`` into the run output.
    """
    if total_decode_shots <= 0:
        return {}
    out: dict = {}
    decoder_total_time_us = float(total_decode_ns) / 1e3
    out["decoder_total_time_us"] = decoder_total_time_us
    out["decoder_runtime_per_shot_us"] = decoder_total_time_us / float(total_decode_shots)
    if total_iterations > 0:
        out["decoder_runtime_per_iteration_us"] = decoder_total_time_us / float(total_iterations)
    if decoder_total_time_us > 0.0:
        decoder_total_time_s = decoder_total_time_us / 1e6
        out["decoder_shots_per_second"] = float(total_decode_shots) / decoder_total_time_s
        if total_iterations > 0:
            out["decoder_iterations_per_second"] = float(total_iterations) / decoder_total_time_s
    out["total_decode_shots"] = int(total_decode_shots)
    out["total_iterations"] = float(total_iterations)
    if total_legs is not None and float(total_legs) > 0:
        out["decoder_runtime_per_leg_us"] = decoder_total_time_us / float(total_legs)
    return out




def _hash_update_arr(hasher: "hashlib._blake2.blake2b", arr) -> None:
    """Update BLAKE2b hasher with array data for deterministic reproducibility.

    This function ensures deterministic results by hashing all relevant data
    in a consistent order and format, enabling reproducible experiments.

    Args:
        hasher: BLAKE2b hasher instance
        arr: NumPy array to hash
    """
    a = np.ascontiguousarray(arr)
    hasher.update(a.tobytes(order="C"))


def parse_args():
    """Parse command line arguments for Relay-BP detailed analysis.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Run Relay-BP decoder with detailed iteration counting"
    )

    # Mode parameter
    parser.add_argument("--mode", choices=["relay", "plain"], default="relay")
    parser.add_argument(
        "--perf",
        choices=["default", "throughput"],
        default="default",
        help="Kernel execution path (default=row-wise)",
    )
    parser.add_argument(
        "--data-type",
        dest="data_type",
        choices=["float16", "float32", "float64"],
        default="float32",
        help="Numeric data type (float16/float32/float64).",
    )
    parser.add_argument(
        "--scheduling",
        choices=["flood-fill", "layered"],
        default="flood-fill",
        help='Iteration scheduling mode: "flood-fill" (default, all checks then all vars) or "layered" (block-layered for first K iterations, then flood-fill). Only supported in throughput mode.',
    )
    parser.add_argument(
        "--layered-k-iters",
        type=int,
        default=0,
        dest="layered_k_iters",
        help='Number of layered iterations per leg (default: 0, disabled). When scheduling="layered" and layered_k_iters > 0, first K iterations use block-layered, remaining use flood-fill. Only used when scheduling="layered" and perf="throughput".',
    )
    parser.add_argument(
        "--layered-block-policy",
        choices=["contiguous", "permute", "strided_interleaving", "color", "color_use_delta_trick"],
        default="contiguous",
        dest="layered_block_policy",
        help='Block policy for layered scheduling: contiguous (default), permute (random), strided_interleaving (cyclic, requires --layered-num-blocks), color (conflict-free layers via graph coloring), or color_use_delta_trick (conflict-free layers + delta-trick, Triton/throughput only). Only used when scheduling="layered".',
    )
    parser.add_argument(
        "--layered-num-blocks",
        type=int,
        default=None,
        dest="layered_num_blocks",
        help='Explicit number of blocks for layered scheduling (default: None, computed from target_edges_per_block). Only used when scheduling="layered".',
    )
    parser.add_argument(
        "--layered-target-edges-per-block",
        type=int,
        default=None,
        dest="layered_target_edges_per_block",
        help='Target edges per block for automatic block sizing (default: 200000). Only used when scheduling="layered" and layered_num_blocks is None.',
    )
    parser.add_argument(
        "--layered-block-seed",
        type=int,
        default=42,
        dest="layered_block_seed",
        help='Random seed for permute block policy (default: 42). Only used when scheduling="layered" and layered_block_policy="permute".',
    )

    # Circuit parameters
    parser.add_argument(
        "--circuit",
        type=str,
        default="bicycle_bivariate_144_12_12_memory_choi_XZ",
        help="Circuit name (default: bicycle_bivariate_144_12_12_memory_XZ)",
    )
    parser.add_argument(
        "--basis",
        type=str,
        choices=["x", "z", "xz"],
        default="xz",
        help="Decoding basis (default: xz)",
    )
    parser.add_argument("--distance", type=int, default=12, help="Code distance (default: 12)")
    parser.add_argument("--rounds", type=int, default=12, help="Number of rounds (default: 12)")
    parser.add_argument(
        "--error-rate", type=float, default=0.003, help="Physical error rate (default: 0.003)"
    )
    parser.add_argument(
        "--target-errors",
        type=int,
        default=10,
        help="Target number of errors to collect (default: 10)",
    )
    parser.add_argument(
        "--batch", type=int, default=2000, help="Batch size for error collection (default: 2000)"
    )
    parser.add_argument(
        "--max-shots", type=int, default=1000000, help="Maximum shots to collect (default: 1000000)"
    )

    # Relay-BP parameters
    parser.add_argument(
        "--gamma0", type=float, default=0.125, help="Ordered memory parameter (default: 0.125)"
    )
    parser.add_argument("--pre-iter", type=int, default=80, help="Pre-iterations (default: 80)")
    parser.add_argument(
        "--num-sets", type=int, default=3, help="Number of Relay-BP sets (default: 10)"
    )
    parser.add_argument(
        "--set-max-iter", type=int, default=60, help="Max iterations per set (default: 60)"
    )
    parser.add_argument(
        "--gamma-dist-min",
        type=float,
        default=-0.24,
        help="Gamma distribution minimum (default: -0.24)",
    )
    parser.add_argument(
        "--gamma-dist-max",
        type=float,
        default=0.66,
        help="Gamma distribution maximum (default: 0.66)",
    )
    parser.add_argument(
        "--gamma-dist-by-error",
        type=str,
        default=None,
        help=(
            "Optional path to per-variable gamma intervals of shape (V, 2). "
            "Each row corresponds to a variable (error column), with [low, high] bounds "
            "for sampling gamma in that column. When provided, it overrides the global "
            "gamma_dist_min/max for that backend, matching Rust gamma_dist_interval_by_error."
        ),
    )
    parser.add_argument(
        "--explicit-gammas",
        type=str,
        default=None,
        help="Path to file containing explicit gamma values. File should contain numpy array of shape (K, V) "
        "where K is number of gamma sets and V is number of variables. Supports .npy, .npz, or .csv formats. "
        "If K < num_sets, gammas will cycle. When provided, uniform random gamma sampling is disabled.",
    )
    parser.add_argument(
        "--stop-nconv", type=int, default=1, help="Stop after N converged solutions (default: 1)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="(plain) Normalized min-sum alpha; None for standard",
    )
    parser.add_argument(
        "--fast-max-unconverged-pct",
        type=float,
        default=None,
        help="Optional early exit threshold (0.0-1.0) for difficult case collection. If set, decoder stops when "
        "unconverged percentage <= this value (e.g., 0.1 = stop when <=10%% unconverged) and returns unconverged indices. "
        "If None, runs full num_sets relay legs. Default: None. "
        "Useful for collecting difficult cases across batches and processing them separately at the end.",
    )

    # Execution parameters
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Enable Relay-BP builtin parallelism (default: False)",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["simple", "json", "csv"],
        default="simple",
        help="Output format (default: simple)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Global seed for all random number generators (default: 0)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["rust", "triton"],
        default=None,
        help="Decoder backend to use: rust (default) or triton",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help='Comma-separated list of GPU IDs to use for Triton backend (e.g., "0,1,2"). Default: "0" (single GPU).',
    )
    # Logging
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument("--debug", action="store_true", help="Shortcut for --log-level DEBUG")
    parser.add_argument(
        "--export-matrices",
        action="store_true",
        help="When set, export H and O sparse matrices to CSV files named <circuit>_H.csv and <circuit>_O.csv",
    )
    parser.add_argument(
        "--hamming-weight",
        action="store_true",
        default=False,
        dest="hamming_weight",
        help="Report Hamming weight stats (total/mean/median/max/min/std) of the "
        "best-candidate decoded error vectors across all shots.",
    )
    # Always measure and report execution time unconditionally

    parser.add_argument(
        "--sample-only",
        action="store_true",
        help="Sample syndromes from Stim and print details (IDs, syndrome weight, error weight, true observables), then exit without decoding.",
    )





    return parser.parse_args()


def run_relay_bp_experiment(
    circuit,
    basis,
    distance,
    rounds,
    error_rate,
    gamma0,
    pre_iter,
    num_sets,
    set_max_iter,
    gamma_dist_min,
    gamma_dist_max,
    stop_nconv,
    target_errors=20,
    batch=2000,
    max_shots=1000000,
    parallel=True,
    seed=0,
    backend: str | None = None,
    perf: str = "default",
    data_type: str = "float32",
    export_matrices: bool = False,
):
    """Deprecated: use run_bp_experiment_core with mode='relay'."""
    return run_bp_experiment_core(
        mode="relay",
        circuit=circuit,
        basis=basis,
        distance=distance,
        rounds=rounds,
        error_rate=error_rate,
        gamma0=gamma0,
        pre_iter=pre_iter,
        num_sets=num_sets,
        set_max_iter=set_max_iter,
        gamma_dist_min=gamma_dist_min,
        gamma_dist_max=gamma_dist_max,
        stop_nconv=stop_nconv,
        alpha=None,
        target_errors=target_errors,
        batch=batch,
        max_shots=max_shots,
        parallel=parallel,
        seed=seed,
        backend=backend,
        perf=perf,
        data_type=data_type,
        export_matrices=export_matrices,
    )


## Single entry point: use run_bp_experiment_core directly


def run_bp_experiment_core(
    *,
    mode: str,
    circuit,
    basis,
    distance,
    rounds,
    error_rate,
    # Relay params
    gamma0=None,
    pre_iter=0,
    num_sets=0,
    set_max_iter=0,
    gamma_dist_min=None,
    gamma_dist_max=None,
    stop_nconv=1,
    explicit_gammas=None,  # Optional numpy array of shape (K, V) or file path
    gamma_dist_interval_by_error=None,  # Optional numpy array or file path for (V, 2) per-variable gamma intervals
    fast_max_unconverged_pct=None,  # Optional early exit threshold (0.0-1.0): stop when unconverged percentage <= this value
    # Plain params
    alpha=None,
    # Common
    target_errors=20,
    batch=2000,
    max_shots=1000000,
    parallel=True,
    seed=0,
    backend: str | None = None,
    perf: str = "default",
    data_type: str = "fp32",
    export_matrices: bool = False,
    debug: bool = False,
    gpus: Optional[str] = None,  # Comma-separated GPU IDs (e.g., "0,1,2") or None for default [0]
    scheduling: str = "flood-fill",  # "flood-fill" or "layered"
    layered_k_iters: int = 0,  # Number of layered iterations per leg
    layered_block_policy: str = "contiguous",  # "contiguous", "permute", or "strided_interleaving"
    layered_num_blocks: Optional[
        int
    ] = None,  # Explicit number of blocks (if None, computed from target_edges_per_block)
    layered_target_edges_per_block: Optional[
        int
    ] = None,  # Target edges per block (default: 200000)
    layered_block_seed: int = 42,  # Random seed for "permute" policy
    sample_only: bool = False,
    hamming_weight: bool = False,  # Report Hamming-weight stats of decoded errors
):
    """Unified core experiment for Relay-BP (mode='relay') and Plain-BP (mode='plain')."""

    # Deterministic seeding using global seed
    random.seed(seed)
    np.random.seed(seed)

    # Get test circuit
    circuit_obj = get_test_circuit(
        circuit=circuit,
        distance=distance,
        rounds=rounds,
        error_rate=error_rate,
    )
    # Beginner-friendly narrative: what is this circuit/code?
    real_qubits = getattr(circuit_obj, "num_qubits", None)
    log.debug(
        "[Step 0] Selected QEC code/circuit '%s' (distance d=%d, rounds=%d, error rate p=%.4g)",
        circuit,
        distance,
        rounds,
        error_rate,
    )
    if real_qubits is not None:
        log.debug("[Step 0] Physical (real) qubits in circuit: %d", real_qubits)

    # Apply basis filtering based on argument
    if basis == "z":
        circuit_obj = filter_detectors_by_basis(circuit_obj, "Z")
    elif basis == "x":
        circuit_obj = filter_detectors_by_basis(circuit_obj, "X")

    # Create detector error model
    dem = circuit_obj.detector_error_model()
    log.debug(
        "[Step 1] Built Detector Error Model (DEM): detectors capture parity-check outcomes, observables encode logicals."
    )
    # Beginner-friendly DEM background
    log.debug(
        "DEM concept: a compressed, noise-only fingerprint of the circuit. It omits gates/qubits and instead maps how noise flips detectors (syndrome bits) and logical observables."
    )
    log.debug(
        "DEM inputs: (1) a compiled noisy circuit with data qubits, measurement ancillas, a measurement schedule over 'rounds', and reset/measure ops\n"
        "            (2) a noise model (e.g., depolarizing) with probabilities like p=%.4g\n (3) a basis '%s' selecting which stabilizers/logicals to track.",
        error_rate,
        basis,
    )
    log.debug(
        "DEM outputs: a set of detectors (M), logical observables (O), and error channels with probabilities specifying which detectors/observables each error flips."
    )

    # Print detector error model statistics
    log.info("DEM summary:")
    log.info(f"  detectors M = {dem.num_detectors}")
    log.info(f"  observables O = {dem.num_observables}")
    log.info(f"  errors N = {CheckMatrices.from_dem(dem).check_matrix.shape[1]}")
    log.info(
        f"  Per-cycle detectors = {dem.num_detectors // rounds} . This is essentially the number of stabilizer parity checks per round, which in hardware are implemented with measurement ancillas."
    )
    log.info(f"  Basis = {basis}")
    # Explain M factorization for beginners: M = rounds × checks_per_round
    M = int(dem.num_detectors)
    R = int(rounds)
    C = M // R if R > 0 else M
    log.debug("Detector count factorization: M=%d = rounds×checks_per_round = %d×%d", M, R, C)
    log.debug(
        "Here, checks_per_round=%d is the number of stabilizer measurements per round (= measurement ancillas).",
        C,
    )

    log.debug(f"  DEM text preview (first 5 lines):\n{str(dem)[:500]}")
    log.debug(
        f"""
        Interpretation:
        With probability 0.00040048, this single error occurs.
        It flips the parity of detectors numbered 0, 1, 23, 74, 137, 143 (these are syndrome bits formed from stabilizer measurement outcomes across time).
        It also flips logical observable L15 (i.e., this error is logically harmful unless corrected).
        Lines without an Lk (e.g., error(...) D0 D5 D73) mean the error only affects detectors, not any tracked logical observable directly.
        """
    )

    log.info(f"Using rounds={rounds} as the number of cycles for per-cycle LER conversion")
    assert dem.num_observables > 0

    # Estimation of qubit counts (data vs measurement ancillas)
    log.debug(
        f"""
    A very common pattern for repeated-syndrome LDPC circuits is:
    real_qubits = data_qubits + ancilla_qubits (+ a small constant)
    This is because the ancillas are reused across rounds.
    That “+1” is typically a small bookkeeping/readout qubit (e.g., a final observable-measurement ancilla, a timing/helper qubit,
    or a single extra ancilla due to boundary/initialization details). In real hardware, this can vary by implementation.
    """
    )
    per_cycle_det = int(dem.num_detectors // rounds)
    # Heuristic: parse data qubits from circuit name like ..._144_12_12_...
    data_guess = None
    m = re.search(r"_(\d+)_\d+_\d+", str(circuit))
    if m:
        data_guess = int(m.group(1))
    # If circuit object exposes num_qubits treat as total (real) qubits
    if real_qubits is None:
        real_qubits = getattr(circuit_obj, "num_qubits", None)
    if data_guess is not None and real_qubits is not None:
        total_est = data_guess + per_cycle_det
        slack = real_qubits - total_est
        log.debug(
            "Qubit accounting (estimate): data=%d + measurement_ancillas=%d + extra=%d -> total=%d (reported real=%s)",
            data_guess,
            per_cycle_det,
            slack,
            total_est + max(0, slack),
            str(real_qubits),
        )
        log.debug(
            "Interpretation: measurement ancillas are reused across rounds; per-cycle detector count approximates the distinct ancilla indices."
        )

    # Check if fast_max_unconverged_pct is supported for this backend
    if fast_max_unconverged_pct is not None and backend == "rust":
        log.warning(
            f"fast_max_unconverged_pct is not supported with Rust backend. Continuing without fast_max_unconverged_pct functionality."
        )
        fast_max_unconverged_pct = None

    # Create check matrices and initialize reproducibility hasher
    check_matrices = CheckMatrices.from_dem(dem)

    # Load explicit gammas and per-variable gamma intervals if provided
    V = int(check_matrices.check_matrix.shape[1])
    explicit_gammas_array = load_explicit_gammas(
        explicit_gammas=explicit_gammas,
        num_variables=V,
        backend=backend,
        num_sets=num_sets if mode == "relay" else None,
        mode=mode,
    )
    gamma_dist_by_error_array = load_gamma_dist_by_error(
        gamma_dist_by_error=gamma_dist_interval_by_error if mode == "relay" else None,
        num_variables=V,
    )

    # Describe matrices (sizes, densities, degree stats, etc.)
    describe_check_matrices(
        check_matrices,
        dem=dem,
        rounds=rounds,
        compute_components=False,
        compute_rank=False,
    )

    # Extra beginner-friendly matrix context
    H_shape = check_matrices.check_matrix.shape
    O_shape = check_matrices.observables_matrix.shape
    priors_len = int(np.asarray(check_matrices.error_priors).size)
    log.debug(
        "[Step 2] Tanner/check matrices built from DEM (The DEM only tracks logical observables, not individual physical-qubit measurements):"
    )
    log.debug("          H (checks x errors) = %s", H_shape)
    log.debug("          O (observables x errors) = %s", O_shape)
    log.debug("          priors length = %d", priors_len)
    obs_count = dem.num_observables
    k_guess = (
        (obs_count // 2) if basis.lower() == "xz" else obs_count
    )  # half the observables if both bases tracked
    log.debug("          Estimated logical qubits k = %d (from observables/basis)", k_guess)
    log.debug(
        "          Narrative: columns of H map physical error bits; rows of H map parity checks (detectors). O maps logical observables."
    )

    hasher = hashlib.blake2b(digest_size=32)

    # Hash problem specification for deterministic reproducibility
    cm = check_matrices.check_matrix
    om = check_matrices.observables_matrix
    # Optional export of H and O matrices to a single CSV for visualization
    if export_matrices:
        try:
            circuit_str = str(circuit)
            ho_path = f"{circuit_str}_HO.csv"
            cm_coo = cm.tocoo()
            om_coo = om.tocoo()
            df_h = pd.DataFrame(
                {
                    "row": cm_coo.row.astype(np.int64),
                    "col": cm_coo.col.astype(np.int64),
                    "value": (cm_coo.data != 0).astype(np.int64),
                }
            )
            df_h["matrix"] = "H"
            df_o = pd.DataFrame(
                {
                    "row": om_coo.row.astype(np.int64),
                    "col": om_coo.col.astype(np.int64),
                    "value": (om_coo.data != 0).astype(np.int64),
                }
            )
            df_o["matrix"] = "O"
            pd.concat([df_h, df_o], ignore_index=True).to_csv(ho_path, index=False)
            log.info(f"Exported combined matrices to '{ho_path}'")
        except Exception as e:
            log.error(f"Failed exporting matrices: {e}")
    _hash_update_arr(hasher, cm.indptr.astype(np.int64, copy=False))
    _hash_update_arr(hasher, cm.indices.astype(np.int64, copy=False))
    _hash_update_arr(hasher, om.indptr.astype(np.int64, copy=False))
    _hash_update_arr(hasher, om.indices.astype(np.int64, copy=False))
    _hash_update_arr(hasher, check_matrices.error_priors.astype(np.float64, copy=False))

    cfg = {
        "circuit": circuit,
        "basis": basis,
        "distance": distance,
        "rounds": rounds,
        "error_rate": error_rate,
        "parallel": bool(parallel),
        "stim_seed": seed,
        # Determinism-affecting runtime knobs
        "backend": backend,
        "perf": perf,
        "data_type": data_type,
        "batch": int(batch),
        "target_errors": int(target_errors),
        "max_shots": int(max_shots),
    }
    if mode == "relay":
        cfg_update = {
            "gamma0": gamma0,
            "pre_iter": pre_iter,
            "num_sets": num_sets,
            "set_max_iter": set_max_iter,
            "gamma_dist": [gamma_dist_min, gamma_dist_max],
            "stop_nconv": stop_nconv,
            "relay_seed": seed,
            # Record stopping criterion implicitly (nconv right now)
        }
        if explicit_gammas_array is not None:
            cfg_update["explicit_gammas"] = f"loaded_shape_{explicit_gammas_array.shape}"
        if gamma_dist_by_error_array is not None:
            cfg_update["gamma_dist_interval_by_error"] = (
                f"loaded_shape_{gamma_dist_by_error_array.shape}"
            )
        cfg.update(cfg_update)
    else:
        cfg.update(
            {
                "set_max_iter": set_max_iter,
                "alpha": alpha,
            }
        )
    hasher.update(json.dumps(cfg, sort_keys=True).encode())

    # --sample-only: sample from Stim, print syndrome details, and return
    if sample_only:
        import scipy.sparse as sp

        sampler = dem.compile_sampler(seed=seed)
        H = sp.csr_matrix(check_matrices.check_matrix, dtype=np.uint8)
        O = sp.csr_matrix(check_matrices.observables_matrix, dtype=np.uint8)
        total = 0
        shot_idx = 0
        log.info(f"[sample-only] Sampling {max_shots} shots in batches of {batch}...")
        while total < max_shots:
            n = min(batch, max_shots - total)
            det, obs, errors = sampler.sample(n, return_errors=True)
            syndromes = np.asarray((H @ errors.astype(np.int8).T) % 2, dtype=np.uint8).T
            true_obs = np.asarray((O @ errors.astype(np.int8).T) % 2, dtype=np.uint8).T
            for i in range(n):
                s_wt = int(syndromes[i].sum())
                e_wt = int(errors[i].sum())
                obs_str = "".join(str(b) for b in true_obs[i])
                log.info(
                    f"  shot {shot_idx}: syndrome_weight={s_wt}, error_weight={e_wt}, true_obs={obs_str}"
                )
                shot_idx += 1
            total += n
        log.info(f"[sample-only] Done. Printed {shot_idx} shots.")
        return {}

    # Initialize batch distribution metrics
    batch_per_gpu_str = None
    batch_per_cpu_str = None

    # Build decoder + runner
    if mode == "relay":
        log.debug(
            "[Step 3] Building Relay-BP decoder (backend=%s, data_type=%s)", backend, data_type
        )
        _RelayDecoderDyn, _ObservableDecoderRunner, _ = _select_backend(
            backend, data_type, require="relay"
        )

        # Build configuration dictionaries for both Rust and Triton backends.
        msl_config = {
            "error_priors": check_matrices.error_priors,
            "alpha": alpha,
            "gamma0": gamma0,
        }
        relay_config = {
            "pre_iter": pre_iter,
            "num_sets": num_sets,
            "set_max_iter": set_max_iter,
            "gamma_dist_interval": (gamma_dist_min, gamma_dist_max),
            "stop_nconv": stop_nconv,
            "logging": False,
            "seed": seed,
        }
        if explicit_gammas_array is not None:
            relay_config["explicit_gammas"] = explicit_gammas_array
        if gamma_dist_by_error_array is not None:
            relay_config["gamma_dist_interval_by_error"] = gamma_dist_by_error_array

        if backend == "triton":
            if not _HAS_TRITON:
                raise ImportError(
                    "Triton backend requested but pelagos.adapter is unavailable"
                )
            # Parse GPU IDs from comma-separated string
            if gpus is None or gpus.strip() == "":
                device_spec = [0]  # Default: single GPU
            else:
                try:
                    device_spec = [int(x.strip()) for x in gpus.split(",")]
                    if not device_spec:
                        raise ValueError("GPU list cannot be empty")
                except ValueError as e:
                    raise ValueError(
                        f"Invalid GPU specification '{gpus}'. Expected comma-separated integers (e.g., '0,1,2'): {e}"
                    )

            backend_config = {
                "data_type": data_type,
                "perf": perf,
                "device": device_spec,
                "algo": "relay",
                "scheduling": scheduling,
                "layered_k_iters": layered_k_iters,
                "layered_block_policy": layered_block_policy,
                "layered_num_blocks": layered_num_blocks,
                "layered_target_edges_per_block": layered_target_edges_per_block,
                "layered_block_seed": layered_block_seed,
            }
            if fast_max_unconverged_pct is not None:
                backend_config["fast_max_unconverged_pct"] = fast_max_unconverged_pct

            decoder = TritonRelayDecoder(
                check_matrices.check_matrix,
                msl_config=msl_config,
                relay_config=relay_config,
                data_type=data_type,
                backend_config=backend_config,
            )
            observable_decoder = TritonObservableRunner(
                decoder,
                check_matrices.observables_matrix,
                include_decode_result=True,
            )
            batch_per_gpu_str = format_batch_per_device_from_decoder(batch, decoder)
        else:
            # Rust backend: keep existing constructor semantics via select_decoder.
            decoder = _RelayDecoderDyn(
                check_matrices.check_matrix,
                **{
                    "error_priors": msl_config["error_priors"],
                    "gamma0": msl_config["gamma0"],
                    "pre_iter": relay_config["pre_iter"],
                    "num_sets": relay_config["num_sets"],
                    "set_max_iter": relay_config["set_max_iter"],
                    "gamma_dist_interval": relay_config["gamma_dist_interval"],
                    "stop_nconv": relay_config["stop_nconv"],
                    "logging": relay_config["logging"],
                    "alpha": msl_config["alpha"],
                    "explicit_gammas": relay_config.get("explicit_gammas"),
                },
            )
            observable_decoder = _ObservableDecoderRunner(
                decoder,
                check_matrices.observables_matrix,
                include_decode_result=True,
            )
            batch_per_gpu_str = None
    else:
        log.debug(
            "[Step 3] Building Plain min-sum BP decoder (backend=%s, data_type=%s, set_max_iter=%d, alpha=%s, gamma0=%s)",
            backend,
            data_type,
            set_max_iter,
            str(alpha),
            str(gamma0),
        )
        if backend == "triton":
            if not _HAS_TRITON:
                raise ImportError(
                    "Triton backend requested but pelagos.adapter is unavailable"
                )
            # Parse GPU IDs for plain mode triton
            if gpus is None or gpus.strip() == "":
                device_spec = [0]  # Default: single GPU
            else:
                try:
                    device_spec = [int(x.strip()) for x in gpus.split(",")]
                    if not device_spec:
                        raise ValueError("GPU list cannot be empty")
                except ValueError as e:
                    raise ValueError(
                        f"Invalid GPU specification '{gpus}'. Expected comma-separated integers (e.g., '0,1,2'): {e}"
                    )

            msl_config_plain = {
                "error_priors": check_matrices.error_priors,
                "alpha": alpha,
                "gamma0": 0.0,
            }
            relay_config_plain = {
                # Emulate plain BP using pre_iter as max_iter and no relay legs.
                "pre_iter": set_max_iter,
                "num_sets": 0,
                "set_max_iter": 0,
                "gamma_dist_interval": (0.0, 0.0),
                "stop_nconv": 1,
                "logging": False,
                "seed": seed,
            }
            backend_config_plain = {
                "data_type": data_type,
                "perf": perf,
                "device": device_spec,
                "algo": "plain",
                "scheduling": scheduling,  # Also support layered for plain mode
                "layered_k_iters": layered_k_iters,
            }

            decoder = TritonRelayDecoder(
                check_matrices.check_matrix,
                msl_config=msl_config_plain,
                relay_config=relay_config_plain,
                data_type=data_type,
                backend_config=backend_config_plain,
            )
            observable_decoder = TritonObservableRunner(
                decoder,
                check_matrices.observables_matrix,
                include_decode_result=True,
            )

            batch_per_gpu_str = format_batch_per_device_from_decoder(batch, decoder)
        else:
            _RelayDecoderDyn_unused, _RustObservableRunner, _MinSumBPDecoderDyn = _select_backend(
                "rust", data_type, require="plain"
            )
            decoder = _MinSumBPDecoderDyn(
                check_matrices.check_matrix,
                check_matrices.error_priors,
                set_max_iter,  # max_iter
                alpha,  # alpha parameter (None, 0.0, or value)
                1.0,  # alpha_iteration_scaling_factor (default)
                gamma0,  # gamma0 parameter for MemBP support
                None,  # data_scale_value
                None,  # max_data_value
                None,  # int_bits
                None,  # frac_bits
            )
            observable_decoder = _RustObservableRunner(
                decoder,
                check_matrices.observables_matrix,
                include_decode_result=True,
            )

    # Warmup (Triton only)
    if backend == "triton":
        try:
            log.debug("[Step 3a] Triton warmup to exclude compilation/autotuning from measurements")
            from pelagos.utils import warmup_build_and_decode

            # Parse GPU IDs for warmup
            if gpus is None or gpus.strip() == "":
                warmup_device = [0]  # Default: single GPU
            else:
                try:
                    warmup_device = [int(x.strip()) for x in gpus.split(",")]
                    if not warmup_device:
                        warmup_device = [0]  # Fallback to single GPU
                except ValueError:
                    warmup_device = [0]  # Fallback to single GPU

            if mode == "relay":
                warmup_build_and_decode(
                    check_matrices=check_matrices,
                    dem=dem,
                    backend=backend,
                    algo="relay",
                    perf=perf,
                    dtype=data_type,
                    pre_iter=1,
                    num_sets=0,
                    set_max_iter=0,
                    stop_nconv=1,
                    gamma0=gamma0,
                    gamma_dist_interval=(gamma_dist_min, gamma_dist_max),
                    alpha=alpha,
                    beta=None,
                    device=warmup_device,
                    seed=seed,
                    repeats=3,
                    batch=batch,
                    parallel=parallel,
                )
            else:
                warmup_build_and_decode(
                    check_matrices=check_matrices,
                    dem=dem,
                    backend=backend,
                    algo="plain",
                    perf=perf,
                    dtype=data_type,
                    pre_iter=1,
                    num_sets=0,
                    set_max_iter=0,
                    stop_nconv=1,
                    gamma0=None,
                    gamma_dist_interval=None,
                    alpha=alpha,
                    beta=None,
                    device=warmup_device,
                    seed=seed,
                    repeats=3,
                    batch=batch,
                    parallel=parallel,
                )
        except Exception:
            pass

    # Stats accumulators
    total_shots = 0  # Actual shots processed (converged in fast pass + difficult batch)
    total_errors = 0
    bp_iters_all = []
    legs_all = []
    total_decode_ns = 0
    total_decode_shots = 0

    # Hamming weight of the best decoded error pattern per shot.
    hamming_weight_all = []


    # Quality metric accumulators (from decoder-internal from_errors path)
    total_unconverged_no_error = 0
    total_unconverged_logical_error = 0
    total_better_decoding_quality_error = 0
    total_worse_decoding_quality_error = 0


    # Collect unconverged indices across batches (for fast_max_unconverged_pct difficult case collection)
    all_unconverged_errors = []  # Store error patterns for unconverged cases
    all_unconverged_global_indices = []  # Store global indices for tracking
    batch_count = 0

    # Track total samples processed for loop condition (includes all cases, converged and unconverged)
    total_samples_processed = 0

    # Progressive LER estimation for fast_max_unconverged_pct optimization
    # Estimate total shots needed based on current LER, updated after each difficult batch
    estimated_max_shots = max_shots  # Start with original max_shots
    using_estimated_shots = False  # Track if we're using estimated values

    def update_progressive_ler_estimate(context: str = ""):
        """Calculate and update progressive LER estimate based on current statistics.

        Args:
            context: Optional context string for logging (e.g., "first batch", "difficult batch")
        """
        nonlocal estimated_max_shots, using_estimated_shots
        if fast_max_unconverged_pct is not None and total_shots > 0 and total_errors > 0:
            progressive_ler = total_errors / total_shots
            # Estimate: if current LER is E/S, to get T errors we need T * S / E shots
            estimated_total_shots = int(target_errors * total_shots / total_errors)
            # Cap at original max_shots (never exceed user's limit)
            estimated_max_shots = min(estimated_total_shots, max_shots)
            using_estimated_shots = estimated_max_shots < max_shots

            context_str = f"After {context}: " if context else ""
            log.info(
                f"[Progressive LER] {context_str}shots={total_shots}, errors={total_errors}, LER={progressive_ler:.6f}"
            )
            log.info(
                f"[Progressive LER] Estimated total shots needed: {estimated_total_shots} (capped at {max_shots} = {estimated_max_shots})"
            )
            if using_estimated_shots:
                log.warning(
                    f"[Progressive LER] Using ESTIMATED max_shots={estimated_max_shots} instead of requested {max_shots} based on progressive LER"
                )
            log.info(f"")

    # First batch optimization: run first batch with full legs (no early exit) to get accurate LER estimate
    # Then enable fast mode from second batch onwards
    first_batch_complete = False  # Track if first batch has been processed
    original_fast_max_unconverged_pct = None  # Store original value
    if fast_max_unconverged_pct is not None and backend == "triton":
        # Temporarily disable fast_max_unconverged_pct for first batch (Triton only)
        original_fast_max_unconverged_pct = observable_decoder.decoder._dec.fast_max_unconverged_pct
        observable_decoder.decoder._dec.fast_max_unconverged_pct = None
        log.info(
            f"[First Batch Optimization] First batch will run with full legs (fast_max_unconverged_pct disabled) to get accurate LER estimate"
        )
        log.info(
            f"[First Batch Optimization] Fast mode (fast_max_unconverged_pct={fast_max_unconverged_pct}) will be enabled from second batch onwards"
        )

    # Sampling loop with difficult batch processing
    # Outer loop: continue until we have enough errors or hit max_shots
    sampler = dem.compile_sampler(seed=seed)
    log.info(f"Collecting until {target_errors} errors or {max_shots} shots...")
    if fast_max_unconverged_pct is not None:
        log.info(
            f"[Progressive LER Optimization] Enabled: will estimate total shots needed based on progressive LER after each difficult batch"
        )
    start_time = time.time()

    # Outer loop: collect batches, process difficult batch when accumulated unconverged >= target_errors, and continue if needed
    # Use estimated_max_shots if fast_max_unconverged_pct is enabled
    effective_max_shots = estimated_max_shots if fast_max_unconverged_pct is not None else max_shots

    # Chunked processing to reduce inter-batch sync overhead
    CHUNK_SIZE = 1  # Process batches in chunks


    while total_errors < target_errors and total_samples_processed < effective_max_shots:
        # Collect chunk of batches on GPU (no CPU sync)
        chunk_gpu_results = []
        chunk_errors_list = []
        chunk_global_starts = []

        for chunk_idx in range(CHUNK_SIZE):
            if total_errors >= target_errors or total_samples_processed >= effective_max_shots:
                break
            det, obs, errors = sampler.sample(batch, return_errors=True)
            batch_count += 1
            global_start = total_samples_processed
            chunk_global_starts.append(global_start)

            # Log progress during collection for large chunks
            if CHUNK_SIZE > 4 and chunk_idx % 4 == 0:
                log.debug(f"  Collecting chunk: {chunk_idx + 1}/{CHUNK_SIZE} batches...")

            t0_ns = time.perf_counter_ns()
            if backend == "triton":
                gpu_result = observable_decoder._decode_gpu_only(errors.astype(np.uint8))
                t1_ns = time.perf_counter_ns()
                chunk_gpu_results.append((gpu_result, t0_ns, t1_ns))
                chunk_errors_list.append(errors)
            else:
                obs_det_batch = observable_decoder.from_errors_decode_observables_detailed_batch(
                    errors.astype(np.uint8), parallel=parallel
                )
                t1_ns = time.perf_counter_ns()
                chunk_gpu_results.append((obs_det_batch, t0_ns, t1_ns))
                chunk_errors_list.append(errors)

        # Process batches in chunk with single batched transfer (one sync for all batches)
        chunk_obs_det_list = None
        if backend == "triton" and len(chunk_gpu_results) > 1:
            # Extract GPU results and true_obs from tuples
            chunk_gpu_results_only = [gpu_result for gpu_result, _, _ in chunk_gpu_results]
            chunk_true_obs_list = [gpu_result["true_obs"] for gpu_result in chunk_gpu_results_only]
            chunk_obs_det_list = observable_decoder._process_gpu_results_batched(
                chunk_gpu_results_only, chunk_errors_list, chunk_true_obs_list
            )

        # Process all batches in chunk
        for batch_idx, ((result_data, t0_ns, t1_ns), errors, global_start) in enumerate(
            zip(chunk_gpu_results, chunk_errors_list, chunk_global_starts)
        ):
            # Get obs_det (backend-specific processing)
            if backend == "triton":
                # Triton: process GPU results
                if chunk_obs_det_list is not None:
                    obs_det = chunk_obs_det_list[batch_idx]
                else:
                    obs_det = observable_decoder._process_gpu_results(result_data)
            else:
                # Rust: results are already processed
                obs_det = result_data

            if hamming_weight:
                batch_hamming = np.fromiter(
                    (int(d.decoding.sum()) for d in obs_det),
                    dtype=np.int32,
                    count=len(obs_det),
                )
                hamming_weight_all.append(batch_hamming)


            # After first batch, enable fast_max_unconverged_pct if it was originally set
            # Note: Progressive LER estimate will be calculated after statistics are updated (see below)
            if (
                fast_max_unconverged_pct is not None
                and backend == "triton"
                and not first_batch_complete
                and batch_count == 1
            ):
                observable_decoder.decoder._dec.fast_max_unconverged_pct = (
                    original_fast_max_unconverged_pct
                )
                first_batch_complete = True
                log.info(
                    f"[First Batch Optimization] First batch complete. Fast mode (fast_max_unconverged_pct={fast_max_unconverged_pct}) now enabled"
                )

            # Check for unconverged_indices (set when fast_max_unconverged_pct is used, same for all items)
            # Only available for Triton backend; Rust backend doesn't support fast_max_unconverged_pct
            # CRITICAL: When fast_max_unconverged_pct is set, the decoder ALWAYS returns unconverged_indices
            # at the end of decode (even if early exit didn't trigger). This contains all cases with found_count < stop_nconv.
            # We need to collect these and process them in the difficult batch to ensure correct iteration counting.
            unconverged_indices = None
            if obs_det and hasattr(obs_det[0], "unconverged_indices"):
                unconverged_indices_raw = obs_det[0].unconverged_indices
                # Handle both None and empty array cases
                if unconverged_indices_raw is not None:
                    unconverged_indices = (
                        unconverged_indices_raw if len(unconverged_indices_raw) > 0 else None
                    )
                else:
                    unconverged_indices = None
            # CRITICAL FIX: When fast_max_unconverged_pct is set, we should always check for unconverged cases.
            # The decoder returns unconverged_indices at the end of decode regardless of whether early exit happened.
            # If unconverged_indices is None or empty but fast_max_unconverged_pct is set, verify by checking converged status.
            # This handles edge cases where unconverged_indices might not be correctly populated.
            if (
                fast_max_unconverged_pct is not None
                and backend == "triton"
                and first_batch_complete
                and (unconverged_indices is None or len(unconverged_indices) == 0)
            ):
                # Fallback: check converged status directly to find unconverged cases
                # This ensures we don't miss unconverged cases even if unconverged_indices wasn't populated
                unconverged_mask = np.array([not r.converged for r in obs_det], dtype=bool)
                if np.any(unconverged_mask):
                    unconverged_indices = np.where(unconverged_mask)[0]
                    log.debug(
                        f"  Batch {batch_count}: Found {len(unconverged_indices)} unconverged cases via fallback check (fast_max_unconverged_pct is set)"
                    )
            if unconverged_indices is not None and len(unconverged_indices) > 0:
                # Collect unconverged cases for difficult batch processing
                unconverged_errors = errors[unconverged_indices]
                all_unconverged_errors.append(unconverged_errors)
                # Track global indices (for logging/debugging)
                global_indices = global_start + unconverged_indices
                all_unconverged_global_indices.extend(global_indices.tolist())
                accumulated_count = sum(len(e) for e in all_unconverged_errors)
                log.info(
                    f"  Batch {batch_count}: {len(unconverged_indices)}/{batch} syndromes did not converge (accumulated: {accumulated_count})"
                )

                # Create mask to exclude unconverged cases from fast-pass statistics
                converged_mask = np.ones(len(obs_det), dtype=bool)
                converged_mask[unconverged_indices] = False
                obs_det_converged = [obs_det[i] for i in range(len(obs_det)) if converged_mask[i]]
                errors_converged = errors[converged_mask]

                # Ensure filtering worked correctly
                if len(obs_det_converged) + len(unconverged_indices) != len(obs_det):
                    log.error(
                        f"  Batch {batch_count}: Filtering mismatch! obs_det={len(obs_det)}, "
                        f"converged={len(obs_det_converged)}, unconverged={len(unconverged_indices)}"
                    )
            else:
                # No unconverged cases, use all results
                converged_mask = np.ones(len(obs_det), dtype=bool)
                obs_det_converged = obs_det
                errors_converged = errors

            total_decode_ns += t1_ns - t0_ns
            # Only count converged shots in fast pass (unconverged will be counted in difficult batch)
            total_decode_shots += len(obs_det_converged)

            # Hash updates (only for converged cases in fast pass)
            _hash_update_arr(hasher, errors_converged.astype(np.uint8, copy=False))
            O = check_matrices.observables_matrix.shape[0]
            pred_obs = np.empty((len(obs_det_converged), O), dtype=np.uint8)
            for i, r in enumerate(obs_det_converged):
                pred_obs[i, :] = np.asarray(r.observables, dtype=np.uint8)
            _hash_update_arr(hasher, pred_obs)

            # Stats (only for converged cases in fast pass)
            # Verify that all cases in obs_det_converged are actually converged
            if unconverged_indices is not None and len(unconverged_indices) > 0:
                unconverged_in_converged = [
                    i for i, r in enumerate(obs_det_converged) if not r.converged
                ]
                if unconverged_in_converged:
                    log.error(
                        f"Fast pass: Found {len(unconverged_in_converged)} unconverged cases in obs_det_converged! "
                        f"This indicates a bug in filtering. Filtering them out now."
                    )
                    # Double-filter: remove any unconverged cases that slipped through
                    obs_det_converged = [r for r in obs_det_converged if r.converged]
                    # Recompute errors_converged to match
                    converged_indices = [
                        i for i in range(len(obs_det)) if i not in unconverged_indices
                    ]
                    errors_converged = errors[converged_indices]

            iters_arr = np.array([r.iterations for r in obs_det_converged], dtype=float)
            iters_i32 = np.fromiter(
                (r.iterations for r in obs_det_converged),
                dtype=np.int32,
                count=len(obs_det_converged),
            )
            succ_arr = np.fromiter(
                (r.converged for r in obs_det_converged),
                dtype=np.uint8,
                count=len(obs_det_converged),
            )
            fail_arr = np.fromiter(
                (r.error_detected for r in obs_det_converged),
                dtype=np.uint8,
                count=len(obs_det_converged),
            )
            _hash_update_arr(hasher, iters_i32)
            _hash_update_arr(hasher, succ_arr)
            _hash_update_arr(hasher, fail_arr)

            bp_iters_all.extend(iters_arr.tolist())
            if mode == "relay" and set_max_iter > 0:
                legs_all.extend(1.0 + np.maximum(0.0, (iters_arr - pre_iter) / set_max_iter))


            # Only count errors from converged cases in fast pass for final statistics
            # Unconverged cases will be counted in the difficult batch (full run result is authoritative)
            errs = int(sum(r.error_detected for r in obs_det_converged))
            n = len(obs_det_converged)
            n_conv = int(sum(r.converged for r in obs_det_converged))
            n_ok = int(sum((not r.error_detected) and r.converged for r in obs_det_converged))
            n_bad_but_conv = int(sum(r.error_detected and r.converged for r in obs_det_converged))
            n_bad_and_nconv = errs - n_bad_but_conv
            n_ok_and_nconv = int(
                sum((not r.error_detected) and (not r.converged) for r in obs_det_converged)
            )

            # Accumulate decoder-internal quality metrics (always populated by from_errors path)
            for r in obs_det_converged:
                total_unconverged_no_error += r.unconverged_no_error
                total_unconverged_logical_error += getattr(r, "unconverged_logical_error", 0) or 0
                total_better_decoding_quality_error += r.better_decoding_quality_error
                total_worse_decoding_quality_error += r.worse_decoding_quality_error
            # Beginner-friendly explanations of statuses
            log.debug(
                "Definitions: error_detected=True means predicted logical observables differ from ground truth (a logical error)."
            )
            log.debug(
                "Definitions: converged=True means BP reached a fixed point/stopping criterion; non-converged does not necessarily imply wrong observables."
            )
            log.debug(
                "Interpretation: success&conv = correct observables with convergence; fail&conv = logical error despite convergence;"
                " fail&nconv = logical error without convergence; success&nconv = correct observables without convergence."
            )
            log.debug(
                "batch breakdown: success&conv=%d, fail&conv=%d, fail&nconv=%d, success&nconv=%d, conv=%d/%d",
                n_ok,
                n_bad_but_conv,
                n_bad_and_nconv,
                n_ok_and_nconv,
                n_conv,
                n,
            )
            total_errors += errs  # For final statistics (only converged in fast pass)
            # Only count converged shots in fast pass (unconverged will be counted in difficult batch)
            total_shots += len(obs_det_converged)
            # Track all samples processed for loop condition (including unconverged that will be reprocessed)
            total_samples_processed += batch
            log.info(f"Collected {total_shots} shots, {total_errors} errors (batch errors: {errs})")

            # Calculate progressive LER estimate immediately after first batch completes
            # This is the key reason for running first batch with full legs - to get early LER estimate
            # Must be called AFTER total_shots and total_errors are updated
            # Check batch_count == 1 to ensure we only do this once after the first batch
            if fast_max_unconverged_pct is not None and batch_count == 1:
                update_progressive_ler_estimate("first batch")

            # Check if accumulated unconverged cases reach batch_size
            # When we have >= batch_size cases, process one full batch and keep leftovers
            # Leftovers will accumulate with new unconverged cases, avoiding padding
            accumulated_unconverged = sum(len(e) for e in all_unconverged_errors)
            if fast_max_unconverged_pct is not None and accumulated_unconverged >= batch:
                log.info(
                    f"Accumulated {accumulated_unconverged} unconverged cases (>= batch_size {batch}). Processing one difficult batch..."
                )
                break  # Exit chunk processing to handle difficult batch

            # Check if we've reached target_errors (converged only) or max_shots
            # Exit inner loop to process difficult batch before final check
            if total_errors >= target_errors:
                log.info(
                    f"Reached target of {target_errors} converged errors. Processing difficult batch before final check..."
                )
                break

            # Check against effective_max_shots (estimated if fast_max_unconverged_pct is enabled)
            effective_max_shots = (
                estimated_max_shots if fast_max_unconverged_pct is not None else max_shots
            )
            if total_samples_processed >= effective_max_shots:
                if fast_max_unconverged_pct is not None and using_estimated_shots:
                    log.info(
                        f"Reached estimated max_shots limit of {effective_max_shots} (based on progressive LER, original was {max_shots}). Processing difficult batch before final check..."
                    )
                else:
                    log.info(
                        f"Reached max_shots limit of {max_shots}. Processing difficult batch before final check..."
                    )
                break

        # Process one difficult batch if we have accumulated >= batch_size cases
        # Process only one full batch at a time, keep leftovers for next iteration
        # This avoids padding by accumulating new cases on top of leftovers
        if fast_max_unconverged_pct is not None and len(all_unconverged_errors) > 0:
            # Concatenate all unconverged error patterns
            difficult_errors = np.concatenate(all_unconverged_errors, axis=0)
            num_difficult = len(difficult_errors)

            # Only process if we have at least batch_size cases
            if num_difficult >= batch:
                # Measure time for difficult batch processing
                difficult_batch_start_ns = time.perf_counter_ns()

                # Temporarily disable fast_max_unconverged_pct on existing decoder for full run
                temp_fast_max_unconverged_pct = None
                if backend == "triton":
                    temp_fast_max_unconverged_pct = (
                        observable_decoder.decoder._dec.fast_max_unconverged_pct
                    )
                    observable_decoder.decoder._dec.fast_max_unconverged_pct = None
                observable_decoder_full = observable_decoder

                # Process only the first batch_size cases (one full batch)
                difficult_batch = difficult_errors[:batch]
                log.info(
                    f"  Processing difficult batch: {batch} syndromes (full run with num_sets={num_sets})"
                )

                # Decode difficult batch with full run (exactly batch_size, no padding needed)
                obs_det_full = (
                    observable_decoder_full.from_errors_decode_observables_detailed_batch(
                        difficult_batch.astype(np.uint8), parallel=parallel
                    )
                )

                # Process results from difficult batch
                iters_arr_full = np.array([r.iterations for r in obs_det_full], dtype=float)
                # Update stats (no hashing for difficult batch)
                bp_iters_all.extend(iters_arr_full.tolist())
                if mode == "relay" and set_max_iter > 0:
                    legs_all.extend(
                        1.0 + np.maximum(0.0, (iters_arr_full - pre_iter) / set_max_iter)
                    )

                # Count errors from difficult batch (full run result is authoritative)
                errs_full = int(sum(r.error_detected for r in obs_det_full))
                n_conv_full = int(sum(r.converged for r in obs_det_full))
                total_errors += errs_full
                total_shots += batch
                total_decode_shots += batch

                # Accumulate quality metrics from difficult batch
                for r in obs_det_full:
                    total_unconverged_no_error += r.unconverged_no_error
                    total_better_decoding_quality_error += r.better_decoding_quality_error
                    total_worse_decoding_quality_error += r.worse_decoding_quality_error

                log.info(
                    f"  Difficult batch complete: {errs_full} errors, {n_conv_full}/{batch} converged"
                )

                # Keep leftovers for next iteration (accumulate with new unconverged cases)
                leftovers = difficult_errors[batch:]
                all_unconverged_errors.clear()
                all_unconverged_global_indices.clear()
                if len(leftovers) > 0:
                    # Store leftovers as a list of arrays (single array in list for consistency)
                    all_unconverged_errors.append(leftovers)
                    log.info(f"  Kept {len(leftovers)} leftover cases for next iteration")

                # Restore original fast_max_unconverged_pct
                if backend == "triton":
                    observable_decoder.decoder._dec.fast_max_unconverged_pct = (
                        temp_fast_max_unconverged_pct
                    )

                # Measure end time and add to total_decode_ns
                difficult_batch_end_ns = time.perf_counter_ns()
                total_decode_ns += difficult_batch_end_ns - difficult_batch_start_ns

                # Progressive LER estimation: calculate expected total shots based on current LER
                # This gives us a better estimate of how many shots we need to reach target_errors
                update_progressive_ler_estimate("difficult batch")

        # After processing difficult batch (if any), check if we've reached target_errors
        # If we have, exit outer loop. Otherwise, outer loop will continue if conditions are met.
        if total_errors >= target_errors:
            log.info(
                f"After processing difficult batch: reached target of {target_errors} errors. Done."
            )
            break
        # If we still need more errors and haven't hit effective_max_shots, the outer loop will continue

    # Process any remaining unconverged cases after outer loop exits
    # Only process if we haven't already reached target_errors (to avoid inflating statistics)
    # If we exited due to target_errors, leftover cases are not needed for statistics
    if fast_max_unconverged_pct is not None and len(all_unconverged_errors) > 0:
        # Check if we've already reached target_errors - if so, skip processing leftovers
        if total_errors >= target_errors:
            num_difficult = sum(len(e) for e in all_unconverged_errors)
            log.info(f"")
            log.info(
                f"[Final Difficult Case Collection] Skipping {num_difficult} leftover unconverged cases (already reached target_errors={target_errors})"
            )
        else:
            # Measure time for entire final difficult batch processing (including overhead)
            final_difficult_batch_start_ns = time.perf_counter_ns()

            # Concatenate all remaining unconverged error patterns
            difficult_errors = np.concatenate(all_unconverged_errors, axis=0)
            num_difficult = len(difficult_errors)
            log.info(f"")
            log.info(
                f"[Final Difficult Case Collection] Processing {num_difficult} remaining unconverged syndromes"
            )

            # Temporarily disable fast_max_unconverged_pct on existing decoder for full run
            original_fast_max_unconverged_pct = None
            if backend == "triton":
                original_fast_max_unconverged_pct = (
                    observable_decoder.decoder._dec.fast_max_unconverged_pct
                )
                observable_decoder.decoder._dec.fast_max_unconverged_pct = None
            observable_decoder_full = observable_decoder

            # Process in chunks of batch_size (last chunk may need padding)
            difficult_batch_count = 0
            final_processed_count = 0
            for i in range(0, num_difficult, batch):
                difficult_batch = difficult_errors[i : i + batch]
                difficult_batch_size = len(difficult_batch)
                difficult_batch_count += 1
                final_processed_count += difficult_batch_size

                # Pad to original batch size only if needed (for last batch)
                needs_padding = difficult_batch_size < batch
                if needs_padding:
                    padding_size = batch - difficult_batch_size
                    dummy_errors = np.zeros(
                        (padding_size, difficult_batch.shape[1]), dtype=np.uint8
                    )
                    difficult_batch_padded = np.concatenate([difficult_batch, dummy_errors], axis=0)
                    log.info(
                        f"  Processing final difficult batch {difficult_batch_count}: {difficult_batch_size} syndromes + {padding_size} padding = {batch} total"
                    )
                else:
                    difficult_batch_padded = difficult_batch
                    log.info(
                        f"  Processing final difficult batch {difficult_batch_count}: {difficult_batch_size} syndromes"
                    )

                obs_det_full = (
                    observable_decoder_full.from_errors_decode_observables_detailed_batch(
                        difficult_batch_padded.astype(np.uint8), parallel=parallel
                    )
                )
                total_decode_shots += difficult_batch_size

                obs_det_actual = obs_det_full[:difficult_batch_size]
                iters_arr_full = np.array([r.iterations for r in obs_det_actual], dtype=float)
                bp_iters_all.extend(iters_arr_full.tolist())
                if mode == "relay" and set_max_iter > 0:
                    legs_all.extend(
                        1.0 + np.maximum(0.0, (iters_arr_full - pre_iter) / set_max_iter)
                    )

                errs_full = int(sum(r.error_detected for r in obs_det_actual))
                total_errors += errs_full
                total_shots += difficult_batch_size

                # Accumulate quality metrics from final difficult batch
                for r in obs_det_actual:
                    total_unconverged_no_error += r.unconverged_no_error
                    total_better_decoding_quality_error += r.better_decoding_quality_error
                    total_worse_decoding_quality_error += r.worse_decoding_quality_error

                log.info(
                    f"  Final difficult batch {difficult_batch_count} complete: {errs_full} errors"
                )

            # Restore original fast_max_unconverged_pct
            if backend == "triton":
                observable_decoder.decoder._dec.fast_max_unconverged_pct = (
                    original_fast_max_unconverged_pct
                )

            # Measure end time and add to total_decode_ns (includes all overhead)
            final_difficult_batch_end_ns = time.perf_counter_ns()
            total_decode_ns += final_difficult_batch_end_ns - final_difficult_batch_start_ns

            # Ensure we processed all remaining unconverged cases
            log.info(
                f"[Final Difficult Case Collection] Complete: processed {final_processed_count} cases in {difficult_batch_count} batch(es)"
            )
            if final_processed_count != num_difficult:
                log.error(
                    f"[Final Difficult Case Collection] MISMATCH: Expected to process {num_difficult} cases but processed {final_processed_count}!"
                )
        log.info(f"")

    # Debug: Log total shots accumulation for diagnosis
    log.debug(
        f"Final total_shots: {total_shots}, total_samples_processed: {total_samples_processed}, batch_count: {batch_count}"
    )

    # Community standard: count unconverged shots as bad trials
    # regardless of whether the decoded observables happened to match.
    total_errors += total_unconverged_no_error

    # Rates and averages
    logical_error_rate = total_errors / total_shots if total_shots > 0 else 0.0
    per_cycle_logical_error_rate = 1 - (1 - logical_error_rate) ** (1 / rounds)

    obs_count = dem.num_observables
    if basis.lower() == "xz":
        k_guess = obs_count // 2
    else:
        k_guess = obs_count
    per_round_per_qubit_rate = logical_error_rate / (rounds * max(1, k_guess))

    if bp_iters_all:
        # Verify that we have the correct number of iteration counts
        if len(bp_iters_all) != total_shots:
            log.warning(
                f"Mismatch: bp_iters_all has {len(bp_iters_all)} entries but total_shots is {total_shots}"
            )
            # Converged cases from fast pass + unconverged cases from difficult batch should equal total_shots
            if fast_max_unconverged_pct is not None:
                log.error(
                    f"[BUG DETECTED] With fast_max_unconverged_pct={fast_max_unconverged_pct}: "
                    f"Iteration count mismatch! This suggests unconverged cases may not be counted correctly."
                )
        avg_bp_iterations = float(np.mean(bp_iters_all))
        avg_legs = float(np.mean(legs_all)) if (mode == "relay" and legs_all) else 0.0
        # Debug: Log iteration statistics when using fast_max_unconverged_pct
        if fast_max_unconverged_pct is not None:
            log.info(
                f"[fast_max_unconverged_pct={fast_max_unconverged_pct}] Iteration stats: total_shots={total_shots}, "
                f"bp_iters_all length={len(bp_iters_all)}, avg_iterations={avg_bp_iterations:.2f}, "
                f"min={min(bp_iters_all):.1f}, max={max(bp_iters_all):.1f}"
            )
            if len(bp_iters_all) != total_shots:
                log.error(
                    f"[fast_max_unconverged_pct={fast_max_unconverged_pct}] CRITICAL: Iteration count mismatch detected!"
                )
            # Warn if we used estimated shots
            if using_estimated_shots:
                log.warning(
                    f"[Progressive LER] Final statistics use ESTIMATED max_shots={estimated_max_shots} (original was {max_shots})"
                )
                log.warning(
                    f"[Progressive LER] Statistics are based on progressive LER estimation and may differ from full run"
                )
    else:
        avg_bp_iterations = 0.0
        avg_legs = 0.0

    # Aggregate Hamming weight stats
    hamming_weight_stats = None
    hamming_weight_arr = (
        np.concatenate(hamming_weight_all) if hamming_weight_all else np.array([], dtype=np.int32)
    )
    if hamming_weight_arr.size > 0:
        hamming_weight_stats = {
            "total": int(hamming_weight_arr.sum()),
            "count": int(hamming_weight_arr.size),
            "mean": float(hamming_weight_arr.mean()),
            "median": float(np.median(hamming_weight_arr)),
            "max": int(hamming_weight_arr.max()),
            "min": int(hamming_weight_arr.min()),
            "std": float(hamming_weight_arr.std()),
        }

    # Calculate batch_per_cpu for Rust backend
    if backend == "rust" and parallel:
        try:
            import multiprocessing

            num_cpu_cores = int(multiprocessing.cpu_count())
            batch_per_cpu_str = format_batch_per_worker(batch, num_cpu_cores)
        except Exception:
            batch_per_cpu_str = None
    else:
        batch_per_cpu_str = None

    # Timing
    end_time = time.time()
    results = {
        "shots": int(total_shots),
        "logical_errors": int(total_errors),
        "logical_error_rate": float(logical_error_rate),
        "per_cycle_logical_error_rate": float(per_cycle_logical_error_rate),
        "per_round_per_qubit_rate": float(per_round_per_qubit_rate),
        "avg_bp_iterations": float(avg_bp_iterations),
        "decode_quality": {
            "unconverged_no_error": int(total_unconverged_no_error),
            "unconverged_logical_error": int(total_unconverged_logical_error),
            "better_decoding_quality_error": int(total_better_decoding_quality_error),
            "worse_decoding_quality_error": int(total_worse_decoding_quality_error),
        },
        "config": {
            "circuit": str(circuit),
            "distance": int(distance),
            "rounds": int(rounds),
            "error_rate": float(error_rate),
            "batch": int(batch),
            "basis": str(basis),
        },
    }

    # Add batch distribution metrics
    if batch_per_gpu_str is not None:
        results["batch_per_gpu"] = batch_per_gpu_str

    if batch_per_cpu_str is not None:
        results["batch_per_cpu"] = batch_per_cpu_str
    log.debug(
        "Summary definitions: logical_error_rate = logical_errors/shots; per_cycle_logical_error_rate rescales per shot to per cycle via rounds;"
        " avg_bp_iterations is the mean number of BP message-passing iterations per shot."
    )
    if mode == "relay":
        results["avg_legs"] = float(avg_legs)
        results["config"].update(
            {
                "gamma0": float(gamma0),
                "num_sets": int(num_sets),
                "stop_nconv": int(stop_nconv),
                "gamma_dist_interval": (float(gamma_dist_min), float(gamma_dist_max)),
            }
        )
    if hamming_weight_stats is not None:
        results["hamming_weight"] = hamming_weight_stats

    # Scheduling info
    results["scheduling"] = scheduling
    results["layered_k_iters"] = layered_k_iters
    results["layered_block_policy"] = layered_block_policy
    results["layered_num_blocks"] = layered_num_blocks
    results["max_intra_block_overlap"] = None  # Default

    # Get max_intra_block_overlap from Triton decoder if available
    if backend == "triton" and scheduling == "layered":
        try:
            dec = observable_decoder.decoder._dec
            if dec.block_overlap_stats is not None:
                results["max_intra_block_overlap"] = dec.block_overlap_stats.get(
                    "max_intra_block_overlap"
                )
        except Exception:
            pass

    # Per-shot and totals
    if start_time and end_time and total_shots > 0:
        total_runtime_seconds = end_time - start_time
        results["runtime_per_shot_us"] = float((total_runtime_seconds * 1e6) / total_shots)
        results["total_runtime_us"] = float(total_runtime_seconds * 1e6)
    if total_decode_shots > 0:
        total_iterations = float(np.sum(bp_iters_all)) if bp_iters_all else 0.0
        total_legs_for_metrics = float(np.sum(legs_all)) if (mode == "relay" and legs_all) else None
        results.update(
            _compute_decoder_timing_metrics(
                total_decode_ns=total_decode_ns,
                total_decode_shots=total_decode_shots,
                total_iterations=total_iterations,
                total_legs=total_legs_for_metrics,
            )
        )

    # Attach diagnostic
    results["per_cycle_ler_alt_half_rounds"] = float(
        1 - (1 - logical_error_rate) ** (1 / max(1, 2 * rounds))
    )
    results["determinism_hash"] = hasher.hexdigest()



    return results


def run_plain_bp_experiment(
    circuit,
    basis,
    distance,
    rounds,
    error_rate,
    set_max_iter=200,
    alpha=None,
    target_errors=20,
    batch=2000,
    max_shots=1000000,
    parallel=True,
    seed=0,
    backend: str | None = None,
    perf: str = "default",
    algo: str = "plain",
    dtype: str = "fp16",
):
    """Deprecated: use run_bp_experiment_core with mode='plain'."""
    return run_bp_experiment_core(
        mode="plain",
        circuit=circuit,
        basis=basis,
        distance=distance,
        rounds=rounds,
        error_rate=error_rate,
        gamma0=None,
        pre_iter=0,
        num_sets=0,
        set_max_iter=set_max_iter,
        gamma_dist_min=None,
        gamma_dist_max=None,
        stop_nconv=1,
        alpha=alpha,
        target_errors=target_errors,
        batch=batch,
        max_shots=max_shots,
        parallel=parallel,
        seed=seed,
        backend=backend,
        perf=perf,
        data_type=dtype,
    )


def main():
    """Main function for Relay-BP detailed analysis script."""
    args = parse_args()

    # Print experiment start information
    level = logging.DEBUG if args.debug else getattr(logging, args.log_level.upper(), logging.INFO)
    setup_logging(level=level)

    log.info("RELAY-BP DETAILED DECODER EXPERIMENT")
    log.info(f"Experiment started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Print all configuration options
    log.info("CONFIGURATION OPTIONS:")

    # Circuit parameters
    log.info("Circuit Parameters:")
    log.info(f"  Circuit: {args.circuit}")
    log.info(f"  Distance: {args.distance}")
    log.info(f"  Rounds: {args.rounds}")
    log.info(f"  Error Rate: {args.error_rate}")
    log.info(f"  Basis: {args.basis}")
    log.info(f"  Mode: {args.mode}")
    log.info(f"  Perf: {args.perf}")
    log.info(f"  Backend: {args.backend or 'rust'}")
    log.info(f"  Data type: {args.data_type}")

    if args.mode == "relay":
        # Relay-BP parameters
        log.info("Relay-BP Parameters:")
        if args.fast_max_unconverged_pct is not None:
            log.info(f"  Fast max unconverged pct (early exit): {args.fast_max_unconverged_pct}")
        log.info(f"  Gamma0: {args.gamma0}")
        log.info(f"  Pre-iterations: {args.pre_iter}")
        log.info(f"  Number of Sets: {args.num_sets}")
        log.info(f"  Set Max Iterations: {args.set_max_iter}")
        if args.explicit_gammas:
            log.info(
                f"  Explicit Gammas: {args.explicit_gammas} (uniform random sampling disabled)"
            )
        if args.gamma_dist_by_error:
            log.info(f"  Gamma Distribution by error (file): {args.gamma_dist_by_error}")
        if not args.explicit_gammas and not args.gamma_dist_by_error:
            log.info(f"  Gamma Distribution: [{args.gamma_dist_min}, {args.gamma_dist_max}]")
        log.info(f"  Stop N Converged: {args.stop_nconv}")
    else:
        # Plain-BP parameters
        log.info("Plain-BP Parameters:")
        log.info(f"  Set Max Iterations: {args.set_max_iter}")
        log.info(f"  Alpha (normalized min-sum): {args.alpha}")
        log.info(f"  Gamma0 (memory strength): {args.gamma0}")

    # Scheduling parameters (only meaningful for throughput mode)
    log.info("Scheduling Parameters:")
    log.info(f"  Mode: {args.scheduling}")
    if args.scheduling != "flood-fill":
        log.info(f"  Block Policy: {args.layered_block_policy}")
        log.info(
            f"  Num Blocks: {args.layered_num_blocks if args.layered_num_blocks is not None else 'auto'}"
        )
        if args.layered_k_iters == 0:
            log.info(f"  Layered K Iterations: 0 (flood-fill only)")
        elif args.layered_k_iters < 0:
            log.info(f"  Layered K Iterations: {args.layered_k_iters} (always layered)")
        else:
            log.info(
                f"  Layered K Iterations: {args.layered_k_iters} (layered first, then flood-fill)"
            )
        if args.layered_target_edges_per_block is not None:
            log.info(f"  Target Edges Per Block: {args.layered_target_edges_per_block}")
        if args.layered_block_policy == "permute":
            log.info(f"  Block Seed: {args.layered_block_seed}")

    # Validate that max_shots is divisible by batch_size (to avoid partial batches)
    if args.max_shots % args.batch != 0:
        raise ValueError(
            f"max_shots ({args.max_shots}) must be divisible by batch_size ({args.batch}) "
            f"without remainder. Current remainder: {args.max_shots % args.batch}. "
            f"Please adjust max_shots or batch_size to avoid partial batches."
        )

    # Execution parameters
    log.info("Execution Parameters:")
    log.info(f"  Target Errors: {args.target_errors}")
    log.info(f"  Batch Size: {args.batch}")
    log.info(f"  Max Shots: {args.max_shots}")
    log.info(f"  Parallel: {args.parallel}")
    log.info(f"  Output Format: {args.output_format}")
    log.info(f"  Seed: {args.seed}")

    # Backend and device information
    backend_str = args.backend or "rust"
    if backend_str == "triton":
        # Parse GPU IDs for display and batch_per_gpu reporting
        if args.gpus and args.gpus.strip():
            try:
                device_ids = [int(x.strip()) for x in args.gpus.split(",")]
                device_ids_str = ",".join(map(str, device_ids))
                num_gpus = max(1, len(device_ids))
            except ValueError:
                device_ids_str = "0"
                num_gpus = 1
        else:
            device_ids_str = "0"
            num_gpus = 1
        log.info(f"  Backend: {backend_str} (GPU devices: {device_ids_str})")
        # Report how batch is split across GPUs
        batch_per_gpu_str = format_batch_per_worker(args.batch, num_gpus)
        if batch_per_gpu_str is not None:
            log.info(f"  Batch per GPU: {batch_per_gpu_str} (for {num_gpus} GPU(s))")
    else:
        log.info(f"  Backend: {backend_str}")
        # For Rust backend with parallel=True, estimate batch_per_cpu
        if backend_str == "rust" and args.parallel:
            try:
                import multiprocessing

                num_cpu_cores = max(1, int(multiprocessing.cpu_count()))
                batch_per_cpu_str = format_batch_per_worker(args.batch, num_cpu_cores)
                if batch_per_cpu_str is not None:
                    log.info(
                        f"  Batch per CPU core: {batch_per_cpu_str} (for {num_cpu_cores} core(s))"
                    )
            except Exception:
                pass

    results = run_bp_experiment_core(
        mode=args.mode,
        circuit=args.circuit,
        basis=args.basis,
        distance=args.distance,
        rounds=args.rounds,
        error_rate=args.error_rate,
        gamma0=args.gamma0,
        pre_iter=args.pre_iter,
        num_sets=args.num_sets,
        set_max_iter=args.set_max_iter,
        gamma_dist_min=args.gamma_dist_min,
        gamma_dist_max=args.gamma_dist_max,
        explicit_gammas=args.explicit_gammas,
        gamma_dist_interval_by_error=args.gamma_dist_by_error,
        stop_nconv=args.stop_nconv,
        fast_max_unconverged_pct=args.fast_max_unconverged_pct,
        alpha=args.alpha,
        target_errors=args.target_errors,
        batch=args.batch,
        max_shots=args.max_shots,
        parallel=args.parallel,
        seed=args.seed,
        backend=args.backend,
        perf=args.perf,
        data_type=args.data_type,
        debug=args.debug,
        export_matrices=args.export_matrices,
        gpus=args.gpus,
        scheduling=args.scheduling,
        layered_k_iters=args.layered_k_iters,
        layered_block_policy=args.layered_block_policy,
        layered_num_blocks=args.layered_num_blocks,
        layered_target_edges_per_block=args.layered_target_edges_per_block,
        layered_block_seed=args.layered_block_seed,
        sample_only=args.sample_only,
        hamming_weight=args.hamming_weight,
    )

    if args.sample_only:
        return

    # Output results based on format
    if args.output_format == "simple":
        log.info(f"Shots: {results['shots']}, Logical Errors: {results['logical_errors']}")
        log.info(f"Logical Error Rate: {results['logical_error_rate']:.2e}")
        log.info(f"Per-Cycle Logical Error Rate: {results['per_cycle_logical_error_rate']:.2e}")
        legs_suffix = f" (legs: {results['avg_legs']:.1f})" if "avg_legs" in results else ""
        log.info(f"Average BP Iterations: {results['avg_bp_iterations']:.1f}{legs_suffix}")
        if "hamming_weight" in results:
            hw = results["hamming_weight"]
            log.info(
                f"Decoded Hamming Weight: mean={hw['mean']:.2f}, median={hw['median']:.1f}, "
                f"min={hw['min']}, max={hw['max']}, std={hw['std']:.2f}, total={hw['total']}"
            )
        dq = results["decode_quality"]
        log.info(
            f"Decode Quality: unconverged_no_error={dq['unconverged_no_error']}, "
            f"unconverged_logical_error={dq.get('unconverged_logical_error', 0)}, "
            f"better_quality_error={dq['better_decoding_quality_error']}, "
            f"worse_quality_error={dq['worse_decoding_quality_error']}"
        )
        if "runtime_per_shot_us" in results:
            log.info(f"Runtime per shot: {_format_time_from_us(results['runtime_per_shot_us'])}")
        # Decoder-only timing metrics if available
        if "decoder_runtime_per_shot_us" in results:
            log.info(
                f"Decoder runtime per shot: {_format_time_from_us(results['decoder_runtime_per_shot_us'])}"
            )
        if "decoder_runtime_per_iteration_us" in results:
            log.info(
                f"Decoder runtime per iteration: {_format_time_from_us(results['decoder_runtime_per_iteration_us'])}"
            )
        if "decoder_runtime_per_leg_us" in results:
            log.info(
                f"Decoder runtime per leg: {_format_time_from_us(results['decoder_runtime_per_leg_us'])}"
            )
        if "total_runtime_us" in results:
            log.info(
                f"Total runtime (Monte Carlo): {_format_time_from_us(results['total_runtime_us'])}"
            )
        if "decoder_total_time_us" in results:
            log.info(
                f"Decoder total time: {_format_time_from_us(results['decoder_total_time_us'])}"
            )
        # Throughput metrics
        wall_time_us = results.get("total_runtime_us")
        shots_val = results.get("shots")
        if wall_time_us is not None and shots_val is not None:
            wall_time_s = float(wall_time_us) / 1e6
            shots = float(shots_val)
            if wall_time_s > 0.0 and shots > 0.0:
                log.info(f"Shots per second (end-to-end): {shots / wall_time_s:.2f} shots/s")

        dec_time_us_val = results.get("decoder_total_time_us")
        total_decode_shots_val = results.get("total_decode_shots")
        total_iterations_val = results.get("total_iterations")
        if dec_time_us_val is not None and float(dec_time_us_val) > 0.0:
            dec_time_s = float(dec_time_us_val) / 1e6
            if total_decode_shots_val is not None and float(total_decode_shots_val) > 0.0:
                shots_per_sec = float(total_decode_shots_val) / dec_time_s
                log.info(f"Decoder shots per second: {shots_per_sec:.2f} shots/s")
                # One shot corresponds to a full trajectory of `rounds` syndrome vectors.
                # Report per-round syndrome throughput as an additional metric.
                syndromes_per_sec = shots_per_sec * float(args.rounds)
                log.info(f"Per-round syndrome throughput: {syndromes_per_sec:.2f} syndromes/s")
            if total_iterations_val is not None and float(total_iterations_val) > 0.0:
                log.info(
                    f"Iterations per second: {float(total_iterations_val) / dec_time_s:.2f} it/s"
                )
        log.info(f"Determinism hash: {results.get('determinism_hash', '-')}")


    elif args.output_format == "json":
        import json

        log.info(json.dumps(results, indent=2))

    elif args.output_format == "csv":
        # Print CSV header and data
        if "runtime_per_shot_ns" in results:
            log.info(
                "shots,logical_errors,logical_error_rate,per_cycle_logical_error_rate,avg_bp_iterations,runtime_per_shot_ns,gamma0,num_sets,stop_nconv"
            )
            log.info(
                f"{results['shots']},{results['logical_errors']},{results['logical_error_rate']:.6e},{results['per_cycle_logical_error_rate']:.6e},{results['avg_bp_iterations']:.1f},{results['runtime_per_shot_ns']:.1f},{results['config']['gamma0']},{results['config']['num_sets']},{results['config']['stop_nconv']}"
            )
        else:
            log.info(
                "shots,logical_errors,logical_error_rate,per_cycle_logical_error_rate,avg_bp_iterations,gamma0,num_sets,stop_nconv"
            )
            log.info(
                f"{results['shots']},{results['logical_errors']},{results['logical_error_rate']:.6e},{results['per_cycle_logical_error_rate']:.6e},{results['avg_bp_iterations']:.1f},{results['config']['gamma0']},{results['config']['num_sets']},{results['config']['stop_nconv']}"
            )


if __name__ == "__main__":
    main()
