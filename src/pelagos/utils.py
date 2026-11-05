# (C) Copyright IBM 2025, 2026
# Modified 2026 by AMD to report ROCm device information.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Utility functions for Relay-BP-S decoding.

This module provides utility functions for quantum error correction decoding,
including error prior computation, gamma sampling, and matrix validation.
These functions support the main Relay-BP algorithm implementation.
"""

from __future__ import annotations
import logging
import torch
import numpy as np
import time
from typing import Any, Dict, Optional, Tuple, Union, List
from pathlib import Path
from scipy.sparse import csc_matrix, csr_matrix, coo_matrix
from scipy.sparse.csgraph import connected_components

log = logging.getLogger(__name__)


def bitpack_errors(errors: torch.Tensor, bits_per_word: int = 32) -> torch.Tensor:
    """Pack error bits into words for memory efficiency.

    Args:
        errors: [B, V] binary tensor (0/1)
        bits_per_word: number of bits per packed word

    Returns:
        [B, (V + bits_per_word - 1) // bits_per_word] packed tensor
    """
    B, V = errors.shape
    words_per_batch = (V + bits_per_word - 1) // bits_per_word

    # Reshape to [B, words_per_batch, bits_per_word]
    padded_errors = torch.zeros(
        B, words_per_batch * bits_per_word, dtype=errors.dtype, device=errors.device
    )
    padded_errors[:, :V] = errors

    # Reshape and pack
    packed = padded_errors.view(B, words_per_batch, bits_per_word)

    # Convert to int32 words (CUDA supports lshift on int32)
    result = torch.zeros(B, words_per_batch, dtype=torch.int32, device=errors.device)

    for i in range(bits_per_word):
        result |= packed[:, :, i].to(torch.int32) << i

    return result


def bitunpack_errors(packed: torch.Tensor, V: int, bits_per_word: int = 32) -> torch.Tensor:
    """Unpack error bits from words.

    Args:
        packed: [B, words_per_batch] packed tensor
        V: number of variables
        bits_per_word: number of bits per packed word

    Returns:
        [B, V] binary tensor (0/1)
    """
    B, words_per_batch = packed.shape

    # Convert to binary representation
    bits = torch.zeros(B, words_per_batch, bits_per_word, dtype=torch.uint8, device=packed.device)

    for i in range(bits_per_word):
        bits[:, :, i] = ((packed >> i) & 1).to(torch.uint8)

    # Flatten and truncate to V
    errors = bits.view(B, -1)[:, :V]

    return errors


def compute_log_prior_ratios(error_priors: np.ndarray) -> np.ndarray:
    """Compute log prior ratios from error probabilities.

    This function converts error probabilities to log-likelihood ratios used
    in belief propagation, corresponding to the log_prior_ratios computation
    in Rust min_sum.rs.

    Args:
        error_priors: [V] error probabilities in (0, 0.5) for each qubit

    Returns:
        [V] log prior ratios log((1-p)/p) for belief propagation
    """
    # Clip probabilities to avoid numerical issues
    p = np.clip(error_priors, 1e-10, 0.5 - 1e-10)

    # Compute log prior ratios
    log_ratios = np.log((1 - p) / p)

    return log_ratios.astype(np.float32)


def compute_decoding_weights(
    hard_dec: torch.Tensor, log_prior_ratios: torch.Tensor
) -> torch.Tensor:
    """Compute decoding weights for solution selection.

    Args:
        hard_dec: [B, V] hard decisions (0/1)
        log_prior_ratios: [V] log prior ratios

    Returns:
        [B] decoding weights (lower is better)
    """
    # Weight = sum of log prior ratios for error bits
    weights = torch.sum(hard_dec.float() * log_prior_ratios.unsqueeze(0), dim=1)

    return weights


def sample_gamma_uniform(
    rows: int, V: int, gamma_min: float, gamma_max: float, device: str, seed: Optional[int] = None
) -> torch.Tensor:
    """Sample gamma values from uniform distribution for disordered memory.

    This function samples memory strength values for the disordered memory phase
    of Relay-BP, corresponding to the gamma sampling in Rust relay.rs.

    Args:
        rows: number of gamma rows to draw (1 for gamma_draw="broadcast", the batch size
            for "per_row")
        V: number of variables (qubits)
        gamma_min: minimum gamma value for memory strength
        gamma_max: maximum gamma value for memory strength
        device: target GPU device
        seed: random seed for reproducibility. Seeds a generator private to this call;
            the global RNG state is neither used nor changed.

    Returns:
        [rows, V] gamma values for memory mixing
    """
    gen = None
    if seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)

    # Sample uniform random values
    uniform = torch.rand(rows, V, device=device, generator=gen)

    # Scale to [gamma_min, gamma_max]
    gamma = gamma_min + (gamma_max - gamma_min) * uniform

    return gamma


def sample_gamma_scalar(B: int, V: int, gamma_value: float, device: str) -> torch.Tensor:
    """Sample scalar gamma values for ordered memory phase.

    This function creates uniform gamma values for the ordered memory phase
    of Relay-BP, corresponding to the gamma0 parameter in Rust relay.rs.

    Args:
        B: batch size
        V: number of variables (qubits)
        gamma_value: uniform gamma value for memory strength
        device: target GPU device

    Returns:
        [B, V] gamma values (all identical)
    """
    return torch.full((B, V), gamma_value, device=device)


def sample_gamma_per_variable(
    rows: int,
    V: int,
    gamma_intervals: torch.Tensor,
    device: str,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Sample gamma values from per-variable uniform intervals.

    Each variable v has its own
    interval [low_v, high_v] and we sample:
        gamma[r, v] ~ Uniform(low_v, high_v)
    independently for each row r.

    Args:
        rows: number of gamma rows to draw (1 for gamma_draw="broadcast", the batch size
            for "per_row")
        V: number of variables (qubits)
        gamma_intervals: [V, 2] tensor with [:,0]=low, [:,1]=high bounds
        device: target GPU device
        seed: random seed for reproducibility. Seeds a generator private to this call;
            the global RNG state is neither used nor changed.

    Returns:
        [rows, V] gamma values for memory mixing
    """
    gen = None
    if seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)

    if gamma_intervals.ndim != 2 or gamma_intervals.shape[0] != V or gamma_intervals.shape[1] != 2:
        raise ValueError(
            f"gamma_intervals must have shape (V={V}, 2), got {tuple(gamma_intervals.shape)}"
        )

    # [rows, V] uniform samples in [0, 1]
    uniform = torch.rand(rows, V, device=device, generator=gen)

    # Broadcast per-variable intervals: low, high are [1, V]
    low = gamma_intervals[:, 0].unsqueeze(0)
    high = gamma_intervals[:, 1].unsqueeze(0)

    gamma = low + (high - low) * uniform
    return gamma


GAMMA_DRAW_MODES = ("broadcast", "per_row")


def validate_error_priors(error_priors: np.ndarray) -> np.ndarray:
    """Validate and clip error priors.

    Args:
        error_priors: [V] error probabilities

    Returns:
        [V] validated and clipped error probabilities
    """
    # Check shape
    if error_priors.ndim != 1:
        raise ValueError("error_priors must be 1D array")

    # Check range and clip
    clipped = np.clip(error_priors, 1e-10, 0.5 - 1e-10)

    # Warn if clipping occurred
    if np.any(clipped != error_priors):
        print("Warning: error_priors were clipped to avoid numerical issues")

    return clipped


def validate_csr_matrix(H_csr) -> bool:
    """Validate CSR matrix format and properties; coerce CSR if needed."""
    from scipy.sparse import csr_matrix, issparse

    if not issparse(H_csr):
        raise ValueError("Input must be a scipy.sparse matrix (any format)")

    # Coerce to CSR if it's e.g. csr_array/coo/csc/etc.
    if not isinstance(H_csr, csr_matrix):
        H_csr = H_csr.tocsr()

    if H_csr.shape[0] == 0 or H_csr.shape[1] == 0:
        raise ValueError("Matrix cannot be empty")
    if H_csr.nnz == 0:
        raise ValueError("Matrix cannot have zero non-zero elements")
    if not np.all((H_csr.data == 0) | (H_csr.data == 1)):
        print("Warning: Matrix contains non-binary values, will be treated as binary")

    return True


def get_device_info(device: str) -> dict:
    """Get device information for debugging.

    Args:
        device: device string ("cuda" or "rocm")

    Returns:
        Dictionary with device information
    """
    if device == "cuda":
        if torch.cuda.is_available():
            return {
                "device": device,
                "available": True,
                "name": torch.cuda.get_device_name(),
                "memory_total": torch.cuda.get_device_properties(0).total_memory,
                "memory_allocated": torch.cuda.memory_allocated(),
                "memory_reserved": torch.cuda.memory_reserved(),
            }
        else:
            return {"device": device, "available": False}

    elif device == "rocm":
        if not torch.version.hip or not torch.cuda.is_available():
            return {"device": device, "available": False}

        # PyTorch exposes HIP devices through the torch.cuda API.
        device_index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device_index)
        return {
            "device": device,
            "available": True,
            "name": properties.name,
            "memory_total": properties.total_memory,
            "memory_allocated": torch.cuda.memory_allocated(device_index),
            "memory_reserved": torch.cuda.memory_reserved(device_index),
            "hip_version": torch.version.hip,
        }

    else:
        raise ValueError(f"Unsupported device: {device}")


def format_memory_size(bytes: int) -> str:
    """Format memory size in human-readable format.

    Args:
        bytes: memory size in bytes

    Returns:
        Formatted string
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes < 1024.0:
            return f"{bytes:.1f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.1f} PB"


def _deg_stats(arr: np.ndarray) -> Tuple[int, float, int]:
    if arr.size == 0:
        return 0, 0.0, 0
    return int(arr.min()), float(arr.mean()), int(arr.max())


def _hist_small_ints(deg: np.ndarray, bins=(2, 3, 4, 5, 6, 8, 12, 16, 24, 32)) -> Dict[str, int]:
    if deg.size == 0:
        return {}
    out = {}
    prev = 0
    for b in bins:
        out[f"≤{b}"] = int((deg <= b).sum() - prev)
        prev = (deg <= b).sum()
    out[f">{bins[-1]}"] = int((deg > bins[-1]).sum())
    return out


def _rank_gf2_via_reduction(H_csr: csr_matrix) -> int:
    """
    Compute rank over GF(2) via sparse row-reduction.
    WARNING: potentially expensive for very large matrices; enable only if you need it.
    """
    # convert to COO for simple row-ops, then to dense bit rows per pivot band
    H = H_csr.copy().astype(np.uint8)
    M, N = H.shape
    pivcol = 0
    rank = 0
    indptr, indices = H.indptr, H.indices
    # Build list of sets for quick XOR row ops
    rows = [set(indices[indptr[i] : indptr[i + 1]]) for i in range(M)]
    while pivcol < N and rank < M:
        # find a row with a 1 in pivcol at or below 'rank'
        pivot = -1
        for r in range(rank, M):
            if pivcol in rows[r]:
                pivot = r
                break
        if pivot == -1:
            pivcol += 1
            continue
        # swap rows (as sets)
        if pivot != rank:
            rows[rank], rows[pivot] = rows[pivot], rows[rank]
        # eliminate below
        for r in range(rank + 1, M):
            if pivcol in rows[r]:
                rows[r] ^= rows[rank]  # symmetric diff = XOR over GF(2)
        rank += 1
        pivcol += 1
    return rank


def describe_check_matrices(
    check_matrices,
    *,
    dem: Optional[Any] = None,  # expects .num_detectors / .num_observables if given
    rounds: Optional[int] = None,
    compute_components: bool = False,  # compute connected components of Tanner graph
    compute_rank: bool = False,  # GF(2) rank of H (can be expensive)
    file=None,
    return_dict: bool = False,
) -> Dict[str, Any]:
    """
    Derive & print detailed stats from real matrices only (no name heuristics).
    """
    H = check_matrices.check_matrix  # usually CSC
    O = check_matrices.observables_matrix  # usually CSC
    pri = np.asarray(check_matrices.error_priors)

    # Ensure both CSR and CSC forms
    H_csr: csr_matrix = H if isinstance(H, csr_matrix) else H.tocsr(copy=False)
    H_csc: csc_matrix = H if isinstance(H, csc_matrix) else H.tocsc(copy=False)

    M, N = H_csr.shape
    E = int(H_csr.nnz)
    density = (E / (M * N)) if (M and N) else 0.0

    # Tanner degrees
    row_deg = np.diff(H_csr.indptr)  # errors per check
    col_deg = np.diff(H_csc.indptr)  # checks per error
    rmin, rmean, rmax = _deg_stats(row_deg)
    cmin, cmean, cmax = _deg_stats(col_deg)
    is_row_reg = rmin == rmax and M > 0
    is_col_reg = cmin == cmax and N > 0
    regularity = (
        f"(d_v={cmin}, d_c={rmin})-regular"
        if (is_row_reg and is_col_reg)
        else ("row-regular" if is_row_reg else ("column-regular" if is_col_reg else None))
    )
    col_hist = _hist_small_ints(col_deg)
    row_hist = _hist_small_ints(row_deg)

    # Observables matrix stats
    O_rows, O_cols = O.shape
    O_nnz = int(O.nnz)
    O_density = (O_nnz / (O_rows * O_cols)) if (O_rows and O_cols) else 0.0

    # Priors
    finite_mask = np.isfinite(pri)
    pri_stats = {
        "length": int(pri.size),
        "finite_fraction": float(finite_mask.mean() if pri.size else 1.0),
        "min": float(np.min(pri[finite_mask])) if finite_mask.any() else float("nan"),
        "mean": float(np.mean(pri[finite_mask])) if finite_mask.any() else float("nan"),
        "max": float(np.max(pri[finite_mask])) if finite_mask.any() else float("nan"),
    }

    # Per-cycle detectors from DEM if provided
    dets_total = getattr(dem, "num_detectors", None) if dem is not None else None
    per_cycle_detectors = (dets_total // rounds) if (dets_total is not None and rounds) else None

    # Optional: connected components of Tanner graph (bipartite graph with M+N nodes)
    comp_info = None
    if compute_components and E > 0:
        # build bipartite adjacency: blocks [0:M) checks, [M:M+N) variables
        H_coo: coo_matrix = H_csr.tocoo(copy=False)
        rows = H_coo.row
        cols = H_coo.col + M
        data = np.ones_like(rows, dtype=np.uint8)
        # symmetric adjacency
        i_idx = np.concatenate([rows, cols])
        j_idx = np.concatenate([cols, rows])
        A = coo_matrix(
            (np.ones_like(i_idx, dtype=np.uint8), (i_idx, j_idx)), shape=(M + N, M + N)
        ).tocsr()
        n_comp, labels = connected_components(A, directed=False, return_labels=True)
        comp_sizes = np.bincount(labels)
        comp_info = {
            "num_components": int(n_comp),
            "component_sizes_desc": np.sort(comp_sizes)[::-1].tolist(),
        }

    # Optional: GF(2) rank and k ≈ N - rank(H)
    rank_info = None
    if compute_rank and E > 0:
        rank_h = _rank_gf2_via_reduction(H_csr.astype(np.uint8))
        k_est = int(N - rank_h)
        rank_info = {"rank_gf2": int(rank_h), "k_estimate": k_est}

    # ----- printing -----
    pf = print if file is None else (lambda *a, **k: print(*a, **k, file=file))

    pf(f"H (checks × errors): {M} × {N}")
    pf(f"  nonzeros (E): {E} | density: {density:.9f}")
    pf(f"  avg column weight (checks per error): {cmean:.2f}")
    pf(f"  avg row    weight (errors per check): {rmean:.2f}")
    pf(f"  column weight min/mean/max: {cmin} / {cmean:.2f} / {cmax}")
    pf(f"  row    weight min/mean/max: {rmin} / {rmean:.2f} / {rmax}")
    if regularity:
        pf(f"  regularity: {regularity}")
    pf(f"  column weight hist: {col_hist}")
    pf(f"  row    weight hist: {row_hist}")

    pf(f"O (observables × errors): {O_rows} × {O_cols} (nnz={O_nnz}, density={O_density:.9f})")

    if rounds is not None:
        if per_cycle_detectors is not None:
            pf(
                f"per-cycle detectors: {per_cycle_detectors}  (from DEM: {dets_total} / rounds: {rounds})"
            )
        else:
            pf(f"per-cycle detectors: unavailable (need DEM with num_detectors)")

    if rank_info:
        pf(f"rank_GF2(H) = {rank_info['rank_gf2']}  =>  k_est ≈ {rank_info['k_estimate']}")

    if comp_info:
        pf(
            f"Tanner graph connected components: {comp_info['num_components']} "
            f"(largest sizes: {comp_info['component_sizes_desc'][:5]})"
        )

    pf(
        f"priors: N={pri_stats['length']}, finite={pri_stats['finite_fraction'] * 100:.1f}% | "
        f"min/mean/max={pri_stats['min']}/{pri_stats['mean']}/{pri_stats['max']}"
    )

    out = {
        "H": {
            "shape": (M, N),
            "nonzeros": E,
            "density": density,
            "avg_column_weight": cmean,
            "avg_row_weight": rmean,
            "column_weight_stats": {"min": cmin, "mean": cmean, "max": cmax},
            "row_weight_stats": {"min": rmin, "mean": rmean, "max": rmax},
            "column_weight_hist": col_hist,
            "row_weight_hist": row_hist,
            "regularity": regularity,
        },
        "O": {"shape": (O_rows, O_cols), "nonzeros": O_nnz, "density": O_density},
        "priors": pri_stats,
        "rounds": rounds,
        "per_cycle_detectors": per_cycle_detectors,
        "rank": rank_info,
        "components": comp_info,
    }
    return out if return_dict else out


def warmup_decoder_options_overrides(options: dict) -> dict:
    """Iteration-limit overrides that turn any relay-bp config into a warmup config."""
    is_relay = (options.get("num_sets") or 0) > 0
    return {
        "pre_iter": 1,
        "num_sets": 1 if is_relay else 0,
        "set_max_iter": 1 if is_relay else 0,
        "stop_nconv": 1,
    }


def warmup_build_and_decode(
    *,
    check_matrices: Any,
    dem: Any,
    backend: str,
    algo: str,
    perf: str,
    dtype: str,
    pre_iter: int,
    num_sets: int,
    set_max_iter: int,
    stop_nconv: int,
    gamma0: float | None,
    gamma_dist_interval: tuple[float, float] | None,
    alpha: float | None,
    beta: float | None,
    device: Union[str, List[int], int] = [0],
    seed: int = 0,
    repeats: int = 3,
    batch: int = 8,
    parallel: bool = True,
) -> None:
    """
    Build a fresh decoder with iteration limits forced to 1 and run warmup decodes.
    Prints start/end and per-repeat timing.
    """
    sampler = dem.compile_sampler(seed=seed)

    if backend == "triton":
        from .adapter import RelayDecoder as _RelayDecoder
        from .adapter import ObservableDecoderRunner as _ObservableDecoderRunner

        _plain = (num_sets == 0) or (algo == "plain")

        # Build Min-sum and Relay config dicts mirroring the Rust bindings, but
        # with iteration limits reduced to 1 for warmup.
        # Note: Preserve alpha value even when _plain=True (num_sets=0) to ensure
        # warmup decoder uses same configuration as actual decoder for autotuning cache.
        msl_config = {
            "error_priors": check_matrices.error_priors,
            "alpha": None if (algo == "plain") else (alpha if alpha not in (None, 0.0) else 0.0),
            "gamma0": (0.0 if _plain else (gamma0 or 0.0)),
        }
        relay_config = {
            "pre_iter": 1,
            "num_sets": 0 if _plain else num_sets,
            "set_max_iter": 0 if _plain else 1,
            "gamma_dist_interval": (0.0, 0.0) if _plain else (gamma_dist_interval or (0.0, 0.0)),
            "stop_nconv": 1 if _plain else max(1, int(stop_nconv)),
            "logging": False,
            "seed": int(seed or 0),
        }
        backend_config = {
            "dtype": dtype,
            "perf": perf,
            "device": device,
            "algo": ("plain" if _plain else "relay"),
        }

        dec = _RelayDecoder(
            check_matrices.check_matrix,
            msl_config=msl_config,
            relay_config=relay_config,
            data_type="float64" if dtype == "fp64" else "float32",
            backend_config=backend_config,
        )

        observable_decoder = _ObservableDecoderRunner(
            dec,
            check_matrices.observables_matrix,
            include_decode_result=True,
        )
    else:
        import relay_bp as _rb

        suffix = {"fp16": "F16", "fp32": "F32", "fp64": "F64"}.get(dtype)
        if suffix is None:
            raise ValueError("Unsupported dtype for rust warmup: " + str(dtype))
        if algo == "plain":
            minsum_name = f"MinSumBPDecoder{suffix}"
            if not hasattr(_rb, minsum_name):
                raise ValueError(f"Requested Rust plain decoder '{minsum_name}' not available")
            MinSum = getattr(_rb, minsum_name)
            dec = MinSum(
                check_matrices.check_matrix,
                error_priors=check_matrices.error_priors,
                max_iter=1,
                alpha=None,
                gamma0=None,
            )
        else:
            relay_name = f"RelayDecoder{suffix}"
            if not hasattr(_rb, relay_name):
                raise ValueError(f"Requested Rust relay decoder '{relay_name}' not available")
            Relay = getattr(_rb, relay_name)
            dec = Relay(
                check_matrices.check_matrix,
                error_priors=check_matrices.error_priors,
                gamma0=(gamma0 or 0.0),
                pre_iter=1,
                num_sets=max(1, int(num_sets)),
                set_max_iter=1,
                gamma_dist_interval=(gamma_dist_interval or (0.0, 0.0)),
                stop_nconv=max(1, int(stop_nconv)),
                seed=int(seed or 0),
                alpha=(alpha if alpha not in (None, 0.0) else 0.0),
                beta=beta,
            )
        from relay_bp import ObservableDecoderRunner as _ObservableDecoderRunner

        observable_decoder = _ObservableDecoderRunner(
            dec,
            check_matrices.observables_matrix,
            include_decode_result=True,
        )

    total_start = time.perf_counter()
    print(
        f"[warmup] starting (repeats={int(repeats)}, batch={int(batch)}, backend={backend}, algo={algo}, perf={perf}, dtype={dtype})"
    )
    for i in range(int(repeats)):
        rep_start = time.perf_counter()
        _det, _obs, _warm_errors = sampler.sample(int(batch), return_errors=True)
        _ = observable_decoder.from_errors_decode_observables_detailed_batch(
            _warm_errors.astype(np.uint8, copy=False), parallel=parallel
        )
        rep_end = time.perf_counter()
        ms = (rep_end - rep_start) * 1e3
        print(f"[warmup] repeat {i + 1}/{int(repeats)} took {ms:.3f} ms")
    total_end = time.perf_counter()
    total_ms = (total_end - total_start) * 1e3
    print(f"[warmup] finished in {total_ms:.3f} ms")


def load_explicit_gammas(
    explicit_gammas: Union[str, np.ndarray, None],
    num_variables: int,
    backend: str = "triton",
    num_sets: Optional[int] = None,
    mode: str = "relay",
) -> Optional[np.ndarray]:
    """Load and validate explicit gamma values for Relay-BP.

    This function loads explicit gamma values from a file or numpy array,
    validates the shape, and converts to the appropriate dtype for the backend.

    Args:
        explicit_gammas: File path (str), numpy array, or None
        num_variables: Number of variables (V) in the code (for validation)
        backend: Backend type ("rust" or "triton") - determines dtype conversion
        num_sets: Number of relay sets (for warning if K < num_sets)
        mode: Decoding mode ("relay" or "plain") - only used for warnings

    Returns:
        numpy array of shape (K, V) with dtype float64 (rust) or float32 (triton),
        or None if explicit_gammas is None

    Raises:
        FileNotFoundError: If file path doesn't exist
        ValueError: If file format is unsupported or shape is invalid
        TypeError: If explicit_gammas is not str, np.ndarray, or None
    """
    if explicit_gammas is None:
        return None

    explicit_gammas_array = None

    if isinstance(explicit_gammas, str):
        # Load from file
        gamma_path = Path(explicit_gammas)
        if not gamma_path.exists():
            raise FileNotFoundError(f"Explicit gammas file not found: {explicit_gammas}")

        if gamma_path.suffix == ".npy":
            explicit_gammas_array = np.load(gamma_path)
        elif gamma_path.suffix == ".npz":
            # For .npz, try common keys or use first array
            data = np.load(gamma_path)
            if "gammas" in data:
                explicit_gammas_array = data["gammas"]
            elif "arr_0" in data:
                explicit_gammas_array = data["arr_0"]
            else:
                keys = list(data.keys())
                if keys:
                    explicit_gammas_array = data[keys[0]]
                else:
                    raise ValueError(f"No arrays found in .npz file: {explicit_gammas}")
            data.close()
        elif gamma_path.suffix == ".csv":
            try:
                import pandas as pd

                explicit_gammas_array = pd.read_csv(gamma_path, header=None).values
            except ImportError:
                raise ImportError(
                    "pandas is required to load .csv files. Install with: pip install pandas"
                )
        else:
            raise ValueError(
                f"Unsupported file format for explicit gammas: {gamma_path.suffix}. Use .npy, .npz, or .csv"
            )

        log.info(
            f"Loaded explicit gammas from '{explicit_gammas}': shape {explicit_gammas_array.shape}"
        )
    elif isinstance(explicit_gammas, np.ndarray):
        explicit_gammas_array = explicit_gammas
    else:
        raise TypeError(
            f"explicit_gammas must be a file path (str) or numpy array, got {type(explicit_gammas)}"
        )

    # Validate shape: must be (K, V) where V matches number of variables
    if explicit_gammas_array.ndim != 2:
        raise ValueError(
            f"explicit_gammas must be 2D array (K, V), got shape {explicit_gammas_array.shape}"
        )
    if explicit_gammas_array.shape[1] != num_variables:
        raise ValueError(
            f"explicit_gammas must have V={num_variables} columns (number of variables), "
            f"got {explicit_gammas_array.shape[1]}"
        )

    K = explicit_gammas_array.shape[0]
    if mode == "relay" and num_sets is not None and num_sets > 0 and K < num_sets:
        log.warning(
            f"Number of gamma sets ({K}) is less than num_sets ({num_sets}). "
            f"Gammas will cycle (relay leg l uses row (l + 1) % {K})."
        )

    # Convert to appropriate dtype for backend
    if backend == "rust":
        explicit_gammas_array = explicit_gammas_array.astype(np.float64)
    else:
        explicit_gammas_array = explicit_gammas_array.astype(np.float32)

    return explicit_gammas_array


def load_gamma_dist_by_error(
    gamma_dist_by_error: Union[str, np.ndarray, None],
    num_variables: int,
) -> Optional[np.ndarray]:
    """Load and validate per-variable gamma intervals for Relay-BP.

    Returns a (V, 2) array of [low, high] intervals per variable.
    """
    if gamma_dist_by_error is None:
        return None

    arr: Optional[np.ndarray] = None

    if isinstance(gamma_dist_by_error, str):
        gamma_path = Path(gamma_dist_by_error)
        if not gamma_path.exists():
            raise FileNotFoundError(f"gamma_dist_by_error file not found: {gamma_dist_by_error}")

        if gamma_path.suffix == ".npy":
            arr = np.load(gamma_path)
        elif gamma_path.suffix == ".npz":
            data = np.load(gamma_path)
            if "gammas" in data:
                arr = data["gammas"]
            elif "arr_0" in data:
                arr = data["arr_0"]
            else:
                keys = list(data.keys())
                if keys:
                    arr = data[keys[0]]
                else:
                    raise ValueError(f"No arrays found in .npz file: {gamma_dist_by_error}")
            data.close()
        elif gamma_path.suffix == ".csv":
            try:
                import pandas as pd

                arr = pd.read_csv(gamma_path, header=None).values
            except ImportError:
                raise ImportError(
                    "pandas is required to load .csv files for gamma_dist_by_error. "
                    "Install with: pip install pandas"
                )
        else:
            raise ValueError(
                f"Unsupported file format for gamma_dist_by_error: {gamma_path.suffix}. "
                "Use .npy, .npz, or .csv"
            )

        log.info(f"Loaded gamma_dist_by_error from '{gamma_dist_by_error}': shape {arr.shape}")
    elif isinstance(gamma_dist_by_error, np.ndarray):
        arr = gamma_dist_by_error
    else:
        raise TypeError(
            f"gamma_dist_by_error must be a file path (str) or numpy array, got {type(gamma_dist_by_error)}"
        )

    # Validate shape: (V, 2)
    if arr.ndim != 2 or arr.shape[0] != num_variables or arr.shape[1] != 2:
        raise ValueError(
            f"gamma_dist_by_error must have shape (V={num_variables}, 2), got {arr.shape}"
        )
    if np.any(arr[:, 0] > arr[:, 1]):
        raise ValueError("gamma_dist_by_error: lower bounds must be <= upper bounds in each row")

    # Use float32 for compatibility with Triton; Rust bindings will up-cast as needed.
    return arr.astype(np.float32)


def format_batch_per_worker(batch: int, num_workers: int) -> Optional[str]:
    """Format batch distribution per worker as 'X' or 'X-Y'.

    This helper is used to report how a batch of size B is split across
    a given number of workers (GPUs or CPU cores), without encoding any
    backend-specific logic. It returns:

    - None if num_workers <= 0
    - 'X'   if B is evenly divisible by num_workers
    - 'X-Y' if not divisible, indicating some workers process X and some X+1
    """
    if num_workers <= 0:
        return None
    base = batch // num_workers
    remainder = batch % num_workers
    if remainder == 0:
        return str(base)
    return f"{base}-{base + 1}"


def format_batch_per_device_from_decoder(batch: int, decoder) -> Optional[str]:
    """Infer device count from decoder (GPUs) and format batch per device."""
    if decoder is None:
        return None
    try:
        if hasattr(decoder, "_dec") and hasattr(decoder._dec, "num_gpus"):
            num_gpus = int(decoder._dec.num_gpus)
        else:
            num_gpus = 1
    except Exception:
        num_gpus = 1
    return format_batch_per_worker(batch, num_gpus)


def has_rust() -> bool:
    """Return True when the Rust ``relay_bp`` reference decoder is importable.

    The Rust implementation is an optional dependency. Callers gate on this
    rather than catching ImportError at each use site.
    """
    try:
        import relay_bp  # noqa: F401
    except Exception:
        return False
    return True


def compare_with_rust(
    H_csr,
    error_priors: np.ndarray,
    syndromes: np.ndarray,
    **decoder_kwargs,
) -> dict:
    """Decode ``syndromes`` with both backends and compare the results.

    Both decoders receive the same check matrix, priors and configuration, so a
    disagreement reflects a real difference rather than a setup skew.

    Pass every parameter that affects gamma sampling explicitly — in particular
    ``gamma_dist_interval`` alongside ``gamma0`` and ``seed``. If one is left to
    a default the two implementations may pick different disordered memory
    strengths, and the comparison will report a mismatch that is a
    configuration difference rather than a decoder disagreement.

    Args:
        H_csr: Check matrix (C x V) in CSR format.
        error_priors: [V] error probabilities in (0, 0.5).
        syndromes: [B, C] uint8 syndromes. The Triton decoder requires a
            torch tensor on its device; a numpy array is accepted here and
            converted for the Rust side.
        **decoder_kwargs: Forwarded to both decoders. ``data_type`` selects the
            Rust precision variant ("float16" / "float32" / "float64").

    Returns:
        A dict with ``agree``, ``mismatched_shots``, ``total_shots``,
        ``triton_e_hat`` and ``rust_e_hat``.

    Raises:
        RuntimeError: if the Rust decoder is not installed, or the installed
            build lacks the precision variant requested.
        ValueError: on an unsupported ``data_type``.
    """
    if not has_rust():
        raise RuntimeError(
            "compare_with_rust requires the Rust reference decoder. "
            "Install it with the 'rust' extra."
        )

    import relay_bp as _rb

    from .decoder import RelayBPDecoder as _TritonDecoder

    dtype = str(decoder_kwargs.get("data_type", "float32"))
    suffix = {"float16": "F16", "float32": "F32", "float64": "F64"}.get(dtype)
    if suffix is None:
        raise ValueError(f"unsupported data_type for comparison: {dtype!r}")

    rust_cls_name = f"RelayDecoder{suffix}"
    if not hasattr(_rb, rust_cls_name):
        raise RuntimeError(f"the installed relay_bp has no {rust_cls_name}")

    triton_out = _TritonDecoder(
        H_csr=H_csr, error_priors=error_priors, **decoder_kwargs
    ).decode(syndromes)
    _triton_raw = triton_out["errors"] if isinstance(triton_out, dict) else triton_out
    # The Triton decoder returns device tensors; bring them to host for comparison.
    triton_e_hat = (
        _triton_raw.cpu().numpy() if hasattr(_triton_raw, "cpu") else np.asarray(_triton_raw)
    )

    _RUST_KEYS = (
        "pre_iter",
        "num_sets",
        "set_max_iter",
        "gamma0",
        "gamma_dist_interval",
        "stop_nconv",
        "normalized_min_sum_alpha",
        "offset_min_sum_beta",
        "seed",
    )
    rust_kwargs = {k: v for k, v in decoder_kwargs.items() if k in _RUST_KEYS}
    rust_decoder = getattr(_rb, rust_cls_name)(
        H_csr, error_priors=error_priors, **rust_kwargs
    )
    # The Rust decoder is a CPU implementation and needs host-side numpy, while
    # the Triton decoder needs a device tensor. Accept either input form.
    if hasattr(syndromes, "cpu"):
        syndromes_np = syndromes.cpu().numpy()
    else:
        syndromes_np = np.asarray(syndromes)
    rust_e_hat = np.asarray(rust_decoder.decode_batch(syndromes_np))

    mismatched = int(np.count_nonzero((triton_e_hat != rust_e_hat).any(axis=-1)))
    return {
        "agree": mismatched == 0,
        "mismatched_shots": mismatched,
        "total_shots": int(triton_e_hat.shape[0]),
        "triton_e_hat": triton_e_hat,
        "rust_e_hat": rust_e_hat,
    }
