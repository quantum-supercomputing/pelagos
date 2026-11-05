# (C) Copyright IBM 2025, 2026
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Relay-BP-S decoder with Triton GPU kernels.

This module implements the Relay-BP (Relay Belief Propagation) algorithm for quantum error correction
using GPU-accelerated Triton kernels. The algorithm performs ensemble decoding with disordered memory
strengths to improve decoding performance on quantum error correction codes.

The implementation closely follows the Rust reference implementation in crates/relay_bp/src/bp/,
providing GPU acceleration while maintaining algorithmic fidelity.
"""

import math

import torch
import numpy as np
from scipy.sparse import csr_matrix
from typing import Dict, Optional, Tuple, Union, List, Any
import os
from dataclasses import dataclass

from .graph import CSRGraph
from . import autotune as ac
from .kernels import (
    c2v_min_sum_kernel,
    v2c_and_marginals_fused_gamma_kernel,
    parity_per_check_kernel,
    parity_from_hard_kernel,
    parity_from_hard_compact_kernel,
    parity_from_hard_compact_with_zero_kernel,
    init_messages_kernel,
    init_messages_with_priors_EB_kernel,
    c2v_min_sum_btile_kernel,
    v2c_and_gamma_btile_kernel,
    post_from_L_kernel,
    reduce_all_ge_kernel,
    freeze_finished_lanes_kernel,
    unfreeze_finished_lanes_kernel,
    count_active_blocks_kernel,
    compact_active_indices_kernel,
    c2v_min_sum_grid_stride_kernel,
    check_and_select_kernel,
    dtype_str_to_enum,
)
from .utils import (
    compute_log_prior_ratios,
    sample_gamma_uniform,
    sample_gamma_scalar,
    sample_gamma_per_variable,
    GAMMA_DRAW_MODES,
    validate_error_priors,
    validate_csr_matrix,
    bitpack_errors,
)
from .autotune import (
    V2C_BTILE_BLOCK_SIZE,
    build_c2v_configs,
    build_v2c_configs,
    build_parity_configs,
    build_init_messages_configs,
    build_c2v_btile_configs,
    build_v2c_btile_configs,
    build_check_select_configs,
    build_c2v_grid_stride_configs,
)


def _build_check_conflict_adjacency(H_csr: csr_matrix) -> List[np.ndarray]:
    """Build adjacency of the check–check conflict graph from CSR H.

    Nodes are checks (rows). Two checks are adjacent if they share at least one variable.
    """
    # Work in CSC so we can iterate variable → incident checks efficiently
    H_csc = H_csr.tocsc()
    C = H_csr.shape[0]

    adj_sets: List[set[int]] = [set() for _ in range(C)]

    indptr = H_csc.indptr
    indices = H_csc.indices  # row indices (checks) for each column

    for v in range(H_csc.shape[1]):
        a, b = indptr[v], indptr[v + 1]
        checks = indices[a:b]
        dv = checks.size
        if dv <= 1:
            continue
        # Add clique among incident checks
        for i in range(dv):
            ci = int(checks[i])
            for j in range(i + 1, dv):
                cj = int(checks[j])
                if ci == cj:
                    continue
                adj_sets[ci].add(cj)
                adj_sets[cj].add(ci)

    adj: List[np.ndarray] = []
    for s in adj_sets:
        if s:
            arr = np.fromiter(s, dtype=np.int32)
            adj.append(np.sort(arr))
        else:
            adj.append(np.empty(0, dtype=np.int32))
    return adj


def _dsatur_coloring(adj: List[np.ndarray]) -> Tuple[np.ndarray, int]:
    """DSATUR heuristic coloring of the check–check conflict graph.

    Returns:
        color: np.int32 array of size C with color IDs [0..L-1]
        L: number of colors (layers)
    """
    C = len(adj)
    color = -np.ones(C, dtype=np.int32)

    # degrees for tie-break
    deg = np.array([len(a) for a in adj], dtype=np.int32)

    # saturation sets: which colors are present among colored neighbors
    sat: List[set[int]] = [set() for _ in range(C)]

    uncolored: set[int] = set(range(C))

    def pick_next() -> int:
        # pick vertex with max saturation, tie-break by degree
        best = None
        best_key = None
        for c in uncolored:
            key = (len(sat[c]), deg[c])
            if best is None or key > best_key:
                best = c
                best_key = key
        return int(best)  # type: ignore[arg-type]

    while uncolored:
        c = pick_next()

        # forbidden colors from colored neighbors
        forbidden: set[int] = set()
        for nb in adj[c]:
            nb_i = int(nb)
            if color[nb_i] >= 0:
                forbidden.add(int(color[nb_i]))

        # smallest available color
        k = 0
        while k in forbidden:
            k += 1
        color[c] = k

        # update saturation of its uncolored neighbors
        for nb in adj[c]:
            nb_i = int(nb)
            if color[nb_i] < 0:
                sat[nb_i].add(k)

        uncolored.remove(c)

    L = int(color.max() + 1) if C > 0 else 0
    return color, L


def _color_layers_from_H(H_csr: csr_matrix) -> Tuple[np.ndarray, np.ndarray]:
    """Compute conflict-free layers (graph coloring) from H_csr.

    Returns:
        check_permutation: np.int32 array of length C giving new row order
                           grouped by layer (layer 0, then layer 1, ...)
        layer_sizes: np.int32 array of length L with number of checks per layer
    """
    adj = _build_check_conflict_adjacency(H_csr)
    color, L = _dsatur_coloring(adj)
    if L == 0:
        C = H_csr.shape[0]
        return np.arange(C, dtype=np.int32), np.zeros(0, dtype=np.int32)

    layers: List[List[int]] = [[] for _ in range(L)]
    for chk, k in enumerate(color):
        layers[int(k)].append(chk)

    layer_sizes = np.array([len(layer) for layer in layers], dtype=np.int32)
    perm_list: List[int] = []
    for layer in layers:
        perm_list.extend(layer)
    check_permutation = np.asarray(perm_list, dtype=np.int32)
    return check_permutation, layer_sizes


def _clone_bench_state(args: Tuple, meta_base: Dict[str, Any]) -> Tuple[Tuple, Dict[str, Any]]:
    """Return (args, meta_base) with every tensor replaced by a scratch clone.

    Autotuning benchmarks a kernel by launching it repeatedly on the argument
    tensors it is handed.
    """
    seen: Dict[int, torch.Tensor] = {}

    def _clone(x):
        if not isinstance(x, torch.Tensor):
            return x
        key = id(x)
        hit = seen.get(key)
        if hit is None:
            hit = x.clone()
            seen[key] = hit
        return hit

    cloned_args = tuple(_clone(a) for a in args)
    cloned_meta = {k: _clone(v) for k, v in meta_base.items()}
    return cloned_args, cloned_meta


# Values accepted by ``unconverged_correction``.
UNCONVERGED_CORRECTIONS = ("first_leg", "last_leg")


@dataclass
class PerfCfg:
    """Performance configuration for kernel tuning.

    Stores autotuned parameters for different performance modes.
    Corresponds to kernel configuration in the Rust implementation.
    """

    perf: str
    msg_dtype: str
    btile: int
    rows_per_chk: int
    rows_per_var: int
    c2v_block: int
    c2v_warps: int
    c2v_stages: int
    v2c_block: int
    v2c_warps: int
    v2c_stages: int


class RelayBPDecoder:
    """Relay-BP-S decoder with Triton GPU kernels.

    Implements the Relay-BP (Relay Belief Propagation) algorithm for quantum error correction
    using GPU-accelerated message passing. The algorithm uses ensemble decoding with disordered
    memory strengths (gamma mixing) to improve decoding performance on quantum error correction codes.

    This implementation closely follows the Rust reference in crates/relay_bp/src/bp/relay.rs
    and crates/relay_bp/src/bp/min_sum.rs, providing GPU acceleration while maintaining
    algorithmic fidelity to the original min-sum belief propagation with memory.

    Key features:
    - Batch processing for multiple syndrome samples
    - Two performance modes: default, throughput
    - Support for both plain BP and Relay-BP algorithms
    - Configurable min-sum parameters (alpha, beta)
    - Memory strength (gamma) mixing for ensemble decoding
    - Early stopping after finding stop_nconv valid solutions
    - Optional explicit per-leg gamma matrices (K×V) for deterministic relay legs
    - unconverged_correction selects the correction for a lane with no valid solution:
      the hard decision at the end of the first leg ("first_leg", default) or of the last
      leg that ran ("last_leg")
    """

    def __init__(
        self,
        H_csr: csr_matrix,
        error_priors: np.ndarray,
        pre_iter: int = 80,
        num_sets: int = 100,
        set_max_iter: int = 60,
        gamma0: float = 0.65,
        gamma_dist_interval: Tuple[float, float] = (-0.24, 0.66),
        stop_nconv: int = 5,
        normalized_min_sum_alpha: Optional[float] = None,
        offset_min_sum_beta: Optional[float] = None,
        data_type: str = "float32",
        device: Union[str, List[int], int] = [0],
        seed: int = 1234,
        gamma_draw: str = "broadcast",
        bitpack_output: bool = False,
        algo: Optional[str] = None,  # "relay" or "plain"
        perf: Optional[str] = None,  # "default" | "throughput"
        alpha_iteration_scaling_factor: Optional[float] = 1.0,
        explicit_gammas: Optional[np.ndarray] = None,
        gamma_dist_interval_by_error: Optional[np.ndarray] = None,
        fast_max_unconverged_pct: Optional[
            float
        ] = None,  # Optional early exit threshold: stop when unconverged percentage <= this value (0.0-1.0)
        scheduling: str = "flood-fill",  # "flood-fill" (default) or "layered" (block-layered for first K iterations)
        layered_k_iters: int = 0,  # Number of layered iterations per leg (0 = disabled, only used when scheduling="layered")
        layered_block_policy: str = "contiguous",  # "contiguous" (default), "permute", "strided_interleaving", "color", or "color_use_delta_trick"
        layered_num_blocks: Optional[
            int
        ] = None,  # Explicit number of blocks (if None, computed from target_edges_per_block)
        layered_target_edges_per_block: Optional[
            int
        ] = None,  # Target edges per block (default: 200000)
        layered_block_seed: int = 42,  # Random seed for "permute" policy
        unconverged_correction: str = "first_leg",
    ):
        """Initialize Relay-BP decoder.

        Args:
            H_csr: Check matrix (C x V) in CSR format representing the quantum code's
                   stabilizer generators. Each row is a check constraint.
            error_priors: [V] error probabilities in (0, 0.5) for each qubit
            pre_iter: T0 - iterations for first set (ordered memory phase)
            num_sets: R - number of relay legs/ensembles (disordered memory phase)
            set_max_iter: Tr - iterations per relay set
            gamma0: γ for first set (ordered memory strength, corresponds to Rust gamma0)
            gamma_dist_interval: (min, max) for disordered γ sampling in relay legs
            stop_nconv: S - number of valid solutions to collect before stopping
            normalized_min_sum_alpha: α for normalized min-sum (0 < α ≤ 1, corresponds to Rust alpha)
            offset_min_sum_beta: β for offset min-sum (mutually exclusive with α)
            data_type: "float16", "float32", or "float64" for uniform precision (storage and computation). Default: "float32".
            device: GPU device selection. Can be:
                - List[int]: Explicit device IDs, e.g. [0, 1, 2, 3] for 4 GPUs
                - int: Number of GPUs to auto-detect, e.g. 4 -> [0, 1, 2, 3]
                - str: Legacy support - "auto" or "cuda" -> [0] (single GPU)
                On ROCm builds, PyTorch uses device type "cuda" for ROCm.
            seed: RNG seed for γ sampling (corresponds to Rust seed)
            gamma_draw: how disordered-leg γ is drawn from gamma_dist_interval or
                gamma_dist_interval_by_error. "broadcast": one [1, V] row per leg from
                seed + leg, shared by every shot. "per_row": one [B, V] draw per leg
                from seed + leg, so a shot's γ depends on its row in the batch.
                Ignored when explicit_gammas is set.
            bitpack_output: whether to return packed error bits for memory efficiency
            algo: "relay" or "plain" algorithm mode
            perf: "default" or "throughput" performance mode
            alpha_iteration_scaling_factor: scaling factor for alpha ramp schedule
            explicit_gammas: optional np.ndarray of shape (K, V) providing per-leg, per-variable
                memory strengths γ. Relay leg l (0-based) uses row ((l + 1) % K), the row Rust
                uses for set l + 1. When provided, uniform γ sampling is disabled.
            fast_max_unconverged_pct: Optional early exit threshold (0.0-1.0) for difficult case collection.
                If set, decoder stops when the percentage of unconverged syndromes drops to this threshold
                or below (e.g., 0.1 = stop when <=10% unconverged) and returns unconverged indices.
                If None, runs full num_sets relay legs. Default: None.
                When using for difficult case collection, collect unconverged_indices across batches
                and process them separately at the end with full num_sets.
            scheduling: Iteration scheduling mode. "flood-fill" (default) processes all checks then all vars.
                "layered" uses block-layered scheduling for first K iterations (Gauss-Seidel style), then
                switches to flood-fill. Only supported in throughput mode.
            layered_k_iters: Number of layered iterations per leg (default: 0, disabled). When scheduling="layered"
                and layered_k_iters > 0, first K iterations use block-layered, remaining use flood-fill.
                Only used when scheduling="layered" and perf="throughput".
            unconverged_correction: Correction returned for a lane with no valid solution in
                any leg. "first_leg" (default, as in the Rust reference) returns the hard
                decision at the end of the pre-iter leg, or of relay leg 1 when pre_iter is 0.
                "last_leg" returns the hard decision at the end of the last leg that ran,
                including a leg after which fast_max_unconverged_pct stops the decode.

        Raises:
            ValueError: If unconverged_correction is not one of UNCONVERGED_CORRECTIONS.
        """
        if unconverged_correction not in UNCONVERGED_CORRECTIONS:
            raise ValueError(
                f"unconverged_correction must be one of {UNCONVERGED_CORRECTIONS} "
                f"(got {unconverged_correction!r})"
            )
        self.unconverged_correction = unconverged_correction
        # Validate inputs
        validate_csr_matrix(H_csr)
        error_priors = validate_error_priors(error_priors)

        if normalized_min_sum_alpha is not None and offset_min_sum_beta is not None:
            raise ValueError("Cannot specify both normalized_min_sum_alpha and offset_min_sum_beta")

        # Allow alpha==0.0 to enable the ramp schedule (Rust-like semantics)
        if normalized_min_sum_alpha is not None and not (0 <= normalized_min_sum_alpha <= 1):
            raise ValueError("normalized_min_sum_alpha must be in [0, 1]")

        if offset_min_sum_beta is not None and offset_min_sum_beta < 0:
            raise ValueError("offset_min_sum_beta must be >= 0")
        # Store parameters
        self.H_csr = H_csr
        # Store check_permutation if used (needed for syndrome permutation in adapter)
        self.check_permutation = None
        self.error_priors = error_priors
        self.pre_iter = pre_iter
        self.num_sets = num_sets
        self.set_max_iter = set_max_iter
        self.gamma0 = gamma0
        self.gamma_dist_interval = gamma_dist_interval
        self.stop_nconv = stop_nconv
        self.fast_max_unconverged_pct = fast_max_unconverged_pct
        if fast_max_unconverged_pct is not None:
            if not (0.0 <= fast_max_unconverged_pct <= 1.0):
                raise ValueError(
                    f"fast_max_unconverged_pct must be between 0.0 and 1.0, got {fast_max_unconverged_pct}"
                )
        self.normalized_min_sum_alpha = normalized_min_sum_alpha
        self.offset_min_sum_beta = offset_min_sum_beta

        # Unified dtype system: support float16, float32, float64
        _dtype_aliases = {
            "float": "float32",
            "half": "float16",
            "double": "float64",
            "fp16": "float16",
            "fp32": "float32",
            "fp64": "float64",
        }
        data_type = _dtype_aliases.get(data_type, data_type)
        if data_type not in ("float16", "float32", "float64"):
            raise ValueError(
                f"data_type must be 'float16', 'float32', or 'float64', got {data_type}"
            )

        self.dtype_str = data_type
        self.dtype = {
            "float16": torch.float16,
            "float32": torch.float32,
            "float64": torch.float64,
        }[data_type]
        # Cache enum once to avoid recomputing it on every kernel launch
        self._dtype_enum = dtype_str_to_enum(self.dtype_str)
        # Parse device specification to list of device IDs
        self.device_ids = self._parse_device_spec(device)
        self.num_gpus = len(self.device_ids)

        # Detect backend (CUDA or ROCm) - PyTorch uses "cuda" device type for both.
        has_gpu = bool(torch.cuda.is_available())
        is_rocm = getattr(torch.version, "hip", None) is not None
        is_cuda_drv = torch.version.cuda is not None
        if not has_gpu:
            raise RuntimeError(
                "GPU not available (install a GPU-enabled PyTorch build: CUDA or ROCm)"
            )
        # Track backend for logging/metrics only (device type is always "cuda" for both CUDA and ROCm)
        self._backend = "rocm" if (is_rocm and not is_cuda_drv) else "cuda"
        # Device type string for PyTorch (works for both CUDA and ROCm)
        self.device_type = "cuda"
        # Cache per-GPU device strings to avoid rebuilding on every use
        self._device_strs: Dict[int, str] = {
            gpu_id: f"{self.device_type}:{gpu_id}" for gpu_id in self.device_ids
        }

        self.seed = seed
        if gamma_draw not in GAMMA_DRAW_MODES:
            raise ValueError(f"gamma_draw must be one of {GAMMA_DRAW_MODES}; got {gamma_draw!r}")
        self.gamma_draw = gamma_draw
        self.bitpack_output = bitpack_output
        # Normalize separate axes
        algo_norm = algo
        if algo_norm not in ("relay", "plain"):
            raise ValueError(f"Unknown algo '{algo_norm}'. Use 'relay' or 'plain'.")
        self.algo = algo_norm

        perf_norm = (perf or "throughput").strip().lower()
        perf_aliases = {
            "default": "default",
            "row": "default",
            "rowwise": "default",
            "throughput": "throughput",
            "btile": "throughput",
        }
        if perf_norm not in perf_aliases:
            raise ValueError(f"Unknown perf '{perf_norm}'. Use 'default'/'throughput'.")
        self.perf = perf_aliases[perf_norm]
        self.alpha_iteration_scaling_factor = float(alpha_iteration_scaling_factor or 1.0)

        # Scheduling validation
        scheduling_norm = (scheduling or "flood-fill").strip().lower()
        if scheduling_norm not in ("flood-fill", "layered"):
            raise ValueError(
                f"Unknown scheduling '{scheduling_norm}'. Use 'flood-fill' or 'layered'."
            )
        self.scheduling = scheduling_norm
        self.layered_k_iters = int(layered_k_iters) if layered_k_iters is not None else 0
        # Layered scheduling only works in throughput mode
        if self.scheduling == "layered" and self.perf != "throughput":
            raise ValueError("scheduling='layered' requires perf='throughput'")
        # Explicit gammas (for relay legs only). Replicate on each GPU as [K,V] fp32.
        self.explicit_gammas_per_gpu = None
        if explicit_gammas is not None:
            eg = np.asarray(explicit_gammas, dtype=np.float32)
            if eg.ndim != 2 or eg.shape[1] != int(H_csr.shape[1]):
                raise ValueError(f"explicit_gammas must have shape (K, V={int(H_csr.shape[1])})")
            # Replicate on each GPU
            self.explicit_gammas_per_gpu = []
            for gpu_id in self.device_ids:
                device_str = self._device_strs[gpu_id]
                eg_gpu = torch.from_numpy(eg).to(device_str, dtype=self.dtype)
                self.explicit_gammas_per_gpu.append(eg_gpu)

        # Per-variable gamma intervals (local gammas feature).
        # Mirrors Rust `gamma_dist_interval_by_error` behavior in crates/relay_bp/src/bp/relay.rs.
        # Stored as [V, 2] tensors (low, high) replicated per GPU.
        self.gamma_dist_interval_by_error_per_gpu = None
        if gamma_dist_interval_by_error is not None:
            gd = np.asarray(gamma_dist_interval_by_error, dtype=np.float32)
            V = int(H_csr.shape[1])
            if gd.ndim != 2 or gd.shape[0] != V or gd.shape[1] != 2:
                raise ValueError(
                    f"gamma_dist_interval_by_error must have shape (V={V}, 2), got {tuple(gd.shape)}"
                )
            if np.any(gd[:, 0] > gd[:, 1]):
                raise ValueError(
                    "gamma_dist_interval_by_error: lower bounds must be <= upper bounds"
                )
            self.gamma_dist_interval_by_error_per_gpu = []
            for gpu_id in self.device_ids:
                device_str = self._device_strs[gpu_id]
                gd_gpu = torch.from_numpy(gd).to(device_str, dtype=self.dtype)
                self.gamma_dist_interval_by_error_per_gpu.append(gd_gpu)

        # Set up device (availability already validated above)

        # Build block metadata for layered scheduling
        # For "permute", "strided_interleaving", and "color" policies, we need to apply
        # a permutation before building graphs
        self.block_metadata = None
        self.block_overlap_stats = None
        self._color_layer_sizes: Optional[np.ndarray] = None
        self.layered_block_policy = layered_block_policy
        self._use_delta_trick: bool = (
            self.scheduling == "layered"
            and self.perf == "throughput"
            and layered_block_policy == "color_use_delta_trick"
        )
        check_permutation: Optional[np.ndarray] = None
        if self.scheduling == "layered" and self.perf == "throughput":
            # Validate layered_block_policy
            if layered_block_policy not in (
                "contiguous",
                "permute",
                "strided_interleaving",
                "color",
                "color_use_delta_trick",
            ):
                raise ValueError(
                    f"layered_block_policy must be 'contiguous', 'permute', 'strided_interleaving', 'color', "
                    f"or 'color_use_delta_trick', got {layered_block_policy!r}"
                )

            # Apply check permutation if needed
            if layered_block_policy == "permute":
                # Random permutation of checks (deterministic with seed)
                # Use default_rng to avoid global RNG state mutation
                rng = np.random.default_rng(layered_block_seed)
                check_permutation = rng.permutation(self.H_csr.shape[0]).astype(np.int32)
                # Store permutation for syndrome reordering in adapter
                self.check_permutation = check_permutation
            elif layered_block_policy == "strided_interleaving":
                # Cyclic interleaving: check k goes to group (k % num_groups)
                # Exploits quasi-cyclic structure of bicycle codes to spread checks across blocks.
                if layered_num_blocks is None:
                    raise ValueError(
                        "layered_block_policy='strided_interleaving' requires layered_num_blocks to be set"
                    )
                C = self.H_csr.shape[0]
                num_groups = int(layered_num_blocks)
                check_permutation = np.array(
                    [i for g in range(num_groups) for i in range(g, C, num_groups)],
                    dtype=np.int32,
                )
                self.check_permutation = check_permutation
            elif layered_block_policy in ("color", "color_use_delta_trick"):
                # Conflict-free layers via DSATUR coloring. Each layer contains checks that do not
                # share variables (overlap=0 within a layer). We build a permutation that groups
                # layers contiguously.
                check_permutation, layer_sizes = _color_layers_from_H(self.H_csr)
                self.check_permutation = check_permutation
                self._color_layer_sizes = layer_sizes

        # Create graph on each GPU (read-only, can be replicated).
        # Apply check_permutation if it was computed above.
        self.graphs = []
        for gpu_id in self.device_ids:
            device_str = self._device_strs[gpu_id]
            graph = CSRGraph(H_csr, device_str, check_permutation=check_permutation)
            self.graphs.append(graph)

        # Set graph shape info
        self.C, self.V = self.graphs[0].C, self.graphs[0].V
        self.E = self.graphs[0].E

        # Build block metadata for layered scheduling
        if self.scheduling == "layered" and self.perf == "throughput":
            graph0 = self.graphs[0]

            # First, construct block check ranges and variable lists depending on policy
            if (
                layered_block_policy in ("color", "color_use_delta_trick")
                and self._color_layer_sizes is not None
            ):
                # One block per color layer; after applying check_permutation, each layer
                # occupies a contiguous range of checks.
                layer_sizes = self._color_layer_sizes
                num_blocks = int(layer_sizes.shape[0])
                block_chk_start_cpu = np.zeros(num_blocks, dtype=np.int32)
                block_chk_end_cpu = np.zeros(num_blocks, dtype=np.int32)
                offset = 0
                for b in range(num_blocks):
                    block_chk_start_cpu[b] = offset
                    offset += int(layer_sizes[b])
                    block_chk_end_cpu[b] = offset

                # Build per-block variable lists using the same CSR-based logic as build_block_metadata
                chk_ptr_host = graph0.chk_ptr.cpu().numpy()
                chk_edges_host = graph0.chk_edges.cpu().numpy()
                edge_var_host = graph0.edge_var.cpu().numpy()

                block_vars_list: List[List[int]] = []
                block_var_ptr_list = [0]

                for b in range(num_blocks):
                    chk_start = int(block_chk_start_cpu[b])
                    chk_end = int(block_chk_end_cpu[b])

                    e0 = chk_ptr_host[chk_start]
                    e1 = chk_ptr_host[chk_end]

                    if e1 > e0:
                        all_edge_ids = chk_edges_host[e0:e1]
                        all_var_ids = edge_var_host[all_edge_ids]
                        block_vars_sorted = np.unique(all_var_ids).astype(np.int32)
                    else:
                        block_vars_sorted = np.array([], dtype=np.int32)

                    block_vars_list.append(block_vars_sorted.tolist())
                    block_var_ptr_list.append(block_var_ptr_list[-1] + len(block_vars_sorted))

                total_vars = block_var_ptr_list[-1]
                block_vars_flat = np.zeros(total_vars, dtype=np.int32)
                for b, var_list in enumerate(block_vars_list):
                    start = block_var_ptr_list[b]
                    end = block_var_ptr_list[b + 1]
                    block_vars_flat[start:end] = var_list

                dev = graph0.chk_ptr.device
                block_chk_start = torch.from_numpy(block_chk_start_cpu).to(dev)
                block_chk_end = torch.from_numpy(block_chk_end_cpu).to(dev)
                block_var_ptr = torch.from_numpy(np.array(block_var_ptr_list, dtype=np.int32)).to(
                    dev
                )
                block_vars = torch.from_numpy(block_vars_flat).to(dev)

                block_var_ptr_cpu = block_var_ptr.cpu().numpy()
            else:
                # Default edge-balanced block construction (contiguous / permute / strided_interleaving)
                block_chk_start, block_chk_end, block_var_ptr, block_vars, num_blocks = (
                    graph0.build_block_metadata(
                        num_blocks=layered_num_blocks,
                        target_edges_per_block=layered_target_edges_per_block,
                    )
                )
                block_chk_start_cpu = block_chk_start.cpu().numpy()
                block_chk_end_cpu = block_chk_end.cpu().numpy()
                block_var_ptr_cpu = block_var_ptr.cpu().numpy()

            # Compute block overlap diagnostics (for debugging/validation)
            block_overlap_stats = self._compute_block_overlap_stats(
                block_chk_start_cpu,
                block_chk_end_cpu,
                block_var_ptr_cpu,
                block_vars.cpu().numpy(),
            )

            # Log block statistics for validation/optimization
            import logging

            logger = logging.getLogger(__name__)
            if logger.isEnabledFor(logging.INFO):
                logger.info(
                    f"Layered scheduling: {num_blocks} blocks, policy='{layered_block_policy}'"
                )
                logger.info(
                    f"  Inter-block overlap (adj blocks share): "
                    f"avg={block_overlap_stats['avg_overlap']:.3f}, "
                    f"min={block_overlap_stats['min_overlap']:.3f}, "
                    f"max={block_overlap_stats['max_overlap']:.3f}"
                )
                if "max_intra_block_overlap" in block_overlap_stats:
                    logger.info(
                        f"  Max intra-block overlap (max checks/var within block): "
                        f"{block_overlap_stats['max_intra_block_overlap']}"
                    )
                # Per-block details at DEBUG level
                if logger.isEnabledFor(logging.DEBUG):
                    chk_ptr_host = graph0.chk_ptr.cpu().numpy()
                    for b in range(num_blocks):
                        chk_start = int(block_chk_start_cpu[b])
                        chk_end = int(block_chk_end_cpu[b])
                        var_start = int(block_var_ptr_cpu[b])
                        var_end = int(block_var_ptr_cpu[b + 1])
                        e0 = chk_ptr_host[chk_start]
                        e1 = chk_ptr_host[chk_end]
                        edges_in_block = e1 - e0
                        vars_in_block = var_end - var_start
                        overlap_ratio = (
                            block_overlap_stats["overlap_ratios"][b - 1] if b > 0 else 0.0
                        )
                        logger.debug(
                            f"  Block {b}: checks [{chk_start}, {chk_end}), "
                            f"edges={edges_in_block}, vars={vars_in_block}, "
                            f"overlap_with_prev={overlap_ratio:.3f}"
                        )

            # Store metadata and CPU copies
            self.block_metadata = {
                "block_chk_start": block_chk_start,
                "block_chk_end": block_chk_end,
                "block_var_ptr": block_var_ptr,
                "block_vars": block_vars,
                "num_blocks": num_blocks,
                "block_chk_start_cpu": block_chk_start_cpu,
                "block_chk_end_cpu": block_chk_end_cpu,
                "block_var_ptr_cpu": block_var_ptr_cpu,
                "overlap_stats": block_overlap_stats,
            }
            self._block_chk_start_cpu = block_chk_start_cpu
            self._block_chk_end_cpu = block_chk_end_cpu
            self._block_var_ptr_cpu = block_var_ptr_cpu
            self.block_overlap_stats = block_overlap_stats

        # Compute log prior ratios (Rust: log_prior_ratios)
        log_ratios = compute_log_prior_ratios(error_priors)
        # Replicate log_prior_ratios on each GPU
        self.log_prior_ratios_per_gpu = []
        for gpu_id in self.device_ids:
            device_str = self._device_strs[gpu_id]
            # Use unified dtype for log_prior_ratios
            log_prior = (
                torch.from_numpy(log_ratios).to(device_str, dtype=self.dtype).unsqueeze(0)
            )  # [1, V]
            self.log_prior_ratios_per_gpu.append(log_prior)

        # Precompute decoding weights for solution selection (matches Rust decoding_quality)
        # Weight = sum of log-priors for error bits, used to select best solutions
        # Rust: decoding_quality += log_prior_ratios[i] when errors[i] == 1 and is_finite(log_prior_ratios[i])
        # log_prior_ratios can be negative when error probability > 0.5
        # Use first GPU's log_prior_ratios (all are identical)
        wj = (
            self.log_prior_ratios_per_gpu[0]
            .squeeze(0)
            .to(self._device_strs[self.device_ids[0]], dtype=self.dtype)
        )  # [V]
        # Filter finite values: set non-finite to 0 (matches Rust's is_finite check)
        wj = torch.where(torch.isfinite(wj), wj, torch.zeros_like(wj))
        self.wj = wj.contiguous()
        # Cache wj on each GPU to avoid .to(...) on every selection call
        self.wj_per_gpu: Dict[int, torch.Tensor] = {}
        for gpu_id in self.device_ids:
            device_str = self._device_strs[gpu_id]
            self.wj_per_gpu[gpu_id] = self.wj.to(device_str)




        # Performance configuration (fallback values, overridden by autotuning)
        self.cfg = PerfCfg(
            perf=self.perf,
            msg_dtype=self.dtype_str,
            btile=64,  # fallback only; tuned meta overrides
            rows_per_chk=16,  # fallback only
            rows_per_var=16,  # fallback only
            c2v_block=0,
            c2v_warps=0,
            c2v_stages=0,
            v2c_block=0,
            v2c_warps=0,
            v2c_stages=0,
        )

        # Set up RNG
        torch.manual_seed(seed)

        # Pre-allocate device tensors (will be allocated in decode)
        # Per-GPU tensor caches for multi-GPU support
        self._device_tensors_per_gpu = {gpu_id: {} for gpu_id in self.device_ids}
        # Cache dummy L tensor per GPU (used when L is not allocated but kernel requires valid pointer)
        self._dummy_L_per_gpu = {}

        # Optional logging for cached tuning usage
        self._tune_log = os.getenv("RELAY_TUNE_VERBOSE", "0") == "1"
        self._tune_logged: set[str] = set()

    def override_llr_columns(self, cols, value):
        """Override the prior log-likelihood-ratio (LLR) on specific variable cols.

        The per-variable prior LLR is ln((1-p)/p). Building from an
        error_priors probability cannot express a belief that a fault did
        fire, because priors are clipped to <= 0.5 (LLR >= 0). This method
        writes such values directly into the kernel's init-LLR tensor
        (log_prior_ratios_per_gpu) and the solution-selection weights
        (wj / wj_per_gpu), bypassing the clip. Useful for staged decoding
        schemes that set per-column priors between rounds.

        Args:
            cols: 1-D int array of variable (column) indices to override.
            value: LLR to write. Negative => biased toward firing. Either a
                scalar (same LLR on all cols) or a length-len(cols) array
                (per-column graded LLRs, e.g. confidence-graded conditioning).

        Returns:
            A restore-callable that reverts exactly these entries to their
            previous values when invoked (so callers can patch-per-shot then
            restore cheaply, no decoder rebuild).
        """
        cols = np.asarray(cols, dtype=np.int64)
        # Save all originals before writing anything: self.wj and
        # wj_per_gpu[gpu] may alias the same tensor (single-GPU case), so
        # interleaving save/write would capture already-overwritten values.
        saved_lpr = [
            (gpu_idx, lpr[0, cols].clone())
            for gpu_idx, lpr in enumerate(self.log_prior_ratios_per_gpu)
        ]
        saved_wj = {gpu_id: wj[cols].clone() for gpu_id, wj in self.wj_per_gpu.items()}
        saved_self_wj = self.wj[cols].clone()

        # value may be a scalar (broadcasts) or a length-len(cols) array
        # (per-column LLRs supplied by a caller).
        # Cast per-tensor to match device/dtype before advanced-index assign.
        for gpu_idx, lpr in enumerate(self.log_prior_ratios_per_gpu):
            lpr[0, cols] = torch.as_tensor(value, dtype=lpr.dtype, device=lpr.device)
        for gpu_id, wj in self.wj_per_gpu.items():
            wj[cols] = torch.as_tensor(value, dtype=wj.dtype, device=wj.device)
        self.wj[cols] = torch.as_tensor(value, dtype=self.wj.dtype, device=self.wj.device)

        def _restore():
            for gpu_idx, old in saved_lpr:
                self.log_prior_ratios_per_gpu[gpu_idx][0, cols] = old
            for gpu_id, old in saved_wj.items():
                self.wj_per_gpu[gpu_id][cols] = old
            self.wj[cols] = saved_self_wj

        return _restore

    def _compute_block_overlap_stats(
        self, block_chk_start, block_chk_end, block_var_ptr, block_vars
    ):
        """Compute block overlap statistics for diagnostics.

        Returns:
            dict with overlap metrics:
            - var_counts: list of variable counts per block
            - overlap_ratios: list of overlap ratios between adjacent blocks
            - avg_overlap: average overlap ratio
            - max_intra_block_overlap: maximum checks-per-variable within any single block
        """
        num_blocks = len(block_chk_start)
        var_counts = []
        overlap_ratios = []

        # Get variable sets for each block
        block_var_sets = []
        for b in range(num_blocks):
            var_start = int(block_var_ptr[b])
            var_end = int(block_var_ptr[b + 1])
            var_set = set(block_vars[var_start:var_end])
            block_var_sets.append(var_set)
            var_counts.append(len(var_set))

        # Compute max intra-block overlap (max checks per variable within a block)
        max_intra = 0
        if hasattr(self, "graphs") and self.graphs:
            chk_ptr_host = self.graphs[0].chk_ptr.cpu().numpy().astype(np.int32)
            edge_var_host = self.graphs[0].edge_var.cpu().numpy().astype(np.int32)
            for b in range(num_blocks):
                chk_start = int(block_chk_start[b])
                chk_end = int(block_chk_end[b])
                var_counts_in_block: Dict[int, int] = {}
                for c in range(chk_start, chk_end):
                    e0 = chk_ptr_host[c]
                    e1 = chk_ptr_host[c + 1]
                    for e in range(e0, e1):
                        v = int(edge_var_host[e])
                        var_counts_in_block[v] = var_counts_in_block.get(v, 0) + 1
                if var_counts_in_block:
                    local_max = max(var_counts_in_block.values())
                    if local_max > max_intra:
                        max_intra = local_max

        # Compute overlap between adjacent blocks
        for b in range(num_blocks - 1):
            var_set_i = block_var_sets[b]
            var_set_j = block_var_sets[b + 1]
            intersection = var_set_i & var_set_j
            min_size = min(len(var_set_i), len(var_set_j))
            if min_size > 0:
                overlap_ratio = len(intersection) / min_size
            else:
                overlap_ratio = 0.0
            overlap_ratios.append(overlap_ratio)

        avg_overlap = np.mean(overlap_ratios) if overlap_ratios else 0.0

        return {
            "var_counts": var_counts,
            "overlap_ratios": overlap_ratios,
            "avg_overlap": float(avg_overlap),
            "min_overlap": float(min(overlap_ratios)) if overlap_ratios else 0.0,
            "max_overlap": float(max(overlap_ratios)) if overlap_ratios else 0.0,
            "max_intra_block_overlap": int(max_intra),
        }

    def _get_tuned_meta(
        self,
        *,
        kernel_name: str,
        problem_key: Dict[str, Any],
        build_cfgs_fn,
        kernel,
        grid,
        args: Tuple,
        meta_base: Dict[str, Any],
        number: int = 20,
        grid_fn=None,
        log_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Shared helper for cache-based autotuning.

        Wraps the common pattern:
          - try_get_saved
          - bench_and_select if missing
          - set_saved
          - optional one-time logging via self._tune_log / self._tune_logged
        """
        saved = ac.try_get_saved(kernel_name, problem_key)
        if saved is None:
            cfgs = [
                dict(c.kwargs, num_warps=c.num_warps, num_stages=c.num_stages)
                for c in build_cfgs_fn()
            ]
            # Tune on scratch clones so benchmarking is side-effect free.
            bench_args, bench_meta = _clone_bench_state(args, meta_base)
            best = ac.bench_and_select(
                kernel=kernel,
                grid=grid,
                args=bench_args,
                meta_base=bench_meta,
                configs=cfgs,
                number=number,
                grid_fn=grid_fn,
            )
            del bench_args, bench_meta
            ac.set_saved(kernel_name, problem_key, best)
            saved = best

        name = log_name or kernel_name
        if self._tune_log and name not in self._tune_logged:
            config_summary = {}
            for key in [
                "BLOCK_SIZE",
                "ROWS_PER_CHK",
                "ROWS_PER_VAR",
                "BTILE",
                "BLOCK_C",
                "BLOCK_V",
                "num_warps",
                "num_stages",
            ]:
                if key in saved:
                    config_summary[key] = saved[key]
            config_str = f" config={config_summary}" if config_summary else ""
            print(
                f"[relay-bp-triton][{name}] Using cached meta from {ac.DEFAULT_CACHE} key={problem_key}{config_str}"
            )
            self._tune_logged.add(name)

        return saved

    def _resolve_configs(self, B, tensors, graph, gpu_idx):
        """Resolve all autotuning configs once per decode call."""
        device_str = self._device_strs[self.device_ids[gpu_idx]]
        log_prior_ratios = self.log_prior_ratios_per_gpu[gpu_idx]
        alpha = self.normalized_min_sum_alpha or 0.0
        beta = self.offset_min_sum_beta or 0.0
        use_alpha = self.normalized_min_sum_alpha is not None
        use_beta = self.offset_min_sum_beta is not None

        # Full-batch active index for benchmarking
        active_idx = torch.arange(B, device=device_str, dtype=torch.int32)
        B_active = B

        if self.cfg.perf == "throughput":
            self._resolve_throughput_configs(
                B,
                B_active,
                active_idx,
                tensors,
                graph,
                gpu_idx,
                device_str,
                log_prior_ratios,
                alpha,
                beta,
                use_alpha,
                use_beta,
            )
        else:
            self._resolve_default_configs(
                B,
                tensors,
                graph,
                gpu_idx,
                device_str,
                log_prior_ratios,
                alpha,
                beta,
                use_alpha,
                use_beta,
            )

    def _resolve_throughput_configs(
        self,
        B,
        B_active,
        active_idx,
        tensors,
        graph,
        gpu_idx,
        device_str,
        log_prior_ratios,
        alpha,
        beta,
        use_alpha,
        use_beta,
    ):
        """Resolve autotuning configs for throughput (btile) mode."""
        # --- C2V btile ---
        c2v_key = {
            "B": B,
            "C": self.C,
            "E": self.E,
            "dtype": self.dtype_str,
            "sparsity_hash": graph.sparsity_hash,
        }
        dummy_L = self._dummy_L_per_gpu.get(gpu_idx)
        if dummy_L is None:
            dummy_L = torch.zeros(1, 1, dtype=self.dtype, device=device_str)
            self._dummy_L_per_gpu[gpu_idx] = dummy_L

        def _c2v_grid_fn(cfg):
            bt = int(cfg.get("BTILE", self.cfg.btile))
            return (self.C, (B_active + bt - 1) // bt)

        saved = self._get_tuned_meta(
            kernel_name="c2v_min_sum_btile_kernel",
            problem_key=c2v_key,
            build_cfgs_fn=build_c2v_btile_configs,
            kernel=c2v_min_sum_btile_kernel,
            grid=(self.C, 1),
            args=(
                tensors["muT"],
                tensors["nuT"],
                graph.chk_ptr,
                graph.chk_edges,
                tensors["syndrome"],
                active_idx,
                tensors["active"],
            ),
            meta_base=dict(
                B=B,
                B_active=B_active,
                C=self.C,
                E=self.E,
                alpha=alpha,
                beta=beta,
                use_alpha=use_alpha,
                use_beta=use_beta,
                dtype_enum=self._dtype_enum,
                chk_start=0,
                chk_end=0,
                edge_var=graph.edge_var,
                L=dummy_L,
                V=self.V,
                use_delta=False,
            ),
            number=20,
            grid_fn=_c2v_grid_fn,
            log_name=f"c2v_min_sum_btile_kernel[gpu={gpu_idx}]",
        )
        self._rc_c2v_btile = int(saved.get("BTILE", self.cfg.btile))
        self._rc_c2v_block = int(saved.get("BLOCK_SIZE", 64))
        self._rc_c2v_warps = int(saved.get("num_warps", 0))
        self._rc_c2v_stages = int(saved.get("num_stages", 0))

        # --- V2C btile ---
        v2c_key = {
            "B": B,
            "V": self.V,
            "E": self.E,
            "dtype": self.dtype_str,
            "WRITE_HARD": True,
            "sparsity_hash": graph.sparsity_hash,
            "BLOCK_SIZE": V2C_BTILE_BLOCK_SIZE,
        }

        def _v2c_grid_fn(cfg):
            bt = int(cfg.get("BTILE", self.cfg.btile))
            return (self.V, (B_active + bt - 1) // bt)

        saved = self._get_tuned_meta(
            kernel_name="v2c_and_gamma_btile_kernel",
            problem_key=v2c_key,
            build_cfgs_fn=build_v2c_btile_configs,
            kernel=v2c_and_gamma_btile_kernel,
            grid=(self.V, 1),
            args=(
                tensors["nuT"],
                tensors["muT"],
                graph.var_ptr,
                graph.var_edges,
                tensors["lambda_"],
                log_prior_ratios,
                tensors["gamma"],
                tensors["posterior_ratios"],
                tensors["hard_dec"],
                active_idx,
                tensors["active"],
            ),
            meta_base=dict(
                B=B,
                B_active=B_active,
                V=self.V,
                E=self.E,
                WRITE_HARD=True,
                dtype_enum=self._dtype_enum,
                block_vars=(
                    graph.var_edges[:1] if graph.var_edges.shape[0] > 0 else graph.var_ptr[:1]
                ),
                block_var_ptr=graph.var_ptr[:2],
                block_idx=-1,
            ),
            number=20,
            grid_fn=_v2c_grid_fn,
            log_name=f"v2c_and_gamma_btile_kernel[gpu={gpu_idx}]",
        )
        self._rc_v2c_btile = int(saved.get("BTILE", self.cfg.btile))
        # Fixed, not tuned: see autotune.V2C_BTILE_BLOCK_SIZE.
        self._rc_v2c_block = V2C_BTILE_BLOCK_SIZE
        self._rc_v2c_warps = int(saved.get("num_warps", 0))
        self._rc_v2c_stages = int(saved.get("num_stages", 0))

        # --- Parity (shared by compact_with_zero and compact kernels) ---
        parity_key = {
            "B": B,
            "C": self.C,
            "V": self.V,
            "E": self.E,
            "ROWS_PER_CHK": self.cfg.rows_per_chk,
            "dtype": self.dtype_str,
            "sparsity_hash": graph.sparsity_hash,
        }
        parity_meta = dict(B=B, C=self.C, V=self.V, E=self.E, ROWS_PER_CHK=self.cfg.rows_per_chk)
        total = B * self.C
        parity_grid = ((total + self.cfg.rows_per_chk - 1) // self.cfg.rows_per_chk,)

        saved = self._get_tuned_meta(
            kernel_name="parity_from_hard_compact_with_zero_kernel",
            problem_key=parity_key,
            build_cfgs_fn=build_parity_configs,
            kernel=parity_from_hard_compact_with_zero_kernel,
            grid=parity_grid,
            args=(
                tensors["hard_dec"],
                graph.chk_ptr,
                graph.chk_edges,
                graph.edge_var,
                tensors["syndrome"],
                tensors["check_ok"],
                tensors["active"],
            ),
            meta_base=parity_meta,
            number=20,
            log_name=f"parity_from_hard_compact_with_zero_kernel[gpu={gpu_idx}]",
        )
        self._rc_parity_block = int(saved["BLOCK_SIZE"])
        self._rc_parity_warps = int(saved.get("num_warps", 0))
        self._rc_parity_stages = int(saved.get("num_stages", 0))

        # --- Select ---
        gpu_id = self.device_ids[gpu_idx]
        wj_gpu = self.wj_per_gpu[gpu_id]
        select_key = {
            "B": B,
            "C": self.C,
            "V": self.V,
            "dtype": self.dtype_str,
            "sparsity_hash": graph.sparsity_hash,
        }

        saved = self._get_tuned_meta(
            kernel_name="check_and_select_kernel",
            problem_key=select_key,
            build_cfgs_fn=build_check_select_configs,
            kernel=check_and_select_kernel,
            grid=(B,),
            args=(
                tensors["check_ok"],
                tensors["posterior_ratios"],
                tensors["hard_dec"],
                wj_gpu,
                tensors["best_weights"],
                tensors["best_errors"],
                tensors["valid_solutions"],
                tensors["found_count"],
                tensors["active"],
            ),
            meta_base=dict(
                B=B,
                C=self.C,
                V=self.V,
                dtype_enum=self._dtype_enum,
                NUM_SOLS=int(self.stop_nconv),
            ),
            number=20,
            grid_fn=lambda _cfg: (B,),
            log_name=f"check_and_select_kernel[gpu={gpu_idx}]",
        )
        self._rc_select_block_c = int(saved.get("BLOCK_C", 256))
        self._rc_select_block_v = int(saved.get("BLOCK_V", 128))
        self._rc_select_warps = int(saved.get("num_warps", 1))

    def _resolve_default_configs(
        self,
        B,
        tensors,
        graph,
        gpu_idx,
        device_str,
        log_prior_ratios,
        alpha,
        beta,
        use_alpha,
        use_beta,
    ):
        """Resolve autotuning configs for default (row-wise) mode."""
        # --- C2V ---
        c2v_key = {
            "B": B,
            "C": self.C,
            "E": self.E,
            "dtype": self.dtype_str,
            "sparsity_hash": graph.sparsity_hash,
        }

        def _c2v_grid_fn(cfg):
            rows = int(cfg.get("ROWS_PER_CHK", self.cfg.rows_per_chk))
            return ((B * self.C + rows - 1) // rows,)

        saved = self._get_tuned_meta(
            kernel_name="c2v_min_sum_kernel",
            problem_key=c2v_key,
            build_cfgs_fn=build_c2v_configs,
            kernel=c2v_min_sum_kernel,
            grid=(1,),
            args=(
                tensors["mu"],
                tensors["nu"],
                graph.chk_ptr,
                graph.chk_edges,
                tensors["syndrome"],
                tensors["active"],
            ),
            meta_base=dict(
                B=B,
                C=self.C,
                E=self.E,
                alpha=alpha,
                beta=beta,
                use_alpha=use_alpha,
                use_beta=use_beta,
                dtype_enum=self._dtype_enum,
            ),
            number=20,
            grid_fn=_c2v_grid_fn,
            log_name=f"c2v_min_sum_kernel[gpu={gpu_idx}]",
        )
        self._rc_c2v_rows = int(saved.get("ROWS_PER_CHK", self.cfg.rows_per_chk))
        self._rc_c2v_block = int(saved.get("BLOCK_SIZE", 64))
        self._rc_c2v_warps = int(saved.get("num_warps", 0))
        self._rc_c2v_stages = int(saved.get("num_stages", 0))

        # --- V2C ---
        v2c_key = {
            "B": B,
            "V": self.V,
            "E": self.E,
            "dtype": self.dtype_str,
            "sparsity_hash": graph.sparsity_hash,
        }

        def _v2c_grid_fn(cfg):
            rows = int(cfg.get("ROWS_PER_VAR", self.cfg.rows_per_var))
            return ((B * self.V + rows - 1) // rows,)

        saved = self._get_tuned_meta(
            kernel_name="v2c_and_marginals_fused_gamma_kernel",
            problem_key=v2c_key,
            build_cfgs_fn=build_v2c_configs,
            kernel=v2c_and_marginals_fused_gamma_kernel,
            grid=(1,),
            args=(
                tensors["nu"],
                tensors["mu"],
                graph.var_ptr,
                graph.var_edges,
                tensors["lambda_"],
                tensors["posterior_ratios"],
                tensors["hard_dec"],
                tensors["active"],
                log_prior_ratios,
                tensors["gamma"],
            ),
            meta_base=dict(B=B, V=self.V, E=self.E, dtype_enum=self._dtype_enum),
            number=20,
            grid_fn=_v2c_grid_fn,
            log_name=f"v2c_and_marginals_fused_gamma_kernel[gpu={gpu_idx}]",
        )
        self._rc_v2c_rows = int(saved.get("ROWS_PER_VAR", self.cfg.rows_per_var))
        self._rc_v2c_block = int(saved.get("BLOCK_SIZE", 64))
        self._rc_v2c_warps = int(saved.get("num_warps", 0))
        self._rc_v2c_stages = int(saved.get("num_stages", 0))

        # --- Parity ---
        parity_key = {
            "B": B,
            "C": self.C,
            "V": self.V,
            "E": self.E,
            "ROWS_PER_CHK": self.cfg.rows_per_chk,
            "dtype": self.dtype_str,
            "sparsity_hash": graph.sparsity_hash,
        }
        parity_meta = dict(B=B, C=self.C, V=self.V, E=self.E, ROWS_PER_CHK=self.cfg.rows_per_chk)
        parity_grid = ((B * self.C + self.cfg.rows_per_chk - 1) // self.cfg.rows_per_chk,)

        saved = self._get_tuned_meta(
            kernel_name="parity_per_check_kernel",
            problem_key=parity_key,
            build_cfgs_fn=build_parity_configs,
            kernel=parity_per_check_kernel,
            grid=parity_grid,
            args=(
                tensors["posterior_ratios"],
                graph.chk_ptr,
                graph.chk_edges,
                graph.edge_var,
                tensors["syndrome"],
                tensors["check_ok"],
            ),
            meta_base=parity_meta,
            number=20,
            log_name=f"parity_per_check_kernel[gpu={gpu_idx}]",
        )
        self._rc_parity_block = int(saved["BLOCK_SIZE"])
        self._rc_parity_warps = int(saved.get("num_warps", 0))
        self._rc_parity_stages = int(saved.get("num_stages", 0))

        # --- Select (shared config for both modes) ---
        gpu_id = self.device_ids[gpu_idx]
        wj_gpu = self.wj_per_gpu[gpu_id]
        select_key = {
            "B": B,
            "C": self.C,
            "V": self.V,
            "dtype": self.dtype_str,
            "sparsity_hash": graph.sparsity_hash,
        }

        saved = self._get_tuned_meta(
            kernel_name="check_and_select_kernel",
            problem_key=select_key,
            build_cfgs_fn=build_check_select_configs,
            kernel=check_and_select_kernel,
            grid=(B,),
            args=(
                tensors["check_ok"],
                tensors["posterior_ratios"],
                tensors["hard_dec"],
                wj_gpu,
                tensors["best_weights"],
                tensors["best_errors"],
                tensors["valid_solutions"],
                tensors["found_count"],
                tensors["active"],
            ),
            meta_base=dict(
                B=B,
                C=self.C,
                V=self.V,
                dtype_enum=self._dtype_enum,
                NUM_SOLS=int(self.stop_nconv),
            ),
            number=20,
            grid_fn=lambda _cfg: (B,),
            log_name=f"check_and_select_kernel[gpu={gpu_idx}]",
        )
        self._rc_select_block_c = int(saved.get("BLOCK_C", 256))
        self._rc_select_block_v = int(saved.get("BLOCK_V", 128))
        self._rc_select_warps = int(saved.get("num_warps", 1))

    def _parse_device_spec(self, device: Union[str, List[int], int]) -> List[int]:
        """Parse device specification into list of device IDs.

        Supports:
        - List[int]: Explicit device IDs, e.g. [0, 1, 2, 3]
        - int: Number of GPUs to auto-detect, e.g. 4 → [0, 1, 2, 3]
        - str: Legacy support - "auto" or "cuda" → [0] (single GPU)

        Args:
            device: Device specification

        Returns:
            List of device IDs
        """
        if isinstance(device, int):
            # Auto-detect N GPUs
            num_gpus = device
            if num_gpus < 1:
                raise ValueError(f"Number of GPUs must be >= 1, got {num_gpus}")
            available = torch.cuda.device_count()
            if num_gpus > available:
                raise ValueError(f"Requested {num_gpus} GPUs but only {available} available")
            return list(range(num_gpus))

        elif isinstance(device, list):
            # Explicit list of device IDs
            if not device:
                raise ValueError("Device list cannot be empty")
            available = torch.cuda.device_count()
            for gpu_id in device:
                if not isinstance(gpu_id, int):
                    raise ValueError(f"Device IDs must be integers, got {type(gpu_id)}")
                if gpu_id < 0 or gpu_id >= available:
                    raise ValueError(f"Device ID {gpu_id} out of range [0, {available - 1}]")
            return device

        elif isinstance(device, str):
            # Legacy string support: "auto" or "cuda" → single GPU [0]
            dev_arg = device.strip().lower()
            if dev_arg in ("auto", "cuda", "rocm"):
                return [0]  # Single GPU
            else:
                raise ValueError(
                    f"Unsupported device string '{device}'. Use List[int], int, or 'auto'/'cuda'."
                )

        else:
            raise ValueError(
                f"Invalid device type: {type(device)}. Expected List[int], int, or str."
            )

    def _allocate_device_tensors(self, B: int, gpu_id: int):
        """Allocate device tensors for batch size B on specific GPU, resetting if reused.

        Args:
            B: Batch size
            gpu_id: GPU device ID

        Returns:
            Tensors dictionary on specified GPU
        """
        device_str = self._device_strs[gpu_id]

        # Safety: Check for integer overflow in E * B calculation
        # Triton uses int32 for pointer arithmetic, so E * B must not exceed int32 max
        # This prevents illegal memory access from overflow in kernels that compute offsets like evec * B + bs
        INT32_MAX = 2**31 - 1
        if self.E > 0 and B > 0:
            max_safe_B = INT32_MAX // self.E
            if B > max_safe_B:
                raise RuntimeError(
                    f"Batch size {B} exceeds int32 overflow limit for E={self.E}: "
                    f"E * B = {self.E * B:,} > int32 max ({INT32_MAX:,}). "
                    f"Maximum safe batch size: {max_safe_B}."
                )

        # Check if we're reusing existing tensors
        if B in self._device_tensors_per_gpu[gpu_id]:
            tensors = self._device_tensors_per_gpu[gpu_id][B]
            # Reset stateful tensors when reusing
            self._reset_reused_tensors(tensors)
            return tensors

        # Allocate new tensors on specified GPU (already zero-initialized by torch.zeros/torch.full)
        tensors = {
            # Beliefs - use unified dtype
            # lambda_: Current mixed beliefs [B,V] (updated each iteration)
            "lambda_": torch.zeros(B, self.V, dtype=self.dtype, device=device_str),
            # posterior_ratios: Posterior ratios/marginals [B,V] (matches Rust: posterior_ratios)
            "posterior_ratios": torch.zeros(B, self.V, dtype=self.dtype, device=device_str),
            "hard_dec": torch.zeros(B, self.V, dtype=torch.uint8, device=device_str),
            # Gamma: Memory strengths [B,V] (Rust equivalent: memory_strengths)
            "gamma": torch.zeros(B, self.V, dtype=self.dtype, device=device_str),
            # Syndrome and validation
            "syndrome": torch.zeros(B, self.C, dtype=torch.uint8, device=device_str),
            "check_ok": torch.zeros(B, self.C, dtype=torch.uint8, device=device_str),
            # Solution tracking
            "best_weights": torch.full((B,), float("inf"), dtype=self.dtype, device=device_str),
            "best_errors": torch.zeros(B, self.V, dtype=torch.uint8, device=device_str),
            # Hard decision at the end of the first leg. Returned for lanes with no solution.
            "first_leg_hard_dec": torch.zeros(B, self.V, dtype=torch.uint8, device=device_str),
            "found_count": torch.zeros(B, dtype=torch.int32, device=device_str),
            "valid_solutions": torch.zeros(B, dtype=torch.uint8, device=device_str),
            "sought_solutions": torch.ones(B, dtype=torch.int32, device=device_str),
            # Device-side active lane management (eliminates host-device syncs)
            "active_idx_buffer": torch.zeros(
                B, dtype=torch.int32, device=device_str
            ),  # Pre-allocated buffer for compacted indices
            "active_count": torch.zeros(
                1, dtype=torch.int32, device=device_str
            ),  # Device tensor for count (no sync needed)
            "block_counts": torch.zeros(
                (B + 255) // 256, dtype=torch.int32, device=device_str
            ),  # Per-block counts (BLOCK_SIZE=256)
            "prefix_sum": torch.zeros(
                (B + 255) // 256 + 1, dtype=torch.int32, device=device_str
            ),  # Prefix sum for compaction
            "old_active_count": torch.zeros(1, dtype=torch.int32, device=device_str),
            "count_changed_flag": torch.zeros(1, dtype=torch.uint8, device=device_str),
            "all_done_flag": torch.zeros(1, dtype=torch.uint8, device=device_str),
            "arrival_counter": torch.zeros(1, dtype=torch.int32, device=device_str),
            "in_leg": torch.zeros(1, dtype=torch.uint8, device=device_str),
        }
        # Messages - use unified dtype
        if self.cfg.perf == "throughput":
            tensors["muT"] = torch.zeros(self.E, B, dtype=self.dtype, device=device_str)
            tensors["nuT"] = torch.zeros(self.E, B, dtype=self.dtype, device=device_str)
            # Optional L tensor for delta-trick layered scheduling (only used when self._use_delta_trick)
            if self._use_delta_trick:
                tensors["L"] = torch.zeros(B, self.V, dtype=self.dtype, device=device_str)
        else:
            tensors["mu"] = torch.zeros(B, self.E, dtype=self.dtype, device=device_str)
            tensors["nu"] = torch.zeros(B, self.E, dtype=self.dtype, device=device_str)


        self._device_tensors_per_gpu[gpu_id][B] = tensors

        # Report max safe batch size
        import logging

        _logger = logging.getLogger(__name__)
        if _logger.isEnabledFor(logging.INFO):
            max_safe = (2**31 - 1) // self.E if self.E > 0 else B
            _logger.info(
                "[GPU %d] Batch size %d allocated (max safe batch size: %d for E=%d)",
                gpu_id,
                B,
                max_safe,
                self.E,
            )

        return tensors


    def _reset_reused_tensors(self, tensors: Dict):
        """Reset stateful tensors when reusing from cache.

        Performance: In-place zero operations are much faster than reallocating tensors.
        Only resets what's necessary for correctness.
        """
        # Reset all message buffers to zero (will be properly initialized by init kernels)
        # These MUST be reset as they contain state from previous decode calls
        if self.cfg.perf == "throughput":
            tensors["muT"].zero_()
            tensors["nuT"].zero_()
            if "L" in tensors:
                tensors["L"].zero_()
        else:
            tensors["mu"].zero_()
            tensors["nu"].zero_()

        # Reset beliefs that are used in first iteration BEFORE being overwritten
        # posterior_ratios is used in gamma mixing in the first iteration, so must be clean
        tensors["posterior_ratios"].zero_()
        # lambda_ is immediately overwritten, but reset for safety
        tensors["lambda_"].zero_()

        # Reset solution tracking (these are NOT immediately overwritten)
        tensors["hard_dec"].zero_()
        tensors["gamma"].zero_()
        tensors["best_weights"].fill_(float("inf"))
        tensors["best_errors"].zero_()
        tensors["first_leg_hard_dec"].zero_()
        tensors["found_count"].zero_()
        tensors["valid_solutions"].zero_()
        tensors["sought_solutions"].fill_(1)
        tensors["check_ok"].zero_()

        # Note: syndrome is NOT zeroed here because it's immediately overwritten
        # with new syndrome values, avoiding redundant work.




    def _get_active_count_from_blocks(self, tensors: Dict, B: int, gpu_idx: int) -> torch.Tensor:
        """Get active count by summing block_counts (avoids separate count kernel).

        Args:
            tensors: Device tensors dictionary
            B: Batch size
            gpu_idx: GPU index for device string

        Returns:
            Active count as device tensor (no sync) - use device-side comparison
        """
        device_str = self._device_strs[self.device_ids[gpu_idx]]
        BLOCK_SIZE = 256
        num_blocks = (B + BLOCK_SIZE - 1) // BLOCK_SIZE

        # Count active per block (reuses block_counts from compaction if available)
        count_active_blocks_kernel[(num_blocks,)](
            tensors["active"],
            tensors["block_counts"],
            B=B,
            BLOCK_SIZE=BLOCK_SIZE,
            num_blocks=num_blocks,
        )
        # Sum block counts on device (no sync) - return tensor for device-side comparison
        if "active_count_tensor" not in tensors:
            tensors["active_count_tensor"] = torch.zeros(1, dtype=torch.int32, device=device_str)
        # torch.sum() returns a 0D scalar tensor, use in-place copy to avoid sync
        total_sum = torch.sum(tensors["block_counts"]).to(torch.int32)
        # Copy scalar to 1D tensor element without sync (both on device)
        if total_sum.dim() == 0:
            # 0D scalar - expand to 1D and copy in-place
            tensors["active_count_tensor"].copy_(total_sum.unsqueeze(0))
        else:
            # Already 1D - direct copy
            tensors["active_count_tensor"].copy_(total_sum)
        return tensors["active_count_tensor"]

    def _compact_active_indices_on_device(
        self, tensors: Dict, B: int, gpu_idx: int
    ) -> Tuple[torch.Tensor, int]:
        """Compact active indices on device (replaces torch.nonzero()).

        Args:
            tensors: Device tensors dictionary
            B: Batch size
            gpu_idx: GPU index (for device string if needed)

        Returns:
            Tuple of (active_idx_buffer slice, B_active count)
        """
        BLOCK_SIZE = 256
        num_blocks = (B + BLOCK_SIZE - 1) // BLOCK_SIZE

        # Phase 1: Count active per block (reuse if already computed)
        count_active_blocks_kernel[(num_blocks,)](
            tensors["active"],
            tensors["block_counts"],
            B=B,
            BLOCK_SIZE=BLOCK_SIZE,
            num_blocks=num_blocks,
        )
        # Phase 2: Compute prefix sum on device (no host sync)
        # Use torch.cumsum which is efficient on device
        torch.cumsum(tensors["block_counts"], dim=0, out=tensors["prefix_sum"][1:])
        tensors["prefix_sum"][0] = 0  # First element is 0

        # Phase 3: Compact indices using prefix sum
        # Validate grid size matches buffer size to catch bugs early
        expected_prefix_sum_size = num_blocks + 1
        if expected_prefix_sum_size != tensors["prefix_sum"].shape[0]:
            raise RuntimeError(
                f"Grid size mismatch: num_blocks={num_blocks} (expects prefix_sum size={expected_prefix_sum_size}) "
                f"but prefix_sum has {tensors['prefix_sum'].shape[0]} elements. B={B}, BLOCK_SIZE={BLOCK_SIZE}"
            )
        compact_active_indices_kernel[(num_blocks,)](
            tensors["active"],
            tensors["active_idx_buffer"],
            tensors["prefix_sum"],
            B=B,
            BLOCK_SIZE=BLOCK_SIZE,
            num_blocks=num_blocks,
        )

        # Get count from last element of prefix sum (sync only when needed)
        B_active = int(tensors["prefix_sum"][num_blocks].item())

        return tensors["active_idx_buffer"][:B_active], B_active

    def _launch_c2v_btile(
        self,
        tensors: Dict,
        B: int,
        active_idx: torch.Tensor = None,
        B_active: int = None,
        graph: "CSRGraph" = None,
        gpu_idx: int = 0,
        chk_start: int = 0,
        chk_end: int = 0,
        use_delta: bool = False,
    ):
        """Launch C2V BTILE kernel with optional block range for layered scheduling.

        Args:
            chk_start: Start check index for block-layered (0 = all checks)
            chk_end: End check index for block-layered (0 = all checks, means C)
            use_delta: If True, use delta-trick mode (requires L tensor to be allocated)
        """
        if graph is None:
            graph = self.graphs[gpu_idx]
        device_str = f"{self.device_type}:{self.device_ids[gpu_idx]}"
        if active_idx is None:
            active_idx = torch.arange(B, device=device_str, dtype=torch.int32)
            B_active = B
        alpha = getattr(self, "_alpha_current", self.normalized_min_sum_alpha or 0.0)
        beta = self.offset_min_sum_beta or 0.0
        use_alpha = self.normalized_min_sum_alpha is not None
        use_beta = self.offset_min_sum_beta is not None

        # Use pre-resolved config (resolved once in _resolve_configs)
        btile = self._rc_c2v_btile
        grid_dim1 = (B_active + btile - 1) // btile
        grid_dim0 = (chk_end - chk_start) if chk_end > 0 else self.C
        grid = (grid_dim0, grid_dim1)

        L_tensor = tensors.get("L")
        if L_tensor is None and use_delta:
            raise RuntimeError("Delta-trick mode requires L tensor to be allocated")
        if L_tensor is None:
            if gpu_idx not in self._dummy_L_per_gpu:
                self._dummy_L_per_gpu[gpu_idx] = torch.zeros(
                    1, 1, dtype=self.dtype, device=device_str
                )
            L_tensor = self._dummy_L_per_gpu[gpu_idx]
        c2v_min_sum_btile_kernel[grid](
            tensors["muT"],
            tensors["nuT"],
            graph.chk_ptr,
            graph.chk_edges,
            tensors["syndrome"],
            active_idx,
            tensors["active"],
            B,
            B_active,
            self.C,
            self.E,
            alpha,
            beta,
            use_alpha,
            use_beta,
            self._dtype_enum,
            self._rc_c2v_block,
            btile,
            chk_start,
            chk_end,
            graph.edge_var,
            L_tensor,
            self.V,
            use_delta,
            num_warps=self._rc_c2v_warps,
            num_stages=self._rc_c2v_stages,
        )

    def _launch_c2v_kernel(self, tensors: Dict, B: int, graph: "CSRGraph" = None, gpu_idx: int = 0):
        """Launch check-to-variable kernel."""
        if graph is None:
            graph = self.graphs[gpu_idx]
        alpha = getattr(self, "_alpha_current", self.normalized_min_sum_alpha or 0.0)
        beta = self.offset_min_sum_beta or 0.0
        use_alpha = self.normalized_min_sum_alpha is not None
        use_beta = self.offset_min_sum_beta is not None

        rows = self._rc_c2v_rows
        grid = (((B * self.C) + rows - 1) // rows,)

        c2v_min_sum_kernel[grid](
            tensors["mu"],
            tensors["nu"],
            graph.chk_ptr,
            graph.chk_edges,
            tensors["syndrome"],
            tensors["active"],
            B=B,
            C=self.C,
            E=self.E,
            alpha=alpha,
            beta=beta,
            use_alpha=use_alpha,
            use_beta=use_beta,
            dtype_enum=self._dtype_enum,
            BLOCK_SIZE=self._rc_c2v_block,
            ROWS_PER_CHK=rows,
            num_warps=self._rc_c2v_warps,
            num_stages=self._rc_c2v_stages,
        )

    def _launch_v2c_fused_gamma_kernel(
        self,
        tensors: Dict,
        B: int,
        graph: "CSRGraph" = None,
        log_prior_ratios: torch.Tensor = None,
        gpu_idx: int = 0,
        prior_batched: bool = False,
    ):
        """Launch variable-to-check and marginals kernel with fused gamma mixing."""
        if graph is None:
            graph = self.graphs[gpu_idx]
        if log_prior_ratios is None:
            log_prior_ratios = self.log_prior_ratios_per_gpu[gpu_idx]

        rows = self._rc_v2c_rows
        grid = (((B * self.V) + rows - 1) // rows,)

        v2c_and_marginals_fused_gamma_kernel[grid](
            tensors["nu"],
            tensors["mu"],
            graph.var_ptr,
            graph.var_edges,
            tensors["lambda_"],
            tensors["posterior_ratios"],
            tensors["hard_dec"],
            tensors["active"],
            log_prior_ratios,
            tensors["gamma"],
            B=B,
            V=self.V,
            E=self.E,
            dtype_enum=self._dtype_enum,
            BLOCK_SIZE=self._rc_v2c_block,
            ROWS_PER_VAR=rows,
            PRIOR_BATCHED=prior_batched,
            num_warps=self._rc_v2c_warps,
            num_stages=self._rc_v2c_stages,
        )

    def _launch_parity_check_kernel(
        self, tensors: Dict, B: int, graph: "CSRGraph" = None, gpu_idx: int = 0
    ):
        """Launch parity check kernel with pre-resolved config."""
        if graph is None:
            graph = self.graphs[gpu_idx]
        assert not (self.cfg.perf == "throughput"), (
            "Throughput mode must use parity_from_hard_kernel"
        )
        total = B * self.C
        grid = ((total + self.cfg.rows_per_chk - 1) // self.cfg.rows_per_chk,)
        meta_base = dict(B=B, C=self.C, V=self.V, E=self.E, ROWS_PER_CHK=self.cfg.rows_per_chk)

        parity_per_check_kernel[grid](
            tensors["posterior_ratios"],
            graph.chk_ptr,
            graph.chk_edges,
            graph.edge_var,
            tensors["syndrome"],
            tensors["check_ok"],
            **meta_base,
            BLOCK_SIZE=self._rc_parity_block,
            num_warps=self._rc_parity_warps,
            num_stages=self._rc_parity_stages,
        )

    def _launch_init_kernel(
        self,
        tensors: Dict,
        B: int,
        graph: "CSRGraph" = None,
        log_prior_ratios: torch.Tensor = None,
        gpu_idx: int = 0,
    ):
        """Launch message initialization kernel for default mode.

        Initializes messages to match Rust:
        - nu (check_to_variable) -> zero
        - mu (variable_to_check) -> log_prior_ratios
        """
        if graph is None:
            graph = self.graphs[gpu_idx]
        if log_prior_ratios is None:
            log_prior_ratios = self.log_prior_ratios_per_gpu[gpu_idx]
        N = B * self.E
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        # Estimate: init kernel uses default 1 warp per program
        num_warps = 1
        dtype_enum = self._dtype_enum
        init_messages_kernel[grid](
            tensors["mu"],
            tensors["nu"],
            graph.edge_var,
            log_prior_ratios.squeeze(0),  # [V] prior LLRs
            B,
            self.E,
            self.V,
            dtype_enum=dtype_enum,
            BLOCK=BLOCK,
        )

    def _launch_init_messages_with_priors_EB(
        self,
        muT: torch.Tensor,
        nuT: torch.Tensor,
        B: int,
        active_idx: Optional[torch.Tensor] = None,
        B_active: Optional[int] = None,
        graph: "CSRGraph" = None,
        log_prior_ratios: Optional[torch.Tensor] = None,
        gpu_idx: int = 0,
        prior_batched: bool = False,
    ):
        """Launch initialization kernel for throughput mode: nu=zero, mu=priors (Rust-compatible).

        This matches Rust's initialization:
        - initialize_check_to_variable() -> nu = zero
        - initialize_variable_to_check() -> mu = log_prior_ratios

        If active_idx and B_active are provided and B_active < B, only initializes messages for active lanes,
        dramatically reducing memory traffic when B_active << B.

        Args:
            muT, nuT: [E,B] message tensors
            B: total batch size
            active_idx: Optional [B_active] compact list of active batch indices
            B_active: Optional number of active lanes (if None, uses B)
            graph: CSRGraph instance (defaults to self.graphs[gpu_idx])
            log_prior_ratios: [1, V] log prior ratios tensor (defaults to self.log_prior_ratios_per_gpu[gpu_idx])
            gpu_idx: GPU index (for device string)
        """
        if graph is None:
            graph = self.graphs[gpu_idx]
        if log_prior_ratios is None:
            log_prior_ratios = self.log_prior_ratios_per_gpu[gpu_idx]
        device_str = self._device_strs[self.device_ids[gpu_idx]]

        prior_arg = log_prior_ratios if prior_batched else log_prior_ratios.squeeze(0)

        # Determine if we should use active lanes only
        use_active = (
            (active_idx is not None)
            and (B_active is not None)
            and (B_active < B)
            and (B_active > 0)
        )

        if use_active:
            # Active lanes path: use hardcoded config for now (will add autotuning later)
            # Note: Kernels use int64 for offset calculations, so overflow is prevented
            total = self.E * B_active
            dummy_active_idx = active_idx
            grid = ((total + 1024 - 1) // 1024,)
            num_warps = 1
            dtype_enum = self._dtype_enum
            init_messages_with_priors_EB_kernel[grid](
                muT,
                nuT,
                graph.edge_var,
                prior_arg,  # [V] shared or [B,V] per-shot prior LLRs
                self.E,
                self.V,
                B,
                dummy_active_idx,  # active_idx
                B_active,  # Number of active lanes
                use_active=use_active,  # Flag to enable active-lanes-only mode
                dtype_enum=dtype_enum,
                BLOCK=1024,
                PRIOR_BATCHED=prior_batched,
                num_warps=num_warps,
                num_stages=1,
            )
        else:
            # Full initialization path: use autotuning
            # Note: Kernels use int64 for offset calculations, so overflow is prevented
            total = self.E * B
            B_active = B  # Use full batch
            # Create dummy tensor (won't be used, but Triton requires it)
            dummy_active_idx = torch.zeros(1, dtype=torch.int32, device=device_str)

            problem_key = {
                "B": B,
                "E": self.E,
                "V": self.V,
                "dtype": self.dtype_str,
                "use_active": False,  # Full path
                "sparsity_hash": graph.sparsity_hash,
            }

            def _grid_fn(cfg):
                block = int(cfg.get("BLOCK", 1024))
                return ((total + block - 1) // block,)

            dtype_enum = self._dtype_enum

            saved = self._get_tuned_meta(
                kernel_name="init_messages_with_priors_EB_kernel",
                problem_key=problem_key,
                build_cfgs_fn=build_init_messages_configs,
                kernel=init_messages_with_priors_EB_kernel,
                grid=(1,),  # Dummy grid, overridden by grid_fn
                args=(
                    muT,
                    nuT,
                    graph.edge_var,
                    prior_arg,  # [V] shared or [B,V] per-shot prior LLRs
                    self.E,
                    self.V,
                    B,
                    dummy_active_idx,  # active_idx (dummy for full path)
                    B_active,  # Number of active lanes (B for full path)
                ),
                meta_base=dict(
                    use_active=False,  # Full path
                    dtype_enum=dtype_enum,
                    PRIOR_BATCHED=prior_batched,
                ),
                number=20,
                grid_fn=_grid_fn,
                log_name=f"init_messages_with_priors_EB_kernel[gpu={gpu_idx}]",
            )

            block = int(saved.get("BLOCK", 1024))
            grid = ((total + block - 1) // block,)
            num_warps = int(saved.get("num_warps", 1))
            num_stages = int(saved.get("num_stages", 1))

            dtype_enum = self._dtype_enum
            init_messages_with_priors_EB_kernel[grid](
                muT,
                nuT,
                graph.edge_var,
                prior_arg,  # [V] shared or [B,V] per-shot prior LLRs
                self.E,
                self.V,
                B,
                dummy_active_idx,  # active_idx (dummy for full path)
                B_active,  # Number of active lanes (B for full path)
                use_active=False,  # Full path
                dtype_enum=dtype_enum,
                BLOCK=block,
                PRIOR_BATCHED=prior_batched,
                num_warps=num_warps,
                num_stages=num_stages,
            )

    def _launch_v2c_btile(
        self,
        tensors: Dict,
        B: int,
        write_hard: bool,
        active_idx: torch.Tensor = None,
        B_active: int = None,
        graph: "CSRGraph" = None,
        log_prior_ratios: torch.Tensor = None,
        gpu_idx: int = 0,
        block_vars: torch.Tensor = None,
        block_var_ptr: torch.Tensor = None,
        block_idx: int = -1,
        prior_batched: bool = False,
    ):
        """Launch variable-to-check BTILE kernel with fused gamma mixing.

        Note: posterior_ratios is always stored (required for next iteration's gamma mixing).

        Args:
            block_vars: Variable indices for block-layered (None = all vars)
            block_var_ptr: Prefix sum for block_vars (None = not used)
            block_idx: Block index for block-layered (-1 = flood-fill, use all vars)
        """
        if graph is None:
            graph = self.graphs[gpu_idx]
        if log_prior_ratios is None:
            log_prior_ratios = self.log_prior_ratios_per_gpu[gpu_idx]
        device_str = self._device_strs[self.device_ids[gpu_idx]]
        if active_idx is None:
            active_idx = torch.arange(B, device=device_str, dtype=torch.int32)
            B_active = B

        # Use pre-resolved config
        btile = self._rc_v2c_btile
        grid_dim1 = (B_active + btile - 1) // btile
        if block_idx >= 0 and block_var_ptr is not None:
            block_var_ptr_cpu = self._block_var_ptr_cpu
            var_start = int(block_var_ptr_cpu[block_idx])
            var_end = int(block_var_ptr_cpu[block_idx + 1])
            grid_dim0 = var_end - var_start
        else:
            grid_dim0 = self.V
        grid = (grid_dim0, grid_dim1)

        # Block-layered vs flood-fill tensor selection
        if block_idx >= 0 and block_vars is not None and block_var_ptr is not None:
            block_vars_actual = block_vars
            block_var_ptr_actual = block_var_ptr
            block_idx_actual = block_idx
        else:
            block_vars_actual = (
                graph.var_edges[:1] if graph.var_edges.shape[0] > 0 else graph.var_ptr[:1]
            )
            block_var_ptr_actual = graph.var_ptr[:2]
            block_idx_actual = -1

        v2c_and_gamma_btile_kernel[grid](
            tensors["nuT"],
            tensors["muT"],
            graph.var_ptr,
            graph.var_edges,
            tensors["lambda_"],
            log_prior_ratios,
            tensors["gamma"],
            tensors["posterior_ratios"],
            tensors["hard_dec"],
            active_idx,
            tensors["active"],
            B=B,
            B_active=B_active,
            V=self.V,
            E=self.E,
            dtype_enum=self._dtype_enum,
            BLOCK_SIZE=self._rc_v2c_block,
            BTILE=btile,
            WRITE_HARD=write_hard,
            block_vars=block_vars_actual,
            block_var_ptr=block_var_ptr_actual,
            block_idx=block_idx_actual,
            PRIOR_BATCHED=prior_batched,
            num_warps=self._rc_v2c_warps,
            num_stages=self._rc_v2c_stages,
        )

    def _launch_parity_from_hard_kernel(
        self, tensors: Dict, B: int, graph: "CSRGraph" = None, gpu_idx: int = 0
    ):
        """Launch parity check using hard_dec (for throughput mode)."""
        if graph is None:
            graph = self.graphs[gpu_idx]
        total = B * self.C
        grid = ((total + self.cfg.rows_per_chk - 1) // self.cfg.rows_per_chk,)
        problem_key = {
            "B": B,
            "C": self.C,
            "V": self.V,
            "E": self.E,
            "ROWS_PER_CHK": self.cfg.rows_per_chk,
            "dtype": self.dtype_str,
            "sparsity_hash": graph.sparsity_hash,
        }
        meta_base = dict(B=B, C=self.C, V=self.V, E=self.E, ROWS_PER_CHK=self.cfg.rows_per_chk)

        saved = self._get_tuned_meta(
            kernel_name="parity_from_hard_kernel",
            problem_key=problem_key,
            build_cfgs_fn=build_parity_configs,
            kernel=parity_from_hard_kernel,
            grid=grid,
            args=(
                tensors["hard_dec"],
                graph.chk_ptr,
                graph.chk_edges,
                graph.edge_var,
                tensors["syndrome"],
                tensors["check_ok"],
            ),
            meta_base=meta_base,
            number=20,
            log_name=f"parity_from_hard_kernel[gpu={gpu_idx}]",
        )

        block_size = int(saved["BLOCK_SIZE"])
        num_warps = int(saved.get("num_warps", 0))
        num_stages = int(saved.get("num_stages", 0))

        parity_from_hard_kernel[grid](
            tensors["hard_dec"],
            graph.chk_ptr,
            graph.chk_edges,
            graph.edge_var,
            tensors["syndrome"],
            tensors["check_ok"],
            **meta_base,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    def _launch_parity_from_hard_compact_with_zero_kernel(
        self, tensors: Dict, B: int, graph: "CSRGraph" = None, gpu_idx: int = 0
    ):
        """Launch fused parity check kernel: zeros inactive lanes and checks parity for active lanes."""
        if graph is None:
            graph = self.graphs[gpu_idx]
        total = B * self.C
        grid = ((total + self.cfg.rows_per_chk - 1) // self.cfg.rows_per_chk,)

        parity_from_hard_compact_with_zero_kernel[grid](
            tensors["hard_dec"],
            graph.chk_ptr,
            graph.chk_edges,
            graph.edge_var,
            tensors["syndrome"],
            tensors["check_ok"],
            tensors["active"],
            B,
            self.C,
            self.V,
            self.E,
            BLOCK_SIZE=self._rc_parity_block,
            ROWS_PER_CHK=self.cfg.rows_per_chk,
            num_warps=self._rc_parity_warps,
            num_stages=self._rc_parity_stages,
        )

    def _launch_parity_from_hard_compact_kernel(
        self,
        tensors: Dict,
        B: int,
        active_idx: torch.Tensor,
        B_active: int,
        graph: "CSRGraph" = None,
        gpu_idx: int = 0,
    ):
        """Launch parity check kernel for active lanes only (compact version).

        Uses active_idx indirection to process only B_active lanes instead of all B.
        """
        if graph is None:
            graph = self.graphs[gpu_idx]
        total = B_active * self.C
        grid = ((total + self.cfg.rows_per_chk - 1) // self.cfg.rows_per_chk,)

        parity_from_hard_compact_kernel[grid](
            tensors["hard_dec"],
            graph.chk_ptr,
            graph.chk_edges,
            graph.edge_var,
            tensors["syndrome"],
            tensors["check_ok"],
            active_idx,
            B,
            B_active,
            self.C,
            self.V,
            self.E,
            BLOCK_SIZE=self._rc_parity_block,
            ROWS_PER_CHK=self.cfg.rows_per_chk,
            num_warps=self._rc_parity_warps,
            num_stages=self._rc_parity_stages,
        )



    def _check_parity_and_select(
        self,
        tensors: Dict,
        B: int,
        active_idx: torch.Tensor = None,
        B_active: int = None,
        *,
        force_hard: bool = False,
        graph: "CSRGraph" = None,
        gpu_idx: int = 0,
        prior_batched: bool = False,
        wj_override: Optional[torch.Tensor] = None,
    ):
        """Check parity and select best solutions fully on device (no host reductions).

        Args:
            tensors: device tensors dict
            B: batch size
            active_idx: optional compact active indices
            B_active: number of active lanes
            force_hard: if True, use hard_dec parity path regardless of perf mode
            graph: CSRGraph instance (defaults to self.graphs[gpu_idx])
            gpu_idx: GPU index (for device string if needed)
        """
        if graph is None:
            graph = self.graphs[gpu_idx]
        if B_active is not None and B_active == 0:
            return
        # Launch the right parity kernel
        if self.cfg.perf == "throughput" or force_hard:
            if (
                self.cfg.perf == "throughput"
                and B_active is not None
                and active_idx is not None
                and B_active < B
            ):
                self._launch_parity_from_hard_compact_kernel(
                    tensors, B, active_idx, B_active, graph=graph, gpu_idx=gpu_idx
                )
            else:
                self._launch_parity_from_hard_compact_with_zero_kernel(
                    tensors, B, graph=graph, gpu_idx=gpu_idx
                )
        else:
            self._launch_parity_check_kernel(tensors, B, graph=graph, gpu_idx=gpu_idx)

        # Device-side select using pre-resolved config
        gpu_id = self.device_ids[gpu_idx]
        wj_gpu = wj_override if prior_batched else self.wj_per_gpu[gpu_id]

        check_and_select_kernel[(B,)](
            tensors["check_ok"],
            tensors["posterior_ratios"],
            tensors["hard_dec"],
            wj_gpu,
            tensors["best_weights"],
            tensors["best_errors"],
            tensors["valid_solutions"],
            tensors["found_count"],
            tensors["active"],
            B,
            self.C,
            self.V,
            dtype_enum=self._dtype_enum,
            BLOCK_C=self._rc_select_block_c,
            BLOCK_V=self._rc_select_block_v,
            NUM_SOLS=int(self.stop_nconv),
            PRIOR_BATCHED=prior_batched,
            num_warps=self._rc_select_warps,
            num_stages=1,
        )

    def decode(
        self,
        syndromes: torch.Tensor,
        *,
        init_posterior: Optional[torch.Tensor] = None,
        init_gamma: Optional[torch.Tensor] = None,
        skip_relay_legs: bool = False,
        pre_iter_override: Optional[int] = None,
        return_last_posterior: bool = False,
        prior_llr: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Decode syndromes using Relay-BP algorithm with multi-GPU support.

        Args:
            syndromes: [B, C] syndrome bits (uint8)
            init_posterior: optional [B, V] tensor; when set, overrides the default
                log_prior_ratios initialization of posterior_ratios so the
                pre-iter phase resumes from a caller-supplied posterior, e.g. to
                continue decoding from a previous call.
            init_gamma: optional [B, V] tensor; when set, overrides the per-lane
                gamma vector used during the pre-iter phase (replaces the scalar
                gamma0 broadcast), so each shot can use its own memory strengths.
            skip_relay_legs: when True, return immediately after the pre-iter
                phase instead of running the relay legs (num_sets loop).
            pre_iter_override: optional override for the number of pre-iter
                iterations (replaces self.pre_iter for this call only).
            return_last_posterior: when True, the result dict additionally
                contains "last_posterior": [B, V]: a clone of
                posterior_ratios at the end of decode.
            prior_llr: optional [B, V] tensor of per-shot prior LLRs replacing the
                shared [1, V] log_prior_ratios for this call (each shot gets
                its own prior).

        Returns:
            Dictionary with keys:
                - "errors": [B, V] decoded errors (uint8) or packed if bitpack_output=True
                - "weights": [B] solution weights (float32)
                - "valid_mask": [B] bool, True where found_count >= stop_nconv
                - "iterations": [B] iteration counts
                - "found_count": [B] int32 tensor of number of valid solutions found per lane
                - "unconverged_indices": [U] int32 tensor of batch indices that didn't converge
                  (found_count < stop_nconv). Empty tensor if all converged or fast_max_unconverged_pct=None.
                  Only non-empty when fast_max_unconverged_pct is set and early exit occurred.
                - "last_posterior" (optional): [B, V] cloned posterior at end of decode.
        """
        B = syndromes.shape[0]

        if syndromes.shape[1] != self.C:
            raise ValueError(f"Expected {self.C} syndrome bits, got {syndromes.shape[1]}")

        # Validate the advanced kwargs.
        advanced_args_used = (
            init_posterior is not None
            or init_gamma is not None
            or skip_relay_legs
            or pre_iter_override is not None
            or return_last_posterior
            or prior_llr is not None
        )
        if advanced_args_used:
            if self.cfg.perf != "throughput":
                raise ValueError(
                    "init_posterior / init_gamma / skip_relay_legs / pre_iter_override / "
                    "return_last_posterior are only supported in throughput perf mode "
                    f"(got perf={self.cfg.perf!r})."
                )
            if self.num_gpus > 1:
                raise ValueError(
                    "init_posterior / init_gamma / skip_relay_legs / pre_iter_override / "
                    "return_last_posterior / prior_llr require single-GPU decode "
                    f"(self.num_gpus={self.num_gpus})."
                )
            if init_posterior is not None and tuple(init_posterior.shape) != (B, self.V):
                raise ValueError(
                    f"init_posterior shape {tuple(init_posterior.shape)} != ({B}, {self.V})"
                )
            if init_gamma is not None and tuple(init_gamma.shape) != (B, self.V):
                raise ValueError(f"init_gamma shape {tuple(init_gamma.shape)} != ({B}, {self.V})")
            if prior_llr is not None and tuple(prior_llr.shape) != (B, self.V):
                raise ValueError(f"prior_llr shape {tuple(prior_llr.shape)} != ({B}, {self.V})")

        # Path for single GPU: avoid thread/gather overhead
        if self.num_gpus == 1:
            gpu_id = self.device_ids[0]
            gpu_idx = 0  # Index in our device_ids list (always 0 for single GPU)
            device_str = f"{self.device_type}:{gpu_id}"

            # Set device context for Triton kernel launches
            with torch.cuda.device(gpu_id):
                # Move syndromes to GPU (if not already there)
                if syndromes.device.type != self.device_type or syndromes.device.index != gpu_id:
                    syndromes = syndromes.to(device_str)

                # Allocate tensors on this GPU
                tensors = self._allocate_device_tensors(B, gpu_id)

                # Use GPU-specific graph and log_prior_ratios
                graph = self.graphs[gpu_idx]
                log_prior_ratios = self.log_prior_ratios_per_gpu[gpu_idx]
                explicit_gammas = (
                    self.explicit_gammas_per_gpu[gpu_idx] if self.explicit_gammas_per_gpu else None
                )

                prior_batched = prior_llr is not None
                wj_override = None
                if prior_batched:
                    plr = prior_llr.to(device=device_str, dtype=self.dtype).contiguous()
                    log_prior_ratios = plr
                    wj_override = torch.where(
                        torch.isfinite(plr), plr, torch.zeros_like(plr)
                    ).contiguous()

                # Run decode loop directly (no thread overhead)
                return self._decode_loop_single_gpu(
                    syndromes,
                    tensors,
                    graph,
                    log_prior_ratios,
                    explicit_gammas,
                    gpu_idx,
                    init_posterior=init_posterior,
                    init_gamma=init_gamma,
                    skip_relay_legs=skip_relay_legs,
                    pre_iter_override=pre_iter_override,
                    return_last_posterior=return_last_posterior,
                    prior_batched=prior_batched,
                    wj_override=wj_override,
                )
        else:
            # Multi-GPU path: split batch and process in parallel
            batch_splits = self._split_batch(B, self.num_gpus)
            results = self._decode_parallel(batch_splits, syndromes)
            return self._gather_results(results, B, batch_splits)

    def _split_batch(self, B: int, num_gpus: int) -> List[Tuple[int, int]]:
        """Split batch into roughly equal chunks across GPUs.

        Args:
            B: Total batch size
            num_gpus: Number of GPUs

        Returns:
            List of (start, end) tuples for each GPU
        """
        batch_per_gpu = B // num_gpus
        remainder = B % num_gpus

        splits = []
        start = 0
        for i in range(num_gpus):
            # Distribute remainder across first few GPUs
            size = batch_per_gpu + (1 if i < remainder else 0)
            end = start + size
            splits.append((start, end))
            start = end

        return splits

    def _decode_parallel(
        self,
        batch_splits: List[Tuple[int, int]],
        syndromes: torch.Tensor,
    ) -> List[Dict[str, torch.Tensor]]:
        """Process batch splits on multiple GPUs in parallel.

        Args:
            batch_splits: List of (start, end) tuples for each GPU
            syndromes: [B, C] syndrome tensor

        Returns:
            List of result dictionaries
        """
        import threading

        results = [None] * self.num_gpus
        threads = []

        def decode_on_gpu(gpu_idx: int, start: int, end: int):
            """Decode batch slice on specific GPU."""
            # Set device context
            with torch.cuda.device(self.device_ids[gpu_idx]):
                # Extract batch slice
                batch_slice = syndromes[start:end].clone()  # Clone to avoid issues
                B_slice = batch_slice.shape[0]

                if B_slice == 0:
                    # Empty batch slice
                    results[gpu_idx] = None
                    return

                # Move to target GPU
                device_str = f"{self.device_type}:{self.device_ids[gpu_idx]}"
                batch_slice = batch_slice.to(device_str)

                # Allocate tensors on this GPU
                tensors = self._allocate_device_tensors(B_slice, self.device_ids[gpu_idx])

                # Use GPU-specific graph and log_prior_ratios
                graph = self.graphs[gpu_idx]
                log_prior_ratios = self.log_prior_ratios_per_gpu[gpu_idx]
                explicit_gammas = (
                    self.explicit_gammas_per_gpu[gpu_idx] if self.explicit_gammas_per_gpu else None
                )

                # Run decode loop on this GPU
                result = self._decode_loop_single_gpu(
                    batch_slice,
                    tensors,
                    graph,
                    log_prior_ratios,
                    explicit_gammas,
                    gpu_idx,
                )
                results[gpu_idx] = result

        # Launch threads for parallel execution
        for gpu_idx, (start, end) in enumerate(batch_splits):
            if start < end:  # Only if batch slice is non-empty
                thread = threading.Thread(target=decode_on_gpu, args=(gpu_idx, start, end))
                thread.start()
                threads.append(thread)

        # Wait for all threads
        for thread in threads:
            thread.join()

        return results

    def _gather_results(
        self,
        results: List[Dict[str, torch.Tensor]],
        B: int,
        batch_splits: List[Tuple[int, int]],
    ) -> Dict[str, torch.Tensor]:
        """Gather results from all GPUs and concatenate.

        Args:
            results: List of result dictionaries from each GPU
            B: Total batch size
            batch_splits: List of (start, end) tuples for each GPU (for unconverged index adjustment)

        Returns:
            Concatenated result dictionary
        """
        # Move all results to first GPU for concatenation
        target_device = f"{self.device_type}:{self.device_ids[0]}"

        gathered = {
            "errors": [],
            "weights": [],
            "valid_mask": [],
            "iterations": [],
            "found_count": [],
            "unconverged_indices": [],
        }

        for gpu_idx, result in enumerate(results):
            if result is not None:
                gathered["errors"].append(result["errors"].to(target_device))
                gathered["weights"].append(result["weights"].to(target_device))
                gathered["valid_mask"].append(result["valid_mask"].to(target_device))
                gathered["iterations"].append(result["iterations"].to(target_device))
                gathered["found_count"].append(result["found_count"].to(target_device))
                # Adjust unconverged indices to global batch coordinates
                if "unconverged_indices" in result and len(result["unconverged_indices"]) > 0:
                    start, _ = batch_splits[gpu_idx]
                    local_indices = result["unconverged_indices"].to(target_device)
                    global_indices = start + local_indices
                    gathered["unconverged_indices"].append(global_indices)


        # Concatenate on target device
        # Handle case where all results might be None (shouldn't happen, but be safe)
        if not gathered["errors"]:
            raise RuntimeError("No valid results from any GPU. All decode threads may have failed.")

        # Concatenate unconverged indices if any
        if gathered["unconverged_indices"]:
            unconverged = torch.cat(gathered["unconverged_indices"], dim=0)
        else:
            unconverged = torch.tensor([], dtype=torch.int32, device=target_device)

        out = {
            "errors": torch.cat(gathered["errors"], dim=0),
            "weights": torch.cat(gathered["weights"], dim=0),
            "valid_mask": torch.cat(gathered["valid_mask"], dim=0),
            "iterations": torch.cat(gathered["iterations"], dim=0),
            "found_count": torch.cat(gathered["found_count"], dim=0),
            "unconverged_indices": unconverged,
        }

        return out

    def _run_bp_iteration(
        self,
        t: int,
        tensors: Dict,
        B: int,
        active_idx: torch.Tensor,
        B_active: int,
        graph: "CSRGraph",
        log_prior_ratios: torch.Tensor,
        gpu_idx: int,
        in_leg_python: bool,
        force_hard: bool = False,
        skip_all_done_sync: bool = False,
        prior_batched: bool = False,
        wj_override: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, int, bool]:
        """Run a single BP iteration (shared between pre-iter and relay legs).

        Args:
            t: Iteration index within the current leg
            tensors: Device tensors dictionary
            B: Batch size
            active_idx: Active lane indices tensor
            B_active: Number of active lanes
            graph: CSRGraph instance
            log_prior_ratios: [1, V] log prior ratios
            gpu_idx: GPU index
            in_leg_python: Whether we're in a relay leg (affects sought_solutions logic)
            force_hard: Whether to force hard_dec parity check (pre-iter only)

        Returns:
            Tuple of (active_idx, B_active, all_done)
        """
        # Make explicit local copies of active_idx / B_active so we can mutate
        # them safely without confusing Python's scoping rules. The function
        # returns the updated versions to the caller.
        local_active_idx = active_idx
        local_B_active = B_active

        # Compute Rust-like alpha schedule
        if self.normalized_min_sum_alpha is None:
            self._alpha_current = 1.0
        else:
            a = float(self.normalized_min_sum_alpha)
            if a == 0.0:
                s = float(self.alpha_iteration_scaling_factor or 1.0)
                self._alpha_current = 1.0 - (2.0 ** (-((t + 1) / s)))
            elif a < 0.0:
                self._alpha_current = 1.0
            else:
                self._alpha_current = a

        # Build active lane compaction for efficiency (only for non-throughput mode)
        # In throughput mode we keep the caller-provided active index set.
        # For batch=1, skip expensive compaction (always active, no sync needed)
        if self.cfg.perf == "default":
            if B == 1:
                # Skip compaction for batch=1 - always active, avoid sync overhead
                device_str = self._device_strs[self.device_ids[gpu_idx]]
                compact_active_idx = torch.tensor([0], dtype=torch.int32, device=device_str)
                compact_B_active = 1
            else:
                compact_active_idx, compact_B_active = self._compact_active_indices_on_device(
                    tensors, B, gpu_idx
                )
        else:
            compact_active_idx, compact_B_active = local_active_idx, local_B_active

        # Message passing: C2V and V2C
        if self.cfg.perf == "throughput":
            if local_B_active == 0:
                pass  # nothing to do for BTILE kernels this iteration
            else:
                # Check if we should use layered scheduling for this iteration.
                # layered_k_iters semantics:
                #   0  -> flood-fill only (no layered iterations)
                #  <0  -> always layered (all iterations in layered mode)
                #  >0  -> use layered for first K iterations, then flood-fill
                use_layered = (
                    self.scheduling == "layered"
                    and self.block_metadata is not None
                    and (self.layered_k_iters < 0 or t < self.layered_k_iters)
                )

                if use_layered:
                    # Block-layered scheduling: process blocks sequentially
                    block_meta = self.block_metadata
                    num_blocks = block_meta["num_blocks"]
                    block_chk_start = block_meta["block_chk_start"]
                    block_chk_end = block_meta["block_chk_end"]
                    block_var_ptr = block_meta["block_var_ptr"]
                    block_vars = block_meta["block_vars"]

                    # Rotate block order (simple rotation: shift by iteration)
                    # This helps with early syndrome drop by varying block processing order
                    block_perm = [(t + b) % num_blocks for b in range(num_blocks)]

                    # Use CPU-side precomputed ranges
                    block_chk_start_cpu = block_meta["block_chk_start_cpu"]
                    block_chk_end_cpu = block_meta["block_chk_end_cpu"]
                    block_var_ptr_cpu = block_meta["block_var_ptr_cpu"]

                    if self.layered_block_policy == "color_use_delta_trick":
                        # Delta-trick layered path: use modified C2V that updates L in-place
                        for block_idx_perm in block_perm:
                            chk_start = int(block_chk_start_cpu[block_idx_perm])
                            chk_end = int(block_chk_end_cpu[block_idx_perm])
                            if chk_end <= chk_start:
                                continue
                            var_start = int(block_var_ptr_cpu[block_idx_perm])
                            var_end = int(block_var_ptr_cpu[block_idx_perm + 1])
                            if var_end <= var_start:
                                continue
                            # C2V delta-trick for this block (updates nuT and L)
                            self._launch_c2v_btile(
                                tensors,
                                B,
                                local_active_idx,
                                local_B_active,
                                graph=graph,
                                gpu_idx=gpu_idx,
                                chk_start=chk_start,
                                chk_end=chk_end,
                                use_delta=True,
                            )
                        # After all blocks, update posterior_ratios and hard_dec from L
                        BLOCK_V = 128
                        bt = BLOCK_V
                        grid_dim1 = (local_B_active + bt - 1) // bt
                        grid = (self.V, grid_dim1)
                        post_from_L_kernel[grid](
                            tensors["L"],
                            tensors["posterior_ratios"],
                            tensors["hard_dec"],
                            local_active_idx,
                            B,
                            local_B_active,
                            self.V,
                            dtype_enum=self._dtype_enum,
                            BLOCK_V=BLOCK_V,
                        )
                    else:
                        for block_idx_perm in block_perm:
                            chk_start = int(block_chk_start_cpu[block_idx_perm])
                            chk_end = int(block_chk_end_cpu[block_idx_perm])

                            # Skip empty blocks (can occur with edge-balanced splitting on irregular degree distributions)
                            if chk_end <= chk_start:
                                continue

                            # Get variable range for this block
                            var_start = int(block_var_ptr_cpu[block_idx_perm])
                            var_end = int(block_var_ptr_cpu[block_idx_perm + 1])

                            # Skip if no variables in this block
                            if var_end <= var_start:
                                continue

                            # C2V for block
                            self._launch_c2v_btile(
                                tensors,
                                B,
                                local_active_idx,
                                local_B_active,
                                graph=graph,
                                gpu_idx=gpu_idx,
                                chk_start=chk_start,
                                chk_end=chk_end,
                            )
                            # V2C for block vars
                            self._launch_v2c_btile(
                                tensors,
                                B,
                                write_hard=True,
                                active_idx=local_active_idx,
                                B_active=local_B_active,
                                graph=graph,
                                log_prior_ratios=log_prior_ratios,
                                gpu_idx=gpu_idx,
                                block_vars=block_vars,
                                block_var_ptr=block_var_ptr,
                                block_idx=block_idx_perm,
                                prior_batched=prior_batched,
                            )
                else:
                    # Flood-fill scheduling: process all checks then all vars
                    self._launch_c2v_btile(
                        tensors,
                        B,
                        local_active_idx,
                        local_B_active,
                        graph=graph,
                        gpu_idx=gpu_idx,
                    )
                    self._launch_v2c_btile(
                        tensors,
                        B,
                        write_hard=True,
                        active_idx=local_active_idx,
                        B_active=local_B_active,
                        graph=graph,
                        log_prior_ratios=log_prior_ratios,
                        gpu_idx=gpu_idx,
                        block_vars=None,
                        block_var_ptr=None,
                        block_idx=-1,
                        prior_batched=prior_batched,
                    )
        else:
            self._launch_c2v_kernel(tensors, B, graph=graph, gpu_idx=gpu_idx)
            # Fused V2C + gamma mixing updates lambda_ internally
            self._launch_v2c_fused_gamma_kernel(
                tensors,
                B,
                graph=graph,
                log_prior_ratios=log_prior_ratios,
                gpu_idx=gpu_idx,
                prior_batched=prior_batched,
            )

        # Check parity and select solutions every iteration
        # Use compacted indices in non-throughput mode; in throughput mode we
        # pass through the caller-provided active index set.
        self._check_parity_and_select(
            tensors,
            B,
            compact_active_idx,
            compact_B_active,
            force_hard=force_hard,
            graph=graph,
            gpu_idx=gpu_idx,
            prior_batched=prior_batched,
            wj_override=wj_override,
        )
        # Device-side freeze + iter_counter increment + all-done reduction (fused)
        freeze_finished_lanes_kernel[(B,)](
            tensors["best_errors"],
            tensors["hard_dec"],
            tensors["gamma"],
            tensors["active"],
            tensors["found_count"],
            tensors["first_iter"],
            tensors["iter_counter"],
            tensors["sought_solutions"],
            self.stop_nconv,
            in_leg_python,
            True,  # count_iteration
            self.V,
            tensors["all_done_flag"],
            tensors["arrival_counter"],
            B,
        )
        # Active compaction eliminated: btile kernels use active[] mask directly to skip
        # frozen lanes, avoiding count_active_blocks + compare + .item() sync overhead.

        # Check flag every iteration to match Rust behavior exactly
        if skip_all_done_sync:
            # Skip sync - will be checked at chunk boundary
            all_done = False
        else:
            all_done = tensors["all_done_flag"].item()
            if all_done:
                tensors["all_done_flag"].fill_(0)
                tensors["in_leg"].fill_(0)

        return local_active_idx, local_B_active, all_done

    def _unconverged_hard_dec(self, tensors: Dict) -> torch.Tensor:
        """Return the [B, V] hard decision used for lanes with no valid solution.

        Args:
            tensors: Device tensors dictionary.

        Returns:
            ``first_leg_hard_dec`` for "first_leg", the live ``hard_dec`` for "last_leg".
        """
        if self.unconverged_correction == "first_leg":
            return tensors["first_leg_hard_dec"]
        return tensors["hard_dec"]

    def _check_fast_max_unconverged_pct(
        self, tensors: Dict, B: int, leg_count: int = -1
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Check if early exit threshold is met and return results if so.

        Args:
            tensors: Device tensors dictionary
            B: Batch size
            leg_count: Current leg count (for debugging/logging, -1 if unknown)

        Returns:
            Result dictionary if early exit threshold met, None otherwise
        """
        if self.fast_max_unconverged_pct is None:
            return None

        unconverged_count = (tensors["found_count"] < self.stop_nconv).sum().item()
        unconverged_pct = unconverged_count / B if B > 0 else 0.0
        # Only exit early if the percentage of unconverged cases is < threshold
        # AND at least some cases have converged (unconverged_pct < 1.0)
        # This ensures we don't exit when all cases are still unconverged
        if unconverged_pct < self.fast_max_unconverged_pct and unconverged_pct < 1.0:
            # Early exit: compute unconverged indices and return
            unconverged_mask = tensors["found_count"] < self.stop_nconv
            unconverged_idx = torch.where(unconverged_mask)[0]
            has_solution = torch.isfinite(tensors["best_weights"])
            fallback_errors = torch.where(
                has_solution.view(-1, 1),
                tensors["best_errors"],
                self._unconverged_hard_dec(tensors),
            )
            errors = bitpack_errors(fallback_errors) if self.bitpack_output else fallback_errors
            # Compute iterations for all lanes (converged lanes use first_iter, unconverged use iter_counter)
            # Note: Unconverged lanes have partial iterations from early exit, but they will be filtered out
            # by the caller and reprocessed with full iterations in the difficult batch.
            iters = torch.where(
                tensors["first_iter"] >= 0, tensors["first_iter"], tensors["iter_counter"]
            )
            result = {
                "errors": errors,
                "weights": tensors["best_weights"],
                "valid_mask": tensors["found_count"] >= self.stop_nconv,
                "iterations": iters,
                "found_count": tensors["found_count"].to(torch.int32),
                "unconverged_indices": unconverged_idx.to(torch.int32),
            }
            return result
        return None

    def _run_bp_iterations_grid_stride(
        self,
        start_iter: int,
        max_iter: int,
        tensors: Dict,
        B: int,
        active_idx: torch.Tensor,
        B_active: int,
        graph: "CSRGraph",
        log_prior_ratios: torch.Tensor,
        gpu_idx: int,
        in_leg_python: bool,
        force_hard: bool = False,
        prior_batched: bool = False,
        wj_override: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, int, bool, int]:
        """Run multiple BP iterations using C2V grid-stride kernel with separate V2C/parity/select phases.

        This method uses c2v_min_sum_grid_stride_kernel (C2V only) with grid-stride parallelism
        to efficiently split work across GPU SMs. V2C, parity, and selection run in separate
        kernel launches after C2V completes.

        Performance:
        - Uses grid-stride parallelism: grid = (B * work_items_per_batch,) splits work across SMs
        - 5 kernel launches per iteration (C2V, V2C, parity, select, freeze)
        - 1 sync per iteration for convergence checking

        Returns:
            Tuple of (active_idx, B_active, all_done, iterations_run)
        """
        if graph is None:
            graph = self.graphs[gpu_idx]
        if log_prior_ratios is None:
            log_prior_ratios = self.log_prior_ratios_per_gpu[gpu_idx]

        gpu_id = self.device_ids[gpu_idx]
        wj_gpu = wj_override if prior_batched else self.wj_per_gpu[gpu_id]

        alpha_base = self.normalized_min_sum_alpha or 0.0
        alpha_scaling = float(self.alpha_iteration_scaling_factor or 1.0)
        use_alpha = self.normalized_min_sum_alpha is not None
        use_beta = self.offset_min_sum_beta is not None
        beta = self.offset_min_sum_beta or 0.0

        # Grid calculation depends on ROWS_PER_CHK/ROWS_PER_VAR (will be determined by autotuner)
        # We need to provide a grid for autotuning, so use a reasonable default
        # The actual grid will be recalculated after we get the tuned config
        work_dim = max(self.C, self.V)
        default_rows_per_chk = 1
        default_rows_per_var = 1
        default_grid = (B * work_dim,)  # Grid for ROWS_PER_CHK=1 (will be adjusted)

        # Autotune iteration-fused kernel
        problem_key = {
            "B": B,
            "C": self.C,
            "V": self.V,
            "E": self.E,
            "sparsity_hash": graph.sparsity_hash,
            "dtype_enum": self._dtype_enum,
            "use_alpha": use_alpha,
            "use_beta": use_beta,
            "in_leg": 1 if in_leg_python else 0,
        }

        # Grid function that adjusts grid based on ROWS_PER_CHK
        def _grid_fn(cfg):
            rows_chk = int(cfg.get("ROWS_PER_CHK", default_rows_per_chk))
            # Each program processes ROWS_PER_CHK consecutive checks/vars
            # work_items_per_batch = (work_dim + ROWS_PER_CHK - 1) // ROWS_PER_CHK
            # grid = B * work_items_per_batch
            work_dim = max(self.C, self.V)
            work_items_per_batch = (work_dim + rows_chk - 1) // rows_chk
            grid_size = B * work_items_per_batch
            return (grid_size,)

        saved = self._get_tuned_meta(
            kernel_name="c2v_min_sum_grid_stride_kernel",
            problem_key=problem_key,
            build_cfgs_fn=build_c2v_grid_stride_configs,
            kernel=c2v_min_sum_grid_stride_kernel,
            grid=default_grid,
            args=(
                tensors["mu"],
                tensors["nu"],
                graph.chk_ptr,
                graph.chk_edges,
                graph.var_ptr,
                graph.var_edges,
                graph.edge_var,
                tensors["lambda_"],
                tensors["posterior_ratios"],
                tensors["hard_dec"],
                tensors["syndrome"],
                log_prior_ratios,
                tensors["gamma"],
                tensors["active"],
                tensors["check_ok"],
                tensors["best_weights"],
                tensors["best_errors"],
                tensors["valid_solutions"],
                tensors["found_count"],
                wj_gpu,
                B,
                self.C,
                self.V,
                self.E,
                0,  # iteration (will vary)
                alpha_base,
                alpha_scaling,
                use_alpha,
                use_beta,
                beta,
                1 if in_leg_python else 0,
                self._dtype_enum,
            ),
            meta_base={},
            number=20,
            grid_fn=_grid_fn,
            log_name=f"c2v_min_sum_grid_stride_kernel[gpu={gpu_idx}]",
        )

        block_size = int(saved["BLOCK_SIZE"])
        num_warps = int(saved.get("num_warps", 4))
        num_stages = int(saved.get("num_stages", 1))
        rows_per_chk = int(saved.get("ROWS_PER_CHK", default_rows_per_chk))
        rows_per_var = int(saved.get("ROWS_PER_VAR", default_rows_per_var))

        # Recalculate grid based on tuned ROWS_PER_CHK
        # Each program processes ROWS_PER_CHK consecutive checks/vars
        work_dim = max(self.C, self.V)
        work_items_per_batch = (work_dim + rows_per_chk - 1) // rows_per_chk
        grid = (B * work_items_per_batch,)

        # Cache other kernel configs to avoid overhead
        from .kernels import (
            check_and_select_kernel,
            v2c_and_marginals_fused_gamma_kernel,
            parity_from_hard_kernel,
            freeze_finished_lanes_kernel,
        )
        from .autotune import build_check_select_configs, build_v2c_configs, build_parity_configs

        # Cache V2C config
        v2c_problem_key = {
            "B": B,
            "V": self.V,
            "E": self.E,
            "dtype": self.dtype_str,
            "sparsity_hash": graph.sparsity_hash,
        }

        def _v2c_grid_fn(cfg):
            rows = int(cfg.get("ROWS_PER_VAR", self.cfg.rows_per_var))
            return (((B * self.V) + rows - 1) // rows,)

        v2c_saved = self._get_tuned_meta(
            kernel_name="v2c_and_marginals_fused_gamma_kernel",
            problem_key=v2c_problem_key,
            build_cfgs_fn=build_v2c_configs,
            kernel=v2c_and_marginals_fused_gamma_kernel,
            grid=(1,),
            args=(
                tensors["nu"],
                tensors["mu"],
                graph.var_ptr,
                graph.var_edges,
                tensors["lambda_"],
                tensors["posterior_ratios"],
                tensors["hard_dec"],
                tensors["active"],
                log_prior_ratios,
                tensors["gamma"],
            ),
            meta_base=dict(
                B=B, V=self.V, E=self.E, dtype_enum=self._dtype_enum, PRIOR_BATCHED=prior_batched
            ),
            number=20,
            grid_fn=_v2c_grid_fn,
            log_name=f"v2c_and_marginals_fused_gamma_kernel[gpu={gpu_idx}]",
        )
        v2c_rows = int(v2c_saved.get("ROWS_PER_VAR", self.cfg.rows_per_var))
        v2c_grid = (((B * self.V) + v2c_rows - 1) // v2c_rows,)
        v2c_block_size = int(v2c_saved.get("BLOCK_SIZE", 64))
        v2c_num_warps = int(v2c_saved.get("num_warps", 0))
        v2c_num_stages = int(v2c_saved.get("num_stages", 0))

        # Cache parity config
        parity_grid = ((B * self.C + self.cfg.rows_per_chk - 1) // self.cfg.rows_per_chk,)
        parity_problem_key = {
            "B": B,
            "C": self.C,
            "V": self.V,
            "E": self.E,
            "ROWS_PER_CHK": self.cfg.rows_per_chk,
            "dtype": self.dtype_str,
            "sparsity_hash": graph.sparsity_hash,
        }
        parity_meta_base = dict(
            B=B, C=self.C, V=self.V, E=self.E, ROWS_PER_CHK=self.cfg.rows_per_chk
        )

        parity_saved = self._get_tuned_meta(
            kernel_name="parity_from_hard_kernel",
            problem_key=parity_problem_key,
            build_cfgs_fn=build_parity_configs,
            kernel=parity_from_hard_kernel,
            grid=parity_grid,
            args=(
                tensors["hard_dec"],
                graph.chk_ptr,
                graph.chk_edges,
                graph.edge_var,
                tensors["syndrome"],
                tensors["check_ok"],
            ),
            meta_base=parity_meta_base,
            number=20,
            log_name=f"parity_from_hard_kernel[gpu={gpu_idx}]",
        )
        parity_block_size = int(parity_saved["BLOCK_SIZE"])
        parity_num_warps = int(parity_saved.get("num_warps", 0))
        parity_num_stages = int(parity_saved.get("num_stages", 0))

        # Cache select config
        select_grid = (B,)
        select_problem_key = {
            "B": B,
            "C": self.C,
            "V": self.V,
            "dtype": self.dtype_str,
            "sparsity_hash": graph.sparsity_hash,
        }
        select_saved = self._get_tuned_meta(
            kernel_name="check_and_select_kernel",
            problem_key=select_problem_key,
            build_cfgs_fn=build_check_select_configs,
            kernel=check_and_select_kernel,
            grid=select_grid,
            args=(
                tensors["check_ok"],
                tensors["posterior_ratios"],
                tensors["hard_dec"],
                wj_gpu,
                tensors["best_weights"],
                tensors["best_errors"],
                tensors["valid_solutions"],
                tensors["found_count"],
                tensors["active"],
            ),
            meta_base=dict(
                B=B,
                C=self.C,
                V=self.V,
                dtype_enum=self._dtype_enum,
                NUM_SOLS=int(self.stop_nconv),
                PRIOR_BATCHED=prior_batched,
            ),
            number=20,
            log_name=f"check_and_select_kernel[gpu={gpu_idx}]",
        )
        select_block_c = int(select_saved.get("BLOCK_C", 256))
        select_block_v = int(select_saved.get("BLOCK_V", 256))
        select_num_warps = int(select_saved.get("num_warps", 1))
        select_num_stages = int(select_saved.get("num_stages", 1))

        C_val = self.C
        V_val = self.V
        E_val = self.E
        dtype_enum_val = self._dtype_enum
        stop_nconv_val = self.stop_nconv
        in_leg_val = 1 if in_leg_python else 0

        # Cache tensor references
        active_tensor = tensors["active"]
        iter_counter_tensor = tensors["iter_counter"]
        mu_tensor = tensors["mu"]
        nu_tensor = tensors["nu"]
        lam_tensor = tensors["lambda_"]
        posterior_tensor = tensors["posterior_ratios"]
        hard_dec_tensor = tensors["hard_dec"]
        syndrome_tensor = tensors["syndrome"]
        gamma_tensor = tensors["gamma"]
        check_ok_tensor = tensors["check_ok"]
        best_weights_tensor = tensors["best_weights"]
        best_errors_tensor = tensors["best_errors"]
        valid_solutions_tensor = tensors["valid_solutions"]
        found_count_tensor = tensors["found_count"]
        all_done_flag_tensor = tensors["all_done_flag"]
        sought_solutions_tensor = tensors["sought_solutions"]
        first_iter_tensor = tensors["first_iter"]

        # Cache graph pointers
        chk_ptr = graph.chk_ptr
        chk_edges = graph.chk_edges
        var_ptr = graph.var_ptr
        var_edges = graph.var_edges
        edge_var = graph.edge_var

        # For B=1, ensure active[0] is always 1 (lane is always active)
        if B == 1:
            active_tensor[0] = 1

        t = start_iter
        all_done = False

        while t < start_iter + max_iter and not all_done:
            # Phase 1: C2V (grid-stride kernel with batch-organized pattern for better cache locality)
            c2v_min_sum_grid_stride_kernel[grid](
                mu_tensor,
                nu_tensor,
                chk_ptr,
                chk_edges,
                var_ptr,
                var_edges,
                edge_var,
                lam_tensor,
                posterior_tensor,
                hard_dec_tensor,
                syndrome_tensor,
                log_prior_ratios,
                gamma_tensor,
                active_tensor,
                check_ok_tensor,
                best_weights_tensor,
                best_errors_tensor,
                valid_solutions_tensor,
                found_count_tensor,
                wj_gpu,
                B=B,
                C=C_val,
                V=V_val,
                E=E_val,
                iteration=t,
                alpha_base=alpha_base,
                alpha_scaling=alpha_scaling,
                use_alpha=use_alpha,
                use_beta=use_beta,
                beta=beta,
                in_leg=in_leg_val,
                dtype_enum=dtype_enum_val,
                BLOCK_SIZE=block_size,
                ROWS_PER_CHK=rows_per_chk,
                ROWS_PER_VAR=rows_per_var,
                num_warps=num_warps,
                num_stages=num_stages,
            )

            # Phase 2: V2C
            v2c_and_marginals_fused_gamma_kernel[v2c_grid](
                nu_tensor,
                mu_tensor,
                var_ptr,
                var_edges,
                lam_tensor,
                posterior_tensor,
                hard_dec_tensor,
                active_tensor,
                log_prior_ratios,
                gamma_tensor,
                B=B,
                V=V_val,
                E=E_val,
                dtype_enum=dtype_enum_val,
                BLOCK_SIZE=v2c_block_size,
                ROWS_PER_VAR=v2c_rows,
                PRIOR_BATCHED=prior_batched,
                num_warps=v2c_num_warps,
                num_stages=v2c_num_stages,
            )

            # Phase 3: Parity
            parity_from_hard_kernel[parity_grid](
                hard_dec_tensor,
                chk_ptr,
                chk_edges,
                edge_var,
                syndrome_tensor,
                check_ok_tensor,
                **parity_meta_base,
                BLOCK_SIZE=parity_block_size,
                num_warps=parity_num_warps,
                num_stages=parity_num_stages,
            )

            # Phase 4: Selection
            check_and_select_kernel[select_grid](
                check_ok_tensor,
                posterior_tensor,
                hard_dec_tensor,
                wj_gpu,
                best_weights_tensor,
                best_errors_tensor,
                valid_solutions_tensor,
                found_count_tensor,
                active_tensor,
                B=B,
                C=C_val,
                V=V_val,
                dtype_enum=dtype_enum_val,
                BLOCK_C=select_block_c,
                BLOCK_V=select_block_v,
                NUM_SOLS=int(self.stop_nconv),
                PRIOR_BATCHED=prior_batched,
                num_warps=select_num_warps,
                num_stages=select_num_stages,
            )

            # Phase 5: Freeze and increment iter_counter
            freeze_finished_lanes_kernel[(B,)](
                best_errors_tensor,
                hard_dec_tensor,
                gamma_tensor,
                active_tensor,
                found_count_tensor,
                first_iter_tensor,
                iter_counter_tensor,
                sought_solutions_tensor,
                stop_nconv_val,
                in_leg_val,
                True,  # count_iteration
                V_val,
                all_done_flag_tensor,
                tensors["arrival_counter"],
                B,
            )

            # all-done check is fused into freeze kernel (last-arriving program does reduction)
            all_done = all_done_flag_tensor.item() != 0
            if all_done:
                all_done_flag_tensor.fill_(0)
                break

            t += 1

        # Get iterations run
        if B == 1:
            iterations_run = int(iter_counter_tensor[0].item())
        else:
            iterations_run = int(iter_counter_tensor.max().item())

        # Update active indices
        if B == 1:
            active_idx = torch.tensor([0], dtype=torch.int32, device=tensors["active"].device)
            B_active = 1
        else:
            active_mask = tensors["active"] != 0
            active_idx = torch.where(active_mask)[0].to(torch.int32)
            B_active = int(active_idx.shape[0])

        if all_done:
            tensors["in_leg"].fill_(0)

        return active_idx, B_active, all_done, iterations_run

    def _decode_loop_single_gpu(
        self,
        syndromes: torch.Tensor,
        tensors: Dict,
        graph: "CSRGraph",
        log_prior_ratios: torch.Tensor,
        explicit_gammas: Optional[torch.Tensor],
        gpu_idx: int,
        *,
        init_posterior: Optional[torch.Tensor] = None,
        init_gamma: Optional[torch.Tensor] = None,
        skip_relay_legs: bool = False,
        pre_iter_override: Optional[int] = None,
        return_last_posterior: bool = False,
        prior_batched: bool = False,
        wj_override: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Execute the decode loop on a single GPU.

        Args:
            syndromes: [B, C] syndrome bits on this GPU
            tensors: Device tensors dictionary for this GPU
            graph: CSRGraph instance for this GPU
            log_prior_ratios: [1, V] log prior ratios on this GPU
            explicit_gammas: [K, V] explicit gammas on this GPU (or None)
            gpu_idx: GPU index (for device string)
            init_posterior, init_gamma, skip_relay_legs, pre_iter_override,
            return_last_posterior: see ``decode``. Validated by caller.

        Returns:
            Result dictionary with errors, weights, valid_mask, iterations, unconverged_indices
        """
        B = syndromes.shape[0]
        device_str = f"{self.device_type}:{self.device_ids[gpu_idx]}"

        # One-time allocations for per-lane freezing and iteration accounting
        if "first_iter" not in tensors:
            tensors["first_iter"] = torch.full((B,), -1, dtype=torch.int32, device=device_str)
        if "iter_counter" not in tensors:
            tensors["iter_counter"] = torch.zeros(B, dtype=torch.int32, device=device_str)
        if "active" not in tensors:
            tensors["active"] = torch.ones(B, dtype=torch.uint8, device=device_str)
        # Reset active lanes at the start of every decode
        tensors["active"].fill_(1)
        # Reset per-decode iteration accounting and frozen state
        tensors["first_iter"].fill_(-1)
        tensors["iter_counter"].fill_(0)
        tensors["all_done_flag"].fill_(0)
        tensors["in_leg"].fill_(0)
        tensors["old_active_count"].fill_(-1)

        # Copy syndromes to device
        tensors["syndrome"][:] = syndromes

        # Initialize beliefs from priors
        # lambda_: Start with prior log-likelihood ratios (Rust: log_prior_ratios)
        tensors["lambda_"].copy_(log_prior_ratios.expand(B, -1))  # Broadcast prior LLRs robustly
        # posterior_ratios: Initialize to log_prior_ratios for first iteration's gamma mixing
        # This matches Rust: posterior_ratios initialized to log_prior_ratios when gamma0 is set
        # (Rust line 157-161: posterior_ratios = log_prior_ratios if gamma0.is_some() else zeros)
        # For Relay-BP, gamma0 is always set, so we initialize posterior_ratios to log_prior_ratios.
        # Caller override: replace with the supplied posterior.
        if init_posterior is not None:
            tensors["posterior_ratios"].copy_(init_posterior.to(self.dtype))
        else:
            tensors["posterior_ratios"].copy_(log_prior_ratios.expand(B, -1))
        tensors["best_weights"].fill_(float("inf"))
        tensors["found_count"].fill_(0)
        tensors["valid_solutions"].fill_(0)
        # Initialize sought_solutions to 1 for pre_iter (will be updated for relay legs)
        tensors["sought_solutions"].fill_(1)

        # Initialize messages (Rust-compatible: nu=zero, mu=priors)
        if self.cfg.perf == "throughput":
            # Rust: initialize_check_to_variable() -> nu = zero
            # Rust: initialize_variable_to_check() -> mu = log_prior_ratios
            self._launch_init_messages_with_priors_EB(
                tensors["muT"],
                tensors["nuT"],
                B,
                graph=graph,
                log_prior_ratios=log_prior_ratios,
                gpu_idx=gpu_idx,
                prior_batched=prior_batched,
            )
            # Delta-trick: initialize L to current marginals (lam_mixed + sum(nu)) for this phase
            # by running one standard V2C pass, then copying posterior_ratios -> L.
            if self._use_delta_trick:
                self._launch_v2c_btile(
                    tensors,
                    B,
                    write_hard=True,
                    active_idx=None,
                    B_active=None,
                    graph=graph,
                    log_prior_ratios=log_prior_ratios,
                    gpu_idx=gpu_idx,
                    block_vars=None,
                    block_var_ptr=None,
                    block_idx=-1,
                    prior_batched=prior_batched,
                )
                tensors["L"].copy_(tensors["posterior_ratios"])
        else:
            self._launch_init_kernel(
                tensors, B, graph=graph, log_prior_ratios=log_prior_ratios, gpu_idx=gpu_idx
            )

        # Pre-iterations (T0) with uniform gamma0 (ordered memory phase)
        if self.algo == "plain":
            # Plain BP: no memory mixing; λ remains λ₀ each iteration
            tensors["gamma"].fill_(0.0)
        elif init_gamma is not None:
            # Caller override: per-lane gamma vector for the pre-iter phase.
            tensors["gamma"][:] = init_gamma.to(self.dtype)
        else:
            # Relay BP: use γ-mixing for ordered memory (even with num_sets=0 for MemBP)
            gamma0_tensor = sample_gamma_scalar(B, self.V, self.gamma0, device_str)
            tensors["gamma"][:] = gamma0_tensor

        # Reset per-set valid flags so only one success can be counted in this set
        tensors["valid_solutions"].fill_(0)

        # Initialize active_idx for pre-iter loop (all lanes start active)
        if self.cfg.perf == "throughput":
            if B == 1:
                # For B=1, skip compaction - lane is always active
                device_str = self._device_strs[self.device_ids[gpu_idx]]
                active_idx = torch.tensor([0], dtype=torch.int32, device=device_str)
                B_active = 1
                # Ensure active[0] is 1 (should already be, but be explicit)
                tensors["active"][0] = 1
                # Store old count for device-side comparison (though we won't use it for B=1)
                tensors["old_active_count"].fill_(1)
            else:
                active_idx, B_active = self._compact_active_indices_on_device(tensors, B, gpu_idx)
                # Store old count for device-side comparison
                tensors["old_active_count"].fill_(B_active)
        else:
            # Non-throughput mode: all lanes are active initially
            device_str = f"cuda:{gpu_idx}"
            if B == 1:
                # For B=1, use simple [0] tensor and ensure active[0] is 1
                active_idx = torch.tensor([0], dtype=torch.int32, device=device_str)
                B_active = 1
                tensors["active"][0] = 1
            else:
                active_idx = torch.arange(B, dtype=torch.int32, device=device_str)
                B_active = B

        # Store in_leg as Python variable to avoid .item() calls in hot loop
        in_leg_python = True  # We're in pre-iter phase

        # Track leg count (pre_iter counts as leg 0, relay legs as 1..num_sets)
        leg_count = 0

        # Resolve all autotuning configs once
        self._resolve_configs(B, tensors, graph, gpu_idx)

        # Pre-iterations loop
        # Use C2V grid-stride kernel for default mode (batch-organized pattern, much faster)
        pre_iter_count = self.pre_iter if pre_iter_override is None else int(pre_iter_override)
        if self.cfg.perf == "default":
            tensors["in_leg"].fill_(1)
            active_idx, B_active, all_done, iterations_run = self._run_bp_iterations_grid_stride(
                0,
                pre_iter_count,
                tensors,
                B,
                active_idx,
                B_active,
                graph,
                log_prior_ratios,
                gpu_idx,
                in_leg_python,
                force_hard=True,
                prior_batched=prior_batched,
                wj_override=wj_override,
            )
            # Update iteration counters (iteration-fused method handles this internally)
        else:
            # Throughput mode: per-iteration loop
            for t in range(pre_iter_count):
                tensors["in_leg"].fill_(1)
                active_idx, B_active, all_done = self._run_bp_iteration(
                    t,
                    tensors,
                    B,
                    active_idx,
                    B_active,
                    graph,
                    log_prior_ratios,
                    gpu_idx,
                    in_leg_python,
                    force_hard=True,
                    prior_batched=prior_batched,
                    wj_override=wj_override,
                )
                if all_done:
                    break

        # Parity and selection already ran after the last pre-iteration, in both perf modes.
        # With pre_iter_count == 0 no iteration has run yet; relay leg 1 overwrites this.
        if self.unconverged_correction == "first_leg":
            tensors["first_leg_hard_dec"].copy_(tensors["hard_dec"])
        # Device-side freeze + all-done check (fused).
        freeze_finished_lanes_kernel[(B,)](
            tensors["best_errors"],
            tensors["hard_dec"],
            tensors["gamma"],
            tensors["active"],
            tensors["found_count"],
            tensors["first_iter"],
            tensors["iter_counter"],
            tensors["sought_solutions"],
            self.stop_nconv,
            in_leg_python,
            False,  # count_iteration
            self.V,
            tensors["all_done_flag"],
            tensors["arrival_counter"],
            B,
        )
        # If every lane has stop_nconv solutions, skip legs entirely and return results now.
        # Also returned here when the caller requested skip_relay_legs.
        reduce_all_ge_kernel[(1,)](
            tensors["found_count"],
            self.stop_nconv,
            tensors["all_done_flag"],
            B,
        )
        if tensors["all_done_flag"].item() or skip_relay_legs:
            # Use lifetime-best solutions where they exist. Otherwise _unconverged_hard_dec.
            # A lane has a valid solution iff its best_weights entry is finite.
            has_solution = torch.isfinite(tensors["best_weights"])
            fallback_errors = torch.where(
                has_solution.view(-1, 1),
                tensors["best_errors"],
                self._unconverged_hard_dec(tensors),
            )
            errors = bitpack_errors(fallback_errors) if self.bitpack_output else fallback_errors
            iters = torch.where(
                tensors["first_iter"] >= 0,
                tensors["first_iter"],
                tensors["iter_counter"].expand_as(tensors["first_iter"]),
            )
            # Compute unconverged indices (should be empty if all done)
            unconverged_mask = tensors["found_count"] < self.stop_nconv
            unconverged_idx = torch.where(unconverged_mask)[0].to(torch.int32)
            result = {
                "errors": errors,
                "weights": tensors["best_weights"],
                "valid_mask": tensors["found_count"] >= self.stop_nconv,
                "iterations": iters,
                "found_count": tensors["found_count"].to(torch.int32),
                "unconverged_indices": unconverged_idx,
            }
            if return_last_posterior:
                result["last_posterior"] = tensors["posterior_ratios"].detach().clone()
            return result

        # Track leg count (pre-iter is leg 0, relay legs are 1..num_sets)
        leg_count = 0  # Pre-iter is leg 0

        # Check fast_max_unconverged_pct early exit after pre-iter (leg_count = 0)
        # Some syndromes may converge during pre-iter, so we should check here too
        early_exit_result = self._check_fast_max_unconverged_pct(tensors, B, leg_count=0)
        if early_exit_result is not None:
            return early_exit_result

        unfreeze_finished_lanes_kernel[(B,)](
            tensors["active"],
            tensors["found_count"],
            self.stop_nconv,
        )

        # Relay legs (R times)
        in_leg_python = True
        for leg in range(self.num_sets):
            # Increment leg count (relay legs are 1..num_sets)
            leg_count = leg + 1
            tensors["in_leg"].fill_(1)
            tensors["sought_solutions"] = tensors["found_count"] + 1

            # Set leg gamma with the following priority:
            # 1) explicit_gammas (K×V) if provided
            # 2) gamma_dist_interval_by_error (V×2) if provided
            # 3) global gamma_dist_interval (scalar (min, max))
            # Cases 2 and 3 draw from seed + leg: one [1, V] row broadcast to all shots
            # (gamma_draw="broadcast") or one row per shot (gamma_draw="per_row").
            gamma_rows = 1 if self.gamma_draw == "broadcast" else B
            if explicit_gammas is not None:
                K = int(explicit_gammas.shape[0])
                row = explicit_gammas[((leg + 1) % K)]
                tensors["gamma"].copy_(row.unsqueeze(0).expand(B, -1))
            elif self.gamma_dist_interval_by_error_per_gpu is not None:
                gamma_intervals = self.gamma_dist_interval_by_error_per_gpu[gpu_idx]
                gamma_leg = sample_gamma_per_variable(
                    gamma_rows, self.V, gamma_intervals, device_str, self.seed + leg
                )
                tensors["gamma"].copy_(gamma_leg.expand(B, -1))
            else:
                gamma_min, gamma_max = self.gamma_dist_interval
                gamma_leg = sample_gamma_uniform(
                    gamma_rows, self.V, gamma_min, gamma_max, device_str, self.seed + leg
                )
                tensors["gamma"].copy_(gamma_leg.expand(B, -1))

            # Rust-equivalent per-leg reinitialization of messages:
            # reset variable<->check messages at the start of each leg while
            # keeping current beliefs (lambda_) carried over
            # Rust: initialize_check_to_variable() -> nu = zero
            # Rust: initialize_variable_to_check() -> mu = log_prior_ratios
            if self.cfg.perf == "throughput":
                # Always reset messages between legs per Relay-BP spec
                # Compute active_idx once for the first iteration of the leg
                if B == 1:
                    # For B=1, skip compaction - lane is always active
                    device_str = self._device_strs[self.device_ids[gpu_idx]]
                    active_idx = torch.tensor([0], dtype=torch.int32, device=device_str)
                    B_active = 1
                    # Ensure active[0] is 1 (should already be, but be explicit)
                    tensors["active"][0] = 1
                    # Store old count for device-side comparison (though we won't use it for B=1)
                    tensors["old_active_count"].fill_(1)
                else:
                    active_idx, B_active = self._compact_active_indices_on_device(
                        tensors, B, gpu_idx
                    )
                    # Update old count for device-side comparison
                    tensors["old_active_count"].fill_(B_active)

                # OPTIMIZATION: Only reinitialize messages for ACTIVE lanes (lazy init)
                # This aims to reduce memory traffic when B_active << B
                # For example, if only 10% of lanes are active, we only write 10% of messages
                self._launch_init_messages_with_priors_EB(
                    tensors["muT"],
                    tensors["nuT"],
                    B,
                    active_idx=active_idx,
                    B_active=B_active,
                    graph=graph,
                    log_prior_ratios=log_prior_ratios,
                    gpu_idx=gpu_idx,
                    prior_batched=prior_batched,
                )
                # Re-initialize L for delta-trick to current marginals (lam_mixed + sum(nu)) for this leg.
                if self._use_delta_trick:
                    self._launch_v2c_btile(
                        tensors,
                        B,
                        write_hard=True,
                        active_idx=active_idx,
                        B_active=B_active,
                        graph=graph,
                        log_prior_ratios=log_prior_ratios,
                        gpu_idx=gpu_idx,
                        block_vars=None,
                        block_var_ptr=None,
                        block_idx=-1,
                        prior_batched=prior_batched,
                    )
                    tensors["L"].copy_(tensors["posterior_ratios"])
            else:
                self._launch_init_kernel(tensors, B, graph=graph, gpu_idx=gpu_idx)
                # Non-throughput mode: initialize active_idx for the relay leg
                # (all lanes start active at the beginning of each leg)
                device_str = f"cuda:{gpu_idx}"
                active_idx = torch.arange(B, dtype=torch.int32, device=device_str)
                B_active = B

            # Reset per-set valid flags so only one success can be counted in this set
            tensors["valid_solutions"].fill_(0)

            # Relay leg iterations loop
            if self.cfg.perf == "default":
                active_idx, B_active, all_done, iterations_run = (
                    self._run_bp_iterations_grid_stride(
                        0,
                        self.set_max_iter,
                        tensors,
                        B,
                        active_idx,
                        B_active,
                        graph,
                        log_prior_ratios,
                        gpu_idx,
                        in_leg_python,
                        force_hard=False,
                        prior_batched=prior_batched,
                        wj_override=wj_override,
                    )
                )
            else:
                # Throughput mode: per-iteration loop (configs cached automatically on first call)
                for t in range(self.set_max_iter):
                    active_idx, B_active, all_done = self._run_bp_iteration(
                        t,
                        tensors,
                        B,
                        active_idx,
                        B_active,
                        graph,
                        log_prior_ratios,
                        gpu_idx,
                        in_leg_python,
                        force_hard=False,
                        prior_batched=prior_batched,
                        wj_override=wj_override,
                    )
                    if all_done:
                        break

            # Parity and selection already ran after the last iteration of the leg.
            # The first leg that ran iterations is relay leg 1 when pre_iter_count == 0.
            if leg == 0 and pre_iter_count == 0 and self.unconverged_correction == "first_leg":
                tensors["first_leg_hard_dec"].copy_(tensors["hard_dec"])
            # Stop the legs only when every lane has stop_nconv solutions.
            reduce_all_ge_kernel[(1,)](
                tensors["found_count"],
                self.stop_nconv,
                tensors["all_done_flag"],
                B,
            )
            # Sync here is acceptable - only called once per leg, not in hot loop
            if tensors["all_done_flag"].item():
                break

            # Check fast_max_unconverged_pct early exit after completing this relay leg (leg_count = leg + 1)
            # We check at leg boundaries (after final parity check) rather than every iteration because:
            # 1. A "leg" is a logical unit (pre_iter phase or a relay leg), matching the threshold semantics
            # 2. Less overhead: check once per leg vs once per iteration (set_max_iter can be 60+)
            # 3. Percentage-based threshold adapts to batch convergence behavior
            early_exit_result = self._check_fast_max_unconverged_pct(
                tensors, B, leg_count=leg_count
            )
            if early_exit_result is not None:
                return early_exit_result

            unfreeze_finished_lanes_kernel[(B,)](
                tensors["active"],
                tensors["found_count"],
                self.stop_nconv,
            )

        # Prepare output
        # A lane with no valid solution returns _unconverged_hard_dec (first or last leg)
        has_solution = torch.isfinite(tensors["best_weights"])
        fallback_errors = torch.where(
            has_solution.view(-1, 1),
            tensors["best_errors"],
            self._unconverged_hard_dec(tensors),
        )
        if self.bitpack_output:
            errors = bitpack_errors(fallback_errors)
        else:
            errors = fallback_errors

        # Build per-lane iteration counts (Rust-like): first_iter if set, else per-lane iter_counter
        iters = torch.where(
            tensors["first_iter"] >= 0, tensors["first_iter"], tensors["iter_counter"]
        )

        # Compute unconverged indices (found_count < stop_nconv)
        # If fast_max_unconverged_pct was set, this will be non-empty for early exits
        # If fast_max_unconverged_pct was None, this will be empty (all ran to completion)
        unconverged_mask = tensors["found_count"] < self.stop_nconv
        unconverged_idx = torch.where(unconverged_mask)[0].to(torch.int32)

        result = {
            "errors": errors,
            "weights": tensors["best_weights"],
            "valid_mask": tensors["found_count"] >= self.stop_nconv,
            "iterations": iters,
            "found_count": tensors["found_count"].to(torch.int32),
            "unconverged_indices": unconverged_idx,
        }
        if return_last_posterior:
            result["last_posterior"] = tensors["posterior_ratios"].detach().clone()
        return result

    def get_stats(self) -> Dict:
        """Get decoder statistics."""
        return {
            "graph": self.graphs[0].get_stats(),
            "parameters": {
                "pre_iter": self.pre_iter,
                "num_sets": self.num_sets,
                "set_max_iter": self.set_max_iter,
                "gamma0": self.gamma0,
                "gamma_dist_interval": self.gamma_dist_interval,
                "stop_nconv": self.stop_nconv,
                "dtype": self.dtype_str,
                "device": self.device_ids,
                "algo": self.algo,
                "perf": self.perf,
            },
        }
