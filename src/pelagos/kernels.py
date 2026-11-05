# (C) Copyright IBM 2025, 2026
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Triton kernels for Relay-BP-S belief propagation decoding.

This module contains GPU kernels implementing the core belief propagation operations
for quantum error correction. The kernels closely follow the Rust implementation
in crates/relay_bp/src/bp/min_sum.rs, providing GPU acceleration for:

- Check-to-variable (C2V) message passing with min-sum algorithm
- Variable-to-check (V2C) message passing with gamma mixing
- Parity checking for syndrome validation
- Memory management and tensor operations

Key kernels:
- c2v_min_sum_kernel: Implements min-sum check node updates
- v2c_and_marginals_fused_gamma_kernel: Variable node updates with memory mixing
- parity_*_kernel: Syndrome validation kernels
- Various utility kernels for memory management and batch processing
"""

import triton
import triton.language as tl


@triton.jit
def c2v_min_sum_kernel(
    mu,
    nu,  # [B,E]
    chk_ptr,
    chk_edges,  # CSR over checks -> edge IDs
    syndrome,  # [B,C]
    active,  # [B] uint8 mask (1=active, 0=frozen)
    B,
    C,
    E,
    alpha,
    beta,
    use_alpha: tl.constexpr,
    use_beta: tl.constexpr,
    dtype_enum: tl.constexpr,  # 0=fp16, 1=fp32, 2=fp64 (compile-time constant)
    BLOCK_SIZE: tl.constexpr,
    ROWS_PER_CHK: tl.constexpr,  # Number of check rows to process per program
):
    """Check-to-variable min-sum kernel implementing belief propagation check node updates.

    This kernel implements the min-sum algorithm for check nodes in belief propagation,
    corresponding to the compute_check_to_variable function in Rust min_sum.rs.

    For each check node, it computes outgoing messages to variable nodes using:
    1. Two-pass algorithm to find min1, min2, and parity
    2. Min-sum rule with optional alpha/beta scaling
    3. Syndrome integration for quantum error correction

    Args:
        mu, nu: Message tensors [B,E] (variable-to-check, check-to-variable)
        chk_ptr, chk_edges: CSR representation of check constraints
        syndrome: [B,C] syndrome bits from quantum measurements
        active: [B] mask for active decoding lanes
        B, C, E: batch size, number of checks, number of edges
        alpha, beta: min-sum scaling parameters
        use_alpha, use_beta: flags for parameter usage
        dtype_enum: compile-time dtype selector (0=fp16, 1=fp32, 2=fp64)
        BLOCK_SIZE: vectorization block size
        ROWS_PER_CHK: number of check rows processed per program
    """
    pid = tl.program_id(axis=0)
    base = pid * ROWS_PER_CHK

    # Process up to ROWS_PER_CHK {b,i} pairs
    for r in tl.static_range(0, ROWS_PER_CHK):
        idx = base + r
        # Only process if within bounds
        if idx < B * C:
            b = idx // C
            i = idx % C

            # Early-out for inactive lanes - check active status first
            act_b = tl.load(active + b)
            if act_b != 0:
                # Only read CSR data for active lanes
                row_start = tl.load(chk_ptr + i)
                row_end = tl.load(chk_ptr + i + 1)
                deg = row_end - row_start
                if deg > 0:
                    deg_is_one = deg == 1
                    # PASS 1: Compute parity and find min1/min2 values (vectorized)
                    neg_parity = tl.zeros((), dtype=tl.int32)
                    # Use native dtype for min1/min2 (no casting)
                    if dtype_enum == 0:  # fp16
                        min1 = tl.full((), 1e30, dtype=tl.float16)
                        min2 = tl.full((), 1e30, dtype=tl.float16)
                    elif dtype_enum == 2:  # fp64
                        min1 = tl.full((), 1e30, dtype=tl.float64)
                        min2 = tl.full((), 1e30, dtype=tl.float64)
                    else:  # fp32
                        min1 = tl.full((), 1e30, dtype=tl.float32)
                        min2 = tl.full((), 1e30, dtype=tl.float32)
                    cnt1 = tl.zeros((), dtype=tl.int32)
                    argmin_e = tl.full((), -1, dtype=tl.int32)

                    tile = row_start
                    while tile < row_end:
                        offs = tile + tl.arange(0, BLOCK_SIZE)
                        m = offs < row_end
                        e = tl.load(chk_edges + offs, mask=m, other=0)
                        mu_e = tl.load(mu + b * E + e, mask=m, other=0.0)

                        # parity via xor of tile parity bits
                        tile_par = tl.sum((mu_e < 0.0) & m, axis=0).to(tl.int32) & 1
                        neg_parity = neg_parity ^ tile_par

                        a = tl.abs(mu_e)
                        a_mask = tl.where(m, a, 1e30)

                        # tile min1/argmin and tie count
                        pos1 = tl.argmax(-a_mask, axis=0)
                        min1_t = tl.max(-a_mask, axis=0) * (-1.0)
                        e1_t = tl.load(chk_edges + tile + pos1)
                        cnt1_t = tl.sum(a_mask == min1_t, axis=0).to(tl.int32)
                        # tile min2 (mask just one occurrence of min1)
                        a_mask2 = tl.where(tl.arange(0, BLOCK_SIZE) == pos1, 1e30, a_mask)
                        min2_t = tl.min(a_mask2, axis=0)

                        # merge into global running mins and counts
                        better1 = min1_t < min1
                        argmin_e = tl.where(better1, e1_t, argmin_e)
                        cnt1 = tl.where(
                            better1, cnt1_t, tl.where(min1_t == min1, cnt1 + cnt1_t, cnt1)
                        )
                        min2 = tl.where(
                            better1,
                            tl.minimum(min1, min2_t),
                            tl.where(
                                min1_t == min1, tl.minimum(min2, min2_t), tl.minimum(min2, min1_t)
                            ),
                        )
                        min1 = tl.where(better1, min1_t, min1)

                        tile += BLOCK_SIZE

                    # include syndrome bit in sign product
                    syn = tl.load(syndrome + b * C + i).to(tl.int32) & 1
                    # parity bit = (neg_parity & 1) XOR syn
                    par_bit = (neg_parity & 1) ^ syn
                    sign_prod = tl.where(par_bit == 0, 1.0, -1.0)

                    # PASS 2: Write outgoing messages (vectorized)
                    tile = row_start
                    while tile < row_end:
                        offs = tile + tl.arange(0, BLOCK_SIZE)
                        m = offs < row_end
                        e = tl.load(chk_edges + offs, mask=m, other=0)
                        mu_e = tl.load(mu + b * E + e, mask=m, other=0.0)
                        sgn_mu = tl.where(mu_e >= 0.0, 1.0, -1.0)

                        is_argmin = e == argmin_e
                        use_min2 = (cnt1 == 1) & is_argmin
                        out_mag = tl.where(use_min2, min2, min1)
                        # Degree-1 checks must send zero.
                        out_mag = tl.where(deg_is_one, 0.0, out_mag)

                        if use_alpha:
                            out_mag = alpha * out_mag
                        if use_beta:
                            out_mag = tl.maximum(out_mag - beta, 0.0)

                        nu_e = (sign_prod * sgn_mu) * out_mag
                        # Store with appropriate dtype based on dtype_enum (compile-time optimized)
                        if dtype_enum == 0:  # fp16
                            tl.store(nu + b * E + e, nu_e.to(tl.float16), mask=m)
                        elif dtype_enum == 2:  # fp64
                            tl.store(nu + b * E + e, nu_e.to(tl.float64), mask=m)
                        else:  # fp32 (dtype_enum == 1)
                            tl.store(nu + b * E + e, nu_e, mask=m)

                        tile += BLOCK_SIZE
                # else: inactive lane — do nothing for this (b,i)


@triton.jit
def v2c_and_marginals_fused_gamma_kernel(
    nu,
    mu,  # [B,E]
    var_ptr,
    var_edges,  # CSR over vars -> edge IDs
    lam,
    posterior_ratios,
    hard_dec,  # [B,V], [B,V], [B,V]
    # lam: Current mixed beliefs [B,V]
    # posterior_ratios: Posterior ratios/marginals [B,V] (matches Rust: posterior_ratios)
    active,  # [B] uint8 mask (1=active)
    log_prior_ratios,
    gamma,  # [1,V] or [B,V], [B,V] - for gamma mixing
    # log_prior_ratios: Prior log-likelihood ratios, [1,V] shared or [B,V] per-shot
    # gamma: Memory strengths [B,V] (matches Rust: memory_strengths)
    B,
    V,
    E,
    dtype_enum: tl.constexpr,  # 0=fp16, 1=fp32, 2=fp64 (compile-time constant)
    BLOCK_SIZE: tl.constexpr,
    ROWS_PER_VAR: tl.constexpr,  # Number of variable rows to process per program
    PRIOR_BATCHED: tl.constexpr = False,  # if True, log_prior_ratios is [B,V], else [1,V]
):
    """Variable-to-check kernel with fused gamma mixing for memory-based belief propagation.

    This kernel implements variable node updates in belief propagation with memory mixing,
    corresponding to the compute_variable_to_check function in Rust min_sum.rs.

    For each variable node, it:
    1. Loads previous iteration's marginal (posterior_prev) from posterior_ratios tensor
    2. Applies gamma mixing: λ_mixed = (1-γ)*log_prior + γ*posterior_prev (Rust-compatible: uses previous posterior)
    3. Computes incoming message sum from check nodes
    4. Updates current marginal: posterior_curr = λ_mixed + sum(nu)
    5. Updates hard decisions
    6. Computes outgoing messages to check nodes

    The gamma mixing implements the "memory" aspect of Relay-BP, matching Rust's behavior:
    Rust uses: λ_mixed = (1-γ)*log_prior + γ*posterior_prev, then posterior = λ_mixed + sum(nu)
    This ensures identical convergence behavior with the Rust reference implementation.

    Note: posterior_ratios is always stored (required for next iteration's gamma mixing).

    Args:
        nu, mu: Message tensors [B,E] (check-to-variable, variable-to-check)
        var_ptr, var_edges: CSR representation of variable constraints
        lam, posterior_ratios, hard_dec: [B,V] beliefs, marginals, hard decisions
        active: [B] mask for active decoding lanes
        log_prior_ratios, gamma: [1,V], [B,V] prior beliefs and memory strengths
        B, V, E: batch size, number of variables, number of edges
        dtype_enum: compile-time dtype selector (0=fp16, 1=fp32, 2=fp64)
        BLOCK_SIZE: vectorization block size
        ROWS_PER_VAR: number of variable rows processed per program
    """
    pid = tl.program_id(axis=0)
    base = pid * ROWS_PER_VAR

    # Process up to ROWS_PER_VAR {b,j} pairs
    for r in tl.static_range(0, ROWS_PER_VAR):
        idx = base + r
        # Only process if within bounds
        if idx < B * V:
            b = idx // V
            j = idx % V

            # Early-out for inactive lanes - check active status first
            act_b = tl.load(active + b)
            if act_b != 0:
                # Only read CSR data for active lanes
                row_start = tl.load(var_ptr + j)
                row_end = tl.load(var_ptr + j + 1)

                # Load previous iteration's marginal (posterior_prev) for Rust-compatible gamma mixing
                # Rust uses: λ_mixed = (1-γ)*log_prior + γ*posterior_prev, then posterior = λ_mixed + sum(nu)
                posterior_bj_prev = tl.load(
                    posterior_ratios + b * V + j
                )  # Previous iteration's marginal/posterior

                # FUSED GAMMA MIXING (Rust-compatible): λ = (1-γ)*log_prior + γ*posterior_prev
                # Use storage dtype directly for all computations (fp16/fp32/fp64)
                g_bj = tl.load(gamma + b * V + j)  # [B,V]
                if PRIOR_BATCHED:
                    log_prior_j = tl.load(log_prior_ratios + b * V + j)  # [B,V]
                else:
                    log_prior_j = tl.load(log_prior_ratios + j)  # [1,V]
                lam_bj_mixed = (1.0 - g_bj) * log_prior_j + g_bj * posterior_bj_prev

                # PASS 1: Sum incoming messages from check nodes (vectorized)
                # Use fp32 accumulators for fp16/fp32 storage, fp64 for fp64
                if dtype_enum == 2:  # fp64
                    sum_nu = tl.zeros((), dtype=tl.float64)
                else:  # fp16 or fp32
                    sum_nu = tl.zeros((), dtype=tl.float32)
                tile = row_start
                while tile < row_end:
                    offs = tile + tl.arange(0, BLOCK_SIZE)
                    m = offs < row_end
                    e = tl.load(var_edges + offs, mask=m, other=0)
                    nu_e = tl.load(nu + b * E + e, mask=m, other=0.0)
                    # nu_e is already in correct dtype, no conversion needed
                    sum_nu += tl.sum(tl.where(m, nu_e, 0.0), axis=0)
                    tile += BLOCK_SIZE

                # Compute current marginal: posterior = λ_mixed + sum(nu) (matches Rust posterior_ratios update)
                posterior_bj_curr = lam_bj_mixed + sum_nu
                # Always store posterior_ratios for next iteration's gamma mixing (required for Rust compatibility)
                tl.store(posterior_ratios + b * V + j, posterior_bj_curr)
                # Match Rust convergence rule Bit::from(posterior <= 0)
                tl.store(hard_dec + b * V + j, (posterior_bj_curr <= 0.0).to(tl.uint8))

                # Store mixed lambda for next iteration (used in message computation)
                tl.store(lam + b * V + j, lam_bj_mixed)

                # PASS 2: Write outgoing messages using mixed lambda (matches Rust schedule)
                tile = row_start
                while tile < row_end:
                    offs = tile + tl.arange(0, BLOCK_SIZE)
                    m = offs < row_end
                    e = tl.load(var_edges + offs, mask=m, other=0)
                    nu_e = tl.load(nu + b * E + e, mask=m, other=0.0)
                    # Use lam_bj_mixed (current iteration's mixed belief) to compute mu, matching Rust
                    # nu_e is already in correct dtype, no conversion needed
                    mu_e = lam_bj_mixed + (sum_nu - nu_e)
                    # mu_e is already in storage dtype (computed from storage dtype values), no conversion needed
                    tl.store(mu + b * E + e, mu_e, mask=m)
                    tile += BLOCK_SIZE
            # else: inactive — do nothing (no loads, no stores)


PARITY_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 32}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_SIZE": 64}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_SIZE": 128}, num_warps=4, num_stages=2),
]


@triton.jit
def parity_per_check_kernel(
    posterior_ratios,  # [B,V] fp32 (read-only)
    chk_ptr,
    chk_edges,  # CSR over checks -> edge IDs
    edge_var,  # [E] var index per edge
    syndrome,  # [B,C] uint8
    check_ok,  # [B,C] uint8 (out)
    B,
    C,
    V,
    E,
    BLOCK_SIZE: tl.constexpr,
    ROWS_PER_CHK: tl.constexpr,  # Number of check rows to process per program
):
    """Parity check kernel for syndrome validation using belief marginals.

    This kernel validates whether the current hard decisions satisfy all check constraints,
    corresponding to the check_convergence function in Rust min_sum.rs.

    For each check constraint, it computes the parity of connected variable bits
    and compares with the syndrome to determine if the constraint is satisfied.

    Args:
        posterior_ratios: [B,V] belief marginals (read-only)
        chk_ptr, chk_edges: CSR representation of check constraints
        edge_var: [E] variable index for each edge
        syndrome: [B,C] syndrome bits from quantum measurements
        check_ok: [B,C] output mask for satisfied constraints
        B, C, V, E: batch size, checks, variables, edges
        BLOCK_SIZE: vectorization block size
        ROWS_PER_CHK: number of check rows processed per program
    """
    pid = tl.program_id(axis=0)
    base = pid * ROWS_PER_CHK

    # Process up to ROWS_PER_CHK {b,i} pairs
    for r in tl.static_range(0, ROWS_PER_CHK):
        idx = base + r
        # Only process if within bounds
        if idx < B * C:
            b = idx // C
            i = idx % C

            row_start = tl.load(chk_ptr + i)
            row_end = tl.load(chk_ptr + i + 1)
            par = tl.zeros((), dtype=tl.int32)

            tile = row_start
            while tile < row_end:
                offs = tile + tl.arange(0, BLOCK_SIZE)
                m = offs < row_end
                e = tl.load(chk_edges + offs, mask=m, other=0)
                v = tl.load(edge_var + e, mask=m, other=0)
                posterior_v = tl.load(posterior_ratios + b * V + v, mask=m, other=0.0)
                # Match Rust: Bit::from(posterior <= 0)
                bits = (posterior_v <= 0.0).to(tl.int32) & 1
                par = par ^ (tl.sum(bits & m.to(tl.int32), axis=0) & 1)
                tile += BLOCK_SIZE

            syn = tl.load(syndrome + b * C + i).to(tl.int32) & 1
            ok = (par == syn).to(tl.uint8)
            tl.store(check_ok + b * C + i, ok)


@triton.jit
def parity_from_hard_kernel(
    hard_dec,  # [B,V] uint8
    chk_ptr,
    chk_edges,  # CSR over checks -> edge IDs
    edge_var,  # [E] var index per edge
    syndrome,  # [B,C] uint8
    check_ok,  # [B,C] uint8 (out)
    B,
    C,
    V,
    E,
    BLOCK_SIZE: tl.constexpr,
    ROWS_PER_CHK: tl.constexpr,  # Number of check rows to process per program
):
    """Parity check kernel for syndrome validation using hard decisions.

    This kernel validates check constraints using pre-computed hard decisions,
    optimized for throughput mode to avoid reading posterior_ratios when hard_dec is already available.

    Args:
        hard_dec: [B,V] hard decisions (0/1 bits)
        chk_ptr, chk_edges: CSR representation of check constraints
        edge_var: [E] variable index for each edge
        syndrome: [B,C] syndrome bits from quantum measurements
        check_ok: [B,C] output mask for satisfied constraints
        B, C, V, E: batch size, checks, variables, edges
        BLOCK_SIZE: vectorization block size
        ROWS_PER_CHK: number of check rows processed per program
    """
    pid = tl.program_id(axis=0)
    base = pid * ROWS_PER_CHK
    for r in tl.static_range(0, ROWS_PER_CHK):
        idx = base + r
        if idx < B * C:
            b = idx // C
            i = idx % C

            row_start = tl.load(chk_ptr + i)
            row_end = tl.load(chk_ptr + i + 1)
            par = tl.zeros((), dtype=tl.int32)

            tile = row_start
            while tile < row_end:
                offs = tile + tl.arange(0, BLOCK_SIZE)
                m = offs < row_end
                e = tl.load(chk_edges + offs, mask=m, other=0)
                v = tl.load(edge_var + e, mask=m, other=0)
                bits = tl.load(hard_dec + b * V + v, mask=m, other=0).to(tl.int32) & 1
                par = par ^ (tl.sum(bits & m.to(tl.int32), axis=0) & 1)
                tile += BLOCK_SIZE

            syn = tl.load(syndrome + b * C + i).to(tl.int32) & 1
            ok = (par == syn).to(tl.uint8)
            tl.store(check_ok + b * C + i, ok)


@triton.jit
def parity_from_hard_compact_kernel(
    hard_dec,  # [B,V] uint8
    chk_ptr,
    chk_edges,  # CSR over checks -> edge IDs
    edge_var,  # [E] var index per edge
    syndrome,  # [B,C] uint8
    check_ok,  # [B,C] uint8 (out)
    active_idx,  # [B_active] int32 - compact list of active batch indices
    B,
    B_active,
    C,
    V,
    E,
    BLOCK_SIZE: tl.constexpr,
    ROWS_PER_CHK: tl.constexpr,  # Number of check rows to process per program
):
    """Parity check kernel for active lanes only (throughput mode optimization).

    This kernel validates check constraints for only the active decoding lanes,
    reducing computation when many lanes have already converged.

    Args:
        hard_dec: [B,V] hard decisions (0/1 bits)
        chk_ptr, chk_edges: CSR representation of check constraints
        edge_var: [E] variable index for each edge
        syndrome: [B,C] syndrome bits from quantum measurements
        check_ok: [B,C] output mask for satisfied constraints
        active_idx: [B_active] indices of active batch lanes
        B, B_active, C, V, E: batch size, active batch size, checks, variables, edges
        BLOCK_SIZE: vectorization block size
        ROWS_PER_CHK: number of check rows processed per program
    """
    pid = tl.program_id(axis=0)
    base = pid * ROWS_PER_CHK
    for r in tl.static_range(0, ROWS_PER_CHK):
        idx = base + r
        if idx < B_active * C:
            b_active = idx // C
            i = idx % C

            # Map active batch index to actual batch index
            b = tl.load(active_idx + b_active)

            row_start = tl.load(chk_ptr + i)
            row_end = tl.load(chk_ptr + i + 1)
            par = tl.zeros((), dtype=tl.int32)

            tile = row_start
            while tile < row_end:
                offs = tile + tl.arange(0, BLOCK_SIZE)
                m = offs < row_end
                e = tl.load(chk_edges + offs, mask=m, other=0)
                v = tl.load(edge_var + e, mask=m, other=0)
                bits = tl.load(hard_dec + b * V + v, mask=m, other=0).to(tl.int32) & 1
                par = par ^ (tl.sum(bits & m.to(tl.int32), axis=0) & 1)
                tile += BLOCK_SIZE

            syn = tl.load(syndrome + b * C + i).to(tl.int32) & 1
            ok = (par == syn).to(tl.uint8)
            tl.store(check_ok + b * C + i, ok)


@triton.jit
def parity_from_hard_compact_with_zero_kernel(
    hard_dec,  # [B,V] uint8
    chk_ptr,
    chk_edges,  # CSR over checks -> edge IDs
    edge_var,  # [E] var index per edge
    syndrome,  # [B,C] uint8
    check_ok,  # [B,C] uint8 (out)
    active,  # [B] uint8 - active lane mask
    B,
    C,
    V,
    E,
    BLOCK_SIZE: tl.constexpr,
    ROWS_PER_CHK: tl.constexpr,  # Number of check rows to process per program
):
    """Parity check over all lanes that writes check_ok for active lanes only.

    Inactive lanes keep the check_ok values from their last active iteration.

    Args:
        hard_dec: [B,V] hard decisions (0/1 bits)
        chk_ptr, chk_edges: CSR representation of check constraints
        edge_var: [E] variable index for each edge
        syndrome: [B,C] syndrome bits from quantum measurements
        check_ok: [B,C] output mask for satisfied constraints
        active: [B] active lane mask (1=active, 0=frozen)
        B, C, V, E: batch size, checks, variables, edges
        BLOCK_SIZE: vectorization block size
        ROWS_PER_CHK: number of check rows processed per program
    """
    pid = tl.program_id(axis=0)
    base = pid * ROWS_PER_CHK
    for r in tl.static_range(0, ROWS_PER_CHK):
        idx = base + r
        if idx < B * C:
            b = idx // C
            i = idx % C

            # Check if lane is active
            is_active = tl.load(active + b) != 0

            if is_active:
                # Active lane: compute parity check
                row_start = tl.load(chk_ptr + i)
                row_end = tl.load(chk_ptr + i + 1)
                par = tl.zeros((), dtype=tl.int32)

                tile = row_start
                while tile < row_end:
                    offs = tile + tl.arange(0, BLOCK_SIZE)
                    m = offs < row_end
                    e = tl.load(chk_edges + offs, mask=m, other=0)
                    v = tl.load(edge_var + e, mask=m, other=0)
                    bits = tl.load(hard_dec + b * V + v, mask=m, other=0).to(tl.int32) & 1
                    par = par ^ (tl.sum(bits & m.to(tl.int32), axis=0) & 1)
                    tile += BLOCK_SIZE

                syn = tl.load(syndrome + b * C + i).to(tl.int32) & 1
                ok = (par == syn).to(tl.uint8)
                tl.store(check_ok + b * C + i, ok)


@triton.jit
def zero_check_ok_inactive_kernel(
    check_ok,  # [B,C] uint8
    active,  # [B] uint8
    B,
    C,
    BLOCK_B: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Zero out check_ok for inactive lanes to ensure clean parity statistics.

    This utility kernel ensures that inactive decoding lanes have zero check_ok values,
    preventing them from contributing to convergence statistics.
    """
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)

    b_idx = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    c_idx = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)

    mb = b_idx < B
    mc = c_idx < C

    # load active flags for this tile's rows
    act = tl.load(active + b_idx, mask=mb, other=0)
    inactive_rows = (act == 0) & mb

    # broadcast to 2D tile
    bb = b_idx[:, None]
    cc = c_idx[None, :]
    mask2d = inactive_rows[:, None] & mc[None, :]

    tl.store(check_ok + bb * C + cc, 0, mask=mask2d)


# fmt: off
@triton.jit
def check_and_select_kernel(
    check_ok,          # [B,C] uint8
    posterior_ratios,  # [B,V] dtype - always used for weight computation (matches Rust)
    hard_dec,          # [B,V] uint8
    wj,                # [V] shared or [B,V] per-shot dtype
    best_weights,      # [B] dtype (in/out)
    best_errors,       # [B,V] uint8 (out when improved)
    valid_solutions,   # [B] uint8 (out flag)
    found_count,       # [B] int32 (in/out)
    active,            # [B] uint8 - active lane mask (skip frozen lanes)
    B,
    C,
    V,
    dtype_enum: tl.constexpr,  # 0=fp16, 1=fp32, 2=fp64 (compile-time constant)
    BLOCK_C: tl.constexpr,
    BLOCK_V: tl.constexpr,
    NUM_SOLS: tl.constexpr,    # stop_nconv
    PRIOR_BATCHED: tl.constexpr = False,  # if True, wj is [B,V], else [V]
):
# fmt: on
    """Device-side solution selection and validation kernel.

    This kernel performs solution selection and validation entirely on device,
    avoiding host-device transfers. For each lane, it:
    1. Checks if all parity constraints are satisfied
    2. Computes solution weight using log-prior ratios (always uses posterior_ratios, matching Rust)
    3. Updates best solution if current solution is better

    This corresponds to the solution selection logic in Rust relay.rs.

    Weight computation matches Rust get_decoding_quality():
    - weight = sum(wj[i] for i where error_bits[i] == 1 and is_finite(wj[i]))
    - wj contains raw log_prior_ratios (can be negative when error prob > 0.5)
    - Only finite values contribute (non-finite values are set to 0 in wj)

    Args:
        check_ok: [B,C] mask of satisfied constraints
        posterior_ratios: [B,V] belief marginals (always used for weight computation)
        hard_dec: [B,V] hard decisions
        wj: [V] raw log-prior ratios for weight computation (not abs, can be negative)
        best_weights: [B] current best weights (in/out)
        best_errors: [B,V] current best error patterns (out)
        valid_solutions: [B] valid solution flags (out)
        found_count: [B] convergence counters (in/out)
        active: [B] active lane mask (1=active, 0=frozen; frozen lanes are skipped)
        B, C, V: batch size, checks, variables
        BLOCK_C, BLOCK_V: vectorization block sizes
    """
    b = tl.program_id(axis=0)
    if b >= B:
        return
    # Skip frozen lanes (avoid redundant parity reads)
    if tl.load(active + b) == 0:
        return

    # 1) all_ok = all(check_ok[b, :])
    all_ok = tl.full((), 1, dtype=tl.int32)
    c0 = 0
    while c0 < C:
        cc = c0 + tl.arange(0, BLOCK_C)
        mc = cc < C
        vals = tl.load(check_ok + b * C + cc, mask=mc, other=0).to(tl.int32)
        # if any zero present in this tile, min will be 0
        tile_all = tl.min(tl.where(mc, vals, 1), axis=0)
        all_ok = tl.minimum(all_ok, tile_all)
        c0 += BLOCK_C

    if all_ok == 0:
        return

    # 2) compute current hard decisions and weight (always use posterior_ratios, matching Rust)
    # Use storage dtype directly for all computations
    if dtype_enum == 0:  # fp16
        weight = tl.zeros((), dtype=tl.float16)
    elif dtype_enum == 2:  # fp64
        weight = tl.zeros((), dtype=tl.float64)
    else:  # fp32
        weight = tl.zeros((), dtype=tl.float32)
    v0 = 0
    while v0 < V:
        vv = v0 + tl.arange(0, BLOCK_V)
        mv = vv < V
        posterior_v = tl.load(posterior_ratios + b * V + vv, mask=mv, other=0.0)
        bits = (posterior_v <= 0.0).to(tl.uint8)
        if PRIOR_BATCHED:
            wtile = tl.load(wj + b * V + vv, mask=mv, other=0.0)
        else:
            wtile = tl.load(wj + vv, mask=mv, other=0.0)
        # Convert bits to match weight dtype (optimize fp32 path)
        if dtype_enum == 0:  # fp16
            weight += tl.sum(wtile * bits.to(tl.float16), axis=0)
        elif dtype_enum == 2:  # fp64
            weight += tl.sum(wtile * bits.to(tl.float64), axis=0)
        else:  # fp32 - bits is uint8, need to convert to float for multiplication
            weight += tl.sum(wtile * bits.to(tl.float32), axis=0)
        v0 += BLOCK_V

    # 3) compare-and-update best
    # Match Rust: First valid solution always becomes the initial best (line 309: min_pm = result.decoding_quality)
    # Subsequent solutions only update if strictly better (line 362: if pm < min_pm)
    prev_best = tl.load(best_weights + b)
    prev_valid = tl.load(valid_solutions + b)  # Load before updating
    # Check if this is the first valid solution ever (prev_best == inf)
    # Use dtype-appropriate threshold for infinity detection
    if dtype_enum == 0:  # fp16: max finite ~65504
        inf_threshold = 60000.0
    elif dtype_enum == 2:  # fp64
        inf_threshold = 1e20
    else:  # fp32
        inf_threshold = 1e20
    is_inf = (prev_best >= inf_threshold).to(tl.int32)
    # First valid solution always updates (matches Rust line 309)
    # Subsequent solutions only update if strictly better (matches Rust line 362)
    better = (weight < prev_best) | (is_inf == 1)
    if better:
        # weight is already in correct dtype, no conversion needed
        tl.store(best_weights + b, weight)
        # write best_errors row (always use posterior_ratios, matching Rust)
        off = 0
        while off < V:
            vv = off + tl.arange(0, BLOCK_V)
            mv = vv < V
            posterior_v2 = tl.load(posterior_ratios + b * V + vv, mask=mv, other=0.0)
            # Match Rust: Bit::from(posterior <= 0) for best_errors storage
            bits2 = (posterior_v2 <= 0.0).to(tl.uint8)
            tl.store(best_errors + b * V + vv, bits2, mask=mv)
            off += BLOCK_V
    # mark lane valid for this cadence
    tl.store(valid_solutions + b, 1)
    # Increment found_count only on first success since last reset (per leg)
    if prev_valid == 0:
        old = tl.load(found_count + b)
        tl.store(found_count + b, old + 1)


# fmt: off
@triton.jit
def freeze_finished_lanes_kernel(
    best_errors,        # [B,V] uint8
    hard_dec,
    gamma,
    active,             # [B,V] uint8, [B,V] fp32, [B] uint8
    found_count,
    first_iter,
    iter_counter,       # [B] int32, [B] int32, [B] int32
    sought_sols,
    stop_nconv,
    in_leg: tl.constexpr,
    count_iteration: tl.constexpr,
    V: tl.constexpr,
    all_done_flag,      # [1] uint8 (out) - set to 1 if all lanes converged
    arrival_counter,    # [1] int32 (atomic) - arrival counter for last-program reduction
    B,                  # scalar int32 - number of batch lanes
):
# fmt: on
    """Freeze converged lanes, increment iter_counter, and check all-done (fused).

    This kernel implements lane freezing for converged solutions, corresponding
    to the convergence handling in Rust relay.rs:
    1. With count_iteration, increments iter_counter for every lane that is active on
       entry, including a lane that converges in this call
    2. For lanes that have found enough valid solutions: records the first convergence
       iteration (outside relay legs), freezes the lane (sets active=0) and resets its
       gamma values
    3. Last-arriving program checks if all lanes converged (fused reduce_all_ge)

    Args:
        best_errors: [B,V] best error patterns found so far
        hard_dec: [B,V] current hard decisions (updated for converged lanes)
        gamma: [B,V] memory strengths (reset to 0 for converged lanes)
        active: [B] active lane mask (set to 0 for converged lanes)
        found_count: [B] convergence counters
        first_iter: [B] first convergence iteration (updated for converged lanes)
        iter_counter: [B] per-lane count of BP iterations run
        stop_nconv: target number of solutions for convergence
        in_leg: use min(sought_sols, stop_nconv) as the stop criterion and do not record first_iter
        count_iteration: count the BP iteration that precedes this call
        V: number of variables
        all_done_flag: [1] output flag (1 if all lanes converged, 0 otherwise)
        arrival_counter: [1] atomic counter for coordinating last-program reduction
        B: number of batch lanes
    """
    b = tl.program_id(axis=0)
    # Process lane only if active (skip frozen lanes)
    if tl.load(active + b) != 0:
        # Increment current_iteration after every iteration, including the one that converges.
        itc = tl.load(iter_counter + b)
        if count_iteration:
            itc = itc + 1
            tl.store(iter_counter + b, itc)
        fc = tl.load(found_count + b)
        if in_leg:
            ss = tl.load(sought_sols + b)
            stop_criterion = tl.minimum(ss, stop_nconv)
        else:
            stop_criterion = stop_nconv
        done_now = fc >= stop_criterion
        if done_now:
            fi = tl.load(first_iter + b)
            if fi < 0:
                if not in_leg:
                    tl.store(first_iter + b, itc)
                col = 0
                while col < V:
                    offs = col + tl.arange(0, 128)
                    m = offs < V
                    be = tl.load(best_errors + b * V + offs, mask=m, other=0)
                    tl.store(hard_dec + b * V + offs, be, mask=m)
                    col += 128
            tl.store(active + b, 0)
            col2 = 0
            while col2 < V:
                offs2 = col2 + tl.arange(0, 128)
                m2 = offs2 < V
                tl.store(gamma + b * V + offs2, 0.0, mask=m2)
                col2 += 128

    # All programs (frozen and active) participate in the all-done reduction.
    # The last program to arrive performs the reduction over found_count.
    old_arrival = tl.atomic_add(arrival_counter, 1)
    if old_arrival == B - 1:
        # Last program: check if all lanes have converged
        all_ok = 1
        tile = 0
        while tile < B:
            offs_r = tile + tl.arange(0, 256)
            m_r = offs_r < B
            fc_r = tl.load(found_count + offs_r, mask=m_r, other=stop_nconv)
            if in_leg:
                ss_r = tl.load(sought_sols + offs_r, mask=m_r, other=stop_nconv)
                sc_r = tl.minimum(ss_r, stop_nconv)
            else:
                sc_r = tl.zeros_like(fc_r) + stop_nconv
            ok_r = (fc_r >= sc_r).to(tl.int32)
            all_ok = tl.minimum(all_ok, tl.min(tl.where(m_r, ok_r, 1), axis=0))
            tile += 256
        tl.store(all_done_flag, all_ok)
        # Reset arrival counter for next call
        tl.store(arrival_counter, 0)


# fmt: off
@triton.jit
def init_messages_kernel(
    mu,
    nu,                         # [B,E] messages - default mode layout
    edge_var,                   # [E] variable index for each edge
    log_prior_ratios,           # [V] shared or [B,V] per-shot prior log-likelihood ratios
    B,
    E,
    V,                          # batch size, number of edges, number of variables
    dtype_enum: tl.constexpr,   # 0=fp16, 1=fp32, 2=fp64
    BLOCK: tl.constexpr,
    PRIOR_BATCHED: tl.constexpr = False,  # if True, log_prior_ratios is [B,V], else [V]
):
# fmt: on
    """Initialize message tensors: nu to zero, mu to prior values (for [B,E] layout).

    Note: This is used for default mode. For throughput mode with [E,B] layout,
    use init_messages_with_priors_EB_kernel.
    """
    pid = tl.program_id(axis=0)
    N = B * E
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < N

    # Compute batch and edge indices for [B,E] layout
    b_idx = offs // E
    e_idx = offs % E

    # Additional safety: ensure indices are within bounds
    m = m & (b_idx < B) & (e_idx < E)

    # Zero nu (check-to-variable messages)
    z = tl.zeros((BLOCK,), dtype=tl.float32)
    tl.store(nu + offs, z, mask=m)

    # Initialize mu (variable-to-check messages) to prior values
    j = tl.load(edge_var + e_idx, mask=m, other=0)
    if PRIOR_BATCHED:
        prior = tl.load(log_prior_ratios + b_idx * V + j, mask=m, other=0.0)
    else:
        prior = tl.load(log_prior_ratios + j, mask=m, other=0.0)

    # Store with appropriate dtype based on dtype_enum (compile-time optimized)
    if dtype_enum == 0:  # fp16
        tl.store(mu + offs, prior.to(tl.float16), mask=m)
    elif dtype_enum == 2:  # fp64
        tl.store(mu + offs, prior.to(tl.float64), mask=m)
    else:  # fp32 (dtype_enum == 1)
        tl.store(mu + offs, prior, mask=m)


# fmt: off
@triton.jit
def init_messages_with_priors_EB_kernel(
    muT,
    nuT,               # [E,B] messages - throughput layout
    edge_var,          # [E] variable index for each edge
    log_prior_ratios,  # [V] shared or [B,V] per-shot prior log-likelihood ratios
    E,
    V,
    B,
    active_idx,        # [B_active] int32 - compact list of active batch indices (ignored if use_active=False)
    B_active,          # int32 - number of active lanes
    use_active: tl.constexpr,  # bool - if True, only initialize active lanes
    dtype_enum: tl.constexpr,  # 0=fp16, 1=fp32, 2=fp64 (compile-time constant)
    BLOCK: tl.constexpr,
    PRIOR_BATCHED: tl.constexpr = False,  # if True, log_prior_ratios is [B,V], else [V]
):
# fmt: on
    """Initialize messages: nu to zero, mu to prior values (throughput [E,B] layout).

    This matches Rust's initialization:
    - check_to_variable (nu) -> zero
    - variable_to_check (mu) -> priors (log_prior_ratios)

    If use_active=True and B_active < B, only initializes messages for active lanes,
    dramatically reducing memory traffic when B_active << B.

    For each edge e:
    - If use_active=False: sets muT[e*B + b] = log_prior_ratios[j] for all batches b
    - If use_active=True: sets muT[e*B + bs[b_active]] = log_prior_ratios[j] for active batches only
    """
    pid = tl.program_id(axis=0)

    if use_active and (B_active < B):
        # Active lanes only: process E * B_active elements
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs < (E * B_active)

        # Compute edge and active batch indices
        edge_idx = offs // B_active
        b_active_idx = offs % B_active

        # Additional safety: ensure edge_idx is within bounds
        m = m & (edge_idx < E) & (b_active_idx < B_active)

        # Load active batch index
        # Safety: b_active_idx is already validated to be < B_active, so active_idx[b_active_idx] is valid
        # If active_idx contains invalid indices (>= B), this will cause out-of-bounds access
        # Validation in decoder.py should catch this before kernel launch
        bs = tl.load(active_idx + b_active_idx, mask=m, other=0)

        # Compute linear offset in [E,B] layout: e*B + b
        # Note: edge_idx is already validated to be < E by the mask
        # Validation in decoder.py ensures E * B <= INT32_MAX, so int32 arithmetic is safe
        linear_offs = edge_idx * B + bs

        # Zero nu (check-to-variable messages) for active lanes only
        z = tl.zeros((BLOCK,), dtype=tl.float32)
        tl.store(nuT + linear_offs, z, mask=m)

        # Initialize mu (variable-to-check messages) to prior values for active lanes only
        j = tl.load(edge_var + edge_idx, mask=m, other=0)
        if PRIOR_BATCHED:
            prior = tl.load(log_prior_ratios + bs * V + j, mask=m, other=0.0)
        else:
            prior = tl.load(log_prior_ratios + j, mask=m, other=0.0)

        # Store with appropriate dtype based on dtype_enum (compile-time optimized)
        if dtype_enum == 0:  # fp16
            tl.store(muT + linear_offs, prior.to(tl.float16), mask=m)
        elif dtype_enum == 2:  # fp64
            tl.store(muT + linear_offs, prior.to(tl.float64), mask=m)
        else:  # fp32 (dtype_enum == 1)
            tl.store(muT + linear_offs, prior, mask=m)
    else:
        # Full initialization: process E * B elements
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs < (E * B)

        # Zero nu (check-to-variable messages)
        z = tl.zeros((BLOCK,), dtype=tl.float32)
        tl.store(nuT + offs, z, mask=m)

        # Initialize mu (variable-to-check messages) to prior values
        edge_idx = offs // B
        j = tl.load(edge_var + edge_idx, mask=m, other=0)
        if PRIOR_BATCHED:
            b_idx = offs % B  # [E,B] layout: batch index is the inner stride
            prior = tl.load(log_prior_ratios + b_idx * V + j, mask=m, other=0.0)
        else:
            prior = tl.load(log_prior_ratios + j, mask=m, other=0.0)

        # Store with appropriate dtype based on dtype_enum (compile-time optimized)
        if dtype_enum == 0:  # fp16
            tl.store(muT + offs, prior.to(tl.float16), mask=m)
        elif dtype_enum == 2:  # fp64
            tl.store(muT + offs, prior.to(tl.float64), mask=m)
        else:  # fp32 (dtype_enum == 1)
            tl.store(muT + offs, prior, mask=m)


# fmt: off
@triton.jit
def reduce_all_ge_kernel(
    found_count,            # [B] int32
    stop_nconv,             # scalar int32
    out_flag,               # [1] uint8 (out)
    B,                      # scalar int32
):
# fmt: on
    """Set out_flag to 1 if every lane has found_count >= stop_nconv, else 0.

    Single-program device-side reduction over the B lanes.
    """
    pid = tl.program_id(axis=0)
    if pid != 0:
        return
    acc = tl.full((), 1, dtype=tl.int32)
    tile = 0
    while tile < B:
        offs = tile + tl.arange(0, 256)
        m = offs < B
        fc = tl.load(found_count + offs, mask=m, other=stop_nconv)
        ok = (fc >= stop_nconv).to(tl.int32)
        acc = tl.minimum(acc, tl.min(tl.where(m, ok, 1), axis=0))
        tile += 256
    tl.store(out_flag, acc.to(tl.uint8))


@triton.jit
def unfreeze_finished_lanes_kernel(active, found_count, stop_nconv):  # [B] uint8  # [B] int32
    """Unfreeze lanes with less than stop_nconv solutions after
    they were frozen for findig previous solution.

    Args:
        active: [B] active lane mask (set to 0 for converged lanes)
        found_count: [B] convergence counters
        stop_nconv: target number of solutions for convergence
    """
    b = tl.program_id(axis=0)
    fc = tl.load(found_count + b)
    done_now = fc >= stop_nconv
    if not done_now:
        tl.store(active + b, 1)




# fmt: off
@triton.jit
def c2v_min_sum_btile_kernel(
    muT,
    nuT,         # [E,B]
    chk_ptr,
    chk_edges,   # CSR over checks -> edge IDs
    syndrome,    # [B,C] (uint8)
    active_idx,  # [B_active] int32 - compact list of active batch indices
    active,      # [B] uint8 - active lane mask (skip frozen lanes)
    B,
    B_active,
    C,
    E,
    alpha,
    beta,
    use_alpha: tl.constexpr,
    use_beta: tl.constexpr,
    dtype_enum: tl.constexpr,  # 0=fp16, 1=fp32, 2=fp64 (compile-time constant)
    BLOCK_SIZE: tl.constexpr,
    BTILE: tl.constexpr,
    chk_start: int,            # Start check index for block-layered (0 = all checks)
    chk_end: int,              # End check index for block-layered (0 = all checks, means C)
    edge_var,                  # [E] int32 - variable index per edge (only used when use_delta=True)
    L,                         # [B,V] - per-lane marginals (only used when use_delta=True)
    V: int,                    # number of variables
    use_delta: tl.constexpr,  # if True, run delta-trick variant
):
# fmt: on
    """Check-to-variable min-sum over [E,B] layout, vectorized across BTILE batch lanes.
    Each program processes one check row i and updates all its edges' nuT[:, b:b+BTILE].

    For block-layered scheduling:
    - chk_start, chk_end: Process only checks in range [chk_start, chk_end)
    - If chk_end == 0, processes all checks (flood-fill mode)
    """
    i = tl.program_id(axis=0)
    btile_id = tl.program_id(axis=1)

    # Block-layered: adjust check index if range is specified
    if chk_end > 0:
        # Block-layered mode: i is relative to block, add chk_start offset
        chk_idx = chk_start + i
        if chk_idx >= chk_end:
            return  # Out of block range
    else:
        # Flood-fill mode: i is absolute check index
        chk_idx = i
        if chk_idx >= C:
            return

    # Use chk_idx instead of i for all check operations
    i = chk_idx
    b0 = btile_id * BTILE
    pos = b0 + tl.arange(0, BTILE)
    mpos = pos < B_active
    bs = tl.load(active_idx + pos, mask=mpos, other=0)
    # Skip frozen lanes via active mask (avoids compaction overhead)
    active_vals = tl.load(active + bs, mask=mpos, other=0)
    mb = mpos & (active_vals != 0)

    row_start = tl.load(chk_ptr + i)
    row_end = tl.load(chk_ptr + i + 1)
    if row_end <= row_start:
        return
    deg_is_one = row_end - row_start == 1

    # Accumulators per batch lane
    neg_parity = tl.zeros((BTILE,), dtype=tl.int32)
    # Use fp32 accumulators for fp16/fp32 storage, fp64 for fp64
    if dtype_enum == 2:  # fp64
        min1 = tl.full((BTILE,), 1e30, dtype=tl.float64)
        min2 = tl.full((BTILE,), 1e30, dtype=tl.float64)
    else:  # fp16 or fp32
        min1 = tl.full((BTILE,), 1e30, dtype=tl.float32)
        min2 = tl.full((BTILE,), 1e30, dtype=tl.float32)
    cnt1 = tl.zeros((BTILE,), dtype=tl.int32)

    # PASS 1: compute min1/min2 and parity per batch lane
    tile = row_start
    while tile < row_end:
        offs = tile + tl.arange(0, BLOCK_SIZE)
        mrow = offs < row_end
        evec = tl.load(chk_edges + offs, mask=mrow, other=0)  # [BLOCK_SIZE]
        # Build 2D mask and gather inputs over edges×BTILE
        mask2d = (mrow[:, None]) & (mb[None, :])
        if use_delta:
            v_idx = tl.load(edge_var + evec, mask=mrow, other=0)
            L_tile = tl.load(L + (bs[None, :] * V) + v_idx[:, None], mask=mask2d, other=0.0)
            nu_old = tl.load(nuT + (evec[:, None]) * B + bs[None, :], mask=mask2d, other=0.0)
            mu_tile = L_tile - nu_old
        else:
            mu_tile = tl.load(
                muT + (evec[:, None]) * B + bs[None, :], mask=mask2d, other=0.0
            )  # [BLOCK_SIZE, BTILE]
        # tile parity via xor of tile parity bits
        tile_par = tl.sum((mu_tile < 0.0) & mask2d, axis=0).to(tl.int32) & 1
        neg_parity = neg_parity ^ tile_par
        # For fp16 storage, do reductions in fp32; otherwise stay in storage dtype
        a = tl.abs(mu_tile.to(tl.float32)) if dtype_enum == 0 else tl.abs(mu_tile)
        a = tl.where(mask2d, a, 1e30)
        # min1 over edges and tie count
        min1_t = tl.min(a, axis=0)
        cnt1_t = tl.sum(a == min1_t[None, :], axis=0).to(tl.int32)
        # min2: mask a single occurrence along edge axis
        pos1 = tl.argmax(-a, axis=0)
        a2 = tl.where(tl.arange(0, BLOCK_SIZE)[:, None] == pos1[None, :], 1e30, a)
        min2_t = tl.min(a2, axis=0)
        # merge with running min1/min2 and counts
        better1 = min1_t < min1
        cnt1 = tl.where(better1, cnt1_t, tl.where(min1_t == min1, cnt1 + cnt1_t, cnt1))
        min2 = tl.where(
            better1,
            tl.minimum(min1, min2_t),
            tl.where(min1_t == min1, tl.minimum(min2, min2_t), tl.minimum(min2, min1_t)),
        )
        min1 = tl.where(better1, min1_t, min1)
        tile += BLOCK_SIZE

    syn = tl.load(syndrome + bs * C + i, mask=mb, other=0).to(tl.int32) & 1
    par_bit = (neg_parity & 1) ^ syn
    sign_prod = tl.where(par_bit == 0, 1.0, -1.0)

    # PASS 2: compute and store nu over edges×BTILE
    tile = row_start
    while tile < row_end:
        offs = tile + tl.arange(0, BLOCK_SIZE)
        mrow = offs < row_end
        evec = tl.load(chk_edges + offs, mask=mrow, other=0)
        mask2d = (mrow[:, None]) & (mb[None, :])
        if use_delta:
            v_idx = tl.load(edge_var + evec, mask=mrow, other=0)
            L_tile = tl.load(L + (bs[None, :] * V) + v_idx[:, None], mask=mask2d, other=0.0)
            nu_old = tl.load(nuT + (evec[:, None]) * B + bs[None, :], mask=mask2d, other=0.0)
            mu_tile = L_tile - nu_old
        else:
            mu_tile = tl.load(muT + (evec[:, None]) * B + bs[None, :], mask=mask2d, other=0.0)
            nu_old = tl.load(nuT + (evec[:, None]) * B + bs[None, :], mask=mask2d, other=0.0)
        sgn = tl.where(mu_tile >= 0.0, 1.0, -1.0)
        # Identify min1 lanes per edge×batch and apply correct tie logic
        a = tl.abs(mu_tile)
        a_mask = tl.where(mask2d, a, 1e30)
        is_min1 = a_mask == min1[None, :]
        use_min2 = (cnt1[None, :] == 1) & is_min1
        out_mag = tl.where(use_min2, min2[None, :], min1[None, :])
        # Degree-1 checks must send zero.
        out_mag = tl.where(deg_is_one, 0.0, out_mag)
        if use_alpha:
            out_mag = alpha * out_mag
        if use_beta:
            out_mag = tl.maximum(out_mag - beta, 0.0)
        nu_tile = (sign_prod[None, :] * sgn) * out_mag
        # Write updated S (nu) back to nuT
        if dtype_enum == 0:  # fp16
            nu_new = nu_tile.to(tl.float16)
        elif dtype_enum == 2:  # fp64
            nu_new = nu_tile.to(tl.float64)
        else:  # fp32 (dtype_enum == 1)
            nu_new = nu_tile
        tl.store(nuT + (evec[:, None]) * B + bs[None, :], nu_new, mask=mask2d)
        # Delta-trick: update L in place for this block (conflict-free layers guarantee no races)
        if use_delta:
            delta = nu_new - nu_old
            new_L = L_tile + delta
            tl.store(L + (bs[None, :] * V) + v_idx[:, None], new_L, mask=mask2d)
        tile += BLOCK_SIZE


# fmt: off
@triton.jit
def v2c_and_gamma_btile_kernel(
    nuT,
    muT,         # [E,B]
    var_ptr,
    var_edges,   # CSR over vars -> edge IDs
    lam,
    log_prior_ratios,
    gamma,       # [B,V], [V], [B,V]
    # lam: Current mixed beliefs [B,V]
    # log_prior_ratios: Prior log-likelihood ratios [V] (matches Rust: log_prior_ratios)
    # gamma: Memory strengths [B,V] (matches Rust: memory_strengths)
    posterior_ratios,
    hard_dec,    # [B,V], [B,V]
    # posterior_ratios: Posterior ratios/marginals [B,V] (matches Rust: posterior_ratios)
    active_idx,  # [B_active] int32 - compact list of active batch indices
    active,      # [B] uint8 - active lane mask (skip frozen lanes)
    B,
    B_active,
    V,
    E,
    dtype_enum: tl.constexpr,  # 0=fp16, 1=fp32, 2=fp64 (compile-time constant)
    BLOCK_SIZE: tl.constexpr,
    BTILE: tl.constexpr,
    WRITE_HARD: tl.constexpr,
    block_vars,      # [num_block_vars] int32 - variable indices for block (can be None pointer for flood-fill)
    block_var_ptr,   # [num_blocks+1] int32 - prefix sum for block_vars (can be None pointer for flood-fill)
    block_idx: int,  # Block index for block-layered (-1 = flood-fill, use all vars)
    PRIOR_BATCHED: tl.constexpr = False,  # if True, log_prior_ratios is [B,V], else [V]
):
# fmt: on
    """Variable-to-check + gamma mixing on [E,B] layout, vectorized across BTILE.

    Rust-compatible gamma mixing: uses previous iteration's marginal (M_prev) in mixing,
    then computes current marginal M_curr = λ_mixed + sum(nu). This matches Rust's
    compute_variable_prior which uses previous posterior_ratios.

    For each variable j and batch tile bs:
    1. Load M_prev (previous iteration's marginal)
    2. Compute λ_mixed = (1-γ)*λ₀ + γ*M_prev
    3. Compute sum_nu (incoming messages)
    4. Compute M_curr = λ_mixed + sum(nu)
    5. Write muT (outgoing messages)

    For block-layered scheduling:
    - block_vars, block_var_ptr, block_idx: Process only variables in the specified block
    - If block_idx == -1, processes all variables (flood-fill mode)
    """
    var_list_idx = tl.program_id(
        axis=0
    )  # Index into block_vars list (or variable index in flood-fill)
    btile_id = tl.program_id(axis=1)

    # Block-layered: get variable index from block_vars list
    if block_idx >= 0:
        # Block-layered mode: var_list_idx is index into block_vars for this block
        var_start = tl.load(block_var_ptr + block_idx)
        var_end = tl.load(block_var_ptr + block_idx + 1)
        if var_list_idx >= (var_end - var_start):
            return  # Out of block variable range
        j = tl.load(block_vars + var_start + var_list_idx)
    else:
        # Flood-fill mode: var_list_idx is absolute variable index
        j = var_list_idx
        if j >= V:
            return
    b0 = btile_id * BTILE
    pos = b0 + tl.arange(0, BTILE)
    mpos = pos < B_active
    # Safety: Only load from active_idx if pos < B_active to prevent out-of-bounds access
    bs = tl.load(active_idx + pos, mask=mpos, other=0)
    # Skip frozen lanes via active mask (avoids compaction overhead)
    active_vals = tl.load(active + bs, mask=mpos, other=0)
    mb = mpos & (active_vals != 0)

    row_start = tl.load(var_ptr + j)
    row_end = tl.load(var_ptr + j + 1)

    # Load previous iteration's marginal (posterior_prev) for Rust-compatible gamma mixing
    # Rust uses: λ_mixed = (1-γ)*log_prior + γ*posterior_prev, then posterior = λ_mixed + sum(nu)
    posterior_bj_prev = tl.load(
        posterior_ratios + bs * V + j, mask=mb, other=0.0
    )  # Previous iteration's marginal/posterior

    # FUSED GAMMA MIXING (Rust-compatible): λ = (1-γ)*log_prior + γ*posterior_prev
    g_bj = tl.load(gamma + bs * V + j, mask=mb, other=0.0)
    if PRIOR_BATCHED:
        log_prior_j = tl.load(log_prior_ratios + bs * V + j, mask=mb, other=0.0)
    else:
        log_prior_j = tl.load(log_prior_ratios + j)
    lam_mixed = (1.0 - g_bj) * log_prior_j + g_bj * posterior_bj_prev

    # sum incoming nu over edges for batch lanes (vectorized)
    # Use fp32 accumulators for fp16/fp32 storage, fp64 for fp64
    if dtype_enum == 2:  # fp64
        sum_nu = tl.zeros((BTILE,), dtype=tl.float64)
    else:  # fp16 or fp32
        sum_nu = tl.zeros((BTILE,), dtype=tl.float32)
    tile = row_start
    while tile < row_end:
        offs = tile + tl.arange(0, BLOCK_SIZE)
        mrow = offs < row_end
        evec = tl.load(var_edges + offs, mask=mrow, other=0)  # [BLOCK_SIZE]
        mask2d = (mrow[:, None]) & (mb[None, :])
        nu_tile = tl.load(nuT + (evec[:, None]) * B + bs[None, :], mask=mask2d, other=0.0)
        sum_nu += tl.sum(tl.where(mask2d, nu_tile, 0.0), axis=0)
        tile += BLOCK_SIZE

    # Compute current marginal: posterior = λ_mixed + sum(nu) (matches Rust posterior_ratios update)
    posterior_bj = lam_mixed + sum_nu
    # Always store posterior_ratios for next iteration's gamma mixing (required for Rust compatibility)
    tl.store(posterior_ratios + bs * V + j, posterior_bj, mask=mb)
    # Optionally update hard_dec if requested (to avoid bandwidth in hot path)
    if WRITE_HARD:
        # Match Rust convergence rule Bit::from(posterior <= 0)
        tl.store(hard_dec + bs * V + j, (posterior_bj <= 0.0).to(tl.uint8), mask=mb)

    # Store mixed lambda for next iteration (used in message computation)
    tl.store(lam + bs * V + j, lam_mixed, mask=mb)

    # write muT for each edge lane (vectorized) using previous iteration beliefs
    tile = row_start
    while tile < row_end:
        offs = tile + tl.arange(0, BLOCK_SIZE)
        mrow = offs < row_end
        evec = tl.load(var_edges + offs, mask=mrow, other=0)
        mask2d = (mrow[:, None]) & (mb[None, :])
        nu_tile = tl.load(nuT + (evec[:, None]) * B + bs[None, :], mask=mask2d, other=0.0)
        # Use lam_mixed (current iteration's mixed belief) to compute mu, matching Rust schedule
        mu_tile = lam_mixed[None, :] + (sum_nu[None, :] - nu_tile)
        tl.store(muT + (evec[:, None]) * B + bs[None, :], mu_tile, mask=mask2d)
        tile += BLOCK_SIZE


# fmt: off
@triton.jit
def post_from_L_kernel(
    L,
    posterior_ratios,
    hard_dec,       # [B,V], [B,V], [B,V]
    active_idx,     # [B_active]
    B,
    B_active,
    V,
    dtype_enum: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
# fmt: on
    """Update posterior_ratios and hard_dec from L for active lanes only.

    This is used by layered delta-trick mode where L already contains the
    current per-variable marginals.
    """
    j = tl.program_id(axis=0)  # variable index
    btile_id = tl.program_id(axis=1)
    if j >= V:
        return
    b0 = btile_id * BLOCK_V
    pos = b0 + tl.arange(0, BLOCK_V)
    mpos = pos < B_active
    bs = tl.load(active_idx + pos, mask=mpos, other=0)

    # Load L and write posterior + hard_dec
    if dtype_enum == 2:  # fp64
        L_tile = tl.load(L + bs * V + j, mask=mpos, other=0.0).to(tl.float64)
    else:
        # For fp16/fp32 storage, do the compare in fp32
        L_tile = tl.load(L + bs * V + j, mask=mpos, other=0.0).to(tl.float32)
    tl.store(posterior_ratios + bs * V + j, L_tile, mask=mpos)
    tl.store(hard_dec + bs * V + j, (L_tile <= 0.0).to(tl.uint8), mask=mpos)


@triton.jit
def permute_rows_kernel(
    srcEB,
    dstEB,  # [E, B]
    perm,  # [E] int32; maps src row e -> dst row p
    E: tl.constexpr,
    B: tl.constexpr,
    BTILE: tl.constexpr,
):
    """Permute edge rows from source layout to destination layout.

    This kernel copies edge rows (all B batch lanes) using a permutation vector,
    enabling efficient layout switching between check-major and variable-major
    orderings for optimized memory access patterns.

    Args:
        srcEB: [E, B] source message tensor
        dstEB: [E, B] destination message tensor
        perm: [E] permutation vector mapping source row e -> destination row p
        E: number of edges
        B: batch size
        BTILE: vectorization tile size for batch dimension
    """
    e = tl.program_id(0)
    if e >= E:
        return

    # destination row for this source row
    p = tl.load(perm + e)

    # copy B entries in BTILE chunks for coalesced IO
    b0 = 0
    while b0 < B:
        bs = b0 + tl.arange(0, BTILE)
        m = bs < B
        vals = tl.load(srcEB + e * B + bs, mask=m, other=0)
        tl.store(dstEB + p * B + bs, vals, mask=m)
        b0 += BTILE


@triton.jit
def count_active_kernel(
    active,  # [B] uint8 mask (1=active, 0=frozen)
    out_count,  # [1] int32 (out) - total count of active lanes
    B: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Count active lanes on device (replaces torch.count_nonzero().item()).

    This kernel performs a device-side reduction to count the number of active
    decoding lanes, avoiding host-device synchronization.

    Args:
        active: [B] active lane mask
        out_count: [1] output count (device tensor)
        B: batch size
        BLOCK_SIZE: vectorization block size
    """
    pid = tl.program_id(axis=0)
    if pid != 0:
        return

    count = tl.zeros((), dtype=tl.int32)
    tile = 0
    while tile < B:
        offs = tile + tl.arange(0, BLOCK_SIZE)
        m = offs < B
        act = tl.load(active + offs, mask=m, other=0).to(tl.int32)
        count += tl.sum(tl.where(m, act, 0), axis=0)
        tile += BLOCK_SIZE

    tl.store(out_count, count)


@triton.jit
def count_active_blocks_kernel(
    active,  # [B] uint8 mask (1=active, 0=frozen)
    block_counts,  # [num_blocks] int32 (out) - count of active per block
    B: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    num_blocks: tl.constexpr,
):
    """Count active lanes per block (for prefix sum compaction).

    Each program counts active elements in its block and writes to block_counts.
    This is used as the first phase of compaction.

    Args:
        active: [B] active lane mask
        block_counts: [num_blocks] output counts per block
        B: batch size
        BLOCK_SIZE: vectorization block size
        num_blocks: number of blocks (should match grid size)
    """
    pid = tl.program_id(axis=0)
    # Note: pid should always be < num_blocks if grid size is set correctly
    base = pid * BLOCK_SIZE

    offs = base + tl.arange(0, BLOCK_SIZE)
    m = offs < B
    act = tl.load(active + offs, mask=m, other=0)
    active_mask = (act != 0) & m

    block_count = tl.sum(active_mask.to(tl.int32), axis=0)
    tl.store(block_counts + pid, block_count)


@triton.jit
def compact_active_indices_kernel(
    active,  # [B] uint8 mask (1=active, 0=frozen)
    active_idx,  # [B] int32 (out) - compact list of active batch indices
    prefix_sum,  # [num_blocks+1] int32 (in) - prefix sum of block counts (from torch.cumsum)
    B: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    num_blocks: tl.constexpr,
):
    """Compact active indices using prefix sum offsets.

    This kernel compacts active batch indices into a contiguous array.
    Requires prefix_sum array computed via torch.cumsum(block_counts) on device.

    Args:
        active: [B] active lane mask
        active_idx: [B] output array for compacted indices (pre-allocated)
        prefix_sum: [num_blocks+1] prefix sum (prefix_sum[i] = sum of counts for blocks 0..i-1)
        B: batch size
        BLOCK_SIZE: vectorization block size
        num_blocks: number of blocks (should match grid size)
    """
    pid = tl.program_id(axis=0)
    # Note: pid should always be < num_blocks if grid size is set correctly
    base = pid * BLOCK_SIZE

    # Get write offset for this block from prefix sum
    # prefix_sum has size num_blocks+1, so prefix_sum[pid] is valid for pid in [0, num_blocks-1]
    write_offset = tl.load(prefix_sum + pid)

    # Process this block
    offs = base + tl.arange(0, BLOCK_SIZE)
    m = offs < B
    act = tl.load(active + offs, mask=m, other=0)

    # Write active indices sequentially starting at write_offset
    local_pos = 0
    for i in tl.static_range(0, BLOCK_SIZE):
        idx = base + i
        if idx < B:
            # Check if this index is active by loading the value directly
            is_active = tl.load(active + idx) != 0
            if is_active:
                write_pos = write_offset + local_pos
                if write_pos < B:  # Safety check
                    tl.store(active_idx + write_pos, idx)
                local_pos += 1


@triton.jit
def compare_count_changed_kernel(
    new_count,  # [1] int32 - new active count
    old_count,  # [1] int32 - old active count
    changed_flag,  # [1] uint8 (out) - 1 if changed, 0 otherwise
):
    """Device-side comparison to check if active count changed (avoids host sync).

    This kernel compares two device tensors and writes a flag, allowing
    device-side control flow without host-device synchronization.
    """
    new = tl.load(new_count)
    old = tl.load(old_count)
    changed = (new != old).to(tl.uint8)
    tl.store(changed_flag, changed)


# ============================================================================
# Data type Enum Helper
# ============================================================================
# All kernels now use dtype_enum (0=fp16, 1=fp32, 2=fp64) instead of msg_is_fp16.
# This is a compile-time constant that enables Triton to optimize dtype-specific
# code paths while maintaining a single kernel implementation.


def dtype_str_to_enum(data_type_str: str) -> int:
    """Convert data_type string to enum value for kernel calls.

    Args:
        data_type_str: "float16", "float32", or "float64"

    Returns:
        Integer enum: 0 for float16, 1 for float32, 2 for float64

    Raises:
        ValueError: If data_type_str is not one of the supported values
    """
    return {"float16": 0, "float32": 1, "float64": 2}[data_type_str]


# ============================================================================
# C2V Kernel with Batch-Organized Grid-Stride Pattern
# ============================================================================


# fmt: off
@triton.jit
def c2v_min_sum_grid_stride_kernel(
    # Message tensors
    mu,
    nu,              # [B,E]
    # CSR graph data
    chk_ptr,
    chk_edges,       # CSR over checks -> edge IDs
    var_ptr,
    var_edges,       # CSR over vars -> edge IDs
    edge_var,        # [E] var index per edge
    # Beliefs and decisions
    lam,
    posterior_ratios,
    hard_dec,        # [B,V], [B,V], [B,V]
    # Inputs
    syndrome,        # [B,C]
    log_prior_ratios,
    gamma,           # [1,V], [B,V]
    active,          # [B] uint8 mask
    # Solution tracking
    check_ok,        # [B,C] uint8 (temporary)
    best_weights,
    best_errors,     # [B] dtype, [B,V] uint8
    valid_solutions,
    found_count,     # [B] uint8, [B] int32
    wj,              # [V] dtype (log prior ratios for weight)
    # Parameters
    B,
    C,
    V,
    E,
    iteration: int,  # Current iteration number (for alpha schedule)
    alpha_base,      # Base alpha value (or 0 for schedule)
    alpha_scaling: float,  # Alpha scaling factor (for schedule)
    use_alpha: tl.constexpr,
    use_beta: tl.constexpr,
    beta: float,
    in_leg: tl.constexpr,
    dtype_enum: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ROWS_PER_CHK: tl.constexpr,
    ROWS_PER_VAR: tl.constexpr,
):
# fmt: on
    """C2V (check-to-variable) kernel with batch-organized grid-stride parallelism.

    This kernel only performs the C2V phase. V2C, parity, and selection phases run
    in separate kernel launches after this kernel completes, providing implicit barriers
    (kernel launch completion) between phases:
    - C2V completes -> V2C starts (V2C reads nu written by C2V)
    - V2C completes -> parity starts (parity reads hard_dec written by V2C)
    - parity completes -> selection starts (selection reads check_ok written by parity)

    Key differences from c2v_min_sum_kernel:
    1. Grid-stride pattern: Batch-organized (processes consecutive checks per batch lane)
       vs flat pattern (processes all (batch,check) pairs in flat order)
    2. Alpha schedule: Computes alpha per iteration internally vs pre-computed alpha passed in
    3. Better cache locality: When ROWS_PER_CHK > 1, consecutive checks per lane improve cache hits



    """
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)

    # Compute work dimensions (grid-stride pattern)
    # Each program processes ROWS_PER_CHK consecutive checks/vars
    work_dim = max(C, V)
    # Calculate which batch lane and starting check/var this program handles
    # Programs are organized as: program handles (batch_lane, starting_check/var)
    # Total work items per batch = (work_dim + ROWS_PER_CHK - 1) // ROWS_PER_CHK
    work_items_per_batch = (work_dim + ROWS_PER_CHK - 1) // ROWS_PER_CHK
    work_item_idx = pid % work_items_per_batch
    b = pid // work_items_per_batch

    if b >= B:
        return

    # Check if lane is active
    active_b = tl.load(active + b)
    if active_b == 0:
        return

    # Starting check/var index for this program
    start_check_var = work_item_idx * ROWS_PER_CHK

    # Compute alpha for this iteration (once per program, not per check/var)
    if use_alpha:
        if alpha_base == 0.0:
            exp_arg = -((iteration + 1) / alpha_scaling) * 0.6931471805599453
            alpha_iter = 1.0 - tl.exp(exp_arg)
        elif alpha_base < 0.0:
            alpha_iter = 1.0
        else:
            alpha_iter = alpha_base
    else:
        alpha_iter = 1.0

    # PHASE 1: C2V (Check-to-Variable) - Process ROWS_PER_CHK consecutive checks
    r = 0
    while r < ROWS_PER_CHK:
        i = start_check_var + r  # Process consecutive checks
        if i < C:
            row_start = tl.load(chk_ptr + i)
            row_end = tl.load(chk_ptr + i + 1)
            deg = row_end - row_start

            if deg > 0:
                deg_is_one = deg == 1
                neg_parity = tl.zeros((), dtype=tl.int32)

                if dtype_enum == 0:  # fp16
                    min1 = tl.full((), 1e30, dtype=tl.float16)
                    min2 = tl.full((), 1e30, dtype=tl.float16)
                elif dtype_enum == 2:  # fp64
                    min1 = tl.full((), 1e30, dtype=tl.float64)
                    min2 = tl.full((), 1e30, dtype=tl.float64)
                else:  # fp32
                    min1 = tl.full((), 1e30, dtype=tl.float32)
                    min2 = tl.full((), 1e30, dtype=tl.float32)
                cnt1 = tl.zeros((), dtype=tl.int32)
                argmin_e = tl.full((), -1, dtype=tl.int32)

                tile = row_start
                while tile < row_end:
                    offs = tile + tl.arange(0, BLOCK_SIZE)
                    m = offs < row_end
                    e = tl.load(chk_edges + offs, mask=m, other=0)
                    mu_e = tl.load(mu + b * E + e, mask=m, other=0.0)

                    tile_par = tl.sum((mu_e < 0.0) & m, axis=0).to(tl.int32) & 1
                    neg_parity = neg_parity ^ tile_par

                    a = tl.abs(mu_e)
                    a_mask = tl.where(m, a, 1e30)

                    pos1 = tl.argmax(-a_mask, axis=0)
                    min1_t = tl.max(-a_mask, axis=0) * (-1.0)
                    e1_t = tl.load(chk_edges + tile + pos1)
                    cnt1_t = tl.sum(a_mask == min1_t, axis=0).to(tl.int32)
                    a_mask2 = tl.where(tl.arange(0, BLOCK_SIZE) == pos1, 1e30, a_mask)
                    min2_t = tl.min(a_mask2, axis=0)

                    better1 = min1_t < min1
                    argmin_e = tl.where(better1, e1_t, argmin_e)
                    cnt1 = tl.where(better1, cnt1_t, tl.where(min1_t == min1, cnt1 + cnt1_t, cnt1))
                    min2 = tl.where(
                        better1,
                        tl.minimum(min1, min2_t),
                        tl.where(
                            min1_t == min1, tl.minimum(min2, min2_t), tl.minimum(min2, min1_t)
                        ),
                    )
                    min1 = tl.where(better1, min1_t, min1)

                    tile += BLOCK_SIZE

                syn = tl.load(syndrome + b * C + i).to(tl.int32) & 1
                par_bit = (neg_parity & 1) ^ syn
                sign_prod = tl.where(par_bit == 0, 1.0, -1.0)

                # PASS 2: Write outgoing messages
                tile = row_start
                while tile < row_end:
                    offs = tile + tl.arange(0, BLOCK_SIZE)
                    m = offs < row_end
                    e = tl.load(chk_edges + offs, mask=m, other=0)
                    mu_e = tl.load(mu + b * E + e, mask=m, other=0.0)
                    sgn_mu = tl.where(mu_e >= 0.0, 1.0, -1.0)

                    is_argmin = e == argmin_e
                    use_min2 = (cnt1 == 1) & is_argmin
                    out_mag = tl.where(use_min2, min2, min1)
                    out_mag = tl.where(deg_is_one, 0.0, out_mag)

                    if use_alpha:
                        out_mag = alpha_iter * out_mag
                    if use_beta:
                        out_mag = tl.maximum(out_mag - beta, 0.0)

                    nu_e = (sign_prod * sgn_mu) * out_mag
                    if dtype_enum == 0:  # fp16
                        tl.store(nu + b * E + e, nu_e.to(tl.float16), mask=m)
                    elif dtype_enum == 2:  # fp64
                        tl.store(nu + b * E + e, nu_e.to(tl.float64), mask=m)
                    else:  # fp32
                        tl.store(nu + b * E + e, nu_e, mask=m)

                    tile += BLOCK_SIZE

        r += 1

    # NOTE: V2C, parity, and selection phases removed from iteration-fused kernel to ensure
    # memory consistency. Different programs can execute in parallel on different SMs, so they
    # can be at different phases simultaneously, causing race conditions. Separate kernel launches
    # provide implicit barriers (kernel launch completion) between phases, ensuring correctness:
    # - C2V completes -> V2C starts (V2C reads nu written by C2V)
    # - V2C completes -> parity starts (parity reads hard_dec written by V2C)
    # - parity completes -> selection starts (selection reads check_ok written by parity)


# ============================================================================
# Persistent BP Kernel (Fully Fused)
# ============================================================================


# fmt: off
@triton.jit
def persistent_bp_iterations_kernel(
    # Message tensors
    mu,
    nu,                    # [B,E]
    # CSR graph data
    chk_ptr,
    chk_edges,             # CSR over checks -> edge IDs
    var_ptr,
    var_edges,             # CSR over vars -> edge IDs
    edge_var,              # [E] var index per edge
    # Beliefs and decisions
    lam,
    posterior_ratios,
    hard_dec,              # [B,V], [B,V], [B,V]
    # Inputs
    syndrome,              # [B,C]
    log_prior_ratios,
    gamma,                 # [1,V], [B,V]
    active,                # [B] uint8 mask
    # Solution tracking
    check_ok,              # [B,C] uint8 (temporary, can reuse)
    best_weights,
    best_errors,           # [B] dtype, [B,V] uint8
    valid_solutions,
    found_count,           # [B] uint8, [B] int32
    first_iter,
    iter_counter,          # [B] int32, [B] int32
    sought_sols,           # [B] int32
    wj,                    # [V] dtype (log prior ratios for weight)
    # Parameters
    B,
    C,
    V,
    E,
    max_iter: int,
    start_iter: int,
    stop_nconv: int,
    alpha_base,            # Base alpha value (or None/0 for schedule)
    alpha_scaling: float,  # Alpha scaling factor (for schedule)
    use_alpha: tl.constexpr,
    use_beta: tl.constexpr,
    beta: float,
    in_leg: tl.constexpr,
    force_hard: tl.constexpr,
    dtype_enum: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
# fmt: on
    """Fully fused persistent BP kernel that runs multiple iterations for one batch lane.

    This kernel processes one batch lane (b) and runs all iterations for that lane
    inside a single kernel launch. It fuses C2V, V2C, parity, select, and freeze
    operations to eliminate kernel launch overhead and host-device syncs.

    Each program_id handles one batch lane:
    - program_id(0) = batch lane 0
    - program_id(1) = batch lane 1
    - etc.

    Args:
        mu, nu: [B,E] message tensors
        chk_ptr, chk_edges: CSR check constraints
        var_ptr, var_edges: CSR variable constraints
        edge_var: [E] variable index per edge
        lam, posterior_ratios, hard_dec: [B,V] beliefs, marginals, decisions
        syndrome: [B,C] syndrome bits
        log_prior_ratios, gamma: [1,V], [B,V] priors and memory strengths
        active: [B] active lane mask
        check_ok: [B,C] temporary parity check results
        best_weights, best_errors: [B] dtype, [B,V] uint8 best solutions
        valid_solutions, found_count: [B] uint8, [B] int32 convergence tracking
        first_iter, iter_counter: [B] int32 iteration tracking
        sought_sols: [B] int32 sought solutions per lane
        wj: [V] dtype log prior ratios for weight computation
        B, C, V, E: batch size, checks, variables, edges
        max_iter: Maximum iterations to run
        start_iter: Starting iteration index (for alpha schedule)
        stop_nconv: Stop after N converged solutions
        alpha_base: Base alpha value (None/0 for schedule)
        alpha_scaling: Alpha scaling factor
        use_alpha, use_beta: Flags for min-sum parameters
        beta: Offset min-sum beta
        in_leg: Whether in relay leg
        force_hard: Whether to force hard_dec parity check
        dtype_enum: Compile-time dtype selector
        BLOCK_SIZE: Vectorization block size
    """
    b = tl.program_id(axis=0)
    if b >= B:
        return

    # Check if lane is active
    active_b = tl.load(active + b)
    if active_b == 0:
        return  # Lane is frozen, skip all iterations

    # Convergence tracking
    converged = False
    converged_iter = -1

    # Run iterations in a loop
    t_offset = 0
    while t_offset < max_iter:
        t = start_iter + t_offset

        # Skip work if already converged (but still need to update iter_counter)
        if not converged:
            # Compute alpha for this iteration (matches decoder logic)
            if use_alpha:
                if alpha_base == 0.0:
                    # Alpha schedule: 1.0 - 2^(-(t+1)/s)
                    # Use exp(-x * ln(2)) instead of 2^(-x) since Triton doesn't support **
                    exp_arg = -((t + 1) / alpha_scaling) * 0.6931471805599453  # ln(2) ≈ 0.693
                    alpha_iter = 1.0 - tl.exp(exp_arg)
                elif alpha_base < 0.0:
                    alpha_iter = 1.0
                else:
                    alpha_iter = alpha_base
            else:
                alpha_iter = 1.0

            # ====================================================================
            # PHASE 1: C2V (Check-to-Variable) - Process all checks for lane b
            # ====================================================================
            i = 0
            while i < C:
                row_start = tl.load(chk_ptr + i)
                row_end = tl.load(chk_ptr + i + 1)
                deg = row_end - row_start

                if deg > 0:
                    deg_is_one = deg == 1

                    # PASS 1: Compute parity and find min1/min2
                    neg_parity = tl.zeros((), dtype=tl.int32)
                    if dtype_enum == 0:  # fp16
                        min1 = tl.full((), 1e30, dtype=tl.float16)
                        min2 = tl.full((), 1e30, dtype=tl.float16)
                    elif dtype_enum == 2:  # fp64
                        min1 = tl.full((), 1e30, dtype=tl.float64)
                        min2 = tl.full((), 1e30, dtype=tl.float64)
                    else:  # fp32
                        min1 = tl.full((), 1e30, dtype=tl.float32)
                        min2 = tl.full((), 1e30, dtype=tl.float32)
                    cnt1 = tl.zeros((), dtype=tl.int32)
                    argmin_e = tl.full((), -1, dtype=tl.int32)

                    tile = row_start
                    while tile < row_end:
                        offs = tile + tl.arange(0, BLOCK_SIZE)
                        m = offs < row_end
                        e = tl.load(chk_edges + offs, mask=m, other=0)
                        mu_e = tl.load(mu + b * E + e, mask=m, other=0.0)

                        # Parity
                        tile_par = tl.sum((mu_e < 0.0) & m, axis=0).to(tl.int32) & 1
                        neg_parity = neg_parity ^ tile_par

                        # Min1/min2
                        a = tl.abs(mu_e)
                        a_mask = tl.where(m, a, 1e30)
                        pos1 = tl.argmax(-a_mask, axis=0)
                        min1_t = tl.max(-a_mask, axis=0) * (-1.0)
                        e1_t = tl.load(chk_edges + tile + pos1)
                        cnt1_t = tl.sum(a_mask == min1_t, axis=0).to(tl.int32)
                        a_mask2 = tl.where(tl.arange(0, BLOCK_SIZE) == pos1, 1e30, a_mask)
                        min2_t = tl.min(a_mask2, axis=0)

                        better1 = min1_t < min1
                        argmin_e = tl.where(better1, e1_t, argmin_e)
                        cnt1 = tl.where(
                            better1, cnt1_t, tl.where(min1_t == min1, cnt1 + cnt1_t, cnt1)
                        )
                        min2 = tl.where(
                            better1,
                            tl.minimum(min1, min2_t),
                            tl.where(
                                min1_t == min1, tl.minimum(min2, min2_t), tl.minimum(min2, min1_t)
                            ),
                        )
                        min1 = tl.where(better1, min1_t, min1)

                        tile += BLOCK_SIZE

                    # Syndrome integration
                    syn = tl.load(syndrome + b * C + i).to(tl.int32) & 1
                    par_bit = (neg_parity & 1) ^ syn
                    sign_prod = tl.where(par_bit == 0, 1.0, -1.0)

                    # PASS 2: Write outgoing messages
                    tile = row_start
                    while tile < row_end:
                        offs = tile + tl.arange(0, BLOCK_SIZE)
                        m = offs < row_end
                        e = tl.load(chk_edges + offs, mask=m, other=0)
                        mu_e = tl.load(mu + b * E + e, mask=m, other=0.0)
                        sgn_mu = tl.where(mu_e >= 0.0, 1.0, -1.0)

                        is_argmin = e == argmin_e
                        use_min2 = (cnt1 == 1) & is_argmin
                        out_mag = tl.where(use_min2, min2, min1)
                        out_mag = tl.where(deg_is_one, 0.0, out_mag)

                        if use_alpha:
                            out_mag = alpha_iter * out_mag
                        if use_beta:
                            out_mag = tl.maximum(out_mag - beta, 0.0)

                        nu_e = (sign_prod * sgn_mu) * out_mag
                        if dtype_enum == 0:  # fp16
                            tl.store(nu + b * E + e, nu_e.to(tl.float16), mask=m)
                        elif dtype_enum == 2:  # fp64
                            tl.store(nu + b * E + e, nu_e.to(tl.float64), mask=m)
                        else:  # fp32
                            tl.store(nu + b * E + e, nu_e, mask=m)

                        tile += BLOCK_SIZE

                i += 1

            # ====================================================================
            # PHASE 2: V2C (Variable-to-Check) with Gamma Mixing - Process all vars for lane b
            # ====================================================================
            j = 0
            while j < V:
                row_start = tl.load(var_ptr + j)
                row_end = tl.load(var_ptr + j + 1)

                # Load previous posterior and compute gamma mixing
                posterior_prev = tl.load(posterior_ratios + b * V + j)
                g_bj = tl.load(gamma + b * V + j)
                log_prior_j = tl.load(log_prior_ratios + j)
                lam_mixed = (1.0 - g_bj) * log_prior_j + g_bj * posterior_prev

                # PASS 1: Sum incoming messages
                if dtype_enum == 2:  # fp64
                    sum_nu = tl.zeros((), dtype=tl.float64)
                else:
                    sum_nu = tl.zeros((), dtype=tl.float32)

                tile = row_start
                while tile < row_end:
                    offs = tile + tl.arange(0, BLOCK_SIZE)
                    m = offs < row_end
                    e = tl.load(var_edges + offs, mask=m, other=0)
                    nu_e = tl.load(nu + b * E + e, mask=m, other=0.0)
                    sum_nu += tl.sum(tl.where(m, nu_e, 0.0), axis=0)
                    tile += BLOCK_SIZE

                # Update posterior and hard_dec
                posterior_curr = lam_mixed + sum_nu
                tl.store(posterior_ratios + b * V + j, posterior_curr)
                tl.store(hard_dec + b * V + j, (posterior_curr <= 0.0).to(tl.uint8))
                tl.store(lam + b * V + j, lam_mixed)

                # PASS 2: Write outgoing messages
                tile = row_start
                while tile < row_end:
                    offs = tile + tl.arange(0, BLOCK_SIZE)
                    m = offs < row_end
                    e = tl.load(var_edges + offs, mask=m, other=0)
                    nu_e = tl.load(nu + b * E + e, mask=m, other=0.0)
                    mu_e = lam_mixed + (sum_nu - nu_e)
                    tl.store(mu + b * E + e, mu_e, mask=m)
                    tile += BLOCK_SIZE

                j += 1

            # ====================================================================
            # PHASE 3: Parity Check - Check all checks for lane b
            # ====================================================================
            i = 0
            while i < C:
                row_start = tl.load(chk_ptr + i)
                row_end = tl.load(chk_ptr + i + 1)
                par = tl.zeros((), dtype=tl.int32)

                tile = row_start
                while tile < row_end:
                    offs = tile + tl.arange(0, BLOCK_SIZE)
                    m = offs < row_end
                    e = tl.load(chk_edges + offs, mask=m, other=0)
                    v = tl.load(edge_var + e, mask=m, other=0)
                    if force_hard:
                        bits = tl.load(hard_dec + b * V + v, mask=m, other=0).to(tl.int32) & 1
                    else:
                        posterior_v = tl.load(posterior_ratios + b * V + v, mask=m, other=0.0)
                        bits = (posterior_v <= 0.0).to(tl.int32) & 1
                    par = par ^ (tl.sum(bits & m.to(tl.int32), axis=0) & 1)
                    tile += BLOCK_SIZE

                syn = tl.load(syndrome + b * C + i).to(tl.int32) & 1
                ok = (par == syn).to(tl.uint8)
                tl.store(check_ok + b * C + i, ok)
                i += 1

            # ====================================================================
            # PHASE 4: Check and Select Solutions
            # ====================================================================
            # Check if all checks are satisfied
            all_ok = tl.full((), 1, dtype=tl.int32)
            i = 0
            while i < C:
                ok_val = tl.load(check_ok + b * C + i).to(tl.int32)
                all_ok = tl.minimum(all_ok, ok_val)
                i += 1

            if all_ok != 0:
                # Compute weight
                if dtype_enum == 0:  # fp16
                    weight = tl.zeros((), dtype=tl.float16)
                elif dtype_enum == 2:  # fp64
                    weight = tl.zeros((), dtype=tl.float64)
                else:  # fp32
                    weight = tl.zeros((), dtype=tl.float32)

                j = 0
                while j < V:
                    posterior_v = tl.load(posterior_ratios + b * V + j)
                    bits = (posterior_v <= 0.0).to(tl.uint8)
                    wj_val = tl.load(wj + j)
                    if dtype_enum == 0:  # fp16
                        weight += wj_val * bits.to(tl.float16)
                    elif dtype_enum == 2:  # fp64
                        weight += wj_val * bits.to(tl.float64)
                    else:  # fp32
                        weight += wj_val * bits.to(tl.float32)
                    j += 1

                # Update best solution
                prev_best = tl.load(best_weights + b)
                prev_valid = tl.load(valid_solutions + b)

                if dtype_enum == 0:  # fp16
                    inf_threshold = 60000.0
                elif dtype_enum == 2:  # fp64
                    inf_threshold = 1e20
                else:  # fp32
                    inf_threshold = 1e20
                is_inf = (prev_best >= inf_threshold).to(tl.int32)
                better = (weight < prev_best) | (is_inf == 1)

                if better:
                    tl.store(best_weights + b, weight)
                    j = 0
                    while j < V:
                        posterior_v = tl.load(posterior_ratios + b * V + j)
                        bits = (posterior_v <= 0.0).to(tl.uint8)
                        tl.store(best_errors + b * V + j, bits)
                        j += 1

                tl.store(valid_solutions + b, 1)
                if prev_valid == 0:
                    old = tl.load(found_count + b)
                    tl.store(found_count + b, old + 1)

            # ====================================================================
            # PHASE 5: Freeze and Check Convergence
            # ====================================================================
            fc = tl.load(found_count + b)
            if in_leg:
                ss = tl.load(sought_sols + b)
                stop_criterion = tl.minimum(ss, stop_nconv)
            else:
                stop_criterion = stop_nconv

            done_now = fc >= stop_criterion
            if done_now:
                fi = tl.load(first_iter + b)
                if fi < 0:
                    if not in_leg:
                        tl.store(first_iter + b, t + 1)
                    # Copy best_errors to hard_dec
                    j = 0
                    while j < V:
                        be = tl.load(best_errors + b * V + j)
                        tl.store(hard_dec + b * V + j, be)
                        j += 1
                tl.store(active + b, 0)
                # Reset gamma
                j = 0
                while j < V:
                    tl.store(gamma + b * V + j, 0.0)
                    j += 1

                converged = True
                converged_iter = t_offset
                # Set iter_counter to convergence iteration (1-indexed) when convergence happens
                # This prevents iter_counter from being updated to max_iter in subsequent loop iterations
                tl.store(iter_counter + b, t + 1)

        # Update iteration counter only if not converged (to avoid overwriting convergence iteration)
        if not converged:
            tl.store(iter_counter + b, t + 1)

        t_offset += 1

    # Store convergence iteration if converged (backup in case first_iter wasn't set in loop)
    # Note: converged_iter is 0-indexed (t_offset), but first_iter should be 1-indexed to match Rust
    # Rust's current_iteration is incremented after each iteration, so it's 1-indexed
    if converged_iter >= 0 and not in_leg:
        fi = tl.load(first_iter + b)
        if fi < 0:
            # Convert to 1-indexed: start_iter + converged_iter + 1
            # This matches the t + 1 assignment in the loop above
            tl.store(first_iter + b, start_iter + converged_iter + 1)

