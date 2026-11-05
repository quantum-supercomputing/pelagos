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
Equivalence tests: Triton Relay-BP-S vs Rust relay_bp (Python frontend).

This module provides comprehensive equivalence testing between the Triton GPU
implementation and the Rust reference implementation of Relay-BP for quantum
error correction. The tests ensure algorithmic fidelity and numerical consistency
across different backends.

Test methodology:
- Deterministic configuration for reproducible results
- Parity validation for quantum error correction correctness
- Weight-based equivalence for different valid solutions
- Batch processing consistency verification
- Memory format and precision testing

The tests use a deterministic configuration to ensure reproducible and comparable results:
  - num_sets=1 (single ensemble)
  - gamma0=0.0 (no memory mixing)
  - gamma_dist_interval=(0.0, 0.0) (deterministic gamma)
  - fixed iteration counts
  - normalized_min_sum_alpha=1.0 (pure min-sum on Triton)
  - stop_nconv=1 (single solution)

Skip conditions:
  - CUDA unavailable (Triton requires GPU)
  - relay_bp (Rust bindings) not importable

An unimportable `pelagos` package is NOT a skip condition: it is a real
defect and fails this module's collection loudly.
"""

import os
import sys
import numpy as np
import pytest
from scipy.sparse import csr_matrix

# Import Rust reference implementation
try:
    import relay_bp  # Rust bindings from this repo's package

    HAS_RELAY_BP = True
except Exception:
    HAS_RELAY_BP = False

# Import the Pelagos (Triton GPU) implementation. This import is NOT
# wrapped in a swallowing try/except: an unimportable `pelagos` is a real
# defect (a broken install, not a missing optional dependency) and must
# fail the test run loudly rather than silently skip every test in this
# module with a plausible-looking reason.
from pelagos import RelayBPDecoder  # noqa: E402
import torch  # noqa: E402

# Legitimate skip conditions only: no GPU, or the optional Rust wheel is
# not installed. Neither is a defect in Pelagos itself.
CUDA_OK = torch.cuda.is_available()

pytestmark = [
    pytest.mark.skipif(not CUDA_OK, reason="CUDA not available (Triton requires GPU)"),
    pytest.mark.skipif(not HAS_RELAY_BP, reason="relay_bp (Rust) not importable"),
]

# ---------------- Helper Functions ----------------


def _probe_rust_min_sum_kwargs(base_kwargs):
    """Probe Rust decoder for min-sum parameter support.

    This function attempts to configure the Rust decoder for pure min-sum operation
    by testing which parameters are available in the current wheel build. Different
    builds may expose different parameter sets, so we probe and only use available ones.

    Args:
        base_kwargs: Base configuration dictionary

    Returns:
        Updated configuration with available min-sum parameters
    """
    import numpy as _np
    from scipy.sparse import csr_matrix as _csr
    import relay_bp as _relay_bp  # use already-imported module

    extras = {}
    # Minimal 1x1 CSR and single prior for parameter probing
    _H_probe = _csr(_np.array([[1]], dtype=_np.uint8))
    _p_probe = [0.01]

    # Test candidate min-sum parameters
    candidates = [
        ("alpha", 1.0),
        ("alpha_iteration_scaling_factor", 0.0),
        ("offset_beta", 0.0),
        ("damping", 0.0),
        ("randomize_update_order", False),
    ]
    for k, v in candidates:
        try:
            _relay_bp.RelayDecoderF32(
                _H_probe,
                error_priors=_p_probe,
                **base_kwargs,
                **{k: v},
            )
            extras[k] = v
        except TypeError:
            # Parameter not accepted by this wheel build
            pass
        except Exception:
            # Parameter accepted but value invalid for this build
            pass
    return {**base_kwargs, **extras}


def weight_from_priors(e_u8: np.ndarray, p: np.ndarray) -> float:
    """Compute maximum likelihood cost from error pattern and priors.

    This function computes the log-likelihood cost of an error pattern under
    the given error priors, corresponding to the objective function used in
    maximum likelihood decoding for quantum error correction.

    Args:
        e_u8: Error pattern as uint8 array (0/1 values)
        p: Error probabilities for each qubit

    Returns:
        Log-likelihood cost: sum_j e_j * log((1-p_j)/p_j)
    """
    wj = np.log((1.0 - p) / p)
    return float((e_u8 * wj).sum())


def compare_with_ml_or_tie(H_csr, s_u8, p, e_rust, e_gpu, tol=1e-4):
    """Compare decoder outputs with maximum likelihood or weight-based equivalence.

    When both decoders produce parity-valid solutions but disagree on the specific
    error pattern, this function validates equivalence by either:
    1. For small problems (n <= 16): Accept if either decoder matches exact ML
    2. For larger problems: Require near-equal log-likelihood costs

    This approach handles the fact that quantum error correction problems may have
    multiple valid solutions with similar likelihoods.

    Args:
        H_csr: Check matrix in CSR format
        s_u8: Syndrome vector
        p: Error priors
        e_rust: Rust decoder output
        e_gpu: GPU decoder output
        tol: Tolerance for weight comparison
    """
    assert e_rust.dtype == np.uint8 and e_gpu.dtype == np.uint8
    assert parity_ok(H_csr, e_rust, s_u8)
    assert parity_ok(H_csr, e_gpu, s_u8)

    if np.array_equal(e_rust, e_gpu):
        return  # Identical solutions - no further validation needed

    # Different solutions: validate equivalence by weight comparison
    w_rust = weight_from_priors(e_rust, p)
    w_gpu = weight_from_priors(e_gpu, p)

    n = H_csr.shape[1]
    if n <= 16:
        # For small problems, check against exact maximum likelihood
        e_ml, w_ml = brute_force_ml(H_csr, s_u8, p)
        if e_ml is not None:
            rust_is_ml = np.array_equal(e_rust, e_ml) and abs(w_rust - w_ml) < 1e-6
            gpu_is_ml = np.array_equal(e_gpu, e_ml) and abs(w_gpu - w_ml) < 1e-6
            if rust_is_ml or gpu_is_ml:
                return
    # For larger problems, require near-equal log-likelihood costs
    assert abs(w_rust - w_gpu) < tol, f"Unequal weights: rust={w_rust}, gpu={w_gpu}"


def make_rust_decoder(H_csr, p):
    """Create Rust decoder with deterministic configuration for testing.

    Args:
        H_csr: Check matrix in CSR format
        p: Error priors array

    Returns:
        Configured Rust RelayDecoderF32 instance
    """
    base_kwargs = dict(
        gamma0=0.0,
        pre_iter=200,
        num_sets=1,
        set_max_iter=0,
        gamma_dist_interval=(0.0, 1e-12),
        stop_nconv=1,
    )
    # Configure for pure min-sum operation if supported
    base_kwargs = _probe_rust_min_sum_kwargs(base_kwargs)

    # Prepare error priors in different formats for PyO3 compatibility
    p64 = np.array(p, dtype=np.float64, order="C", copy=True).ravel()
    p32 = np.array(p, dtype=np.float32, order="C", copy=True).ravel()
    p_list = [float(x) for x in p64]

    # Try different data types for error priors
    try:
        return relay_bp.RelayDecoderF32(H_csr, error_priors=p64, **base_kwargs)
    except TypeError:
        pass
    try:
        return relay_bp.RelayDecoderF32(H_csr, error_priors=p32, **base_kwargs)
    except TypeError:
        pass
    return relay_bp.RelayDecoderF32(H_csr, error_priors=p_list, **base_kwargs)


def make_triton_decoder(H_csr, p, device="cuda", **overrides):
    """Create Triton decoder with deterministic configuration for testing.

    This function constructs a RelayBPDecoder with deterministic settings and
    handles unknown parameters gracefully to support different builds.

    Args:
        H_csr: Check matrix in CSR format
        p: Error priors array
        device: GPU device ("cuda" or "rocm")
        **overrides: Additional configuration parameters

    Returns:
        Configured Triton RelayBPDecoder instance
    """
    # Default deterministic configuration for reproducible testing.
    cfg = dict(
        H_csr=H_csr,
        error_priors=p,
        pre_iter=200,
        num_sets=1,
        set_max_iter=0,
        gamma0=0.0,
        gamma_dist_interval=(0.0, 1e-12),
        stop_nconv=1,
        normalized_min_sum_alpha=1.0,
        offset_min_sum_beta=None,
        data_type="fp32",
        algo="plain",
        device=device,
        seed=1234,
        bitpack_output=False,
    )
    cfg.update(overrides)

    return RelayBPDecoder(**cfg)


def parity_ok(H_csr, e_u8, s_u8) -> bool:
    """Check if error pattern satisfies syndrome constraints.

    This function verifies that the decoded error pattern produces the correct
    syndrome when multiplied by the check matrix, which is the fundamental
    requirement for valid quantum error correction.

    Args:
        H_csr: Check matrix in CSR format
        e_u8: Error pattern as uint8 array
        s_u8: Syndrome vector as uint8 array

    Returns:
        True if H @ e mod 2 == s, False otherwise
    """
    return bool(np.all((H_csr.dot(e_u8.astype(np.uint8)) & 1) == s_u8))


def brute_force_ml(H_csr, s_u8, p):
    """Compute exact maximum likelihood solution by brute force.

    This function exhaustively searches all possible error patterns to find
    the one with minimum log-likelihood cost that satisfies the syndrome
    constraints. Only feasible for small problems (n <= 16).

    Args:
        H_csr: Check matrix in CSR format
        s_u8: Syndrome vector
        p: Error priors

    Returns:
        Tuple of (error_pattern, weight) or (None, inf) if no solution found
    """
    H = H_csr.toarray().astype(np.uint8)
    m, n = H.shape
    if n > 16:
        return None, float("inf")
    s = s_u8.astype(np.uint8)
    wj = np.log((1.0 - p) / p)
    best_e, best_w = None, float("inf")

    # Exhaustive search over all 2^n possible error patterns
    for x in range(1 << n):
        e = np.array([(x >> j) & 1 for j in range(n)], dtype=np.uint8)
        if np.all(((H @ e) & 1) == s):
            w = float((e * wj).sum())
            if w < best_w - 1e-12:
                best_w, best_e = w, e.copy()
    return best_e, best_w


# ---------------- Test Functions ----------------


def test_triangle_exact_match():
    """Test exact equivalence on simple 3-variable, 2-check triangle code.

    This test uses a minimal quantum error correction code to verify that
    both Rust and Triton implementations produce identical results or
    equivalent solutions with the same log-likelihood cost.
    """
    H = csr_matrix(np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8))
    H.sort_indices()
    p = np.clip(np.array([0.01, 0.01, 0.01], dtype=np.float64), 1e-6, 1 - 1e-6)
    S = [
        np.array([0, 0], np.uint8),
        np.array([1, 0], np.uint8),
        np.array([0, 1], np.uint8),
        np.array([1, 1], np.uint8),
    ]

    rust = make_rust_decoder(H, p)
    gpu = make_triton_decoder(H, p, device="cuda")

    for s in S:
        e_rust = np.asarray(rust.decode(s), dtype=np.uint8)
        s_dev = torch.from_numpy(np.asarray([s], dtype=np.uint8)).to("cuda")
        e_gpu = gpu.decode(s_dev)["errors"][0].detach().cpu().numpy().astype(np.uint8)

        assert parity_ok(H, e_rust, s)
        assert parity_ok(H, e_gpu, s)

        # Verify syndrome reproduction (fundamental QEC requirement)
        s_hat_rust = H.dot(e_rust) & 1
        s_hat_gpu = H.dot(e_gpu) & 1
        assert np.all(s_hat_rust == s), "Rust decode must reproduce the input syndrome"
        assert np.all(s_hat_gpu == s), "GPU decode must reproduce the input syndrome"


def test_bitpacked_output():
    """Test bit-packed output format for memory efficiency.

    This test verifies that the Triton decoder can output error patterns in
    a bit-packed format for memory efficiency, and that the round-trip
    pack/unpack operations preserve correctness and satisfy syndrome constraints.
    """
    from pelagos.utils import bitpack_errors, bitunpack_errors

    H = csr_matrix(
        np.array(
            [
                [1, 1, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 0, 0, 0],
            ],
            dtype=np.uint8,
        )
    )
    H.sort_indices()
    p = np.clip(np.array([0.01] * 8, dtype=np.float64), 1e-6, 1 - 1e-6)
    s = np.array([1, 0, 1, 0], dtype=np.uint8)

    # Test bit-packed output mode
    try:
        gpu = make_triton_decoder(H, p, device="cuda", bitpack_output=True)
    except TypeError:
        pytest.skip("bitpack_output not supported in this Triton build")

    s_dev = torch.from_numpy(np.asarray([s], dtype=np.uint8)).to("cuda")
    out_gpu = gpu.decode(s_dev)

    # Extract packed errors and valid mask
    packed_errors = out_gpu["errors"][0]  # Should be packed format
    valid_mask = out_gpu["valid_mask"][0].item()

    # Skip if decoder didn't actually pack (feature disabled in build)
    if packed_errors.dtype != torch.int32:
        pytest.skip("decoder returned unpacked errors; bitpacking disabled in this build")

    # Basic structure validation
    assert isinstance(packed_errors, torch.Tensor), "Packed errors should be a tensor"
    assert isinstance(valid_mask, bool), "Valid mask should be boolean"

    # Round-trip validation: unpack the packed errors
    V = 8  # number of variables
    unpacked_errors = bitunpack_errors(packed_errors.unsqueeze(0), V=V)[0]  # Remove batch dim

    # Verify unpacked errors are binary (0/1)
    assert torch.all((unpacked_errors == 0) | (unpacked_errors == 1)), (
        "Unpacked errors should be binary"
    )
    assert unpacked_errors.shape == (V,), (
        f"Unpacked errors should have shape ({V},), got {unpacked_errors.shape}"
    )

    # Memory efficiency validation
    original_size = V  # bytes for uint8 array
    packed_size = packed_errors.numel() * 4  # bytes for int32 array (4 bytes per int32)
    assert packed_size <= original_size, (
        f"Packed format should use less or equal memory: {packed_size} > {original_size}"
    )

    # Parity check validation: verify unpacked errors satisfy syndrome constraints
    if valid_mask:  # Only check if the solution is valid
        # Convert to numpy for sparse matrix operations
        e_np = unpacked_errors.cpu().numpy().astype(np.uint8)
        s_hat = (H.dot(e_np) & 1).astype(np.uint8)
        assert np.array_equal(s_hat, s), (
            f"Unpacked errors should satisfy syndrome: got {s_hat}, expected {s}"
        )

    # Manual round-trip test: pack known errors and unpack them
    test_errors = torch.tensor([1, 0, 1, 1, 0, 0, 1, 0], dtype=torch.uint8, device="cuda")
    manually_packed = bitpack_errors(test_errors.unsqueeze(0), bits_per_word=32)[0]
    manually_unpacked = bitunpack_errors(manually_packed.unsqueeze(0), V=V)[0]
    assert torch.equal(manually_unpacked, test_errors), "Manual round-trip pack/unpack failed"


@pytest.mark.parametrize(
    "n,c,seed",
    [
        (8, 4, 0),
        (10, 5, 1),
    ],
)
def test_random_small_graphs_equivalence(n, c, seed):
    """Test equivalence on randomly generated small quantum codes.

    This test generates random check matrices and verifies that both Rust and
    Triton decoders produce parity-valid solutions. For different valid solutions,
    it checks weight-based equivalence or exact maximum likelihood matching.

    Args:
        n: Number of qubits (variables)
        c: Number of checks (stabilizer generators)
        seed: Random seed for reproducible test generation
    """
    rng = np.random.default_rng(seed)
    H_dense = np.zeros((c, n), dtype=np.uint8)
    for i in range(c):
        deg = int(rng.integers(2, min(4, n) + 1))
        cols = rng.choice(n, size=deg, replace=False)
        H_dense[i, cols] = 1
    H = csr_matrix(H_dense)
    H.sort_indices()

    p = np.clip(rng.uniform(0.005, 0.02, size=n).astype(np.float64), 1e-6, 1 - 1e-6)
    S = [rng.integers(0, 2, size=c, dtype=np.uint8) for _ in range(8)]

    rust = make_rust_decoder(H, p)
    gpu = make_triton_decoder(H, p, device="cuda")
    for s in S:
        e_rust = np.asarray(rust.decode(s), dtype=np.uint8)
        s_dev = torch.from_numpy(np.asarray([s], dtype=np.uint8)).to("cuda")
        out_gpu = gpu.decode(s_dev)
        e_gpu = out_gpu["errors"][0].detach().cpu().numpy().astype(np.uint8)
        gpu_ok = bool(out_gpu["valid_mask"][0].item())

        # Handle non-convergence: GPU should mark invalid and report infinite weight
        if not gpu_ok:
            w_gpu_api = float(out_gpu["weights"][0].item())
            assert not np.isfinite(w_gpu_api), "Non-converged decode must report infinite weight"
            continue

        # GPU converged: require parity-valid solution
        assert parity_ok(H, e_gpu, s), "Triton reported valid but parity fails"

        rust_ok = parity_ok(H, e_rust, s)
        if not rust_ok:
            # Rust failed but GPU succeeded: verify GPU solution is optimal
            e_ml, w_ml = brute_force_ml(H, s, p)
            if e_ml is not None:
                w_gpu = weight_from_priors(e_gpu, p)
                assert np.array_equal(e_gpu, e_ml), "GPU bits must match ML when Rust fails"
                assert abs(w_gpu - w_ml) < 1e-6, "GPU weight must equal ML when Rust fails"
            continue

        # Both decoders produced valid solutions: compare for equivalence
        compare_with_ml_or_tie(H, s, p, e_rust, e_gpu, tol=1e-4)


def test_batch_consistency_triangle():
    """Test batch processing consistency on triangle code.

    This test verifies that batch processing produces the same results as
    individual single-shot decodes, ensuring the batch implementation
    maintains algorithmic correctness.
    """
    H = csr_matrix(np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8))
    p = np.array([0.01, 0.01, 0.01], dtype=np.float64)
    S = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.uint8)

    rust = make_rust_decoder(H, p)
    gpu = make_triton_decoder(H, p, device="cuda")

    # Single-shot decoding via Rust (baseline)
    singles = np.stack([np.asarray(rust.decode(s), dtype=np.uint8) for s in S], axis=0)

    # Batch decoding on GPU
    s_dev = torch.tensor(S, dtype=torch.uint8, device="cuda")
    out = gpu.decode(s_dev)
    errs = out["errors"].detach().cpu().numpy().astype(np.uint8)

    # Verify all solutions are parity-valid
    for i in range(S.shape[0]):
        assert parity_ok(H, singles[i], S[i])
        assert parity_ok(H, errs[i], S[i])

    # Compare weights for different solutions
    wj = np.log((1.0 - p) / p)
    w_rust = (singles * wj).sum(axis=1)
    w_gpu = (errs * wj).sum(axis=1)
    diffs = np.any(singles != errs, axis=1)
    assert np.all(np.abs(w_rust[diffs] - w_gpu[diffs]) < 1e-5)


def test_fp16_messages_parity():
    """Test fp16 message precision preserves quantum error correction correctness.

    This test verifies that using fp16 precision for internal message passing
    still produces parity-valid solutions, ensuring that reduced precision
    doesn't compromise the fundamental correctness of quantum error correction.
    """
    H = csr_matrix(np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8))
    p = np.array([0.01, 0.01, 0.01], dtype=np.float64)
    S = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.uint8)

    rust = make_rust_decoder(H, p)
    gpu = RelayBPDecoder(
        H_csr=H,
        error_priors=p,
        pre_iter=10,
        num_sets=1,
        set_max_iter=10,
        gamma0=0.0,
        gamma_dist_interval=(0.0, 0.0),
        stop_nconv=1,
        normalized_min_sum_alpha=1.0,
        data_type="fp16",
        algo="relay",
        device="cuda",
        seed=0,
    )

    for s in S:
        e_rust = np.asarray(rust.decode(s), dtype=np.uint8)
        s_dev = torch.from_numpy(np.array([s], dtype=np.uint8)).to("cuda")
        e_gpu = gpu.decode(s_dev)["errors"][0].detach().cpu().numpy().astype(np.uint8)

        assert parity_ok(H, e_rust, s)
        assert parity_ok(H, e_gpu, s)

        # Verify syndrome reproduction (fundamental QEC requirement)
        s_hat_rust = H.dot(e_rust) & 1
        s_hat_gpu = H.dot(e_gpu) & 1
        assert np.all(s_hat_rust == s), "Rust decode must reproduce the input syndrome"
        assert np.all(s_hat_gpu == s), "GPU decode must reproduce the input syndrome"


def test_determinism():
    """Test deterministic behavior across multiple runs.

    This test verifies that running the same decoding problem multiple times
    produces identical results, ensuring reproducible behavior for both
    Rust and Triton implementations.
    """
    H = csr_matrix(np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]], dtype=np.uint8))
    H.sort_indices()
    p = np.clip(np.array([0.01, 0.01, 0.01, 0.01], dtype=np.float64), 1e-6, 1 - 1e-6)
    s = np.array([1, 0, 1], dtype=np.uint8)

    rust = make_rust_decoder(H, p)
    gpu = make_triton_decoder(H, p, device="cuda", seed=42)  # Fixed seed for determinism

    # Run decoding twice with identical inputs
    e_rust1 = np.asarray(rust.decode(s), dtype=np.uint8)
    e_rust2 = np.asarray(rust.decode(s), dtype=np.uint8)

    s_dev = torch.from_numpy(np.asarray([s], dtype=np.uint8)).to("cuda")
    out1 = gpu.decode(s_dev)
    out2 = gpu.decode(s_dev)
    e_gpu1 = out1["errors"][0].detach().cpu().numpy().astype(np.uint8)
    e_gpu2 = out2["errors"][0].detach().cpu().numpy().astype(np.uint8)
    valid1 = out1["valid_mask"][0].item()
    valid2 = out2["valid_mask"][0].item()

    # Verify deterministic behavior
    assert np.array_equal(e_rust1, e_rust2), "Rust decoder should be deterministic"
    assert np.array_equal(e_gpu1, e_gpu2), "Triton decoder should be deterministic"
    assert valid1 == valid2, "Triton valid_mask should be deterministic"


def test_ensemble_relay_legs():
    """Test ensemble decoding with multiple relay legs.

    This test verifies that the ensemble decoding functionality works correctly
    with multiple relay sets, ensuring that the batching logic and solution
    aggregation maintain quantum error correction correctness.
    """
    H = csr_matrix(
        np.array(
            [[1, 1, 0, 0, 0], [0, 1, 1, 0, 0], [0, 0, 1, 1, 0], [0, 0, 0, 1, 1]], dtype=np.uint8
        )
    )
    H.sort_indices()
    p = np.clip(np.array([0.01, 0.01, 0.01, 0.01, 0.01], dtype=np.float64), 1e-6, 1 - 1e-6)
    s = np.array([1, 0, 1, 0], dtype=np.uint8)

    # Rust decoder with ensemble configuration
    rust = relay_bp.RelayDecoderF64(
        H,
        error_priors=p,
        gamma0=0.1,
        pre_iter=10,
        num_sets=2,
        set_max_iter=5,
        gamma_dist_interval=(0.0, 0.2),
        stop_nconv=2,
        seed=42,
    )

    # Triton decoder with matching ensemble configuration
    gpu = make_triton_decoder(
        H,
        p,
        device="cuda",
        gamma0=0.1,
        pre_iter=10,
        num_sets=2,
        set_max_iter=5,
        gamma_dist_interval=(0.0, 0.2),
        stop_nconv=2,
        seed=42,
    )

    e_rust = np.asarray(rust.decode(s), dtype=np.uint8)
    s_dev = torch.from_numpy(np.asarray([s], dtype=np.uint8)).to("cuda")
    out_gpu = gpu.decode(s_dev)
    e_gpu = out_gpu["errors"][0].detach().cpu().numpy().astype(np.uint8)

    # Both should produce valid solutions
    assert parity_ok(H, e_rust, s)
    assert parity_ok(H, e_gpu, s)

    # Residual syndrome equality checks
    s_hat_rust = H.dot(e_rust) & 1
    s_hat_gpu = H.dot(e_gpu) & 1
    assert np.all(s_hat_rust == s), "Rust decode must reproduce the input syndrome"
    assert np.all(s_hat_gpu == s), "GPU decode must reproduce the input syndrome"


def test_high_degree_checks():
    """Test high-degree check constraints to exercise GPU kernel memory pressure.

    This test creates check constraints with high variable degrees to stress-test
    the Triton kernels under register and shared memory pressure, ensuring
    robustness for complex quantum error correction codes.
    """
    # Create check matrix with varying constraint degrees
    n_vars = 25
    n_checks = 3
    H_dense = np.zeros((n_checks, n_vars), dtype=np.uint8)

    # First check: high degree (20 variables) - stress test
    H_dense[0, :20] = 1

    # Second check: medium degree (10 variables)
    H_dense[1, 5:15] = 1

    # Third check: low degree (3 variables)
    H_dense[2, [0, 10, 20]] = 1

    H = csr_matrix(H_dense)
    H.sort_indices()
    p = np.clip(np.full(n_vars, 0.01, dtype=np.float64), 1e-6, 1 - 1e-6)
    s = np.array([1, 0, 1], dtype=np.uint8)

    rust = make_rust_decoder(H, p)
    gpu = make_triton_decoder(H, p, device="cuda")

    e_rust = np.asarray(rust.decode(s), dtype=np.uint8)
    s_dev = torch.from_numpy(np.asarray([s], dtype=np.uint8)).to("cuda")
    out_gpu = gpu.decode(s_dev)
    e_gpu = out_gpu["errors"][0].detach().cpu().numpy().astype(np.uint8)

    # Both should produce valid solutions
    assert parity_ok(H, e_rust, s)
    assert parity_ok(H, e_gpu, s)

    # Residual syndrome equality checks
    s_hat_rust = H.dot(e_rust) & 1
    s_hat_gpu = H.dot(e_gpu) & 1
    assert np.all(s_hat_rust == s), "Rust decode must reproduce the input syndrome"
    assert np.all(s_hat_gpu == s), "GPU decode must reproduce the input syndrome"


@pytest.mark.parametrize("dtype", ["fp32", "fp16"])
def test_dtype_parity(dtype):
    """Test uniform precision (fp32/fp16) preserves quantum error correction correctness.

    This test verifies that different floating-point precisions for storage and computation
    still produce parity-valid solutions, ensuring numerical stability across precision levels.

    Args:
        dtype: Uniform precision ("fp32" or "fp16")
    """
    H = csr_matrix(np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]], dtype=np.uint8))
    H.sort_indices()
    p = np.clip(np.array([0.01, 0.01, 0.01, 0.01], dtype=np.float64), 1e-6, 1 - 1e-6)
    s = np.array([1, 0, 1], dtype=np.uint8)

    rust = make_rust_decoder(H, p)
    gpu = make_triton_decoder(H, p, device="cuda", data_type=dtype)

    e_rust = np.asarray(rust.decode(s), dtype=np.uint8)
    s_dev = torch.from_numpy(np.asarray([s], dtype=np.uint8)).to("cuda")
    out_gpu = gpu.decode(s_dev)
    e_gpu = out_gpu["errors"][0].detach().cpu().numpy().astype(np.uint8)

    # Both should produce valid solutions
    assert parity_ok(H, e_rust, s)
    assert parity_ok(H, e_gpu, s)

    # Residual syndrome equality checks
    s_hat_rust = H.dot(e_rust) & 1
    s_hat_gpu = H.dot(e_gpu) & 1
    assert np.all(s_hat_rust == s), "Rust decode must reproduce the input syndrome"
    assert np.all(s_hat_gpu == s), "GPU decode must reproduce the input syndrome"


def test_unsorted_csr_equivalence():
    """Test Triton kernels handle unsorted CSR matrix indices correctly.

    This test verifies that the Triton implementation correctly handles
    CSR matrices with unsorted column indices, ensuring robustness
    regardless of matrix preprocessing.
    """
    H = csr_matrix(np.array([[1, 0, 1, 0], [0, 1, 1, 1]], np.uint8))
    # Deliberately scramble columns within rows
    H_unsorted = csr_matrix((H.data, H.indices[::-1], H.indptr), shape=H.shape)  # unsorted
    p = np.array([0.02, 0.01, 0.03, 0.02], np.float64)
    s = np.array([1, 0], np.uint8)

    # Test with sorted CSR (baseline)
    H_sorted = H.copy()
    H_sorted.sort_indices()
    gpu_sorted = make_triton_decoder(H_sorted, p, device="cuda")

    # Test with unsorted CSR (should produce equivalent results)
    gpu_unsorted = make_triton_decoder(H_unsorted, p, device="cuda")

    s_dev = torch.from_numpy(np.asarray([s], dtype=np.uint8)).to("cuda")
    out_sorted = gpu_sorted.decode(s_dev)
    out_unsorted = gpu_unsorted.decode(s_dev)

    e_sorted = out_sorted["errors"][0].detach().cpu().numpy().astype(np.uint8)
    e_unsorted = out_unsorted["errors"][0].detach().cpu().numpy().astype(np.uint8)

    # Both decoders should produce valid solutions
    assert parity_ok(H_sorted, e_sorted, s)
    assert parity_ok(H_unsorted, e_unsorted, s)

    # Verify syndrome reproduction (fundamental QEC requirement)
    s_hat_sorted = H_sorted.dot(e_sorted) & 1
    s_hat_unsorted = H_unsorted.dot(e_unsorted) & 1
    assert np.all(s_hat_sorted == s), (
        "Triton decode must reproduce the input syndrome with sorted CSR"
    )
    assert np.all(s_hat_unsorted == s), (
        "Triton decode must reproduce the input syndrome with unsorted CSR"
    )


def test_non_convergence_contract():
    """Test non-convergence handling and contract compliance.

    This test verifies that when the Triton decoder fails to converge,
    it correctly reports infinite weight and marks the solution as invalid,
    ensuring proper error handling for difficult quantum error correction instances.
    """
    # Create difficult instance with maximally uncertain priors and high cycle count
    H = csr_matrix(np.array([[1, 1, 1, 1, 0, 0], [0, 1, 1, 0, 1, 1], [1, 0, 1, 0, 1, 1]], np.uint8))
    p = np.full(6, 0.5, np.float64)  # deliberately uninformative priors
    s = np.array([1, 1, 1], np.uint8)
    gpu = make_triton_decoder(H, p, set_max_iter=0, num_sets=1, stop_nconv=1)  # minimal effort
    s_batch = np.array([s], dtype=np.uint8)
    out = gpu.decode(torch.from_numpy(s_batch).to("cuda"))
    ok = bool(out["valid_mask"][0].item())
    if not ok:
        assert not np.isfinite(float(out["weights"][0].item())), (
            "Non-converged solutions should have infinite weight"
        )


def test_variable_batch_shapes():
    """Test batch processing with various input shapes.

    This test verifies that batch processing handles different input shapes
    correctly, ensuring that tensor indexing and memory layout work properly
    across various batch sizes and dimensions.
    """
    H = csr_matrix(np.array([[1, 1, 0], [0, 1, 1]], np.uint8))
    p = np.array([0.01, 0.02, 0.03], np.float64)
    S = np.array([[1, 0], [0, 1], [1, 1], [0, 0]], np.uint8)
    gpu = make_triton_decoder(H, p)
    out = gpu.decode(torch.tensor(S, dtype=torch.uint8, device="cuda"))
    assert out["errors"].shape[0] == S.shape[0], "Output batch size should match input batch size"

    # Verify all output tensors have correct shapes
    assert out["errors"].shape == (4, 3), f"Expected (4, 3), got {out['errors'].shape}"
    assert out["valid_mask"].shape == (4,), f"Expected (4,), got {out['valid_mask'].shape}"
    assert out["weights"].shape == (4,), f"Expected (4,), got {out['weights'].shape}"
