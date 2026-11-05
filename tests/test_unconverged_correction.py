# (C) Copyright IBM 2026
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Correction returned for a shot that found no solution in any leg.

With ``unconverged_correction="first_leg"`` (the default) the decoder returns the hard
decision at the end of the first leg that ran: the pre-iter leg when it runs, otherwise
relay leg 1. This is the Rust reference behaviour (``relay.rs`` keeps the pre-iter
outcome and replaces it only with a converged leg). ``"last_leg"`` returns the hard
decision at the end of the last leg that ran.

The reference for "first_leg" is the same decoder with ``num_sets=0``: the pre-iter leg
uses a fixed gamma0 and no random draw, so its hard decision is the same whether or not
relay legs follow. All tests need a GPU; the Rust comparison also needs the ``rust`` extra.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix

torch = pytest.importorskip("torch", reason="torch not installed (install the 'gpu' extra)")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available (Triton requires a GPU)"
)

from pelagos import RelayBPDecoder, RelayDecoder  # noqa: E402

PERFS = ["throughput", "default"]


def _problem(C: int = 24, V: int = 40, deg: int = 4, seed: int = 7) -> csr_matrix:
    rng = np.random.default_rng(seed)
    H = np.zeros((C, V), dtype=np.uint8)
    for c in range(C):
        for v in rng.choice(V, deg, replace=False):
            H[c, v] = 1
    return csr_matrix(H)


def _syn(B: int, C: int, seed: int = 3) -> np.ndarray:
    # Random syndromes: most are not reachable by a low-weight error, so many shots never
    # converge, which is what these tests need.
    return np.random.default_rng(seed).integers(0, 2, size=(B, C)).astype(np.uint8)


def _decoder(H: csr_matrix, num_sets: int, perf: str, **kw) -> RelayBPDecoder:
    kw.setdefault("pre_iter", 4)
    return RelayBPDecoder(
        H, np.full(H.shape[1], 0.06), num_sets=num_sets, set_max_iter=4, gamma0=0.1,
        stop_nconv=1, device=[0], seed=42, algo="relay", perf=perf, **kw,
    )


def _to_np(out: dict) -> dict:
    return {k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in out.items()}


def _decode(H: csr_matrix, syn: np.ndarray, num_sets: int, perf: str, **kw) -> dict:
    return _to_np(_decoder(H, num_sets, perf, **kw).decode(torch.from_numpy(syn)))


# --- first_leg (default) --------------------------------------------------------------


@pytest.mark.parametrize("perf", PERFS)
def test_unconverged_shot_returns_leg0_hard_decision(perf):
    """With relay legs, a shot with found_count == 0 returns the num_sets=0 decoding."""
    H = _problem()
    syn = _syn(512, H.shape[0])
    full = _decode(H, syn, num_sets=3, perf=perf)
    leg0 = _decode(H, syn, num_sets=0, perf=perf)
    never = full["found_count"] == 0
    assert never.sum() > 0, "fixture produced no shot without a solution"
    np.testing.assert_array_equal(full["errors"][never], leg0["errors"][never])


@pytest.mark.parametrize("perf", PERFS)
def test_unconverged_shots_match_rust(perf):
    """Unconverged shots return exactly what the relay-bp Rust reference returns."""
    relay_bp = pytest.importorskip("relay_bp", reason="relay-bp not installed (install the 'rust' extra)")
    H = _problem()
    syn = _syn(256, H.shape[0])
    rust = relay_bp.RelayDecoderF32(
        H, error_priors=np.full(H.shape[1], 0.06), gamma0=0.1, pre_iter=4, num_sets=3,
        set_max_iter=4, gamma_dist_interval=(-0.24, 0.66), stop_nconv=1, seed=42,
    ).decode_detailed_batch(syn)
    rust_ok = np.array([bool(r.success) for r in rust])
    rust_e = np.stack([np.asarray(r.decoding) for r in rust]).astype(np.uint8)
    full = _decode(H, syn, num_sets=3, perf=perf)
    both_never = ~rust_ok & (full["found_count"] == 0)
    assert both_never.sum() > 0, "fixture produced no shot unconverged in both decoders"
    np.testing.assert_array_equal(full["errors"][both_never], rust_e[both_never])


def test_early_exit_unconverged_shot_returns_leg0_hard_decision():
    """The fast_max_unconverged_pct exit after relay leg 1 returns the leg-0 decoding too.

    The threshold lies between the unconverged fraction after leg 0 and after leg 1,
    so the check after leg 0 does not exit and the check after leg 1 does.
    """
    H = _problem()
    syn = _syn(512, H.shape[0])
    leg0 = _decode(H, syn, num_sets=0, perf="throughput")
    leg1 = _decode(H, syn, num_sets=1, perf="throughput")
    u0, u1 = float((leg0["found_count"] == 0).mean()), float((leg1["found_count"] == 0).mean())
    assert u1 < u0, "leg 1 solved no extra shot; the threshold cannot separate the two checks"
    full = _decode(H, syn, num_sets=3, perf="throughput", fast_max_unconverged_pct=(u0 + u1) / 2)
    assert len(full["unconverged_indices"]) > 0, "early exit did not trigger"
    never = full["found_count"] == 0
    assert never.sum() > 0, "fixture produced no shot without a solution"
    np.testing.assert_array_equal(full["errors"][never], leg0["errors"][never])


@pytest.mark.parametrize("perf", PERFS)
def test_pre_iter_zero_returns_relay_leg1_hard_decision(perf):
    """pre_iter=0: a shot with found_count == 0 returns the hard decision after relay leg 1.

    The reference is the same decoder with num_sets=1 and "last_leg", whose last leg is
    relay leg 1 with the same gamma draw. The rows must also be non-zero on at least one
    shot, so they are not the empty pre-iter leg's decision.
    """
    H = _problem()
    syn = _syn(512, H.shape[0])
    full = _decode(H, syn, num_sets=3, perf=perf, pre_iter=0)
    leg1 = _decode(H, syn, num_sets=1, perf=perf, pre_iter=0, unconverged_correction="last_leg")
    never = full["found_count"] == 0
    assert never.sum() > 0, "fixture produced no shot without a solution"
    np.testing.assert_array_equal(full["errors"][never], leg1["errors"][never])
    assert full["errors"][never].any(axis=1).any()


@pytest.mark.parametrize("perf", PERFS)
def test_converged_shots_still_satisfy_the_syndrome(perf):
    """Shots with a solution keep their best solution, which satisfies H e = s."""
    H = _problem()
    syn = _syn(512, H.shape[0])
    full = _decode(H, syn, num_sets=3, perf=perf)
    solved = full["found_count"] > 0
    assert solved.sum() > 0, "fixture produced no solved shot"
    e = full["errors"][solved].astype(np.int64)
    np.testing.assert_array_equal((H @ e.T % 2).T, syn[solved])


# --- last_leg ---------------------------------------------------------------------------


@pytest.mark.parametrize("perf", PERFS)
def test_last_leg_unconverged_shot_returns_final_hard_decision(perf):
    """last_leg: a shot with found_count == 0 returns the decoder's hard_dec after the last leg.

    The reference is the decoder's ``hard_dec`` device tensor read after ``decode`` returns,
    which holds each still-active lane's hard decision at the end of the last leg. The rows
    must also differ from the leg-0 (num_sets=0) decoding on at least one shot.
    """
    H = _problem()
    syn = _syn(512, H.shape[0])
    dec = _decoder(H, num_sets=3, perf=perf, unconverged_correction="last_leg")
    full = _to_np(dec.decode(torch.from_numpy(syn)))
    final_hard = dec._device_tensors_per_gpu[0][syn.shape[0]]["hard_dec"].cpu().numpy()
    leg0 = _decode(H, syn, num_sets=0, perf=perf)
    never = full["found_count"] == 0
    assert never.sum() > 0, "fixture produced no shot without a solution"
    np.testing.assert_array_equal(full["errors"][never], final_hard[never])
    assert (full["errors"][never] != leg0["errors"][never]).any(axis=1).any()


def test_last_leg_early_exit_returns_leg1_hard_decision():
    """last_leg with the fast_max_unconverged_pct exit after relay leg 1.

    The reference is a num_sets=1 decode with last_leg and no early exit: its
    unconverged rows are the hard decisions at the end of relay leg 1.
    """
    H = _problem()
    syn = _syn(512, H.shape[0])
    leg0 = _decode(H, syn, num_sets=0, perf="throughput")
    leg1 = _decode(H, syn, num_sets=1, perf="throughput", unconverged_correction="last_leg")
    u0, u1 = float((leg0["found_count"] == 0).mean()), float((leg1["found_count"] == 0).mean())
    assert u1 < u0, "leg 1 solved no extra shot; the threshold cannot separate the two checks"
    full = _decode(H, syn, num_sets=3, perf="throughput",
                   fast_max_unconverged_pct=(u0 + u1) / 2, unconverged_correction="last_leg")
    assert len(full["unconverged_indices"]) > 0, "early exit did not trigger"
    never = full["found_count"] == 0
    assert never.sum() > 0, "fixture produced no shot without a solution"
    np.testing.assert_array_equal(full["errors"][never], leg1["errors"][never])
    assert (full["errors"][never] != leg0["errors"][never]).any(axis=1).any()


# --- configuration --------------------------------------------------------------------


@pytest.mark.parametrize("perf", PERFS)
def test_default_is_first_leg(perf):
    """Without the argument the decoder uses first_leg and decodes identically."""
    H = _problem()
    syn = _syn(256, H.shape[0])
    assert _decoder(H, num_sets=3, perf=perf).unconverged_correction == "first_leg"
    a = _decode(H, syn, num_sets=3, perf=perf)
    b = _decode(H, syn, num_sets=3, perf=perf, unconverged_correction="first_leg")
    np.testing.assert_array_equal(a["errors"], b["errors"])


def test_invalid_unconverged_correction_raises():
    """An unknown value raises ValueError naming the allowed values."""
    with pytest.raises(ValueError, match="first_leg.*last_leg"):
        _decoder(_problem(), num_sets=3, perf="throughput", unconverged_correction="x")


@pytest.mark.parametrize("mode", ["first_leg", "last_leg"])
def test_adapter_passes_unconverged_correction(mode):
    """RelayDecoder reads unconverged_correction from relay_config; the default is first_leg."""
    H = _problem()
    relay = {"pre_iter": 4, "num_sets": 3, "set_max_iter": 4, "stop_nconv": 1, "seed": 42}
    if mode == "last_leg":
        relay["unconverged_correction"] = mode
    adapter = RelayDecoder(
        H,
        msl_config={"error_priors": np.full(H.shape[1], 0.06), "gamma0": 0.1},
        relay_config=relay,
        backend_config={"device": [0], "data_type": "float32"},
    )
    assert adapter._dec.unconverged_correction == mode


def test_detailed_batch_results_report_found_count():
    """Both RelayDecoder detailed-batch methods report each shot's found_count."""
    H = _problem()
    syn = _syn(64, H.shape[0])
    adapter = RelayDecoder(
        H,
        msl_config={"error_priors": np.full(H.shape[1], 0.06), "gamma0": 0.1},
        relay_config={"pre_iter": 4, "num_sets": 3, "set_max_iter": 4, "stop_nconv": 2, "seed": 42},
        backend_config={"device": [0], "data_type": "float32"},
    )
    expected = adapter._decode_inner(torch.from_numpy(syn))["found_count"].cpu().numpy()
    for method in (adapter.decode_detailed_batch, adapter.decode_detailed_batch_from_syndromes):
        got = np.array([r.found_count for r in method(syn)])
        np.testing.assert_array_equal(got, expected)
        assert set(got.tolist()) <= {0, 1, 2}
