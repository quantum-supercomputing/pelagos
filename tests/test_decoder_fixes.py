# (C) Copyright IBM 2026
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Regression tests for decoder fixes: stopping, iteration counts, and gamma draws.

* ``valid_mask`` is True exactly for shots with ``stop_nconv`` solutions.
* Each lane stops on its own ``found_count``, so a shot decodes the same alone, in a small
  batch, or in a large batch.
* With ``gamma_draw="broadcast"`` (the default) a shot's result does not depend on its row
  in the batch; ``"per_row"`` keeps the old position-dependent draw.
* ``iterations`` counts every BP iteration run, as the Rust reference does, and relay leg
  ``l`` uses ``explicit_gammas[(l + 1) % K]``, as Rust's set ``l + 1`` does.

All tests need a GPU. Comparisons against Rust also need the ``rust`` extra.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix

torch = pytest.importorskip("torch", reason="torch not installed (install the 'gpu' extra)")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available (Triton requires a GPU)"
)

import pelagos.decoder as dmod  # noqa: E402
from pelagos import RelayBPDecoder, RelayDecoder  # noqa: E402

_DATA = Path(__file__).resolve().parent / "testdata" / "bicycle_bivariate"
GAMMA0 = 0.65
GAMMA_INTERVAL = (-0.24, 0.66)


# --- shared problem builders -----------------------------------------------------------


def _bb_18_4_3(num_shots: int):
    """[[18,4,3]] memory-X circuit at p=0.005: H, priors, and sampled syndromes."""
    stim = pytest.importorskip("stim", reason="stim not installed (install the 'stim' extra)")
    [path] = _DATA.glob("*18_4_3_memory_X,*error_rate=0.005,*")
    dem = stim.Circuit(gzip.decompress(path.read_bytes()).decode()).detector_error_model(
        decompose_errors=False
    )
    rows, cols, priors = [], [], []
    j = 0
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                rows.append(t.val)
                cols.append(j)
        priors.append(inst.args_copy()[0])
        j += 1
    H = csr_matrix((np.ones(len(rows), np.uint8), (rows, cols)), shape=(dem.num_detectors, j))
    syn = dem.compile_sampler(seed=1).sample(num_shots)[0].astype(np.uint8)
    return H, np.array(priors), torch.from_numpy(syn)


def _regular_code(seed: int, V: int = 600, C: int = 300) -> csr_matrix:
    """(3, 6)-regular check matrix: every variable in 3 checks, every check on 6 variables."""
    rng = np.random.default_rng(seed)
    while True:
        sockets = np.repeat(np.arange(C), 6)
        rng.shuffle(sockets)
        rows = sockets.reshape(V, 3)
        if all(len(set(r)) == 3 for r in rows):
            break
    return csr_matrix(
        (np.ones(3 * V, np.uint8), (rows.ravel(), np.repeat(np.arange(V), 3))),
        shape=(C, V),
    )


@pytest.fixture(scope="module")
def regular_case():
    """(3, 6)-regular code, non-uniform priors, and 128 syndromes of increasing difficulty."""
    H = _regular_code(3)
    V = H.shape[1]
    priors = np.random.default_rng(4).uniform(0.01, 0.03, V)
    scale = np.repeat([0.5, 1.0, 1.5, 3.0], 32)[:, None]
    E = (np.random.default_rng(5).random((128, V)) < priors * scale).astype(np.uint8)
    S = ((H @ E.T) % 2).T.astype(np.uint8)
    return H, priors, S


def _rust_relay(H, priors, **kw):
    """The relay-bp Rust reference decoder (CPU)."""
    relay_bp = pytest.importorskip("relay_bp", reason="relay-bp not installed (install the 'rust' extra)")
    return relay_bp.RelayDecoderF32(
        H,
        error_priors=priors,
        gamma0=GAMMA0,
        gamma_dist_interval=GAMMA_INTERVAL,
        logging=False,
        alpha=None,
        **kw,
    )


# --- valid_mask --------------------------------------------------------------------------


@pytest.mark.parametrize("stop_nconv", [1, 2, 4])
def test_valid_mask_equals_found_count_reaching_stop_nconv(stop_nconv):
    """valid_mask == (found_count >= stop_nconv), including shots that finished in earlier legs."""
    H, priors, syn = _bb_18_4_3(256)
    dec = RelayDecoder(
        H,
        msl_config={"error_priors": priors, "alpha": 1, "gamma0": 0.1},
        relay_config={"pre_iter": 20, "num_sets": 20, "set_max_iter": 20,
                      "stop_nconv": stop_nconv, "seed": 7},
        backend_config={"device": [0], "data_type": "float32"},
    )
    out = dec._decode_inner(syn)
    found = out["found_count"].cpu()
    assert (found >= stop_nconv).any()
    assert torch.equal(out["valid_mask"].cpu(), found >= stop_nconv)


# --- per-lane stopping: results do not depend on the batch ---------------------------------

_PER_LANE_B = 512
_PER_LANE_KEYS = ("errors", "weights", "found_count", "iterations")


@pytest.fixture(scope="module")
def bb_case():
    return _bb_18_4_3(_PER_LANE_B)


@pytest.fixture
def fixed_launch_configs(monkeypatch):
    """Use the first candidate config of each kernel at every batch size (no tuning)."""
    chosen: dict = {}

    def fixed(self, *, kernel_name, build_cfgs_fn, **kwargs):
        if kernel_name not in chosen:
            c = build_cfgs_fn()[0]
            chosen[kernel_name] = dict(c.kwargs, num_warps=c.num_warps, num_stages=c.num_stages)
        return dict(chosen[kernel_name])

    monkeypatch.setattr(dmod.RelayBPDecoder, "_get_tuned_meta", fixed)


def _per_lane_decoder(H, priors, stop_nconv: int) -> RelayDecoder:
    # Identical explicit gammas for every shot, so only the stopping logic can make
    # results depend on the batch.
    return RelayDecoder(
        H,
        msl_config={"error_priors": priors, "alpha": 1, "gamma0": 0.1},
        relay_config={
            "pre_iter": 10, "num_sets": 30, "set_max_iter": 30, "stop_nconv": stop_nconv,
            "seed": 7,
            "explicit_gammas": np.random.default_rng(0).uniform(-0.24, 0.66, size=(30, H.shape[1])),
        },
        backend_config={"device": [0], "data_type": "float32"},
    )


def _run(dec, syn: torch.Tensor, keys) -> dict:
    out = dec._decode_inner(syn) if isinstance(dec, RelayDecoder) else dec.decode(syn)
    return {k: out[k].cpu().clone() for k in keys}


def _rows_differing(full: dict, rows: torch.Tensor, part: dict) -> dict:
    n = len(rows)
    return {k: int((full[k][rows] != part[k]).reshape(n, -1).any(1).sum()) for k in part}


@pytest.mark.parametrize("stop_nconv", [1, 2, 4])
def test_single_shot_matches_batch(bb_case, fixed_launch_configs, stop_nconv):
    """Each of 24 shots decoded alone (B=1) matches the same shot in the B=512 batch."""
    H, priors, syn = bb_case
    dec = _per_lane_decoder(H, priors, stop_nconv)
    full = _run(dec, syn, _PER_LANE_KEYS)
    rows = torch.arange(24)
    parts = [_run(dec, syn[i : i + 1], _PER_LANE_KEYS) for i in rows.tolist()]
    part = {k: torch.cat([p[k] for p in parts]) for k in _PER_LANE_KEYS}
    assert _rows_differing(full, rows, part) == dict.fromkeys(_PER_LANE_KEYS, 0)


@pytest.mark.parametrize("stop_nconv", [1, 2, 4])
@pytest.mark.parametrize("B_sub", [2, 10, 64])
def test_small_batch_matches_large_batch(bb_case, fixed_launch_configs, stop_nconv, B_sub):
    """A subset decoded as its own small batch matches the same shots in the B=512 batch."""
    H, priors, syn = bb_case
    dec = _per_lane_decoder(H, priors, stop_nconv)
    full = _run(dec, syn, _PER_LANE_KEYS)
    rows = torch.from_numpy(np.random.default_rng(4).permutation(_PER_LANE_B)[:B_sub])
    part = _run(dec, syn[rows], _PER_LANE_KEYS)
    assert _rows_differing(full, rows, part) == dict.fromkeys(_PER_LANE_KEYS, 0)


# --- gamma_draw ------------------------------------------------------------------------------

_GD_KEYS = ("errors", "found_count", "iterations")
_GD_B = 256
_GD_P = 0.03


def _ldpc(C: int, V: int, col_weight: int, seed: int) -> csr_matrix:
    rng = np.random.default_rng(seed)
    H = np.zeros((C, V), dtype=np.uint8)
    for v in range(V):
        H[rng.choice(C, col_weight, replace=False), v] = 1
    return csr_matrix(H)


_GD_H = _ldpc(90, 180, 3, 0)


def _gd_syndromes(B: int, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    err = (rng.random((B, _GD_H.shape[1])) < _GD_P).astype(np.uint8)
    return torch.from_numpy((_GD_H @ err.T % 2).T.astype(np.uint8))


def _gd_decoder(stop_nconv: int, **kw) -> RelayBPDecoder:
    return RelayBPDecoder(
        _GD_H, np.full(_GD_H.shape[1], _GD_P), pre_iter=5, num_sets=20, set_max_iter=20,
        stop_nconv=stop_nconv, gamma0=0.1, gamma_dist_interval=GAMMA_INTERVAL,
        data_type="float32", device=[0], algo="relay", seed=7, **kw,
    )


def _n_rows_differ(a: dict, rows: torch.Tensor, b: dict) -> int:
    return sum(_rows_differing(a, rows, b).values())


def test_result_invariant_to_permutation():
    """Permuting the batch leaves every shot's errors, found_count and iterations unchanged."""
    syn = _gd_syndromes(_GD_B, 1)
    perm = torch.from_numpy(np.random.default_rng(2).permutation(_GD_B))
    dec = _gd_decoder(stop_nconv=2)
    assert _n_rows_differ(_run(dec, syn, _GD_KEYS), perm, _run(dec, syn[perm], _GD_KEYS)) == 0


@pytest.mark.parametrize("B_sub", [1, 10, 64])
def test_subset_rerun_matches_full_batch(B_sub):
    """A subset of shots re-decoded alone gives the same per-shot results as the full batch."""
    syn = _gd_syndromes(_GD_B, 3)
    sub = torch.from_numpy(np.random.default_rng(4).permutation(_GD_B)[:B_sub])
    dec = _gd_decoder(stop_nconv=1)
    assert _n_rows_differ(_run(dec, syn, _GD_KEYS), sub, _run(dec, syn[sub], _GD_KEYS)) == 0


def test_per_variable_interval_subset_rerun_matches_full_batch():
    """Same subset invariance with gamma_dist_interval_by_error."""
    V = _GD_H.shape[1]
    intervals = np.stack([np.full(V, -0.3), np.linspace(0.3, 0.8, V)], axis=1)
    syn = _gd_syndromes(_GD_B, 5)
    sub = torch.from_numpy(np.random.default_rng(6).permutation(_GD_B)[:10])
    dec = _gd_decoder(stop_nconv=1, gamma_dist_interval_by_error=intervals)
    assert _n_rows_differ(_run(dec, syn, _GD_KEYS), sub, _run(dec, syn[sub], _GD_KEYS)) == 0


def test_gamma_draw_default_is_broadcast_and_rejects_unknown():
    assert _gd_decoder(stop_nconv=1).gamma_draw == "broadcast"
    with pytest.raises(ValueError, match="gamma_draw"):
        _gd_decoder(stop_nconv=1, gamma_draw="shared")


def test_per_row_depends_on_position():
    """"per_row": permuting the batch changes some shots' results."""
    syn = _gd_syndromes(_GD_B, 1)
    perm = torch.from_numpy(np.random.default_rng(2).permutation(_GD_B))
    dec = _gd_decoder(stop_nconv=2, gamma_draw="per_row")
    assert _n_rows_differ(_run(dec, syn, _GD_KEYS), perm, _run(dec, syn[perm], _GD_KEYS)) > 0


@pytest.mark.parametrize("mode", ["broadcast", "per_row"])
def test_adapter_passes_gamma_draw(mode):
    """The adapter reads gamma_draw from relay_config; the default is "broadcast"."""
    relay = {"pre_iter": 5, "num_sets": 2, "set_max_iter": 5, "stop_nconv": 1, "seed": 7}
    if mode == "per_row":
        relay["gamma_draw"] = mode
    adapter = RelayDecoder(
        _GD_H,
        msl_config={"error_priors": np.full(_GD_H.shape[1], _GD_P), "alpha": 1, "gamma0": 0.1},
        relay_config=relay,
        backend_config={"device": [0], "data_type": "float32"},
    )
    assert adapter._dec.gamma_draw == mode


# --- iteration counts match Rust ----------------------------------------------------------


def _triton(H, priors, perf, pre_iter, num_sets, set_max_iter) -> RelayBPDecoder:
    return RelayBPDecoder(
        H, priors, pre_iter=pre_iter, num_sets=num_sets, set_max_iter=set_max_iter,
        stop_nconv=1, gamma0=GAMMA0, gamma_dist_interval=GAMMA_INTERVAL, device=[0],
        seed=7, algo="relay", perf=perf,
    )


def _rust_and_triton(case, perf: str, pre_iter: int):
    """Success and iterations per shot from Rust and Triton, pre-iterations only."""
    H, priors, S = case
    rust = _rust_relay(H, priors, pre_iter=pre_iter, num_sets=0, set_max_iter=20,
                       stop_nconv=1).decode_detailed_batch(S)
    out = _triton(H, priors, perf, pre_iter, 0, 20).decode(torch.from_numpy(S))
    return (
        np.array([bool(r.success) for r in rust]),
        np.array([int(r.iterations) for r in rust]),
        out["found_count"].cpu().numpy() > 0,
        out["iterations"].cpu().numpy(),
    )


@pytest.mark.parametrize("perf", ["default", "throughput"])
@pytest.mark.parametrize("pre_iter", [3, 10, 30])
def test_pre_iterations_count_matches_rust_per_shot(regular_case, perf, pre_iter):
    """Without relay legs, iterations equal Rust's on every shot both decoders agree on."""
    rust_ok, rust_it, tri_ok, tri_it = _rust_and_triton(regular_case, perf, pre_iter)
    agree = rust_ok == tri_ok
    assert agree.mean() > 0.95
    assert rust_ok[agree].any() and not rust_ok[agree].all()
    assert tri_it[agree].tolist() == rust_it[agree].tolist()


@pytest.mark.parametrize("perf", ["default", "throughput"])
@pytest.mark.parametrize("pre_iter", [3, 10, 30])
def test_pre_iterations_success_matches_rust_per_shot(regular_case, perf, pre_iter):
    """The same shots converge as with Rust."""
    rust_ok, _, tri_ok, _ = _rust_and_triton(regular_case, perf, pre_iter)
    assert tri_ok.tolist() == rust_ok.tolist()


@pytest.mark.parametrize("pre_iter", [1, 3])
def test_throughput_posterior_after_pre_iterations_matches_rust(regular_case, pre_iter):
    """perf="throughput": the posterior after pre_iter iterations equals Rust's.

    Compared on shots neither decoder has converged on, since a converged lane stops
    updating its posterior.
    """
    H, priors, S = regular_case
    rust = _rust_relay(H, priors, pre_iter=pre_iter, num_sets=0, set_max_iter=20,
                       stop_nconv=1).decode_detailed_batch(S)
    out = _triton(H, priors, "throughput", pre_iter, 0, 20).decode(
        torch.from_numpy(S),
        skip_relay_legs=True,
        pre_iter_override=pre_iter,
        return_last_posterior=True,
    )
    both_open = ~np.array([bool(r.success) for r in rust]) & (out["found_count"].cpu().numpy() == 0)
    assert both_open.any()
    rust_post = np.stack([np.asarray(r.posterior_ratios) for r in rust])[both_open]
    tri_post = out["last_posterior"].float().cpu().numpy()[both_open]
    np.testing.assert_allclose(tri_post, rust_post, rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize("perf", ["default", "throughput"])
def test_never_converged_shot_counts_every_iteration(regular_case, perf):
    """A shot that never converges reports pre_iter + num_sets * set_max_iter."""
    H, priors, S = regular_case
    out = _triton(H, priors, perf, 30, 5, 20).decode(torch.from_numpy(S))
    never = out["found_count"].cpu() == 0
    assert bool(never.any())
    assert set(out["iterations"].cpu()[never].tolist()) == {30 + 5 * 20}


# --- explicit gammas: the row each relay leg uses ---------------------------------------------

_EG_K = 3
_EG_NUM_SETS = 7


@pytest.fixture(scope="module")
def gamma_case():
    """Check matrix, non-uniform priors and K rows of explicit gammas."""
    H = _regular_code(3)
    V = H.shape[1]
    priors = np.random.default_rng(4).uniform(0.01, 0.03, V)
    gammas = np.random.default_rng(6).uniform(-0.24, 0.66, (_EG_K, V))
    return H, priors, gammas


def _explicit_decoder(H, priors, gammas, perf: str, stop_nconv: int) -> RelayBPDecoder:
    return RelayBPDecoder(
        H, priors, pre_iter=10, num_sets=_EG_NUM_SETS, set_max_iter=15, stop_nconv=stop_nconv,
        gamma0=GAMMA0, device=[0], seed=7, algo="relay", perf=perf, explicit_gammas=gammas,
    )


@pytest.mark.parametrize("perf", ["default", "throughput"])
def test_each_leg_uses_rust_row(gamma_case, perf, monkeypatch):
    """Leg L starts with memory strengths explicit_gammas[(L + 1) % K]."""
    H, priors, gammas = gamma_case
    E = (np.random.default_rng(8).random((1, H.shape[1])) < 0.15).astype(np.uint8)
    S = torch.from_numpy(((H @ E.T) % 2).T.astype(np.uint8))
    seen: list[np.ndarray] = []
    if perf == "throughput":
        orig_it = dmod.RelayBPDecoder._run_bp_iteration

        def per_iteration(self, t, tensors, *args, **kwargs):
            if t == 0:
                seen.append(tensors["gamma"][0].float().cpu().numpy().copy())
            return orig_it(self, t, tensors, *args, **kwargs)

        monkeypatch.setattr(dmod.RelayBPDecoder, "_run_bp_iteration", per_iteration)
    else:
        orig_gs = dmod.RelayBPDecoder._run_bp_iterations_grid_stride

        def per_phase(self, start_iter, max_iter, tensors, *args, **kwargs):
            seen.append(tensors["gamma"][0].float().cpu().numpy().copy())
            return orig_gs(self, start_iter, max_iter, tensors, *args, **kwargs)

        monkeypatch.setattr(dmod.RelayBPDecoder, "_run_bp_iterations_grid_stride", per_phase)
    out = _explicit_decoder(H, priors, gammas, perf, stop_nconv=1).decode(S)
    assert int(out["found_count"][0]) == 0, "the shot must not converge, so every leg runs"
    legs = seen[1:]  # seen[0] is the pre-iteration phase (uniform gamma0)
    assert len(legs) == _EG_NUM_SETS
    for leg, g in enumerate(legs):
        np.testing.assert_allclose(g, gammas[(leg + 1) % _EG_K].astype(np.float32), rtol=0, atol=0)


@pytest.mark.parametrize("perf", ["default", "throughput"])
@pytest.mark.parametrize("stop_nconv", [1, 3])
def test_legs_agree_with_rust(gamma_case, perf, stop_nconv):
    """With relay legs and explicit gammas: success and iterations agree with Rust on most shots."""
    H, priors, gammas = gamma_case
    V = H.shape[1]
    scale = np.repeat([0.5, 1.0, 1.5, 3.0], 32)[:, None]
    E = (np.random.default_rng(5).random((128, V)) < priors * scale).astype(np.uint8)
    S = ((H @ E.T) % 2).T.astype(np.uint8)
    rust = _rust_relay(H, priors, pre_iter=10, num_sets=_EG_NUM_SETS, set_max_iter=15,
                       stop_nconv=stop_nconv, explicit_gammas=gammas).decode_detailed_batch(S)
    out = _explicit_decoder(H, priors, gammas, perf, stop_nconv).decode(torch.from_numpy(S))
    rust_ok = np.array([bool(r.success) for r in rust])
    rust_it = np.array([int(r.iterations) for r in rust])
    tri_ok = out["found_count"].cpu().numpy() > 0
    tri_it = out["iterations"].cpu().numpy()
    agree = rust_ok == tri_ok
    assert agree.mean() >= 0.97
    assert (tri_it == rust_it)[agree].mean() >= 0.85
