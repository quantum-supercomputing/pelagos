# (C) Copyright IBM 2026
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""CPU-only unit tests: input validation, priors, bit packing, DEM conversion.

These run on GitHub-hosted runners, which have no GPU. The numerical decoder
tests live in test_smoke.py and test_triton_against_rust.py and need a GPU.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csc_matrix, csr_matrix

torch = pytest.importorskip("torch", reason="torch not installed (install the 'gpu' extra)")

from pelagos.utils import (  # noqa: E402
    bitpack_errors,
    bitunpack_errors,
    compute_log_prior_ratios,
    validate_csr_matrix,
    validate_error_priors,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CIRCUIT_DIR = REPO_ROOT / "tests" / "testdata" / "bicycle_bivariate"
TUTORIALS_DIR = REPO_ROOT / "tutorials"


# --- priors ------------------------------------------------------------------


def test_log_prior_ratios_match_formula():
    p = np.array([0.001, 0.01, 0.1, 0.3])
    llr = compute_log_prior_ratios(p)
    assert llr.dtype == np.float32
    np.testing.assert_allclose(llr, np.log((1 - p) / p), rtol=1e-6)


def test_log_prior_ratios_are_finite_for_zero_prior():
    llr = compute_log_prior_ratios(np.array([0.0, 0.05]))
    assert np.all(np.isfinite(llr))
    assert llr[0] > llr[1]


def test_validate_error_priors_passes_valid_priors_through():
    p = np.array([0.001, 0.2, 0.45])
    np.testing.assert_array_equal(validate_error_priors(p), p)


def test_validate_error_priors_rejects_2d_input():
    with pytest.raises(ValueError, match="1D"):
        validate_error_priors(np.full((2, 3), 0.01))


# --- check matrix validation --------------------------------------------------


def test_validate_csr_matrix_accepts_other_sparse_formats():
    H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    assert validate_csr_matrix(csr_matrix(H))
    assert validate_csr_matrix(csc_matrix(H))


@pytest.mark.parametrize(
    "H, message",
    [
        (np.array([[1, 0], [0, 1]]), "scipy.sparse"),
        (csr_matrix((0, 3), dtype=np.uint8), "empty"),
        (csr_matrix((2, 3), dtype=np.uint8), "zero non-zero"),
    ],
)
def test_validate_csr_matrix_rejects_bad_input(H, message):
    with pytest.raises(ValueError, match=message):
        validate_csr_matrix(H)


# --- bit packing --------------------------------------------------------------


@pytest.mark.parametrize("V", [1, 31, 32, 33, 100])
def test_bitpack_roundtrip(V):
    rng = np.random.default_rng(V)
    errors = torch.from_numpy(rng.integers(0, 2, size=(4, V), dtype=np.uint8))
    packed = bitpack_errors(errors)
    assert packed.shape == (4, (V + 31) // 32)
    assert torch.equal(bitunpack_errors(packed, V), errors)


# --- stim DEM -> check matrices ----------------------------------------------


def test_check_matrices_reproduce_sampled_syndromes_and_observables():
    stim = pytest.importorskip("stim", reason="stim not installed (install the 'stim' extra)")
    pytest.importorskip("beliefmatching", reason="install the 'stim' extra")
    from pelagos.stim.sinter.check_matrices import CheckMatrices

    dem = stim.Circuit.generated(
        "surface_code:rotated_memory_x", distance=3, rounds=3,
        after_clifford_depolarization=0.001,
    ).detector_error_model(decompose_errors=False)
    cm = CheckMatrices.from_dem(dem)
    H, O = cm.check_matrix.tocsr(), cm.observables_matrix.tocsr()

    # Every mechanism has a non-zero prior, so none are pruned and columns line up with the DEM.
    assert H.shape == (dem.num_detectors, dem.num_errors)
    assert O.shape == (dem.num_observables, dem.num_errors)
    assert np.all((cm.error_priors > 0) & (cm.error_priors < 0.5))

    det, obs, err = dem.compile_sampler(seed=0).sample(200, return_errors=True)
    e = err.astype(np.uint8).T
    np.testing.assert_array_equal(((H @ e) % 2).T.astype(bool), det)
    np.testing.assert_array_equal(((O @ e) % 2).T.astype(bool), obs)


def test_bundled_circuits_load():
    stim = pytest.importorskip("stim", reason="stim not installed (install the 'stim' extra)")
    paths = sorted(CIRCUIT_DIR.glob("*.stim.gz"))
    assert len(paths) == 24
    for path in paths:
        circuit = stim.Circuit(gzip.decompress(path.read_bytes()).decode())
        assert circuit.num_detectors > 0 and circuit.num_observables > 0, path.name


# --- tutorials ---------------------------------------------------------------


@pytest.mark.parametrize("path", sorted(TUTORIALS_DIR.glob("*.ipynb")), ids=lambda p: p.name)
def test_tutorial_notebook_is_well_formed(path):
    """Notebooks parse, their code compiles, and no saved output is an error."""
    nb = json.loads(path.read_text())
    assert nb["nbformat"] == 4
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert code_cells, "notebook has no code cells"
    for i, cell in enumerate(code_cells):
        compile("".join(cell["source"]), f"{path.name}[code cell {i}]", "exec")
        errors = [o for o in cell.get("outputs", []) if o["output_type"] == "error"]
        assert not errors, f"{path.name} code cell {i} has a saved error output"
