# (C) Copyright IBM 2026
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Smoke tests: import surface and a minimal decode.

`import pelagos` pulls in the Triton kernels, which import torch at module
scope, so the import-surface tests need the `gpu` extra installed even though
they never touch a GPU. The decode tests additionally need a working GPU.
Both are guarded, so a CPU-only environment skips rather than fails.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


requires_torch = pytest.mark.skipif(
    not _torch_available(),
    reason="torch not installed (install the 'gpu' extra)",
)

requires_cuda = pytest.mark.skipif(
    not _cuda_available(), reason="CUDA not available (Triton requires a GPU)"
)


@requires_torch
def test_public_exports():
    import pelagos

    assert set(pelagos.__all__) == {
        "RelayBPDecoder",
        "RelayDecoder",
        "ObservableDecoderRunner",
        "RelayDecoderF32",
        "RelayDecoderF64",
    }


@requires_torch
def test_rust_aliases_point_at_the_adapter():
    import pelagos

    assert pelagos.RelayDecoderF32 is pelagos.RelayDecoder
    assert pelagos.RelayDecoderF64 is pelagos.RelayDecoder


@requires_cuda
def test_decodes_a_triangle_code():
    """Three checks over three variables; a single-bit error is corrected.

    ``decode()`` requires a torch CUDA tensor (not numpy), ``algo`` must be
    passed explicitly, and the result dict's error-pattern key is
    ``"errors"`` (not ``"e_hat"``) — see the module docstring below for the
    verified reference call this is modeled on.
    """
    import torch

    from pelagos import RelayBPDecoder

    H = csr_matrix(
        np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=np.uint8)
    )
    priors = np.full(3, 0.05, dtype=np.float64)
    decoder = RelayBPDecoder(
        H_csr=H,
        error_priors=priors,
        pre_iter=10,
        num_sets=2,
        set_max_iter=10,
        stop_nconv=1,
        data_type="float32",
        algo="relay",
    )
    error = np.array([[1, 0, 0]], dtype=np.uint8)
    syndrome = (H.toarray() @ error.T % 2).T.astype(np.uint8)
    syndrome_gpu = torch.from_numpy(syndrome).to("cuda:0")
    result = decoder.decode(syndrome_gpu)
    e_hat = result["errors"].cpu().numpy().astype(np.uint8)
    assert e_hat.shape[-1] == 3
    residual = (H.toarray() @ e_hat[0] % 2).astype(np.uint8)
    assert np.array_equal(residual, syndrome[0]), "decoded pattern must explain the syndrome"


@requires_cuda
def test_batch_of_syndromes_returns_one_row_each():
    import torch

    from pelagos import RelayBPDecoder

    H = csr_matrix(np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=np.uint8))
    priors = np.full(3, 0.05, dtype=np.float64)
    decoder = RelayBPDecoder(
        H_csr=H,
        error_priors=priors,
        pre_iter=10,
        num_sets=2,
        set_max_iter=10,
        stop_nconv=1,
        data_type="float32",
        algo="relay",
    )
    syndromes = np.zeros((4, 3), dtype=np.uint8)
    syndromes_gpu = torch.from_numpy(syndromes).to("cuda:0")
    result = decoder.decode(syndromes_gpu)
    e_hat = result["errors"].cpu().numpy().astype(np.uint8)
    assert e_hat.shape[0] == 4
