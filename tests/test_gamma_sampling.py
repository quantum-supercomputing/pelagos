# (C) Copyright IBM 2026
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Seeded gamma samplers in ``pelagos.utils`` use a private generator.

``sample_gamma_uniform`` and ``sample_gamma_per_variable`` with a ``seed`` return the
same values as ``torch.manual_seed(seed)`` followed by ``torch.rand``, leave the global
CPU and CUDA RNG state unchanged, and return the same values when called concurrently
from several threads. Runs on CPU, and also on the GPU when one is available.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

import pytest

torch = pytest.importorskip("torch", reason="torch not installed (install the 'gpu' extra)")

from pelagos.utils import sample_gamma_per_variable, sample_gamma_uniform  # noqa: E402

DEVICES = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])
B, V = 64, 257


def _uniform(device: str, seed: int) -> torch.Tensor:
    """sample_gamma_uniform on [0.1, 0.9]."""
    return sample_gamma_uniform(B, V, 0.1, 0.9, device, seed)


def _per_variable(device: str, seed: int) -> torch.Tensor:
    """sample_gamma_per_variable with intervals [v / V, v / V + 0.5]."""
    low = torch.arange(V, dtype=torch.float32, device=device) / V
    intervals = torch.stack([low, low + 0.5], dim=1)
    return sample_gamma_per_variable(B, V, intervals, device, seed)


SAMPLERS: list[Callable[[str, int], torch.Tensor]] = [_uniform, _per_variable]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("sampler", SAMPLERS)
def test_matches_manual_seed_draw(device: str, sampler: Callable[[str, int], torch.Tensor]) -> None:
    """The draw equals torch.rand after torch.manual_seed(seed)."""
    seed = 1237
    got = sampler(device, seed)
    torch.manual_seed(seed)
    uniform = torch.rand(B, V, device=device)
    if sampler is _uniform:
        want = 0.1 + (0.9 - 0.1) * uniform
    else:
        low = (torch.arange(V, dtype=torch.float32, device=device) / V).unsqueeze(0)
        high = low + 0.5
        want = low + (high - low) * uniform
    assert torch.equal(got, want)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("sampler", SAMPLERS)
def test_global_rng_state_unchanged(
    device: str,
    sampler: Callable[[str, int], torch.Tensor],
) -> None:
    """A seeded call leaves the global CPU and CUDA RNG state unchanged."""
    torch.manual_seed(999)
    cpu_before = torch.random.get_rng_state()
    cuda_before = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    sampler(device, 5)
    assert torch.equal(torch.random.get_rng_state(), cpu_before)
    cuda_after = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    assert all(torch.equal(a, b) for a, b in zip(cuda_after, cuda_before))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("sampler", SAMPLERS)
def test_concurrent_threads_match_sequential(
    device: str,
    sampler: Callable[[str, int], torch.Tensor],
) -> None:
    """Two threads drawing with different seeds each get their sequential result."""
    seeds = (11, 22)
    want = {s: sampler(device, s) for s in seeds}
    mismatches = {s: 0 for s in seeds}

    def work(seed: int) -> None:
        for _ in range(200):
            if not torch.equal(sampler(device, seed), want[seed]):
                mismatches[seed] += 1

    threads = [threading.Thread(target=work, args=(s,)) for s in seeds]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert mismatches == {s: 0 for s in seeds}
