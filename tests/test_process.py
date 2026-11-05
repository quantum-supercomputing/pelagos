# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the process transport; GPU comparison is a separate smoke."""
import importlib.util
import sys
import threading
import types
from pathlib import Path

import numpy as np
import pytest

# Avoid importing Triton so these host-only checks also run on CPU CI.
spec = importlib.util.spec_from_file_location(
    "process_under_test", Path(__file__).parents[1] / "src/pelagos/process.py")
process = importlib.util.module_from_spec(spec)
spec.loader.exec_module(process)


@pytest.mark.parametrize("shape", [(64, 1872), (4096, 17), (5114, 17), (1707, 31), (0, 3)])
def test_copy(shape):
    array = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape[::-1]).T
    result = process.layout_copy(array)
    np.testing.assert_array_equal(result, array)
    assert result.flags.c_contiguous


@pytest.mark.parametrize("batch,workers", [(0, 2), (1, 8), (9, 4), (16, 2)])
def test_splits(batch, workers):
    splits = process.split_batch(batch, workers)
    assert [i for start, end in splits for i in range(start, end)] == list(range(batch))
    assert max(end - start for start, end in splits) - min(end - start for start, end in splits) <= 1


def test_visibility():
    env = process.worker_environment(1, True, {"ROCR_VISIBLE_DEVICES": "4,2", "HIP_VISIBLE_DEVICES": "0,1"})
    assert env["ROCR_VISIBLE_DEVICES"] == "2"
    assert env["HIP_VISIBLE_DEVICES"] == "0"
    assert process.worker_environment(0, False, {"CUDA_VISIBLE_DEVICES": "3,5"})["CUDA_VISIBLE_DEVICES"] == "3"
    with pytest.raises(ValueError):
        process.worker_environment(2, True, {"ROCR_VISIBLE_DEVICES": "4,2"})


def test_parallel_preparation():
    from scipy.sparse import csr_matrix
    h = csr_matrix(np.array([[1, 0, 1], [0, 1, 1]], dtype=np.uint8))
    obs = csr_matrix(np.array([[1, 1, 0]], dtype=np.uint8))
    errors = np.arange(27, dtype=np.uint8).reshape(9, 3) % 2
    actual_obs, syndromes = process.prepare_batch(h, obs, errors, 4, lambda a, b: np.asarray((a @ b) % 2))
    np.testing.assert_array_equal(actual_obs, ((obs @ errors.T) % 2).T)
    np.testing.assert_array_equal(syndromes, ((h @ errors.T) % 2).T)
    assert syndromes.flags.c_contiguous


def test_finite_outputs():
    good = {"weights": np.array([1., np.inf]), "valid_mask": np.array([True, False]),
            "iterations": np.array([2, 3])}
    process.validate_payload(good)
    with pytest.raises(ValueError):
        process.validate_payload(dict(good, weights=np.array([np.inf, np.inf])))
    with pytest.raises(ValueError):
        process.validate_payload(dict(good, weights=np.array([1., np.nan])))


def test_transport_order_offsets_and_reuse(monkeypatch):
    class FakeTensor:
        pass

    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(
        Tensor=FakeTensor, from_numpy=lambda array: array))

    class Connection:
        def send(self, message):
            shard, _ = message
            self.payload = {"errors": shard.copy(), "iterations": np.ones(len(shard), dtype=np.int32),
                            "unconverged_indices": np.array([len(shard) - 1], dtype=np.int32)}

        def poll(self, timeout):
            return True

        def recv(self):
            return "ok", self.payload

    decoder = process.ProcessRelayBPDecoder.__new__(process.ProcessRelayBPDecoder)
    decoder._workers = [(None, Connection()), (None, Connection())]
    decoder._lock = threading.Lock()
    decoder._closed = False
    decoder._shape = (3, 3)
    decoder._timeout = 1
    decoder.num_gpus = 2
    decoder.bitpack_output = True
    syndromes = np.arange(15, dtype=np.uint8).reshape(5, 3) % 2
    try:
        first = decoder.decode(syndromes)
        np.testing.assert_array_equal(first["errors"], syndromes)
        np.testing.assert_array_equal(first["unconverged_indices"], [2, 4])
        second = decoder.decode(1 - syndromes)
        np.testing.assert_array_equal(second["errors"], 1 - syndromes)
        np.testing.assert_array_equal(first["errors"], syndromes)
        with pytest.raises(ValueError):
            decoder.decode(syndromes, prior_llr=np.ones((5, 3)))
        with pytest.raises(ValueError):
            decoder.decode(np.full((5, 3), 2))
    finally:
        del decoder._workers


def test_receive_failure_and_timeout():
    decoder = process.ProcessRelayBPDecoder.__new__(process.ProcessRelayBPDecoder)
    connection = types.SimpleNamespace(poll=lambda timeout: False)
    with pytest.raises(TimeoutError):
        decoder._receive(connection, "ok", 0)
    connection = types.SimpleNamespace(poll=lambda timeout: True, recv=lambda: ("error", "test failure"))
    with pytest.raises(RuntimeError, match="test failure"):
        decoder._receive(connection, "ok", float("inf"))
