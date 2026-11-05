# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in process-per-GPU orchestration; the single-GPU decoder is unchanged.

Results are owned CPU tensors. Each decoder instance reuses its workers until
close(). Device indices refer to the caller's visible device list.
"""
from __future__ import annotations

import inspect
import os
from pathlib import Path
import socket
import subprocess
import sys
import shutil
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import shared_memory
from multiprocessing.connection import Connection

import numpy as np


def layout_copy(array):
    """Return a C-contiguous copy of a matrix.

    A strided (e.g. transposed) uint8 matrix is copied in blocks of 32 columns, which
    keeps memory access cache-friendly: for syndrome batches this is 3-8x faster than
    ``np.ascontiguousarray``. Other inputs fall back to ``np.ascontiguousarray``.
    """
    array = np.asarray(array)
    if array.ndim != 2 or array.dtype != np.uint8 or array.flags.c_contiguous:
        return np.ascontiguousarray(array)
    result = np.empty(array.shape, dtype=array.dtype)
    for column in range(0, array.shape[1], 32):
        result[:, column:column + 32] = array[:, column:column + 32]
    return result


def split_batch(batch, workers):
    if batch < 0 or workers < 1:
        raise ValueError("batch must be nonnegative and workers positive")
    boundaries = []
    start = 0
    for rank in range(workers):
        end = start + batch // workers + (rank < batch % workers)
        boundaries.append((start, end))
        start = end
    return boundaries


def prepare_batch(check_matrix, observables, errors, workers, multiply_mod2):
    """Prepare disjoint host shards concurrently, preserving original row order."""
    def prepare(bounds):
        start, end = bounds
        shard = errors[start:end]
        obs = multiply_mod2(observables, shard.T).T
        syndromes = np.asarray((check_matrix @ shard.T) % 2, dtype=np.uint8).T
        return obs, syndromes

    splits = [(start, end) for start, end in split_batch(len(errors), workers) if start < end]
    if not splits:
        raise ValueError("empty error batch")
    if len(splits) == 1:
        obs, syndromes = prepare(splits[0])
    else:
        with ThreadPoolExecutor(max_workers=len(splits)) as executor:
            parts = list(executor.map(prepare, splits))
        obs = np.concatenate([p[0] for p in parts], axis=0)
        syndromes = np.concatenate([p[1] for p in parts], axis=0)
    return obs, layout_copy(syndromes)


def worker_environment(device, hip, environment=None):
    env = dict(os.environ if environment is None else environment)
    key = "ROCR_VISIBLE_DEVICES" if hip else "CUDA_VISIBLE_DEVICES"
    # HIP visibility is relative to ROCR visibility. Resolve it before narrowing.
    visible = env.get(key)
    devices = visible.split(",") if visible is not None else None
    if hip:
        secondary = env.get("HIP_VISIBLE_DEVICES", env.get("CUDA_VISIBLE_DEVICES"))
        if secondary is not None:
            indices = [int(x) for x in secondary.split(",") if x]
            if any(x < 0 for x in indices):
                raise ValueError("No usable visible GPU")
            devices = [devices[x] if devices is not None else str(x) for x in indices]
    if device < 0 or (devices is not None and device >= len(devices)):
        raise ValueError("device is outside the caller's visible GPU list")
    env[key] = devices[device] if devices is not None else str(device)
    if hip:
        env["HIP_VISIBLE_DEVICES"] = "0"
        env["CUDA_VISIBLE_DEVICES"] = "0"
    return env


def validate_payload(payload):
    for key, array in payload.items():
        if not np.issubdtype(array.dtype, np.inexact):
            continue
        bad = ~np.isfinite(array)
        if not bad.any():
            continue
        valid = payload.get("valid_mask")
        if (key != "weights" or valid is None or valid.shape != array.shape
                or np.any(bad & valid) or not np.all(np.isposinf(array[bad]))):
            raise ValueError(f"nonfinite decoder output: {key}")
    if np.any(payload["iterations"] < 0):
        raise ValueError("negative iteration count")


class ProcessRelayBPDecoder:
    """Persistent Linux GPU workers with static sharding and shared error output.

    Constructor arguments match RelayBPDecoder, plus worker_timeout_s. Advanced
    per-call decode overrides and layered scheduling are explicitly unsupported.
    No global monkeypatches, shared pool cache, or autotuning policy changes.
    """

    def __init__(self, *args, worker_timeout_s=300, **kwargs):
        import torch
        from .decoder import RelayBPDecoder

        if os.name != "posix":
            raise RuntimeError("Process backend requires POSIX descriptor passing")
        bound = inspect.signature(RelayBPDecoder).bind(*args, **kwargs)
        bound.apply_defaults()
        ctor = dict(bound.arguments)
        devices = ctor.pop("device")
        if isinstance(devices, int):
            devices = list(range(devices))
        elif isinstance(devices, str):
            devices = [0] if devices in ("auto", "cuda") else [int(x) for x in devices.split(",")]
        self.device_ids = list(devices)
        if not self.device_ids or len(set(self.device_ids)) != len(self.device_ids):
            raise ValueError("device list must be nonempty and unique")
        if any(not isinstance(x, int) or x < 0 for x in self.device_ids):
            raise ValueError("device indices must be nonnegative integers")
        if ctor["scheduling"] != "flood-fill":
            raise ValueError("process backend currently supports flood-fill only")
        if not np.isfinite(worker_timeout_s) or worker_timeout_s <= 0:
            raise ValueError("worker_timeout_s must be finite and positive")
        self.num_gpus = len(self.device_ids)
        self.pre_iter = ctor["pre_iter"]
        self.num_sets = ctor["num_sets"]
        self.fast_max_unconverged_pct = ctor["fast_max_unconverged_pct"]
        self.bitpack_output = ctor["bitpack_output"]
        self.check_permutation = None
        self._shape = ctor["H_csr"].shape
        self._timeout = worker_timeout_s
        self._workers = []
        self._shared = None
        self._shared_array = None
        self._lock = threading.Lock()
        self._closed = False
        self._cache = tempfile.TemporaryDirectory(prefix="pelagos-process-")
        try:
            for device in self.device_ids:
                env = worker_environment(device, torch.version.hip is not None)
                # Preserve the selected checkout when launched without installation.
                root = str(Path(__file__).resolve().parents[1])
                env["PYTHONPATH"] = os.pathsep.join([root, env.get("PYTHONPATH", "")])
                cache = Path(self._cache.name) / str(device)
                cache.mkdir()
                env["TRITON_CACHE_DIR"] = str(cache / "triton")
                tune = env.get("RELAY_TUNE_CACHE")
                if tune and Path(tune).is_file():
                    shutil.copyfile(tune, cache / "tune.json")
                env["RELAY_TUNE_CACHE"] = str(cache / "tune.json")
                env["RELAY_TUNE_LOG_CSV"] = str(cache / "autotune.csv")
                left, right = socket.socketpair()
                try:
                    proc = subprocess.Popen(
                        [sys.executable, "-m", "pelagos._process_worker", str(right.fileno())],
                        pass_fds=(right.fileno(),), env=env,
                    )
                    conn = Connection(left.detach())
                finally:
                    left.close()
                    right.close()
                self._workers.append((proc, conn))
                conn.send(ctor)
            deadline = time.monotonic() + self._timeout
            for _, conn in self._workers:
                metadata = self._receive(conn, "ready", deadline)
                self.wj = torch.from_numpy(metadata["wj"])
        except BaseException:
            self.close()
            raise

    def _receive(self, conn, expected, deadline):
        if not conn.poll(max(0, deadline - time.monotonic())):
            raise TimeoutError("process decoder worker timed out")
        try:
            status, payload = conn.recv()
        except EOFError as exc:
            raise RuntimeError("process decoder worker exited") from exc
        if status != expected:
            raise RuntimeError(f"process decoder worker: {status}: {payload}")
        return payload

    def decode(self, syndromes, **kwargs):
        import torch
        if kwargs:
            raise ValueError("process backend does not support per-call decode overrides")
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("concurrent decode calls are not supported")
        try:
            if self._closed:
                raise RuntimeError("decoder is closed")
            if isinstance(syndromes, torch.Tensor):
                if syndromes.device.type != "cpu":
                    raise ValueError("process backend requires host syndromes")
                syndromes = syndromes.detach().numpy()
            array = np.asarray(syndromes)
            if array.ndim != 2 or array.shape[1] != self._shape[0] or not array.shape[0]:
                raise ValueError("expected nonempty [batch, checks] syndromes")
            if not np.all((array == 0) | (array == 1)):
                raise ValueError("syndromes must be binary")
            array = layout_copy(array.astype(np.uint8, copy=False))
            batch = len(array)
            if not self.bitpack_output:
                shape = (batch, self._shape[1])
                if self._shared_array is None or self._shared_array.shape != shape:
                    self._release_shared()
                    self._shared = shared_memory.SharedMemory(create=True, size=int(np.prod(shape)))
                    self._shared_array = np.ndarray(shape, dtype=np.uint8, buffer=self._shared.buf)
            parts = []
            deadline = time.monotonic() + self._timeout
            for (_, conn), (start, end) in zip(self._workers, split_batch(batch, self.num_gpus)):
                if start == end:
                    continue
                shared = None if self.bitpack_output else (self._shared.name, self._shared_array.shape, start, end)
                conn.send((array[start:end].copy(), shared))
                parts.append((start, conn))
            results = [(start, self._receive(conn, "ok", deadline)) for start, conn in parts]
            keys = set(results[0][1])
            if any(set(result) != keys for _, result in results):
                raise RuntimeError("inconsistent worker result keys")
            output = {}
            for key in keys:
                chunks = [value[key] + start if key == "unconverged_indices" else value[key]
                          for start, value in results]
                output[key] = torch.from_numpy(np.concatenate(chunks, axis=0))
            if not self.bitpack_output:
                output["errors"] = torch.from_numpy(self._shared_array.copy())
            return output
        except (RuntimeError, TimeoutError, EOFError, BrokenPipeError):
            self.close()
            raise
        finally:
            self._lock.release()

    def _release_shared(self):
        self._shared_array = None
        if self._shared is not None:
            self._shared.close()
            self._shared.unlink()
            self._shared = None

    def close(self):
        self._closed = True
        for proc, conn in self._workers:
            conn.close()
            if proc.poll() is None:
                proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        self._workers.clear()
        self._release_shared()
        if hasattr(self, "_cache"):
            self._cache.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        if hasattr(self, "_workers"):
            self.close()
