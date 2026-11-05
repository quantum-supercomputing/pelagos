# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Process-backend correctness smoke test (not a throughput benchmark).

Run with PYTHONPATH=src python tests/process_gpu_smoke.py --output DIR.
Requires two allocated GPUs. Stock references run in separate subprocesses so
the process-backend parent must never initialize a GPU context.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


def case_data(case):
    from scipy.sparse import csr_matrix
    if case == "triangle":
        h = csr_matrix(np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=np.uint8))
        priors = np.full(3, .05)
        errors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8)
    else:
        import stim
        from pelagos.stim.sinter.check_matrices import CheckMatrices
        n, distance = (144, 12) if case == "gross144" else (288, 18)
        files = list((Path(__file__).parent / "testdata/bicycle_bivariate").glob(
            f"*{n}_12_{distance}_memory_choi_XZ,*error_rate=0.003,*"))
        assert len(files) == 1
        circuit = stim.Circuit(gzip.decompress(files[0].read_bytes()).decode())
        matrices = CheckMatrices.from_dem(circuit.detector_error_model())
        h = matrices.check_matrix.tocsr()
        priors = matrices.error_priors
        errors = (np.random.default_rng(0).random((128, h.shape[1])) < priors).astype(np.uint8)
    syndromes = np.asarray((h @ errors.T) % 2, dtype=np.uint8).T
    config = dict(H_csr=h, error_priors=priors, pre_iter=80, num_sets=50,
                  set_max_iter=60, gamma0=.125,
                  gamma_dist_interval=(-.161, .815) if case == "gross288" else (-.24, .66),
                  stop_nconv=1, algo="relay", perf="throughput", seed=0)
    return syndromes, config


def reference(args):
    import torch
    from pelagos import RelayBPDecoder
    from pelagos.process import split_batch
    syndromes, config = case_data(args.case)
    decoder = RelayBPDecoder(**config, device=[0])
    for repeat in range(2):
        results = []
        for start, end in split_batch(len(syndromes), args.gpus):
            result = decoder.decode(torch.from_numpy(np.ascontiguousarray(syndromes[start:end])))
            results.append((start, {k: v.detach().cpu().numpy() for k, v in result.items()}))
        combined = {key: np.concatenate([value[key] + start if key == "unconverged_indices" else value[key]
                                        for start, value in results]) for key in results[0][1]}
        np.savez(args.output / f"{args.case}-g{args.gpus}-r{repeat}.npz", **combined)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--case", choices=["triangle", "gross144", "gross288"])
    parser.add_argument("--gpus", type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.reference:
        reference(args)
        return
    import torch
    from pelagos.process import ProcessRelayBPDecoder, validate_payload
    assert not torch.cuda.is_initialized()
    report = {"purpose": "process-backend correctness smoke test", "cases": []}
    started = time.monotonic()
    try:
        for case in ("triangle", "gross144", "gross288"):
            for gpus in (1, 2):
                env = dict(os.environ, RELAY_TUNE_CACHE=str(args.output / f"{case}-g{gpus}-tune.json"),
                           RELAY_TUNE_LOG_CSV=str(args.output / f"{case}-g{gpus}-tune.csv"))
                subprocess.run([sys.executable, __file__, "--reference", "--case", case,
                                "--gpus", str(gpus), "--output", str(args.output)],
                               env=env, check=True, timeout=240)
                os.environ["RELAY_TUNE_CACHE"] = env["RELAY_TUNE_CACHE"]
                tune = Path(env["RELAY_TUNE_CACHE"])
                before = hashlib.sha256(tune.read_bytes()).hexdigest() if tune.exists() else None
                syndromes, config = case_data(case)
                with ProcessRelayBPDecoder(**config, device=list(range(gpus)), worker_timeout_s=240) as decoder:
                    pids = [proc.pid for proc, _ in decoder._workers]
                    retained = None
                    for repeat in range(2):
                        output = decoder.decode(syndromes)
                        payload = {key: value.numpy() for key, value in output.items()}
                        validate_payload(payload)
                        with np.load(args.output / f"{case}-g{gpus}-r{repeat}.npz") as expected:
                            assert set(expected.files) == set(payload)
                            for key in payload:
                                np.testing.assert_array_equal(payload[key], expected[key], err_msg=f"{case} G{gpus} {key}")
                        if retained is not None:
                            np.testing.assert_array_equal(retained[0], retained[1])
                        retained = (output["errors"].numpy(), output["errors"].numpy().copy())
                        assert pids == [proc.pid for proc, _ in decoder._workers]
                    children = [proc for proc, _ in decoder._workers]
                assert all(proc.poll() is not None for proc in children)
                assert not torch.cuda.is_initialized()
                assert before == (hashlib.sha256(tune.read_bytes()).hexdigest() if tune.exists() else None)
                report["cases"].append({"case": case, "gpus": gpus, "repeats": 2,
                                         "exact_outputs": True, "workers_reused": True,
                                         "parent_gpu_context": False, "input_shape": list(syndromes.shape)})
                print("SMOKE_PASS", case, gpus, flush=True)
        report["passed"] = True
    except BaseException as exc:
        report.update(passed=False, error=repr(exc))
        raise
    finally:
        report["elapsed_s"] = time.monotonic() - started
        report["torch"] = torch.__version__
        (args.output / "summary.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
