<!-- Process-per-GPU documentation contributed by AMD; subsequently revised by IBM. -->

# Process-per-GPU execution

The optional process backend runs one persistent worker process per GPU. Each call splits the
batch statically across the workers, prepares observables and syndromes on several host
threads, and returns decoded errors through shared memory. It uses the same Triton kernels as
the default backend and does not change autotuning.

```python
from pelagos import RelayDecoder, ObservableDecoderRunner

with RelayDecoder(
    check_matrix,
    msl_config=msl_config,
    relay_config=relay_config,
    data_type="float32",
    backend_config={"execution": "process", "device": [0, 1],
                    "perf": "throughput", "algo": "relay"},
) as decoder:
    runner = ObservableDecoderRunner(decoder, observables_matrix)
    results = runner.from_errors_decode_observables_detailed_batch(errors)
```

The default remains `execution="thread"`. For the low-level API, import
`ProcessRelayBPDecoder` from `pelagos.process`; its constructor accepts the
`RelayBPDecoder` arguments plus `worker_timeout_s` (default 300).

## Usage notes

* Inputs must be host (CPU) arrays or tensors; results are CPU tensors owned by the caller.
* Workers start once and are reused across calls. Reuse one instance across batches, and
  close it when done, or use it as a context manager as above.
* Concurrent calls on one instance, layered scheduling, and the advanced per-call `decode()`
  arguments are not supported. They raise an error rather than being silently ignored.
* The process backend requires a POSIX system (Linux).

## Devices and caches

GPU IDs refer to the devices visible to the calling process, so they respect
`CUDA_VISIBLE_DEVICES` (and `ROCR_VISIBLE_DEVICES` / `HIP_VISIBLE_DEVICES` on ROCm). Each
worker sees only its own GPU and has private temporary Triton and autotuning caches. If
`RELAY_TUNE_CACHE` points to an existing tuning file, each worker starts from a copy of it; the
original file is never modified. Kernels without a cached entry are autotuned as usual.

## Testing

The CPU tests run anywhere, including CI:

```bash
uv run pytest tests/test_process.py -v
```

The GPU smoke test needs at least two GPUs:

```bash
export PYTHONPATH="$PWD/src"
export RELAY_TUNE_MAX_CFG=4          # bounds autotuning time for the test only
export TRITON_CACHE_DIR="$(mktemp -d)/triton"
python tests/process_gpu_smoke.py --output /tmp/process-smoke
```

It decodes three codes (a toy code and the [[144,12,12]] and [[288,12,18]] bivariate bicycle
codes) on one and two GPUs, and compares every result field with the default single-GPU
decoder on identically split inputs. It also checks that workers are reused across calls,
that outputs stay valid after later calls, that the tuning cache is left unchanged, and that
the calling process never creates a GPU context. It is a correctness check, not a throughput
benchmark.
