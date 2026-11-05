<!-- Modified 2026 by AMD to document process-per-GPU execution. -->

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/logo-dark.svg">
    <img src="docs/logo-light.svg" alt="Pelagos" width="140" height="140">
  </picture>
</p>

<h1 align="center">Pelagos</h1>

<p align="center">
  GPU-accelerated <strong>Relay-BP</strong> decoder for qLDPC codes,
  implemented in <a href="https://github.com/triton-lang/triton">Triton</a>.
</p>

<p align="center">
  <a href="https://github.com/quantum-supercomputing/pelagos/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/quantum-supercomputing/pelagos/actions/workflows/ci.yml/badge.svg?branch=main"></a>
  <a href="LICENSE"><img alt="License: Apache 2.0" src="https://img.shields.io/badge/license-Apache_2.0-blue.svg"></a>
  <img alt="Python 3.11 | 3.12 | 3.13" src="https://img.shields.io/badge/python-3.11_|_3.12_|_3.13-3776AB?logo=python&logoColor=white">
  <a href="https://ieeexplore.ieee.org/document/11662198"><img alt="Paper: IEEE QSW 2026" src="https://img.shields.io/badge/paper-IEEE_QSW_2026-00629B?logo=ieee&logoColor=white"></a>
</p>

<p align="center">
  <a href="https://github.com/triton-lang/triton"><img alt="Triton" src="https://img.shields.io/badge/kernels-Triton-6E4AFF"></a>
  <a href="https://pytorch.org"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white"></a>
  <img alt="AMD ROCm" src="https://img.shields.io/badge/AMD-ROCm-ED1C24?logo=amd&logoColor=white">
  <img alt="NVIDIA CUDA" src="https://img.shields.io/badge/NVIDIA-CUDA-76B900?logo=nvidia&logoColor=white">
  <a href="https://github.com/quantumlib/Stim"><img alt="stim" src="https://img.shields.io/badge/noise_models-stim-4285F4"></a>
  <a href="tutorials/"><img alt="Tutorials: Jupyter" src="https://img.shields.io/badge/tutorials-Jupyter-F37626?logo=jupyter&logoColor=white"></a>
</p>

Pelagos implements min-sum belief propagation with disordered memory
strengths (Relay-BP-S), batched across syndromes and executed as fused
Triton kernels. It is interface-compatible with (and cross-validated
against) the `relay-bp` Rust reference implementation, available on PyPI.

## Hardware support

Pelagos is vendor-neutral: the kernels are written in Triton, which targets
both NVIDIA (CUDA), AMD (ROCm/HIP) and Intel (SPIR-V) GPUs. The decoder 
detects the active backend at construction time and adapts launch parameters 
accordingly.

## Install

```bash
pip install pelagos[gpu]           # decoder + torch/triton
pip install pelagos[gpu,rust]      # + the Rust CPU reference, side by side
pip install pelagos[gpu,stim]      # + stim / DEM support
```

Development uses [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra all
uv run pytest
```

A GPU is required to decode; there is no CPU fallback (see
[Known limitations](#known-limitations)). The package imports and its
structural tests run anywhere.

## Quickstart

```python
import numpy as np
import torch
from scipy.sparse import csr_matrix
from pelagos import RelayBPDecoder

# A tiny code: 3 checks over 3 variables.
H = csr_matrix(np.array([[1, 1, 0],
                          [0, 1, 1],
                          [1, 0, 1]], dtype=np.uint8))
priors = np.full(3, 0.05)

decoder = RelayBPDecoder(
    H_csr=H,
    error_priors=priors,
    pre_iter=80,          # T0: ordered-memory iterations
    num_sets=100,         # R: relay legs
    set_max_iter=60,      # Tr: iterations per leg
    gamma0=0.65,          # ordered memory strength
    gamma_dist_interval=(-0.24, 0.66),
    stop_nconv=5,         # stop after 5 valid solutions
    data_type="float32",
    algo="relay",         # required: "relay" or "plain"
)

# decode() takes a torch GPU tensor, not a numpy array.
syndrome = np.array([[1, 0, 1]], dtype=np.uint8)
syndrome_gpu = torch.from_numpy(syndrome).to("cuda:0")
result = decoder.decode(syndrome_gpu)

# The decoded error pattern is under the "errors" key.
e_hat = result["errors"].cpu().numpy()
print(e_hat)
```

Three details that are easy to get wrong (all enforced by
`tests/test_smoke.py`):

- `algo` must be passed explicitly — `algo="relay"` or `algo="plain"`.
  Omitting it raises `ValueError: Unknown algo 'None'`.
- `decode()` requires a torch tensor already on the GPU
  (`torch.from_numpy(...).to("cuda:0")` — PyTorch uses the `"cuda"` device
  type for ROCm builds as well, so the same call works on AMD),
  not a numpy array.
- The result dict's error-pattern key is `"errors"`, not `"e_hat"`.

### From a stim detector error model

```python
import stim
from pelagos.stim.sinter.check_matrices import CheckMatrices
from pelagos import RelayBPDecoder

dem = stim.Circuit.generated(
    "surface_code:rotated_memory_x", distance=3, rounds=3,
    after_clifford_depolarization=0.001,
).detector_error_model(decompose_errors=False)

cm = CheckMatrices.from_dem(dem)
decoder = RelayBPDecoder(H_csr=cm.check_matrix, error_priors=cm.error_priors, algo="relay")
```

Note the field is `cm.error_priors`, not `cm.priors`. This path requires
the `stim` extra (`pip install pelagos[stim]`), which pulls in
`beliefmatching`.

## Tutorials

Step-by-step Jupyter notebooks live in [`tutorials/`](tutorials/). Start with
[Tutorial 1: single-GPU quickstart](tutorials/01_single_gpu_quickstart.ipynb). It goes
from a stim circuit to a logical error rate on the [[72,12,6]] bivariate bicycle code,
compares plain BP with Relay-BP, and measures throughput:

```bash
uv sync --extra tutorials            # or: pip install -e ".[tutorials]"
uv run jupyter lab tutorials/
```

## Benchmarking

`pelagos_benchmark.py` is a full-featured benchmark CLI. Run
`python pelagos_benchmark.py --help` for the complete flag list; the
important ones are `--backend {rust,triton}`, `--circuit`, `--basis`,
`--distance`, `--rounds`, `--error-rate`, `--data-type`, and `--gpus`. It
needs `pandas`, which ships in the `bench` extra:

```bash
pip install "pelagos[gpu,bench]"     # or: uv sync --extra gpu --extra bench
```

```bash
python pelagos_benchmark.py --backend triton \
    --circuit bicycle_bivariate_144_12_12_memory_X --basis x \
    --distance 12 --rounds 12 --error-rate 0.003 --data-type float32
```

There is no single `--compare-rust` flag; to cross-check against the Rust
reference, run the same command twice with `--backend rust` and
`--backend triton` (the `rust` extra must be installed for the former).

## Included circuits

`tests/testdata/bicycle_bivariate/` ships gzipped stim circuits for the
bivariate bicycle codes [[18,4,3]], [[72,12,6]], [[144,12,12]], and
[[288,12,18]], in `X` and `choi_XZ` bases, at physical error rates 0.002,
0.003, and 0.005 (24 files, 2.4 MB total).

`stim.Circuit.from_file` does not transparently decompress gzip, so the
bundled loader in `tests/testdata/circuits.py` handles both forms:

```python
import sys; sys.path.insert(0, "tests")
from testdata import get_test_circuit

circuit = get_test_circuit(
    "bicycle_bivariate_144_12_12_memory_X", 0.003, distance=12, rounds=12
)
dem = circuit.detector_error_model(decompose_errors=False)
```

## Testing

Continuous integration runs on GitHub-hosted runners, which have **no GPU**. On every pull
request, CI checks packaging, imports, lint (syntax errors and undefined names), the
CPU-side code (input validation, bit packing, the conversion from a stim DEM to check
matrices) and the well-formedness of the tutorial notebooks. The decoder tests, including
the bit-exact comparison against the Rust reference, **are skipped in CI**.

Before each release, and for any pull request that touches the kernels or the decoder, a
maintainer runs the full suite on a GPU:

```bash
uv sync --extra gpu --extra rust --extra stim --extra test
uv run pytest tests/ -v -rs          # nothing should be reported as skipped
```

## Known limitations

- **A GPU is required.** There is no CPU fallback; use the Rust
  implementation (`relay-bp` on PyPI) for CPU decoding.
- **Multi-GPU support is limited to one node.** The default multi-device path
  (`device=[0, 1, ...]`, `--gpus`) runs one thread per GPU in a single process and
  does not scale linearly with the number of GPUs. The opt-in
  [process-per-GPU backend](docs/process-backend.md) runs one worker process per GPU
  instead; its smoke test checks results against the single-GPU decoder on two GPUs.
  Scaling across nodes is not supported.

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for the development setup,
testing, and sign-off requirements. Everyone taking part is expected to follow the
[Code of Conduct](CODE_OF_CONDUCT.md). Please report security issues privately, as described in
[SECURITY.md](SECURITY.md). Release notes are in [CHANGELOG.md](CHANGELOG.md).

## Governance

Pelagos is maintained by contributors from IBM and AMD under a lightweight,
maintainer-led model. See [GOVERNANCE.md](GOVERNANCE.md) for how decisions are made and
[MAINTAINERS.md](MAINTAINERS.md) for the current maintainers.

## Citation

If you use Pelagos in your research, please cite:

> D. Diamantopoulos, L. von Niederhäusern, A. S. Anghel, A. Kayi, R. Pletka, T. Müller,
> N. Papandreou, J. Blanusa, S. Sarafijanovic, N. M. Gonzalez, F. Abel, C. Hagleitner,
> S. Seelam, B. Johnson, R. Haas, and H. Pozidis, "Pêlagos: An Adaptive GPU Decoder for
> Error Correction Simulations of Quantum LDPC Codes," in *2026 IEEE International
> Conference on Quantum Software (QSW)*, 2026, pp. 152–165.
> [IEEE Xplore](https://ieeexplore.ieee.org/document/11662198)

```bibtex
@INPROCEEDINGS{11662198,
  author={Diamantopoulos, Dionysios and von Niederhäusern, Luc and Anghel, Andreea Simona and Pletka, Roman and Kayi, Apo and Müller, Tristan and Papandreou, Nikolaos and Blanusa, Jovan and Sarafijanovic, Slavisa and Gonzalez, Nelson Mimura and Abel, Francois and Hagleitner, Christoph and Seelam, Seetharami and Johnson, Blake and Haas, Robert and Pozidis, Haralampos},
  booktitle={2026 IEEE International Conference on Quantum Software (QSW)},
  title={Pêlαγos: An Adaptive GPU Decoder for Error Correction Simulations of Quantum LDPC Codes},
  year={2026},
  pages={152-165},
  doi={10.1109/QSW72780.2026.00025}}

```

## License

Pelagos is jointly owned by IBM and AMD.

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).
