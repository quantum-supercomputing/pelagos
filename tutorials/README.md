<!-- Modified 2026 by AMD to add the multi-GPU quickstart. -->

# Tutorials

Hands-on Jupyter notebooks for Pelagos. Run them from a clone of the repository, because they
load the stim circuits bundled in `tests/testdata/`.

```bash
uv sync --extra tutorials          # or: pip install -e ".[tutorials]"
uv run jupyter lab tutorials/
```

Add the `rust` extra (`uv sync --extra tutorials --extra rust`) to enable the optional
cross-check against the `relay-bp` CPU reference.

| # | Notebook | What you learn | Runtime* |
|---|---|---|---|
| 1 | [Single-GPU quickstart](01_single_gpu_quickstart.ipynb) | Decoder API; from a stim circuit to a logical error rate on the [[72,12,6]] bivariate bicycle code; plain BP vs. Relay-BP; throughput vs. batch size; sweeping the physical error rate; comparing against the Rust reference | ~3–4 min |
| 2 | [Multi-GPU quickstart](02_multi_gpu_quickstart.ipynb) | Process-per-GPU decoding; host inputs and outputs; syndrome and logical-error checks; scaling with 1, 2, 4, and 8 GPUs at 1,024 shots per GPU | Depends on GPU count and hardware |

\* Measured on a laptop RTX 3000 Ada (8 GB). A GPU (CUDA or ROCm) is required.

Tutorial 2 uses up to eight AMD GPUs with ROCm on the same machine, with at
least two needed for the multi-GPU examples. It adapts to the available GPU count.
