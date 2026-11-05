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

\* Measured on a laptop RTX 3000 Ada (8 GB). A GPU (CUDA or ROCm) is required.
