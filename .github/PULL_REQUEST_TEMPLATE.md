## Summary

<!-- What does this change, and why? Link related issues, e.g. "Fixes #123". -->

## Checklist

- [ ] Commits are signed off (`git commit -s`); see [CONTRIBUTING.md](../CONTRIBUTING.md).
- [ ] Tests added or updated where it makes sense.
- [ ] `uv run pytest tests/ -v -rs` passes locally.
- [ ] **If kernels, decoder, or adapter changed:** full suite run on a GPU with no skips.
      GPU used: <!-- e.g. NVIDIA A100 / AMD MI300X, or "none, maintainer please run" -->
- [ ] Tutorial notebooks re-run and committed with outputs (if changed).
