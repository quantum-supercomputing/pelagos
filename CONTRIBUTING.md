# Contributing to Pelagos

Thank you for your interest in Pelagos! Bug reports, fixes, new tutorials, and performance
improvements are all welcome. By taking part you agree to follow our
[Code of Conduct](CODE_OF_CONDUCT.md). How decisions are made is described in
[GOVERNANCE.md](GOVERNANCE.md).

## Reporting issues

Please open a [GitHub issue](https://github.com/quantum-supercomputing/pelagos/issues)
using one of the templates. For a bug, include the GPU model, the driver version, the
backend (CUDA or ROCm), and a minimal reproducer. Report security vulnerabilities privately,
as described in [SECURITY.md](SECURITY.md), not in a public issue.

## Development setup

Pelagos uses [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/quantum-supercomputing/pelagos.git
cd pelagos
uv sync --extra all
```

## Testing

CI runs on GitHub-hosted runners, which have no GPU. It runs the lint step and the CPU tests;
the decoder tests are skipped there. Before you open a pull request, run the checks CI runs:

```bash
uvx ruff check --isolated --select E9,F63,F7,F82 src tests tutorials pelagos_benchmark.py
uv run pytest tests/ -v -rs
```

If your change touches the kernels, the decoder, or the adapter, also run the full suite on a
GPU (CUDA or ROCm) with the `gpu`, `rust`, `stim` and `test` extras installed. Nothing should
be reported as skipped. Say in the pull request which GPU you used. If you do not have a GPU,
say so, and a maintainer will run it.

If you change a tutorial notebook, re-run it top to bottom and commit it with its outputs.

## Pull requests

* Keep each pull request focused on one change, and add or update tests where possible.
* Match the style of the surrounding code.
* New source files carry an Apache 2.0 license header, either the IBM header used throughout
  the repository or your own organization's copyright line with
  `SPDX-License-Identifier: Apache-2.0`. When you modify a file, add a line to its header
  noting the change.
* Every commit must be signed off (see below).
* A maintainer reviews every pull request. CI must pass before merging.

## Developer Certificate of Origin

Pelagos uses the [Developer Certificate of Origin](https://developercertificate.org/)
(DCO). By signing off a commit, you certify that you wrote the change, or otherwise have the
right to submit it under the project's [Apache 2.0 license](LICENSE). To sign off, add a
`Signed-off-by` line to each commit message, which `git commit -s` does for you:

```
Signed-off-by: Jane Doe <jane.doe@example.com>
```

The name and email must match the commit author. To sign off commits you already made, run
`git rebase --signoff main` and force-push your branch.

## License

Contributions are licensed under the [Apache License, Version 2.0](LICENSE).
