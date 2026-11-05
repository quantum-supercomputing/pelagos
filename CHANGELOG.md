<!-- Modified 2026 by AMD to document process-per-GPU execution. -->

# Changelog

All notable changes to Pelagos are documented in this file. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Opt-in process-per-GPU execution (`backend_config={"execution": "process"}` in
  `RelayDecoder`, or `pelagos.process.ProcessRelayBPDecoder`), with one persistent worker per
  GPU, parallel host-side preparation of syndromes and observables, and decoded errors
  returned through shared memory. The default backend and the decoding kernels are unchanged.
  See [docs/process-backend.md](docs/process-backend.md).

## [0.1.0] - Unreleased

First public release.

### Added

- Relay-BP-S and plain min-sum BP decoders for qLDPC codes, as fused Triton kernels
  (`RelayBPDecoder`), batched across syndromes, with `float16`, `float32` and `float64` data types.
- CUDA and ROCm support through Triton; the backend is detected at construction time.
- Batch-independent results: each shot stops on its own `stop_nconv`, and the `gamma_draw`
  option (default `"broadcast"`) shares each relay leg's memory strengths across the batch, so
  a shot decodes the same regardless of batch size or position. `gamma_draw="per_row"` draws
  per shot instead.
- Iteration counts and explicit-gamma rows that match the `relay-bp` Rust reference, and a
  `valid_mask` that is True exactly where `found_count >= stop_nconv`.
- `unconverged_correction` option for shots that find no solution: `"first_leg"` (default,
  as in the Rust reference) returns the hard decision at the end of the pre-iter leg, or of
  relay leg 1 when `pre_iter=0`; `"last_leg"` returns the one from the last leg that ran.
- `DecodeResult.found_count`: the number of solutions found for each shot, reported by both
  `RelayDecoder` detailed-batch methods.
- `RelayDecoder` and `ObservableDecoderRunner` adapters with an interface compatible with the
  `relay-bp` Rust package.
- `CheckMatrices.from_dem` for building decoding problems from stim detector error models.
- `pelagos_benchmark.py` benchmark CLI, with a Rust backend for side-by-side comparison.
- Bundled stim circuits for the [[18,4,3]], [[72,12,6]], [[144,12,12]] and [[288,12,18]]
  bivariate bicycle codes.
- Tutorial 1: a single-GPU quickstart notebook.
- Tests, including bit-exact comparison against the Rust reference (GPU), and CPU-only CI.

[Unreleased]: https://github.com/quantum-supercomputing/pelagos/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/quantum-supercomputing/pelagos/releases/tag/v0.1.0
