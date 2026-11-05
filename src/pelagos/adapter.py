# (C) Copyright IBM 2026
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
#
# Modified 2026 by AMD: opt-in process-per-GPU dispatch and parallel preparation.

"""
Triton adapter for Relay-BP decoder providing compatibility with relay_bp interface.

This module provides adapter classes that wrap the Triton GPU implementation
to match the interface expected by the relay_bp package, enabling seamless
integration with existing benchmarking and analysis tools.

The adapter handles:
- Interface compatibility between Triton and Rust backends
- Observable decoding with error detection
- Batch processing with detailed iteration tracking
- Matrix format conversion and validation
"""

import numpy as np
import torch
from dataclasses import dataclass
from typing import Dict, List, Optional

try:
    import scipy.sparse as sp
except Exception:
    sp = None

from .decoder import RelayBPDecoder as TritonDecoder


@dataclass
class DecodeResult:
    """Result container for decoding with detailed statistics.

    This is intentionally shaped to look like the Rust ``DecodeResult`` /
    ``ObservableDecodeResult`` exposed through the relay_bp Python bindings.
    This allows higher-level code to treat Triton
    and Rust results uniformly.

    Canonical (Rust-compatible) fields:
        decoding: Physical error pattern (Array1<Bit> in Rust)
        success:  Whether the decoder successfully satisfied all checks
        iterations: Number of BP iterations performed

    Additional Triton-only convenience fields:
        converged: Whether the lane converged according to Relay-BP stopping rule
        error_detected: Whether a logical error was detected (from observables mismatch)
        observables: Predicted observable values
        unconverged_indices: Optional batch-level array of indices for unconverged syndromes.
        found_count: Number of valid solutions found for the shot (0..stop_nconv). Set by the
            detailed-batch methods of RelayDecoder.
    """

    # Rust-compatible public API
    decoding: np.ndarray
    success: bool
    iterations: int

    # Quality metric
    decoding_quality: float = float("inf")

    # Only populated when true errors are available (from_errors path).
    unconverged_no_error: Optional[bool] = None
    unconverged_logical_error: Optional[bool] = None
    better_decoding_quality_error: Optional[bool] = None
    worse_decoding_quality_error: Optional[bool] = None

    # Triton-specific extras
    converged: bool = False
    error_detected: bool = False
    observables: np.ndarray = None  # type: ignore[assignment]
    unconverged_indices: Optional[np.ndarray] = None
    found_count: Optional[int] = None



class RelayDecoder:
    """Adapter for Triton Relay-BP decoder providing relay_bp interface compatibility.

        This class wraps the Triton GPU implementation to match the interface expected
    by the relay_bp package, enabling seamless integration with existing tools.

    The adapter handles matrix format conversion, parameter mapping, and provides
    the same interface as the Rust implementation for benchmarking consistency.
    """

    def __init__(
        self,
        check_matrix,
        *,
        # High-level configuration dictionaries
        msl_config: Optional[dict] = None,
        relay_config: Optional[dict] = None,
        data_type: Optional[str] = None,
        backend_config: Optional[dict] = None,
        **kwargs,
    ):
        """Create a Triton Relay-BP decoder.

        High-level *config* style:
            - ``msl_config``: Min-sum configuration dict (error_priors, alpha, gamma0, ...)
            - ``relay_config``: Relay-BP configuration dict
            - ``data_type``: decoder numeric type ("float", "float32", "int", ...)
            - ``backend_config``: Triton backend-specific knobs (dtype, perf, device, fast_max_unconverged_pct, ...)
        """
        if msl_config is None or relay_config is None:
            raise ValueError("msl_config and relay_config must be provided to Triton RelayDecoder")

        msl_config = dict(msl_config)
        relay_config = dict(relay_config)
        backend_config = dict(backend_config or {})

        # Min-sum / memory configuration
        error_priors = msl_config.get("error_priors")
        if error_priors is None:
            raise ValueError("msl_config must contain 'error_priors'")

        alpha = msl_config.get("alpha", None)
        beta = msl_config.get("beta", None)
        alpha_iteration_scaling_factor = msl_config.get("alpha_iteration_scaling_factor", 1.0)
        gamma0 = msl_config.get("gamma0", 0.1)

        # Relay configuration
        pre_iter = relay_config.get("pre_iter", 80)
        num_sets = relay_config.get("num_sets", 300)
        set_max_iter = relay_config.get("set_max_iter", 60)
        gamma_dist_interval = tuple(relay_config.get("gamma_dist_interval", (-0.24, 0.66)))
        explicit_gammas = relay_config.get("explicit_gammas", None)
        gamma_dist_interval_by_error = relay_config.get("gamma_dist_interval_by_error", None)
        stop_nconv = int(relay_config.get("stop_nconv", 1))
        self.stop_nconv = stop_nconv
        seed = relay_config.get("seed", 0)
        gamma_draw = str(relay_config.get("gamma_draw", "broadcast"))
        # Correction for shots with no valid solution: "first_leg" or "last_leg".
        unconverged_correction = str(relay_config.get("unconverged_correction", "first_leg"))

        # Backend/Triton-specific configuration
        data_type_cfg = backend_config.get("data_type", data_type)
        # Normalize common aliases
        _dtype_aliases = {
            "float": "float32",
            "half": "float16",
            "double": "float64",
            "fp16": "float16",
            "fp32": "float32",
            "fp64": "float64",
        }
        data_type_cfg = _dtype_aliases.get(data_type_cfg, data_type_cfg)
        if data_type_cfg not in ("float16", "float32", "float64"):
            raise ValueError(f"Unsupported data_type for Triton backend: {data_type_cfg!r}")
        self.data_type = data_type_cfg

        perf = backend_config.get("perf", "throughput")
        device = backend_config.get("device", (0,))
        # Normalize device specification so that the underlying Triton decoder
        # always sees a type it can handle. Its _parse_device_spec expects a
        # List[int], int, or str. Older call sites (and some defaults here)
        # may use a tuple, so we convert tuples to lists.
        # Also handle edge case where device might be integer 0 (should be [0])
        if isinstance(device, tuple):
            device = list(device)
        elif isinstance(device, int):
            # If device is a single integer, treat it as a single GPU ID [device]
            # (not "auto-detect device GPUs" which would fail for device=0)
            device = [device]
        fast_max_unconverged_pct = backend_config.get("fast_max_unconverged_pct", None)
        bitpack_output = bool(backend_config.get("bitpack_output", False))
        scheduling = backend_config.get("scheduling", "flood-fill")
        layered_k_iters = backend_config.get("layered_k_iters", 0)
        layered_block_policy = backend_config.get("layered_block_policy", "contiguous")
        layered_num_blocks = backend_config.get("layered_num_blocks", None)
        layered_target_edges_per_block = backend_config.get("layered_target_edges_per_block", None)
        layered_block_seed = backend_config.get("layered_block_seed", 42)

        # Algorithm mode: "relay" (default) or "plain"
        algo = backend_config.get("algo", "relay")
        # Convert input to CSR matrix format (handles various sparse matrix types)
        if sp is not None:
            self.H_csr = sp.csr_matrix(check_matrix)
        else:
            # Fallback for environments without SciPy (not recommended)
            H = np.asarray(check_matrix, dtype=np.uint8)
            from scipy.sparse import csr_matrix as _csr  # raises if SciPy missing

            self.H_csr = _csr(H)

        self.N = int(self.H_csr.shape[1])


        _plain = num_sets == 0
        _gamma_interval = (0.0, 0.0) if _plain else tuple(gamma_dist_interval)

        # Initialize underlying Triton decoder with mapped parameters
        execution = backend_config.get("execution", "thread")
        decoder_type = TritonDecoder
        process_options = {}
        if execution == "process":
            from .process import ProcessRelayBPDecoder
            decoder_type = ProcessRelayBPDecoder
            process_options["worker_timeout_s"] = backend_config.get("worker_timeout_s", 300)
        elif execution != "thread":
            raise ValueError("execution must be 'thread' or 'process'")
        self._process_execution = execution == "process"
        self._dec = decoder_type(
            self.H_csr,
            error_priors=error_priors,
            pre_iter=int(pre_iter),
            num_sets=int(num_sets),
            set_max_iter=int(set_max_iter),
            gamma0=float(gamma0),
            gamma_dist_interval=_gamma_interval,
            stop_nconv=int(stop_nconv),
            normalized_min_sum_alpha=alpha,
            offset_min_sum_beta=beta,
            data_type=self.data_type,
            device=device,
            seed=int(seed) if seed is not None else 0,
            gamma_draw=gamma_draw,
            bitpack_output=bitpack_output,
            algo=algo,
            perf=perf,
            alpha_iteration_scaling_factor=float(alpha_iteration_scaling_factor),
            explicit_gammas=explicit_gammas,
            gamma_dist_interval_by_error=gamma_dist_interval_by_error,
            fast_max_unconverged_pct=fast_max_unconverged_pct,
            scheduling=scheduling,
            layered_k_iters=layered_k_iters,
            layered_block_policy=layered_block_policy,
            layered_num_blocks=layered_num_blocks,
            layered_target_edges_per_block=layered_target_edges_per_block,
            layered_block_seed=layered_block_seed,
            unconverged_correction=unconverged_correction,
            **process_options,
        )

    def close(self):
        """Release persistent workers when using process execution."""
        if self._process_execution:
            self._dec.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _decode_inner(self, S_torch):
        """Single seam: dispatch to the Triton decoder.

        Returns a result dict with the keys consumed by the rest of the
        adapter (errors, iterations, found_count, valid_mask,
        unconverged_indices).
        """
        return self._dec.decode(S_torch)

    def log_prior_ratios(self) -> np.ndarray:
        """Return the decoder's log-prior-ratios: ln((1-p)/p).

        These match the Rust ``Decoder::log_prior_ratios()`` semantics and
        are used for quality metric computation.  Non-finite values
        are already zeroed during decoder initialisation.
        """
        return self._dec.wj.cpu().numpy().astype(np.float64)

    def override_llr_columns(self, cols, value: float):
        """Override prior LLR on given columns (see TritonDecoder.override_llr_columns).

        Returns a restore-callable, so a caller can patch per-column
        priors for a round and then revert them.
        """
        return self._dec.override_llr_columns(cols, value)

    def decode(self, syndrome: np.ndarray | torch.Tensor):
        """Decode syndrome using Triton GPU implementation.

        Args:
            syndrome: Syndrome vector as numpy array or torch tensor

        Returns:
            Dict containing decoding results with relay_bp interface compatibility.
            If fast_max_unconverged_pct is set in decoder, also includes "unconverged_indices" key.
        """
        # Triton decoder expects batched input; wrap single syndrome as batch of size 1.
        # Keep on CPU. decode() handles GPU placement.
        if isinstance(syndrome, np.ndarray):
            s = torch.from_numpy(syndrome.astype(np.uint8, copy=False)).view(1, -1)
        else:
            s = syndrome.to(dtype=torch.uint8).view(1, -1)

        out = self._dec.decode(s)
        bits = out["errors"][0].to(torch.int8)
        # Use found_count > 0 for converged (at least one valid solution found).
        converged = bool(out["found_count"][0].item() > 0)
        iters = int(out.get("iterations", torch.tensor([self._dec.pre_iter]))[0].item())

        result = {
            "bits": bits,
            "posterior_llr": None,
            "converged": converged,
            "num_ensembles": 0,
            "iterations": iters,
            "selected_idx": 0,
        }
        # Include unconverged_indices if present (when fast_max_unconverged_pct is set)
        if "unconverged_indices" in out:
            result["unconverged_indices"] = out["unconverged_indices"].cpu().numpy()
        return result

    def decode_detailed(self, detectors: np.ndarray | torch.Tensor):
        """Decode detectors with detailed results (matches Rust interface).

        Args:
            detectors: Detector vector as numpy array or torch tensor

        Returns:
            DecodeResult object with detailed decoding information
        """
        result = self.decode(detectors)
        bits = result["bits"].cpu().numpy()
        converged = bool(result["converged"])
        iters = int(result["iterations"])

        # For single-shot decode we expose "decoding" as the physical error pattern and
        # "success" as convergence. This matches the Rust DecodeResult fields consumed
        # by BaseRelayDecoder._produce_output.
        return DecodeResult(
            decoding=bits,
            success=converged,
            iterations=iters,
            converged=converged,
            error_detected=not converged,
            observables=bits,
            unconverged_indices=result.get("unconverged_indices"),
        )

    def decode_batch(self, detectors: np.ndarray | torch.Tensor, return_iterations=False):
        """Decode batch of detectors (matches Rust interface).

        Args:
            detectors: Batch of detector vectors as numpy array or torch tensor
            return_iterations: when True, also return the per-shot BP iteration

        Returns:
            Batch of decoded error vectors as numpy array.
            If fast_max_unconverged_pct is set in decoder, returns tuple (errors, unconverged_indices).
            If return_iterations is True, the per-shot iteration array is appended.
        """
        # Keep on CPU. decode() handles GPU placement.
        if isinstance(detectors, np.ndarray):
            detectors = torch.from_numpy(detectors.astype(np.uint8, copy=False))
        else:
            detectors = detectors.to(dtype=torch.uint8)

        out = self._dec.decode(detectors)
        errors = out["errors"].to(torch.int8).cpu().numpy()
        result = [errors]
        if self._dec.fast_max_unconverged_pct is not None and "unconverged_indices" in out:
            result.append(out["unconverged_indices"].cpu().numpy())
        if return_iterations:
            iters_t = out.get("iterations")
            if iters_t is None:
                iters = np.full(errors.shape[0], int(self._dec.pre_iter), dtype=np.int32)
            else:
                iters = iters_t.to(torch.int32).cpu().numpy()
            result.append(iters)
        return result[0] if len(result) == 1 else tuple(result)

    def decode_batch_with_prior_tensors(
        self, detectors: torch.Tensor, prior_llr: torch.Tensor, *, return_iterations=False
    ):
        """Decode a batch with a per-shot [B, V] prior LLR, returning also the
        posterior. GPU-resident.
        """
        out = self._dec.decode(
            detectors, return_last_posterior=True, prior_llr=prior_llr
        )
        errors = out["errors"]
        post = out["last_posterior"]
        if return_iterations:
            iters_t = out.get("iterations")
            if iters_t is None:
                iters_t = torch.full(
                    (detectors.shape[0],),
                    int(self._dec.pre_iter),
                    dtype=torch.int32,
                    device=errors.device,
                )
            return errors, post, iters_t
        return errors, post

    def decode_detailed_batch(self, detectors: np.ndarray | torch.Tensor):
        """Decode batch of detectors with detailed results (matches Rust interface).

        Args:
            detectors: Batch of detector vectors as numpy array or torch tensor

        Returns:
            List of DecodeResult objects with detailed decoding information.
            If fast_max_unconverged_pct is set in decoder, returns tuple (results, unconverged_indices).
        """
        # Keep on CPU. decode() handles GPU placement.
        if isinstance(detectors, np.ndarray):
            detectors = torch.from_numpy(detectors.astype(np.uint8, copy=False))
        else:
            detectors = detectors.to(dtype=torch.uint8)

        out = self._dec.decode(detectors)

        # Extract unconverged_indices if available
        unconverged_indices = None
        if self._dec.fast_max_unconverged_pct is not None and "unconverged_indices" in out:
            unconverged_indices = out["unconverged_indices"].cpu().numpy()

        # Use found_count > 0 for success (at least one valid solution found).
        B = detectors.shape[0]
        found_count = out["found_count"].cpu().numpy()
        success_mask = found_count > 0

        results: List[DecodeResult] = []
        for i in range(B):
            bits = out["errors"][i].to(torch.int8)
            success = bool(success_mask[i])
            iters = int(
                out.get(
                    "iterations",
                    torch.tensor([self._dec.pre_iter]),
                )[i].item()
            )
            bits_np = bits.cpu().numpy()
            results.append(
                DecodeResult(
                    decoding=bits_np,
                    success=success,
                    iterations=iters,
                    converged=success,
                    error_detected=not success,
                    observables=bits_np,
                    unconverged_indices=unconverged_indices,
                    found_count=int(found_count[i]),
                )
            )
        return results

    def decode_detailed_batch_from_syndromes(
        self, detectors: np.ndarray | torch.Tensor
    ) -> List[DecodeResult]:
        """Decode a batch of detector syndromes and return Rust-compatible results.

        This is the primary entry point for higher-level decoders
        that expect DecodeResult-like objects given detector error model syndromes.

        Success/converged is determined by found_count > 0 (at least one valid solution found)
        """
        # Keep on CPU. decode() handles GPU placement.
        if isinstance(detectors, np.ndarray):
            S_torch = torch.from_numpy(detectors.astype(np.uint8, copy=False))
        else:
            S_torch = detectors.to(dtype=torch.uint8)

        out = self._decode_inner(S_torch)
        B = S_torch.shape[0]


        errors = out["errors"].to(torch.int8)
        iters = out.get(
            "iterations",
            torch.full((B,), int(self._dec.pre_iter), dtype=torch.int32),
        )
        found_count = out["found_count"].cpu().numpy()
        success_mask = found_count > 0

        # Also get unconverged_indices for compatibility (used by fast_max_unconverged_pct feature)
        unconverged_indices_gpu = out.get("unconverged_indices", None)
        unconverged_indices_np = None
        if unconverged_indices_gpu is not None and unconverged_indices_gpu.numel() > 0:
            unconverged_indices_np = unconverged_indices_gpu.cpu().numpy()

        # Transfer decoded errors to CPU and recompute quality
        all_bits = errors.cpu().numpy()  # (B, V) int8
        all_iters = iters.cpu().numpy()
        wj_np = self.log_prior_ratios()  # (N,) float64


        results: List[DecodeResult] = []
        for i in range(B):
            bits_np = all_bits[i]
            success = bool(success_mask[i])
            iterations = int(all_iters[i])
            # Recompute quality
            dq = float(np.dot(bits_np.astype(np.float64), wj_np)) if success else float("inf")
            result = DecodeResult(
                decoding=bits_np,
                success=success,
                iterations=iterations,
                decoding_quality=dq,
                converged=success,
                error_detected=not success,
                observables=bits_np,
                unconverged_indices=unconverged_indices_np,
                found_count=int(found_count[i]),
            )
            results.append(result)
        return results


class ObservableDecoderRunner:
    """Observable decoder runner providing batch processing with error detection.

    This class handles batch decoding of error patterns and computes observable
    predictions with logical error detection, matching the relay_bp interface.
    """

    def __init__(
        self,
        decoder: RelayDecoder,
        observables_matrix,
        include_decode_result: bool = True,
        **kwargs,
    ):
        self.decoder = decoder
        if sp is not None and sp.issparse(observables_matrix):
            self._obs = observables_matrix.tocsr()
        else:
            self._obs = (
                sp.csr_matrix(observables_matrix)
                if sp is not None
                else np.asarray(observables_matrix, dtype=np.uint8)
            )
        self.O, self.N = self._obs.shape
        self.M = int(decoder.H_csr.shape[0])
        # Cache for batched concatenation to avoid allocation overhead
        self._cached_concat_errors = None
        self._cached_concat_iterations = None
        self._cached_concat_valid = None

    def _mul_mod2(self, A_csr, x: np.ndarray | torch.Tensor) -> np.ndarray:
        """Compute matrix-vector multiplication modulo 2 for observable computation.

        Args:
            A_csr: Sparse matrix (observables or check matrix)
            x: Vector (error pattern or decoded bits)

        Returns:
            Result of A @ x mod 2 as uint8 array
        """
        if sp is not None and sp.issparse(A_csr):
            y = A_csr @ (np.asarray(x, dtype=np.int8))
            return (np.asarray(y) % 2).astype(np.uint8)
        A = np.asarray(A_csr, dtype=np.uint8)
        y = (A @ np.asarray(x, dtype=np.int8)) % 2
        return y.astype(np.uint8)

    def from_errors_decode_observables_detailed_batch(
        self, errors: np.ndarray, parallel: bool = False
    ) -> List[DecodeResult]:
        """Decode batch of error patterns and compute observable predictions with error detection.

        This method implements the core benchmarking functionality, processing a batch
        of error patterns and returning detailed statistics for each pattern including
        iteration counts, convergence status, and logical error detection.

        Args:
            errors: [B,N] array of error patterns to decode
            parallel: Whether to use parallel processing (unused in Triton implementation)

        Returns:
            List of DecodeResult objects containing detailed decoding statistics.
            If fast_max_unconverged_pct is set and unconverged cases exist, the list will have an
            'unconverged_indices' attribute containing the indices of unconverged syndromes.
        """
        # Use new methods to avoid code duplication
        gpu_result = self._decode_gpu_only(errors)
        return self._process_gpu_results(gpu_result)

    def _decode_gpu_only(self, errors: np.ndarray) -> Dict[str, torch.Tensor]:
        """Decode batch on GPU and return GPU tensors (no CPU transfer).

        Args:
            errors: [B,N] array of error patterns to decode

        Returns:
            Dictionary with GPU tensors: errors, true_obs, and decoder output dict
        """
        B, N = errors.shape
        assert N == self.decoder.N, f"N mismatch: errors has {N}, decoder has {self.decoder.N}"

        if self.decoder._process_execution:
            from .process import prepare_batch
            true_obs, syndromes = prepare_batch(
                self.decoder.H_csr, self._obs, errors, self.decoder._dec.num_gpus,
                self._mul_mod2,
            )
            return {"out": self.decoder._decode_inner(torch.from_numpy(syndromes)),
                    "true_obs": true_obs, "errors": errors}

        # Compute true observables from error patterns
        true_obs = self._mul_mod2(self._obs, errors.T).T  # (B,O)

        # Build syndromes from errors via H @ e mod 2
        H = self.decoder.H_csr
        if sp is not None and sp.issparse(H):
            S = (H @ errors.T) % 2
            S = np.asarray(S, dtype=np.uint8).T
        else:
            H_np = np.asarray(H, dtype=np.uint8)
            S = ((H_np @ errors.T) % 2).astype(np.uint8).T

        # If decoder uses check permutation, apply it to syndromes to match permuted graph
        # The graph was built with H[check_permutation, :], so syndromes need to be permuted too
        if self.decoder._dec.check_permutation is not None:
            perm = self.decoder._dec.check_permutation
            S = S[:, perm]

        # Decode on GPU (multi-GPU handled transparently by decode()).
        # Keep tensor on CPU here. decode() places each GPU's slice directly,
        # avoiding a redundant CPU -> GPU 0 -> GPU N round-trip.
        S_torch = torch.from_numpy(S)
        out = self.decoder._decode_inner(S_torch)

        return {
            "out": out,
            "true_obs": true_obs,
            "errors": errors,
        }


    def _process_gpu_results(self, gpu_result: Dict[str, torch.Tensor]) -> List[DecodeResult]:
        """Transfer GPU results to CPU and process into DecodeResult list.

        Args:
            gpu_result: Dictionary from _decode_gpu_only() with out, true_obs, errors

        Returns:
            List of DecodeResult objects (same as from_errors_decode_observables_detailed_batch)
        """
        out = gpu_result["out"]
        true_obs = gpu_result["true_obs"]
        errors = gpu_result["errors"]
        B = errors.shape[0]

        # Batch CPU transfers (single sync point)
        errors_gpu = out["errors"].to(torch.uint8)
        iterations_gpu = out.get(
            "iterations",
            torch.full(
                (B,),
                int(self.decoder._dec.pre_iter),
                device=out["errors"].device,
                dtype=torch.int32,
            ),
        ).to(torch.int32)
        found_count_gpu = out["found_count"].to(torch.int32)
        unconverged_indices_gpu = out.get("unconverged_indices", None)

        # Single batched transfer (all tensors transferred together)
        bits = errors_gpu.cpu().numpy()
        iters = iterations_gpu.cpu().numpy()
        found_count = found_count_gpu.cpu().numpy()

        # Use found_count > 0 for success (at least one valid solution found).
        success_mask = found_count > 0

        unconverged_indices = None
        if (
            self.decoder._dec.fast_max_unconverged_pct is not None
            and unconverged_indices_gpu is not None
        ):
            unconverged_indices = unconverged_indices_gpu.cpu().numpy()

        pred_obs = self._mul_mod2(self._obs, bits.T).T

        # Compute quality metrics
        # Rust: get_decoding_quality(errors) = sum(log_prior_ratios[i] for i where errors[i]==1 and is_finite)
        # Our log_prior_ratios() already zeros non-finite entries, so the dot product is equivalent.
        wj_np = self.decoder.log_prior_ratios()  # (N,) float64
        decoded_quality = (bits.astype(np.float64) * wj_np).sum(axis=1)  # (B,) float64
        true_quality = (errors.astype(np.float64) * wj_np).sum(axis=1)  # (B,) float64

        res: List[DecodeResult] = []
        for i in range(B):
            err_det = bool((true_obs[i] != pred_obs[i]).any())
            success = bool(success_mask[i])
            it = int(iters[i])
            obs = pred_obs[i].astype(np.uint8)
            dq = float(decoded_quality[i]) if success else float("inf")
            better_quality = dq < true_quality[i]
            res.append(
                DecodeResult(
                    decoding=bits[i],
                    success=success,
                    iterations=it,
                    decoding_quality=dq,
                    unconverged_no_error=bool(not err_det and not success),
                    unconverged_logical_error=bool(err_det and not success),
                    better_decoding_quality_error=bool(success and err_det and better_quality),
                    worse_decoding_quality_error=bool(success and err_det and not better_quality),
                    converged=success,
                    error_detected=err_det,
                    observables=obs,
                    unconverged_indices=unconverged_indices,
                )
            )

        return res

    def _process_gpu_results_batched(
        self,
        chunk_gpu_results: List[Dict[str, torch.Tensor]],
        chunk_errors_list: List[np.ndarray],
        chunk_true_obs_list: List[np.ndarray],
    ) -> List[List[DecodeResult]]:
        """Process multiple GPU results with a single batched transfer.

        Args:
            chunk_gpu_results: List of GPU result dictionaries from _decode_gpu_only()
            chunk_errors_list: List of error arrays (CPU, already available)
            chunk_true_obs_list: List of true observable arrays (CPU, already available)

        Returns:
            List of DecodeResult lists (one per batch)
        """
        if not chunk_gpu_results:
            return []

        # Concatenate all batches for single transfer
        # Ensure all tensors are on the same device before concatenation
        errors_list = []
        iterations_list = []
        found_count_list = []
        batch_sizes = []

        for r in chunk_gpu_results:
            B = r["out"]["errors"].shape[0]
            batch_sizes.append(B)
            errors_list.append(r["out"]["errors"].to(torch.uint8))
            iterations_list.append(
                r["out"]
                .get(
                    "iterations",
                    torch.full(
                        (B,),
                        int(self.decoder._dec.pre_iter),
                        device=r["out"]["errors"].device,
                        dtype=torch.int32,
                    ),
                )
                .to(torch.int32)
            )
            found_count_list.append(r["out"]["found_count"].to(torch.int32))

        # Concatenate using pre-allocated buffers to avoid allocation overhead
        total_B = sum(batch_sizes)
        V = errors_list[0].shape[1]
        device_str = str(errors_list[0].device)

        # Pre-allocate buffers if needed
        if self._cached_concat_errors is None or self._cached_concat_errors.shape[0] < total_B:
            self._cached_concat_errors = torch.empty(
                (total_B, V), dtype=torch.uint8, device=device_str
            )
            self._cached_concat_iterations = torch.empty(
                (total_B,), dtype=torch.int32, device=device_str
            )
            self._cached_concat_valid = torch.empty(
                (total_B,), dtype=torch.int32, device=device_str
            )

        # Copy into pre-allocated buffers
        offset = 0
        for err, it, fc in zip(errors_list, iterations_list, found_count_list):
            B = err.shape[0]
            self._cached_concat_errors[offset : offset + B] = err
            self._cached_concat_iterations[offset : offset + B] = it
            self._cached_concat_valid[offset : offset + B] = fc
            offset += B

        all_errors_gpu = self._cached_concat_errors[:total_B]
        all_iterations_gpu = self._cached_concat_iterations[:total_B]
        all_found_count_gpu = self._cached_concat_valid[:total_B]

        # single batched transfer (one sync point for all batches)
        all_bits = all_errors_gpu.cpu().numpy()
        all_iters = all_iterations_gpu.cpu().numpy()
        all_found_count = all_found_count_gpu.cpu().numpy()

        # Gamma-scaled LLRs for quality computation
        wj_np = self.decoder.log_prior_ratios()  # (N,) float64

        # Process each batch (unconverged_indices handled per batch since they're batch-local)
        results = []
        batch_start = 0
        for batch_idx, (gpu_result, errors, true_obs) in enumerate(
            zip(chunk_gpu_results, chunk_errors_list, chunk_true_obs_list)
        ):
            B = errors.shape[0]
            batch_end = batch_start + B

            # Extract batch slice from batched transfer
            bits = all_bits[batch_start:batch_end]
            iters = all_iters[batch_start:batch_end]
            found_count = all_found_count[batch_start:batch_end]

            # Use found_count > 0 for success (at least one valid solution found).
            success_mask = found_count > 0

            # Handle unconverged_indices for this batch (small, transfer separately)
            unconverged_indices = None
            if self.decoder._dec.fast_max_unconverged_pct is not None:
                uidx_gpu = gpu_result["out"].get("unconverged_indices", None)
                if uidx_gpu is not None and len(uidx_gpu) > 0:
                    unconverged_indices = uidx_gpu.cpu().numpy()

            pred_obs = self._mul_mod2(self._obs, bits.T).T

            # Compute quality
            decoded_quality = (bits.astype(np.float64) * wj_np).sum(axis=1)
            true_quality = (errors.astype(np.float64) * wj_np).sum(axis=1)

            res: List[DecodeResult] = []
            for i in range(B):
                err_det = bool((true_obs[i] != pred_obs[i]).any())
                success = bool(success_mask[i])
                it = int(iters[i])
                obs = pred_obs[i].astype(np.uint8)
                dq = float(decoded_quality[i]) if success else float("inf")
                better_quality = dq < true_quality[i]
                res.append(
                    DecodeResult(
                        decoding=bits[i],
                        success=success,
                        iterations=it,
                        decoding_quality=dq,
                        unconverged_no_error=bool(not err_det and not success),
                        unconverged_logical_error=bool(err_det and not success),
                        better_decoding_quality_error=bool(success and err_det and better_quality),
                        worse_decoding_quality_error=bool(
                            success and err_det and not better_quality
                        ),
                        converged=success,
                        error_detected=err_det,
                        observables=obs,
                        unconverged_indices=unconverged_indices,
                    )
                )

            results.append(res)
            batch_start = batch_end

        return results
