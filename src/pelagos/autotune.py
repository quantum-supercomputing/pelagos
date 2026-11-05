# (C) Copyright IBM 2026
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import os
import json
import hashlib
import platform
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Callable
import itertools, random
import threading
import triton
import torch
import csv

DEFAULT_CACHE = Path(os.getenv("RELAY_TUNE_CACHE", "~/.cache/relay_bp_tune.json")).expanduser()

# Global lock to make cache writes thread-safe across decoder threads
_CACHE_LOCK = threading.Lock()

# Optional CSV logging (set RELAY_TUNE_LOG_CSV to enable)
RUN_ID = os.getenv("RELAY_TUNE_RUN_ID", datetime.now(timezone.utc).isoformat())
LOG_CSV_PATH = os.getenv("RELAY_TUNE_LOG_CSV", "./autotuner_" + RUN_ID + ".csv").strip()


def _csv_log_enabled() -> bool:
    return bool(LOG_CSV_PATH)


def _safe_str(x):
    try:
        return json.dumps(x, sort_keys=True)
    except Exception:
        return str(x)


def _append_csv_row(row: dict):
    if not _csv_log_enabled():
        return
    path = Path(LOG_CSV_PATH).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = (not path.exists()) or (path.exists() and path.stat().st_size == 0)
    fieldnames = [
        "timestamp",
        "run_id",
        "kernel",
        "backend",
        "device_name",
        "device_index",
        "sm",
        "torch",
        "triton",
        "rep",
        "trial_idx",
        "trial_total",
        "grid",
        "BLOCK_SIZE",
        "ROWS_PER_CHK",
        "ROWS_PER_VAR",
        "BTILE",
        "BLOCK_C",
        "BLOCK_V",
        "num_warps",
        "num_stages",
        "time_s",
        "throughput_units_per_s",
        "work_units",
        "best_so_far",
        "problem_key",
        "meta_base",
    ]
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fieldnames})


def _fingerprint() -> Dict[str, Any]:
    backend = "cpu"
    device_info: Dict[str, Any] = {}
    if torch.cuda.is_available():
        try:
            dev = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(dev)
            major = getattr(props, "major", 0)
            minor = getattr(props, "minor", 0)
            sm = major * 10 + minor
            device_info = {
                "device_name": getattr(props, "name", "unknown"),
                "capability": (major, minor),
                "sm": sm,
                "total_mem": int(getattr(props, "total_memory", 0)),
                "multi_processor_count": int(getattr(props, "multi_processor_count", 0)),
            }
            backend = "cuda" if torch.version.cuda is not None else "rocm"
        except Exception:
            backend = (
                "cuda"
                if torch.version.cuda is not None
                else ("rocm" if getattr(torch.version, "hip", None) else "gpu")
            )
            dev = torch.cuda.current_device()
            device_info = {
                "device_name": torch.cuda.get_device_name(dev),
            }
    else:
        device_info = {"device_name": "cpu"}

    return {
        "backend": backend,
        **device_info,
        "driver_cuda": torch.version.cuda,
        "driver_hip": getattr(torch.version, "hip", None),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "python": platform.python_version(),
        "arch_override": os.getenv("SM", ""),
    }


def _make_cache_key(kernel_name: str, problem_key: Dict[str, Any]) -> str:
    fp = _fingerprint()
    blob = json.dumps({"kernel": kernel_name, "fp": fp, "shape": problem_key}, sort_keys=True)
    return hashlib.blake2b(blob.encode(), digest_size=24).hexdigest()


def load_cache() -> Dict[str, Dict[str, Any]]:
    """Load cache file with timeout/retry to prevent blocking on distributed filesystems."""
    import time

    max_retries = 5
    retry_delay = 0.05  # 50ms

    if not DEFAULT_CACHE.exists():
        return {}

    for attempt in range(max_retries):
        try:
            return json.loads(DEFAULT_CACHE.read_text())
        except (json.JSONDecodeError, OSError, IOError):
            # File might be in the middle of being written/renamed
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            # Last attempt failed, return empty dict (will trigger autotuning)
            return {}
        except Exception:
            # Other unexpected errors, return empty dict
            return {}

    return {}


def save_cache(table: Dict[str, Dict[str, Any]]):
    """Save cache using atomic file operations (temp file + rename).

    Uses atomic write (temp file + rename) with retry logic to handle concurrent
    writes from multiple processes. Temp file includes PID for uniqueness per process.

    Args:
        table: Cache data to save
    """
    import time
    import os

    max_retries = 10
    retry_delay = 0.1  # seconds

    # Ensure directory exists
    DEFAULT_CACHE.parent.mkdir(parents=True, exist_ok=True)

    # Use PID in temp filename to make it unique per process
    tmp = DEFAULT_CACHE.parent / f"{DEFAULT_CACHE.stem}.tmp.{os.getpid()}"
    cache_data = json.dumps(table, indent=2, sort_keys=True)

    for attempt in range(max_retries):
        try:
            # Use atomic write: write to temp file, then rename
            # This is safe even with concurrent writes (last write wins, which is fine)
            tmp.write_text(cache_data)
            if not tmp.exists():
                raise FileNotFoundError(f"Temp file {tmp} was not created")
            # Atomic rename - works on most filesystems and is safe for concurrent access
            tmp.replace(DEFAULT_CACHE)
            return  # Success
        except (OSError, FileNotFoundError) as e:
            if attempt < max_retries - 1:
                # Wait with exponential backoff before retry
                time.sleep(retry_delay * (2 ** min(attempt, 4)))  # Cap exponential growth
                # Clean up any stale temp file from previous attempt
                if tmp.exists():
                    try:
                        tmp.unlink()
                    except OSError:
                        pass
                continue
            else:
                # Last attempt failed, log and re-raise
                import warnings

                warnings.warn(f"Failed to save autotune cache after {max_retries} attempts: {e}")
                raise
        finally:
            # Clean up temp file only if rename failed (file still exists)
            # If rename succeeded, tmp no longer exists (it became DEFAULT_CACHE)
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass


def try_get_saved(kernel_name: str, problem_key: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Check if cached tuning results exist for a kernel/problem combination.

    Reads cache based on atomic file operations and accept that duplicate autotuning
    might occasionally occur.
    """
    try:
        table = load_cache()
        key = _make_cache_key(kernel_name, problem_key)
        entry = table.get(key)
        if entry is None:
            return None
        return entry["selected"]
    except Exception:
        # Any error reading cache, return None (will trigger autotuning)
        return None


def set_saved(kernel_name: str, problem_key: Dict[str, Any], meta: Dict[str, Any]):
    """Save tuned meta for a kernel/problem combination.

    This function is called from multiple decoder processes (one per GPU) during
    autotuning. We use atomic file operations (temp file + rename).

    """
    table = load_cache()
    key = _make_cache_key(kernel_name, problem_key)
    record = {
        "kernel": str(kernel_name),
        "problem": problem_key,
        "selected": meta,
        "fingerprint": _fingerprint(),
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "format": 2,
    }
    table[key] = record
    # Save - atomic operations handle concurrency
    save_cache(table)


def _is_invalid_config_error(e: Exception) -> bool:
    """Check if an exception indicates an invalid configuration error (common on AMD HIP).

    Args:
        e: Exception to check

    Returns:
        True if the exception indicates an invalid configuration error
    """
    error_msg = str(e).lower()
    error_repr = repr(e).lower()
    # Check the exception itself
    is_invalid_config = (
        "invalid configuration" in error_msg
        or "hip" in error_msg
        or "code: 9" in error_msg
        or "invalid configuration" in error_repr
        or "hip" in error_repr
        or "code: 9" in error_repr
    )
    # Also check the exception's cause/context (for wrapped exceptions)
    # Python uses __cause__ for explicit chaining, __context__ for implicit chaining
    if not is_invalid_config:
        for exc in [e.__cause__, e.__context__]:
            if exc is not None:
                cause_msg = str(exc).lower()
                cause_repr = repr(exc).lower()
                if (
                    "invalid configuration" in cause_msg
                    or "hip" in cause_msg
                    or "code: 9" in cause_msg
                    or "invalid configuration" in cause_repr
                    or "hip" in cause_repr
                    or "code: 9" in cause_repr
                ):
                    is_invalid_config = True
                    break
    return is_invalid_config


def bench_and_select(
    kernel,
    grid,
    args: Tuple,
    meta_base: Dict[str, Any],
    configs: List[Dict[str, Any]],
    number: int = 20,
    grid_fn: Optional[Callable[[Dict[str, Any]], Tuple[int, ...]]] = None,
) -> Dict[str, Any]:
    """Benchmark kernel configurations and select the best.

    This function minimizes raw wall time, which is fair because:
    - grid_fn adjusts grid size to keep total work constant (e.g., BTILE doubles -> grid dimension halves)
    - For BTILE kernels: grid = (C, (B + bt - 1) // bt), so total work = B * C is constant
    - For ROWS_PER_CHK kernels: grid = ((B*C + rows - 1) // rows,), so total work = B * C is constant

    We log normalized throughput (work_units/time) for analysis, but selection uses wall time
    since work is constant across configs.
    """
    verbose = os.getenv("RELAY_TUNE_VERBOSE", "0") == "1"
    # Timeout per config: skip if single iteration takes > 5 seconds (likely hanging/compilation issue)
    timeout_per_iter_ms = float(os.getenv("RELAY_TUNE_TIMEOUT_MS", "5000.0"))

    # Detect backend for grid size limits (AMD HIP has stricter limits)
    fp = _fingerprint()
    is_amd = fp.get("backend") == "rocm" or fp.get("driver_hip") is not None
    # AMD GPUs may have practical grid size limits lower than theoretical max
    # We use a conservative limit to avoid "invalid configuration" errors
    max_grid_dim = 2**24 if is_amd else 2**31 - 1  # 16M for AMD, 2B for NVIDIA

    best_t = float("inf")
    best_cfg: Optional[Dict[str, Any]] = None
    total = len(configs)
    kname = getattr(kernel, "__name__", "kernel")
    if verbose:
        import threading

        thread_name = threading.current_thread().name
        try:
            dev = torch.cuda.current_device()
        except Exception:
            dev = None
        dev_str = f", gpu={dev}" if dev is not None else ""
        backend_str = " (AMD)" if is_amd else ""
        print(
            f"[relay-bp-triton][thread={thread_name}{dev_str}] Tuning {kname} with {total} configs, reps={number}{backend_str}"
        )
    for idx, cfg in enumerate(configs, start=1):
        conf_kwargs = {k: v for k, v in cfg.items() if k.isupper()}
        num_warps = cfg.get("num_warps")
        num_stages = cfg.get("num_stages")
        grid_local = grid_fn(cfg) if grid_fn is not None else grid

        # Check for invalid grid sizes (e.g., too large for AMD HIP backend)
        # AMD GPUs have stricter practical limits than NVIDIA
        if any(dim > max_grid_dim for dim in grid_local):
            if verbose:
                print(
                    f"  [{idx}/{total}] SKIP: grid size {grid_local} exceeds maximum grid dimension {max_grid_dim}"
                )
            continue

        # Try to launch kernel - catch invalid configuration errors (e.g., AMD HIP limits)
        # Note: AMD HIP errors can manifest as RuntimeError or SystemError (wrapped)
        try:
            kernel[grid_local](
                *args, **meta_base, **conf_kwargs, num_warps=num_warps, num_stages=num_stages
            )
            torch.cuda.synchronize()
            for _ in range(2):
                kernel[grid_local](
                    *args, **meta_base, **conf_kwargs, num_warps=num_warps, num_stages=num_stages
                )
            torch.cuda.synchronize()
        except (RuntimeError, SystemError) as e:
            # Primary handler for expected exception types
            # Skip configurations that fail with invalid configuration errors (common on AMD HIP)
            if _is_invalid_config_error(e):
                if verbose:
                    print(
                        f"  [{idx}/{total}] SKIP: invalid configuration for grid={grid_local}, warps={num_warps}, stages={num_stages}: {e}"
                    )
                continue
            # Re-raise other errors (unexpected errors)
            raise
        except Exception as e:
            # Defensive catch-all: check if it's an invalid config error in any form
            if _is_invalid_config_error(e):
                if verbose:
                    print(
                        f"  [{idx}/{total}] SKIP: invalid configuration (caught as {type(e).__name__}) for grid={grid_local}, warps={num_warps}, stages={num_stages}: {e}"
                    )
                continue
            # Re-raise unexpected errors
            raise

        def run():
            kernel[grid_local](
                *args, **meta_base, **conf_kwargs, num_warps=num_warps, num_stages=num_stages
            )

        # Benchmark with a fixed number of repetitions (iterations), not a time budget
        iters = max(1, int(number))
        times_ms = []
        skipped_due_to_timeout = False
        skipped_due_to_error = False
        for iter_idx in range(iters):
            try:
                start_evt = torch.cuda.Event(enable_timing=True)
                end_evt = torch.cuda.Event(enable_timing=True)
                start_evt.record()
                run()
                end_evt.record()
                torch.cuda.synchronize()
                iter_time = start_evt.elapsed_time(end_evt)
                # Skip configs that take too long per iteration (likely register spilling/hanging)
                if iter_time > timeout_per_iter_ms:
                    if verbose:
                        print(
                            f"  [{idx}/{total}] SKIP: iteration {iter_idx + 1} took {iter_time:.1f}ms > {timeout_per_iter_ms:.1f}ms timeout"
                        )
                    skipped_due_to_timeout = True
                    break
                times_ms.append(iter_time)
            except (RuntimeError, SystemError) as e:
                # Catch runtime/system errors during benchmarking (e.g., invalid configuration discovered during run)
                if _is_invalid_config_error(e):
                    if verbose:
                        print(
                            f"  [{idx}/{total}] SKIP: invalid configuration during benchmark: {e}"
                        )
                    skipped_due_to_error = True
                    break
                # Re-raise other errors (unexpected errors)
                raise
            except Exception as e:
                # Defensive catch-all: check if it's an invalid config error in any form
                if _is_invalid_config_error(e):
                    if verbose:
                        print(
                            f"  [{idx}/{total}] SKIP: invalid configuration during benchmark (caught as {type(e).__name__}): {e}"
                        )
                    skipped_due_to_error = True
                    break
                # Re-raise unexpected errors
                raise

        if skipped_due_to_timeout or skipped_due_to_error:
            continue

        if not times_ms:
            if verbose:
                print(f"  [{idx}/{total}] SKIP: no valid timings")
            continue

        t = sum(times_ms) / len(times_ms)

        # Compute normalized throughput metrics for analysis
        # Note: grid_fn adjusts grid size to keep total work constant (e.g., BTILE doubles -> grid halves),
        # so raw wall time comparison is valid. But we log normalized metrics for visibility.
        total_programs = 1
        for dim in grid_local:
            total_programs *= dim
        # Estimate work: B*C for C2V, B*V for V2C, B*E for message ops
        # Use meta_base values if available, otherwise estimate from grid
        B = meta_base.get("B", 1)
        C = meta_base.get("C", grid_local[0] if len(grid_local) > 0 else 1)
        V = meta_base.get("V", grid_local[0] if len(grid_local) > 0 else 1)
        E = meta_base.get("E", 1)
        # Approximate work units (kernels process different amounts, but rough estimate)
        kl = kname.lower()
        if "c2v" in kl:
            work_units = B * C  # one check per batch item
        elif "v2c" in kl:
            work_units = B * V  # one variable per batch item
        elif "parity" in kl:
            work_units = B * C  # one check per batch item
        elif "check_and_select" in kl:
            work_units = B * C  # validates checks and selects per lane
        else:
            work_units = B * E  # fallback: edges processed
        throughput = work_units / t if t > 0 else 0.0

        if verbose:
            cfg_str = ", ".join([f"{k}={v}" for k, v in conf_kwargs.items()])
            print(
                f"  [{idx}/{total}] {cfg_str}, warps={num_warps}, stages={num_stages} -> {t:.3e} s (throughput: {throughput:.1e} units/s)"
            )
        _append_csv_row(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "run_id": RUN_ID,
                "kernel": kname,
                "backend": fp.get("backend"),
                "device_name": fp.get("device_name"),
                "device_index": fp.get("device_index"),
                "sm": fp.get("sm"),
                "torch": fp.get("torch"),
                "triton": fp.get("triton"),
                "rep": number,
                "trial_idx": idx,
                "trial_total": total,
                "grid": _safe_str(grid_local),
                "BLOCK_SIZE": conf_kwargs.get("BLOCK_SIZE"),
                "ROWS_PER_CHK": conf_kwargs.get("ROWS_PER_CHK"),
                "ROWS_PER_VAR": conf_kwargs.get("ROWS_PER_VAR"),
                "BTILE": conf_kwargs.get("BTILE"),
                "BLOCK_C": conf_kwargs.get("BLOCK_C"),
                "BLOCK_V": conf_kwargs.get("BLOCK_V"),
                "num_warps": num_warps,
                "num_stages": num_stages,
                "time_s": t,
                "throughput_units_per_s": throughput,
                "work_units": work_units,
                "best_so_far": (t < best_t),
                "problem_key": _safe_str({}),
                "meta_base": _safe_str(meta_base),
            }
        )
        if t < best_t:
            best_t = t
            best_cfg = {"num_warps": num_warps, "num_stages": num_stages, **conf_kwargs}
            if verbose:
                print(f"    -> new best")
    if best_cfg is None:
        raise RuntimeError("No config benchmarked")
    if verbose:
        import threading

        thread_name = threading.current_thread().name
        try:
            dev = torch.cuda.current_device()
        except Exception:
            dev = None
        dev_str = f", gpu={dev}" if dev is not None else ""
        best_str = ", ".join([f"{k}={v}" for k, v in best_cfg.items()])
        # Include simple key summary to avoid misleading cross-run comparisons
        key_str = f"B={meta_base.get('B')}, C={meta_base.get('C')}, V={meta_base.get('V')}"
        print(f"[relay-bp-triton][thread={thread_name}{dev_str}] Selected: {best_str}  [{key_str}]")
    return best_cfg


# ---- Dynamic config builders for bench path ----
def _parse_list(env: str, default: list[int]) -> list[int]:
    v = os.getenv(env)
    if v is None:
        return default
    try:
        return [int(x) for x in v.replace(" ", "").split(",") if x]
    except Exception:
        return default


def _cap(n: int | None = None, env: str = "RELAY_TUNE_MAX_CFG", default: int = 512) -> int:
    try:
        return int(os.getenv(env, default if n is None else n))
    except Exception:
        return default if n is None else n


def _sample(
    cfgs: list[triton.Config],
    env: str = "RELAY_TUNE_SAMPLE",
    default: int | None = None,
    seed: int = 0,
) -> list[triton.Config]:
    k = os.getenv(env)
    if k is None and default is None:
        return cfgs
    try:
        kint = int(k if k is not None else default)
    except Exception:
        return cfgs
    if kint and kint < len(cfgs):
        rng = random.Random(int(os.getenv("RELAY_TUNE_SEED", seed)))
        return rng.sample(cfgs, kint)
    return cfgs


def _make_configs(space: dict[str, list[int]], cap: int | None = None) -> list[triton.Config]:
    keys = list(space.keys())
    vals = [space[k] for k in keys]
    out: list[triton.Config] = []
    for combo in itertools.product(*vals):
        kw = dict(zip(keys, combo))
        bs = kw.get("BLOCK_SIZE", 0)
        warps = kw.get("num_warps", 0)
        out.append(
            triton.Config(kw, num_warps=kw.get("num_warps", 2), num_stages=kw.get("num_stages", 2))
        )
    out = out[: _cap(cap)]
    out = _sample(out)
    return out


def build_c2v_configs() -> list[triton.Config]:
    bs = _parse_list("RELAY_SWEEP_C2V_BLOCK", [1, 2, 4, 8, 16])
    r = _parse_list("RELAY_SWEEP_ROWS_PER_CHK", [1, 2, 4, 8])
    wp = _parse_list("RELAY_SWEEP_C2V_WARPS", [1, 2, 4])
    st = _parse_list("RELAY_SWEEP_C2V_STAGES", [1, 2])
    space = {"BLOCK_SIZE": bs, "ROWS_PER_CHK": r, "num_warps": wp, "num_stages": st}
    return _make_configs(space)


def build_v2c_configs() -> list[triton.Config]:
    bs = _parse_list("RELAY_SWEEP_V2C_BLOCK", [1, 2, 4, 8, 16, 32])
    r = _parse_list("RELAY_SWEEP_ROWS_PER_VAR", [1, 2, 4, 8])
    wp = _parse_list("RELAY_SWEEP_V2C_WARPS", [1, 2, 4])
    st = _parse_list("RELAY_SWEEP_V2C_STAGES", [1, 2])
    space = {"BLOCK_SIZE": bs, "ROWS_PER_VAR": r, "num_warps": wp, "num_stages": st}
    return _make_configs(space)


def build_c2v_btile_configs() -> list[triton.Config]:
    """Configs for C2V BTILE kernels (check-major, dimension ~C).

    Environment overrides (C2V-specific):
      - RELAY_SWEEP_C2V_BT_BLOCK   (BLOCK_SIZE)
      - RELAY_SWEEP_C2V_BT_WARPS   (num_warps)
      - RELAY_SWEEP_C2V_BT_STAGES  (num_stages)
      - RELAY_SWEEP_C2V_BTILE      (BTILE)
    Falls back to the shared BT defaults if unset.
    """
    bs = _parse_list(
        "RELAY_SWEEP_C2V_BT_BLOCK", _parse_list("RELAY_SWEEP_BT_BLOCK", [1, 2, 4, 8, 16, 32])
    )
    wp = _parse_list("RELAY_SWEEP_C2V_BT_WARPS", _parse_list("RELAY_SWEEP_BT_WARPS", [1, 2, 4]))
    stg = _parse_list("RELAY_SWEEP_C2V_BT_STAGES", _parse_list("RELAY_SWEEP_BT_STAGES", [1, 2]))
    bt = _parse_list("RELAY_SWEEP_C2V_BTILE", _parse_list("RELAY_SWEEP_BTILE", [16, 32, 64]))
    space = {"BLOCK_SIZE": bs, "BTILE": bt, "num_warps": wp, "num_stages": stg}
    return _make_configs(space)


V2C_BTILE_BLOCK_SIZE = 1


def build_v2c_btile_configs() -> list[triton.Config]:
    """Configs for V2C BTILE kernels (var-major, dimension ~V).

    BLOCK_SIZE is always V2C_BTILE_BLOCK_SIZE and has no environment override.

    Environment overrides (V2C-specific):
      - RELAY_SWEEP_V2C_BT_WARPS   (num_warps)
      - RELAY_SWEEP_V2C_BT_STAGES  (num_stages)
      - RELAY_SWEEP_V2C_BTILE      (BTILE)
    Falls back to the shared BT defaults if unset.
    """
    # BLOCK_SIZE stays 1: larger tiles change the summation order of a variable's
    # incoming messages (with num_warps), so results would depend on the config.
    bs = [V2C_BTILE_BLOCK_SIZE]
    wp = _parse_list("RELAY_SWEEP_V2C_BT_WARPS", _parse_list("RELAY_SWEEP_BT_WARPS", [1, 2, 4]))
    stg = _parse_list("RELAY_SWEEP_V2C_BT_STAGES", _parse_list("RELAY_SWEEP_BT_STAGES", [1, 2]))
    bt = _parse_list("RELAY_SWEEP_V2C_BTILE", _parse_list("RELAY_SWEEP_BTILE", [16, 32, 64]))
    space = {"BLOCK_SIZE": bs, "BTILE": bt, "num_warps": wp, "num_stages": stg}
    return _make_configs(space)


def build_check_select_configs() -> list[triton.Config]:
    """Build configs for the check_and_select kernel.

    Environment overrides:
      - RELAY_SWEEP_CS_BLOCK_C: comma list for BLOCK_C (default: 64,128,256)
      - RELAY_SWEEP_CS_BLOCK_V: comma list for BLOCK_V (default: 64,128,256)
      - RELAY_SWEEP_CS_WARPS:   comma list for num_warps (default: 1,2)
      - RELAY_SWEEP_CS_STAGES:  comma list for num_stages (default: 1,2)
    """
    bc = _parse_list("RELAY_SWEEP_CS_BLOCK_C", [16, 32, 64])
    bv = _parse_list("RELAY_SWEEP_CS_BLOCK_V", [16, 32, 64])
    wp = _parse_list("RELAY_SWEEP_CS_WARPS", [1, 2, 4, 8])
    st = _parse_list("RELAY_SWEEP_CS_STAGES", [1, 2, 3, 4])
    space = {"BLOCK_C": bc, "BLOCK_V": bv, "num_warps": wp, "num_stages": st}
    return _make_configs(space)


def build_btile_transpose_configs() -> list[triton.Config]:
    btile = _parse_list("RELAY_SWEEP_TR_BTILE", [8, 16, 32, 64, 128, 256])
    wp = _parse_list("RELAY_SWEEP_TR_WARPS", [1, 2, 4])
    stg = _parse_list("RELAY_SWEEP_TR_STAGES", [1, 2])
    space = {"BTILE": btile, "num_warps": wp, "num_stages": stg}
    return _make_configs(space)


def build_parity_configs() -> list[triton.Config]:
    bs = _parse_list("RELAY_SWEEP_PAR_BLOCK", [1, 2, 4, 16, 32, 64, 128, 256])
    wp = _parse_list("RELAY_SWEEP_PAR_WARPS", [1, 2, 4])
    stg = _parse_list("RELAY_SWEEP_PAR_STAGES", [1, 2])
    space = {"BLOCK_SIZE": bs, "num_warps": wp, "num_stages": stg}
    return _make_configs(space)


def build_permute_configs() -> list[triton.Config]:
    """Build autotune configurations for permute_rows_kernel.

    The permute kernel copies edge rows in BTILE chunks, so we tune:
    - BTILE: batch tile size for vectorized copying
    - num_warps: number of warps per program
    - num_stages: pipeline stages
    """
    btile = _parse_list("RELAY_SWEEP_PERM_BTILE", [64, 128, 256, 512])
    wp = _parse_list("RELAY_SWEEP_PERM_WARPS", [1, 2, 4, 8])
    stg = _parse_list("RELAY_SWEEP_PERM_STAGES", [1, 2])
    space = {"BTILE": btile, "num_warps": wp, "num_stages": stg}
    return _make_configs(space)


def build_init_messages_configs() -> list[triton.Config]:
    """Build autotune configurations for init_messages_with_priors_EB_kernel.

    The init kernel initializes messages (nu=zero, mu=priors) for all edges and batches.
    This is a memory-bound operation, so we tune:
    - BLOCK: block size for vectorized operations (affects memory coalescing)
    - num_warps: number of warps per program (affects occupancy)
    - num_stages: pipeline stages (affects memory latency hiding)

    Environment overrides:
      - RELAY_SWEEP_INIT_BLOCK: comma list for BLOCK (default: 256,512,1024,2048)
      - RELAY_SWEEP_INIT_WARPS: comma list for num_warps (default: 1,2,4,8)
      - RELAY_SWEEP_INIT_STAGES: comma list for num_stages (default: 1,2,3,4)
    """
    bs = _parse_list("RELAY_SWEEP_INIT_BLOCK", [128, 256, 512, 1024, 2048])
    wp = _parse_list("RELAY_SWEEP_INIT_WARPS", [2, 4, 8, 16])
    stg = _parse_list("RELAY_SWEEP_INIT_STAGES", [1, 2, 3, 4])
    space = {"BLOCK": bs, "num_warps": wp, "num_stages": stg}
    return _make_configs(space)


def build_persistent_bp_configs() -> list[triton.Config]:
    """Build autotune configurations for persistent_bp_iterations_kernel.

    The persistent kernel processes one batch lane at a time, running all iterations
    inside a single kernel launch. It fuses C2V, V2C, parity, and selection operations.

    We tune:
    - BLOCK_SIZE: Vectorization block size for memory operations (affects coalescing)
    - num_warps: Number of warps per program (affects occupancy and parallelism)
    - num_stages: Pipeline stages (affects memory latency hiding)

    Environment overrides:
      - RELAY_SWEEP_PERSISTENT_BLOCK: comma list for BLOCK_SIZE (default: 16,32,64,128,256)
      - RELAY_SWEEP_PERSISTENT_WARPS: comma list for num_warps (default: 1,2,4,8)
      - RELAY_SWEEP_PERSISTENT_STAGES: comma list for num_stages (default: 1,2,3,4)
    """
    bs = _parse_list("RELAY_SWEEP_PERSISTENT_BLOCK", [16, 32, 64, 128, 256])
    wp = _parse_list("RELAY_SWEEP_PERSISTENT_WARPS", [1, 2, 4, 8])
    stg = _parse_list("RELAY_SWEEP_PERSISTENT_STAGES", [1, 2, 3, 4])
    space = {"BLOCK_SIZE": bs, "num_warps": wp, "num_stages": stg}
    return _make_configs(space)


def build_c2v_grid_stride_configs() -> list[triton.Config]:
    """Build autotune configurations for c2v_min_sum_grid_stride_kernel.

    This kernel only performs the C2V (check-to-variable) phase to ensure
    memory consistency. V2C, parity, and selection phases run in separate kernel launches
    after this kernel completes, providing implicit barriers (kernel launch completion)
    between phases.

    This kernel uses grid-stride parallelism (like CUDA) to split work across SMs efficiently.

    We tune:
    - BLOCK_SIZE: Vectorization block size for memory operations
    - num_warps: Number of warps per program
    - num_stages: Pipeline stages
    - ROWS_PER_CHK: Number of checks processed per program (default: 1 for grid-stride)
    - ROWS_PER_VAR: Number of vars processed per program (default: 1, unused in C2V-only)

    Environment overrides (kept for backward compatibility):
      - RELAY_SWEEP_ITERATION_FUSED_BLOCK: comma list for BLOCK_SIZE (default: 16,32,64,128)
      - RELAY_SWEEP_ITERATION_FUSED_WARPS: comma list for num_warps (default: 1,2,4,8)
      - RELAY_SWEEP_ITERATION_FUSED_STAGES: comma list for num_stages (default: 1,2,3)
      - RELAY_SWEEP_ITERATION_FUSED_ROWS_CHK: comma list for ROWS_PER_CHK (default: 1,2,4,8)
      - RELAY_SWEEP_ITERATION_FUSED_ROWS_VAR: comma list for ROWS_PER_VAR (default: 1,2,4,8)

    Note: For small matrices, ROWS_PER_CHK=1 maximizes parallelism. For huge matrices,
    ROWS_PER_CHK>1 can improve cache locality and reduce overhead. The autotuner will
    select the optimal value based on actual performance.
    """
    bs = _parse_list("RELAY_SWEEP_ITERATION_FUSED_BLOCK", [16, 32, 64, 128])
    wp = _parse_list("RELAY_SWEEP_ITERATION_FUSED_WARPS", [1, 2, 4, 8])
    stg = _parse_list("RELAY_SWEEP_ITERATION_FUSED_STAGES", [1, 2, 3])
    # Make ROWS_PER_CHK and ROWS_PER_VAR tunable (like regular C2V/V2C kernels)
    # Default includes 1 for grid-stride, but also allows 2,4,8 for better cache locality on huge matrices
    r_chk = _parse_list("RELAY_SWEEP_ITERATION_FUSED_ROWS_CHK", [1, 2, 4, 8])
    r_var = _parse_list("RELAY_SWEEP_ITERATION_FUSED_ROWS_VAR", [1, 2, 4, 8])
    space = {
        "BLOCK_SIZE": bs,
        "ROWS_PER_CHK": r_chk,
        "ROWS_PER_VAR": r_var,
        "num_warps": wp,
        "num_stages": stg,
    }
    return _make_configs(space)
