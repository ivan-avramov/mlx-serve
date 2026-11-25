"""
process_manager.py — subprocess lifecycle for the active mlx model.

State machine: IDLE -> LOADING -> READY / FAILED -> IDLE (on inactivity/shutdown)
"""

import asyncio
import contextlib
import json
import logging
import os
import pathlib
import re
import stat
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum

import httpx
import psutil

from . import config, events, metrics

logger = logging.getLogger("mlx-serve.process")

# Resolve executable paths relative to the running venv so subprocesses
# inherit the same environment that has mlx_lm / mlx_vlm installed.
_VENV_BIN = pathlib.Path(sys.executable).parent
_MLX_LM_SERVER = _VENV_BIN / "mlx_lm.server"
_MLX_VLM_SERVER = _VENV_BIN / "mlx_vlm.server"

_LOG_DIR = pathlib.Path(os.environ.get("SABLE_MODEL_LOG_DIR") or pathlib.Path(tempfile.gettempdir()) / "mlx-manager-logs")
_LOG_MODEL_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}\Z")
_PREVIOUS_LOG_LIMIT = 32 * 1024
_VERIFY_PREFERENCE_ENV = "SABLE_MTP_VERIFY_PREFERENCE"
_VERIFY_WORKER_ENV = "SABLE_MTP_VERIFY_WORKER"
_VERIFY_RETRY_ENV = "SABLE_MTP_VERIFY_RETRY"
_VERIFY_PROBE_TIMEOUT_EXIT = 76  # mlx_vlm optional synthetic joint probe only
_VERIFY_PROBE_TIMEOUT_LINE = b"SABLE_MTP_VERIFY_PROBE_TIMEOUT"

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class ModelState(Enum):
    IDLE = "idle"
    LOADING = "loading"
    DOWNLOADING = "downloading"
    READY = "ready"
    FAILED = "failed"


@dataclass
class _WorkerAttempt:
    id: str
    model: str
    started_at: float
    failure_reason: str | None = None
    failure_at: float | None = None


_attempt: _WorkerAttempt | None = None
_last_failed_attempt: _WorkerAttempt | None = None
_state: ModelState = ModelState.IDLE
_active_model: str | None = None
_process: subprocess.Popen | None = None
_ready_event: asyncio.Event = asyncio.Event()
_load_timeout: int | None = None
_switch_lock: asyncio.Lock = asyncio.Lock()
_last_request_at: datetime | None = None
_started_at: datetime = datetime.now(UTC)
_stderr_log_path: pathlib.Path | None = None
_stderr_log_handle = None
_inactivity_timeout: int = config.INACTIVITY_TIMEOUT  # overridable per-request
_loading_started_at: float | None = None  # monotonic time when loading began

# Failure tracking (circuit breaker / cooldown)
_last_failure_at: float | None = None  # monotonic timestamp of last failure
_last_failed_model: str | None = None
_last_failure_reason: str | None = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _record_failure(model_name: str, reason: str) -> None:
    """Record a model load failure for cooldown tracking."""
    global _last_failure_at, _last_failed_model, _last_failure_reason
    _last_failure_at = time.monotonic()
    _last_failed_model = model_name
    _last_failure_reason = reason
    _mark_attempt_failure(reason)


def _mark_attempt_failure(reason: str) -> None:
    global _last_failed_attempt
    if _attempt is not None:
        _attempt.failure_reason = reason
        _attempt.failure_at = time.time()
        _last_failed_attempt = _attempt


def _attempt_detail() -> dict:
    return {"attempt_id": _attempt.id} if _attempt else {}


def _clear_failure() -> None:
    """Reset failure tracking state."""
    global _last_failure_at, _last_failed_model, _last_failure_reason
    _last_failure_at = None
    _last_failed_model = None
    _last_failure_reason = None


def _failure_cooldown_remaining(model_name: str) -> float:
    """Return seconds remaining in the failure cooldown for this model, or 0.0."""
    if _last_failure_at is None or _last_failed_model != model_name:
        return 0.0
    elapsed = time.monotonic() - _last_failure_at
    if elapsed >= config.FAILURE_COOLDOWN:
        return 0.0
    return config.FAILURE_COOLDOWN - elapsed


def _is_model_cached(model_cfg: config.ModelConfig) -> bool:
    """Return True if the model's weights are already in the HuggingFace cache."""
    try:
        from huggingface_hub import try_to_load_from_cache
        from huggingface_hub.file_download import _CACHED_NO_EXIST

        result = try_to_load_from_cache(model_cfg.hf_path, "config.json")
        return isinstance(result, str) and result is not _CACHED_NO_EXIST
    except Exception:
        return False


def _readiness_timeout(model_cfg: config.ModelConfig) -> int:
    """Return the appropriate readiness timeout for this model."""
    # All callers joining this load see the same interval, even if the files
    # become cached between the first worker and its one permitted retry.
    if model_cfg.name == _active_model and _load_timeout is not None:
        return _load_timeout
    return config.STARTUP_TIMEOUT if _is_model_cached(model_cfg) else config.DOWNLOAD_TIMEOUT


def _build_command(model_cfg: config.ModelConfig) -> list[str]:
    executable = _MLX_VLM_SERVER if model_cfg.type == "vision" else _MLX_LM_SERVER
    if not executable.exists():
        pkg = "mlx-vlm" if model_cfg.type == "vision" else "mlx-lm"
        extra = "vision" if model_cfg.type == "vision" else "text"
        raise RuntimeError(
            f"{pkg} is not installed (expected: {executable}). Run: pip install mlx-serve[{extra}]"
        )
    cmd = [
        str(executable),
        "--model",
        model_cfg.hf_path,
        "--host",
        "127.0.0.1",
        "--port",
        str(config.MLX_PORT),
    ]
    if model_cfg.context_length > 0:
        cmd += ["--max-tokens", str(model_cfg.context_length)]
    if model_cfg.max_kv_cache_size > 0:
        if model_cfg.type == "vision":
            cmd += ["--max-kv-size", str(model_cfg.max_kv_cache_size)]
        else:
            cmd += ["--max-kv-cache-size", str(model_cfg.max_kv_cache_size)]
    if model_cfg.kv_prealloc_tokens > 0 and model_cfg.type == "vision":
        cmd += ["--kv-prealloc-tokens", str(model_cfg.kv_prealloc_tokens)]
    if model_cfg.type == "vision" and model_cfg.cache_session_shrink is not None:
        cmd += ["--cache-session-shrink", "on" if model_cfg.cache_session_shrink else "off"]
    if model_cfg.type == "vision" and model_cfg.attention_policy not in ("", "auto"):
        cmd += ["--attention-policy", model_cfg.attention_policy]
    if model_cfg.type == "vision" and model_cfg.lazy_prompt_embeddings is True:
        cmd += ["--lazy-prompt-embeddings"]
    if model_cfg.type == "vision" and model_cfg.mtp_verify_scan not in ("", "per_query"):
        cmd += ["--mtp-verify-scan", model_cfg.mtp_verify_scan]
        if model_cfg.mtp_verify_ab:
            cmd += ["--mtp-verify-ab"]

    if model_cfg.kv_bits > 0:
        cmd += ["--kv-bits", str(model_cfg.kv_bits)]
    if model_cfg.kv_quant_scheme:
        cmd += ["--kv-quant-scheme", model_cfg.kv_quant_scheme]
    if model_cfg.moe_expand:
        cmd += ["--moe-expand", model_cfg.moe_expand]
    if model_cfg.kv_quant_mode:
        cmd += ["--kv-quant-mode", model_cfg.kv_quant_mode]
    if model_cfg.prefill_step_size:
        cmd += ["--prefill-step-size", str(model_cfg.prefill_step_size)]
    # vision models don't support these, only lm
    if model_cfg.type != "vision":
        if model_cfg.reasoning_parser:
            cmd += ["--reasoning-parser", model_cfg.reasoning_parser]
        if model_cfg.tool_call_parser:
            cmd += ["--tool-call-parser", model_cfg.tool_call_parser]
    # Per-model generation defaults, forwarded opaquely as one JSON arg (mlx-serve does not
    # enumerate the keys). mlx-vlm applies each entry as a default only when the request
    # omits it. enable_thinking travels inside this block now.
    if model_cfg.generation_defaults:
        cmd += ["--generation-defaults", json.dumps(model_cfg.generation_defaults)]
    # Speculative decoding flags (mlx-vlm / vision only). draft_kind drives it;
    # the suffix knobs are no-ops for other kinds. Omitted fields keep defaults.
    if model_cfg.type == "vision" and model_cfg.draft_kind:
        cmd += ["--draft-kind", model_cfg.draft_kind]
        # mtp/dflash/eagle3 need a separate split drafter folder; suffix is drafter-free.
        if model_cfg.draft_model:
            cmd += ["--draft-model", model_cfg.draft_model]
        if model_cfg.draft_block_size > 0:
            cmd += ["--draft-block-size", str(model_cfg.draft_block_size)]
        if model_cfg.suffix_min_match > 0:
            cmd += ["--suffix-min-match", str(model_cfg.suffix_min_match)]
        if model_cfg.draft_cooldown > 0:
            cmd += ["--draft-cooldown", str(model_cfg.draft_cooldown)]
    cmd += ["--quantized-kv-start", str(model_cfg.quantized_kv_start)]
    if model_cfg.cache_limit_gb and model_cfg.cache_limit_gb > 0:
        cmd += ["--cache-limit-gb", str(model_cfg.cache_limit_gb)]
    if model_cfg.memory_limit_frac and model_cfg.memory_limit_frac > 0:
        cmd += ["--memory-limit-frac", str(model_cfg.memory_limit_frac)]
    return cmd


def _worker_verify_preference(model_cfg: config.ModelConfig) -> str | None:
    preference = os.environ.get(_VERIFY_PREFERENCE_ENV)
    if (
        preference in ("joint-first", "per-query")
        and model_cfg.type == "vision"
        and model_cfg.mtp_verify_scan == "joint_v1"
        and model_cfg.draft_kind == "mtp"
        and not model_cfg.mtp_verify_ab
    ):
        return preference
    return None


def _log_name(model: str) -> str:
    if not _LOG_MODEL_NAME.fullmatch(model):
        raise ValueError("invalid model name for worker log")
    return f"{model}.log"


def _open_log_directory() -> int:
    return os.open(_LOG_DIR, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)


def _open_attempt_log(attempt: _WorkerAttempt):
    name = _log_name(attempt.model)
    _LOG_DIR.mkdir(mode=0o700, exist_ok=True)
    directory = _open_log_directory()
    tmp = f".{attempt.id}.log"
    try:
        fd = os.open(tmp, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory)
        log = os.fdopen(fd, "w")
        try:
            log.write(f"SABLE_MODEL_ATTEMPT id={attempt.id}\n")
            log.flush()
            # One atomic replacement publishes the new attempt. Readers of the old
            # descriptor keep a coherent view of the preceding attempt.
            try:
                _archive_attempt_log(directory, name, attempt.id)
            except OSError as exc:
                raise OSError(f"Cannot archive {name}; keeping the current worker log") from exc
            os.replace(tmp, name, src_dir_fd=directory, dst_dir_fd=directory)
        except BaseException:
            log.close()
            with contextlib.suppress(OSError):
                os.unlink(tmp, dir_fd=directory)
            raise
        return log
    finally:
        os.close(directory)


def _archive_attempt_log(directory: int, name: str, new_attempt_id: str) -> None:
    try:
        source = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
    except FileNotFoundError:
        return
    try:
        info = os.fstat(source)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or info.st_nlink != 1:
            raise OSError("Unsafe current worker log")
        if info.st_size == 0:
            return
        offset = max(0, info.st_size - _PREVIOUS_LOG_LIMIT)
        count = info.st_size - offset
        tail = os.pread(source, count, offset)
        if len(tail) != count:
            raise OSError("Short read of current worker log")
        if offset:
            # Keep the prior marker even when the tail starts past the header.
            head = os.pread(source, 128, 0)
            if len(head) != 128:
                raise OSError("Short read of current worker log header")
            marker, newline, _ = head.partition(b"\n")
            if newline and marker.startswith(b"SABLE_MODEL_ATTEMPT id="):
                tail = marker + b"\n" + tail
            else:
                # Legacy logs have no attempt marker.
                tail = b"\n" + tail
    finally:
        os.close(source)

    temp = f".{new_attempt_id}.previous.tmp"
    fd = os.open(
        temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory
    )
    try:
        with os.fdopen(fd, "wb") as archive:
            if archive.write(tail) != len(tail):
                raise OSError("Short write of previous worker log")
        os.replace(temp, f"_previous.{name}", src_dir_fd=directory, dst_dir_fd=directory)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temp, dir_fd=directory)


def _tail(fd: int, limit: int) -> bytes:
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or info.st_nlink != 1:
        return b""
    return os.pread(fd, limit, max(0, info.st_size - limit))


def model_log_tail(model_name: str) -> list[str]:
    name = _log_name(model_name)
    directory = _open_log_directory()
    try:
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        try:
            return _tail(fd, 32 * 1024).decode("utf-8", errors="replace").splitlines()[-100:]
        finally:
            os.close(fd)
    finally:
        os.close(directory)


def _saw_verify_probe_timeout() -> bool:
    if _stderr_log_handle is None:
        return False
    try:
        return _VERIFY_PROBE_TIMEOUT_LINE in _tail(_stderr_log_handle.fileno(), 4096).splitlines()
    except OSError:
        return False


def _spawn_worker(
    model_name: str, model_cfg: config.ModelConfig, *, verification_retry: bool = False
) -> subprocess.Popen:
    global _attempt, _stderr_log_path, _stderr_log_handle
    _attempt = _WorkerAttempt(str(uuid.uuid4()), model_name, time.time())
    _stderr_log_path = None
    cmd = _build_command(model_cfg)
    preference = _worker_verify_preference(model_cfg)
    env = None
    if preference or _VERIFY_WORKER_ENV in os.environ or _VERIFY_RETRY_ENV in os.environ:
        env = os.environ.copy()
        env.pop(_VERIFY_WORKER_ENV, None)
        env.pop(_VERIFY_RETRY_ENV, None)
        if preference:
            env[_VERIFY_WORKER_ENV] = preference
        if verification_retry:
            env[_VERIFY_RETRY_ENV] = "probe_timeout"

    _stderr_log_handle = _open_attempt_log(_attempt)
    _stderr_log_path = _LOG_DIR / _log_name(model_name)
    logger.info(f"Spawning: {' '.join(cmd)}")
    try:
        process = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=_stderr_log_handle, env=env
        )
    except BaseException:
        _stderr_log_handle.close()
        _stderr_log_handle = None
        raise

    events.emit(
        events.EventType.MODEL_LOADING,
        model=model_name,
        detail={"hf_path": model_cfg.hf_path, "type": model_cfg.type, "pid": process.pid, **_attempt_detail()},
    )
    events.emit(
        events.EventType.SUBPROCESS_SPAWNED,
        model=model_name,
        detail={"pid": process.pid, "command": cmd, **_attempt_detail()},
    )
    return process


def _diagnose_failure(timeout_seconds: int | None = None) -> dict:
    """Inspect subprocess state and stderr to determine failure reason.

    ``timeout_seconds`` is the readiness deadline that was actually applied
    (download vs. startup), reported back when the failure is a timeout.
    """
    detail: dict = _attempt_detail()

    if _process is not None and _process.poll() is not None:
        detail["exit_code"] = _process.returncode
        detail["reason"] = "process_crash"
    else:
        detail["reason"] = "startup_timeout"
        detail["timeout_seconds"] = (
            timeout_seconds if timeout_seconds is not None else config.STARTUP_TIMEOUT
        )

    if _stderr_log_handle is not None:
        with contextlib.suppress(OSError):
            lines = _tail(_stderr_log_handle.fileno(), 32 * 1024).decode("utf-8", errors="replace").splitlines()
            detail["stderr_tail"] = [line[-1000:] for line in lines[-15:]]
            text = "\n".join(lines).lower()
            if any(kw in text for kw in ("out of memory", "memoryerror", "malloc", "mlock failed")):
                detail["reason"] = "oom"
            elif any(kw in text for kw in ("no such file", "not found", "does not exist")):
                detail["reason"] = "model_not_found"
            elif "address already in use" in text:
                detail["reason"] = "port_conflict"

    vm = psutil.virtual_memory()
    detail["ram_percent_at_failure"] = vm.percent
    if vm.percent > 90:
        detail["memory_pressure"] = "critical"

    return detail


def _fail_spawn(model_name: str) -> None:
    global _state
    _state = ModelState.FAILED
    _record_failure(model_name, "worker_launch")
    events.emit(
        events.EventType.MODEL_FAILED, model=model_name,
        detail={"reason": "worker_launch", **_attempt_detail()},
    )
    _ready_event.set()


async def _terminate_current() -> None:
    global _attempt, _process, _state, _active_model, _stderr_log_handle, _inactivity_timeout, _ready_event
    prev_model = _active_model
    worker, _process = _process, None
    _state = ModelState.IDLE
    _active_model = None
    _attempt = None
    _ready_event.set()
    _ready_event = asyncio.Event()
    _inactivity_timeout = config.INACTIVITY_TIMEOUT
    if worker is not None:
        logger.info(f"Terminating subprocess for model={prev_model} pid={worker.pid}")
        worker.terminate()
        try:
            await asyncio.wait_for(
                asyncio.get_running_loop().run_in_executor(None, worker.wait),
                timeout=5,
            )
        except TimeoutError:
            logger.warning(f"Subprocess did not exit gracefully, killing pid={worker.pid}")
            worker.kill()
            await asyncio.get_running_loop().run_in_executor(None, worker.wait)
    if _stderr_log_handle is not None:
        _stderr_log_handle.close()
        _stderr_log_handle = None

    if prev_model is not None:
        # Memory snapshot after unload to show freed memory
        snap = metrics.take_memory_snapshot(active_model=None, event="model_unloaded")
        events.emit(
            events.EventType.MODEL_UNLOADED,
            model=prev_model,
            detail={
                "ram_used_gb": snap.ram_used_gb,
                "ram_available_gb": snap.ram_available_gb,
            },
        )


def _reap_exited_worker() -> bool:
    """Notice a READY worker that has exited on its own (crash, OOM kill, external kill).

    The readiness loop only watches the subprocess until it becomes READY; afterwards a dead
    worker stayed "ready" with its stale pid and every proxied request failed with 500. Reap it
    here: close handles, emit the crash event and return to IDLE so the next request reloads.
    No failure cooldown is recorded: this is not a load failure, and a reload is what callers want.
    Returns True when a dead worker was reaped.
    """
    global _process, _state, _active_model, _stderr_log_handle
    if _process is None or _state != ModelState.READY:
        return False
    exit_code = _process.poll()
    if exit_code is None:
        return False
    dead_model, pid = _active_model, _process.pid
    logger.error(
        f"Model {dead_model} worker pid={pid} exited with code {exit_code} while READY; reaping"
    )
    _process = None
    if _stderr_log_handle is not None:
        _stderr_log_handle.close()
        _stderr_log_handle = None
    _state = ModelState.IDLE
    _active_model = None
    _mark_attempt_failure("worker_exit")
    events.emit(
        events.EventType.SUBPROCESS_CRASH,
        model=dead_model,
        detail={"pid": pid, "exit_code": exit_code, "phase": "ready", **_attempt_detail()},
    )
    return True


async def _health_check_loop(worker=None) -> None:
    """Poll /health every 2s until READY or startup timeout."""
    global _process, _state, _stderr_log_handle
    if worker is None:
        worker = _process
    model_name = _active_model
    model_cfg = config.MODELS.get(model_name) if model_name else None
    readiness_timeout = _readiness_timeout(model_cfg) if model_cfg else config.STARTUP_TIMEOUT
    deadline = time.monotonic() + readiness_timeout
    verification_retried = False
    async with httpx.AsyncClient() as client:
        while True:
            if _process is not worker or _active_model != model_name:
                return
            # Detect early process exit
            exit_code = worker.poll() if worker is not None else None
            if exit_code is not None:
                if (
                    exit_code == _VERIFY_PROBE_TIMEOUT_EXIT
                    and not verification_retried
                    and model_cfg is not None
                    and _worker_verify_preference(model_cfg) == "joint-first"
                    and _saw_verify_probe_timeout()
                ):
                    async with _switch_lock:
                        if _process is not worker or _active_model != model_name:
                            return
                        verification_retried = True
                        events.emit(
                            events.EventType.SUBPROCESS_CRASH,
                            model=model_name,
                            detail={
                                "pid": worker.pid,
                                "exit_code": exit_code,
                                "phase": "verification_probe",
                                **_attempt_detail(),
                            },
                        )
                        logger.warning(
                            "Joint verification probe timed out for %s; retrying with per-query",
                            model_name,
                        )
                        if _stderr_log_handle is not None:
                            _stderr_log_handle.close()
                            _stderr_log_handle = None
                        try:
                            _process = worker = _spawn_worker(
                                model_name, model_cfg, verification_retry=True
                            )
                            deadline = time.monotonic() + readiness_timeout
                        except (OSError, RuntimeError, ValueError):
                            _process = None
                            logger.exception("Could not start per-query worker for %s", model_name)
                            _fail_spawn(model_name)
                            return
                    continue
                _state = ModelState.FAILED
                _ready_event.set()
                detail = _diagnose_failure(timeout_seconds=readiness_timeout)
                load_ms = (
                    (time.monotonic() - _loading_started_at) * 1000 if _loading_started_at else None
                )
                _record_failure(_active_model, detail.get("reason", "unknown"))
                events.emit(
                    events.EventType.MODEL_FAILED,
                    model=_active_model,
                    detail=detail,
                    duration_ms=load_ms,
                )
                if detail.get("reason") == "oom":
                    events.emit(
                        events.EventType.OOM_DETECTED,
                        model=_active_model,
                        detail=detail,
                    )
                logger.error(
                    f"Model {_active_model} failed to load: {detail.get('reason', 'unknown')} "
                    f"(logs: {_stderr_log_path})"
                )
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                resp = await client.get(
                    f"http://127.0.0.1:{config.MLX_PORT}/health",
                    timeout=min(2, remaining),
                )
                if _process is not worker or _active_model != model_name:
                    return
                if resp.status_code == 200 and worker.poll() is None:
                    _state = ModelState.READY
                    _ready_event.set()
                    _clear_failure()
                    load_ms = (
                        (time.monotonic() - _loading_started_at) * 1000
                        if _loading_started_at
                        else None
                    )

                    # Memory snapshot after model is loaded
                    pid = _process.pid if _process else None
                    snap = metrics.take_memory_snapshot(
                        active_model=_active_model,
                        event="model_loaded",
                        subprocess_pid=pid,
                    )
                    events.emit(
                        events.EventType.MODEL_READY,
                        model=_active_model,
                        detail={
                            "ram_used_gb": snap.ram_used_gb,
                            "subprocess_rss_mb": snap.subprocess_rss_mb,
                            "metal_active_mb": snap.metal_active_mb,
                            **_attempt_detail(),
                        },
                        duration_ms=load_ms,
                    )
                    events.emit(
                        events.EventType.MODEL_SWITCH_COMPLETE,
                        model=_active_model,
                        duration_ms=load_ms,
                    )
                    logger.info(
                        f"Model {_active_model} ready in {load_ms:.0f}ms | "
                        f"RAM={snap.ram_used_gb}GB | subprocess_rss={snap.subprocess_rss_mb}MB"
                    )
                    return
            except httpx.RequestError:
                pass
            await asyncio.sleep(min(2, max(0, deadline - time.monotonic())))

    # Timed out
    if _process is not worker or _active_model != model_name:
        return
    _state = ModelState.FAILED
    _ready_event.set()
    detail = _diagnose_failure(timeout_seconds=readiness_timeout)
    load_ms = (time.monotonic() - _loading_started_at) * 1000 if _loading_started_at else None
    _record_failure(_active_model, detail.get("reason", "unknown"))
    events.emit(
        events.EventType.MODEL_FAILED,
        model=_active_model,
        detail=detail,
        duration_ms=load_ms,
    )
    logger.error(
        f"Model {_active_model} timed out after {readiness_timeout}s (logs: {_stderr_log_path})"
    )


async def _switch_model(model_name: str) -> None:
    global _process, _state, _active_model
    global _loading_started_at, _load_timeout

    async with _switch_lock:
        # Another coroutine may have already loaded this model while we waited
        if _active_model == model_name and _state in (ModelState.READY, ModelState.LOADING, ModelState.DOWNLOADING):
            return

        prev_model = _active_model

        if prev_model is not None:
            events.emit(
                events.EventType.MODEL_SWITCH_START,
                model=model_name,
                detail={"from_model": prev_model, "to_model": model_name},
            )
            logger.info(f"Switching model: {prev_model} -> {model_name}")

        await _terminate_current()

        model_cfg = config.MODELS[model_name]
        cached = _is_model_cached(model_cfg)
        _load_timeout = config.STARTUP_TIMEOUT if cached else config.DOWNLOAD_TIMEOUT
        _state = ModelState.DOWNLOADING if not cached else ModelState.LOADING
        _active_model = model_name
        _loading_started_at = time.monotonic()

        if not cached:
            events.emit(
                events.EventType.MODEL_DOWNLOADING,
                model=model_name,
                detail={"hf_path": model_cfg.hf_path},
            )

        try:
            _process = _spawn_worker(model_name, model_cfg)
        except (OSError, RuntimeError, ValueError):
            logger.exception("Could not start worker for %s", model_name)
            _fail_spawn(model_name)
            return

        # Snapshot before model loads to capture baseline memory
        metrics.take_memory_snapshot(active_model=model_name, event="model_loading")

        asyncio.create_task(_health_check_loop(_process))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def ensure_model(model_name: str) -> bool:
    """
    Ensure the named model is READY before the caller proxies a request.
    Returns True if the model was freshly loaded (cold start).
    """
    global _last_request_at

    _reap_exited_worker()
    if _active_model == model_name and _state == ModelState.READY:
        _last_request_at = datetime.now(UTC)
        return False  # already warm

    cooldown = _failure_cooldown_remaining(model_name)
    if cooldown > 0:
        from fastapi import HTTPException

        raise HTTPException(
            status_code=503,
            detail=(
                f"Model {model_name} recently failed ({_last_failure_reason}); "
                f"retry in {cooldown:.0f}s"
            ),
        )

    cold_start = True

    if _active_model != model_name or _state in (ModelState.IDLE, ModelState.FAILED):
        await _switch_model(model_name)

    ready_event = _ready_event
    if _state in (ModelState.LOADING, ModelState.DOWNLOADING):
        model_cfg = config.MODELS[model_name]
        attempts = 2 if _worker_verify_preference(model_cfg) == "joint-first" else 1
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(ready_event.wait(), timeout=attempts * _readiness_timeout(model_cfg) + 5)

    if ready_event is not _ready_event or _active_model != model_name or _state != ModelState.READY:
        from fastapi import HTTPException

        raise HTTPException(status_code=503, detail=f"Model {model_name} failed to load")

    _last_request_at = datetime.now(UTC)
    return cold_start


def set_keep_alive(seconds: int) -> None:
    """Override the inactivity timeout for the current model session."""
    global _inactivity_timeout
    _inactivity_timeout = seconds


async def start_inactivity_watcher() -> None:
    """Background task: unload idle models every 30s check."""
    while True:
        await asyncio.sleep(30)
        async with _switch_lock:
            _reap_exited_worker()
            if (
                _state == ModelState.READY
                and _last_request_at is not None
                and datetime.now(UTC) - _last_request_at > timedelta(seconds=_inactivity_timeout)
            ):
                idle_secs = int((datetime.now(UTC) - _last_request_at).total_seconds())
                events.emit(
                    events.EventType.MODEL_INACTIVITY_UNLOAD,
                    model=_active_model,
                    detail={"idle_seconds": idle_secs},
                )
                logger.info(f"Model {_active_model} idle for {idle_secs}s, unloading")
                await _terminate_current()


async def unload() -> None:
    """Terminate the active subprocess and reset to IDLE."""
    async with _switch_lock:
        await _terminate_current()


# ---------------------------------------------------------------------------
# Status accessors
# ---------------------------------------------------------------------------


def get_status(*, reap: bool = True) -> dict:
    if reap:
        _reap_exited_worker()
    unload_at = None
    if _last_request_at is not None and _inactivity_timeout > 0 and _state == ModelState.READY:
        unload_at = (_last_request_at + timedelta(seconds=_inactivity_timeout)).isoformat()
    status = {
        "active_model": _active_model,
        "state": _state.value,
        "pid": _process.pid if _process else None,
        "last_request_at": _last_request_at.isoformat() if _last_request_at else None,
        "unload_at": unload_at,
        "uptime_seconds": int((datetime.now(UTC) - _started_at).total_seconds()),
        "inactivity_timeout_seconds": _inactivity_timeout,
    }
    if not reap:
        # A diagnostic read may precede lifecycle reaping. Project what we observed
        # without clearing the worker, emitting events, or starting another attempt.
        exited = _state == ModelState.READY and _process is not None and _process.poll() is not None
        if exited:
            status.update(state="failed", pid=None, unload_at=None)
        failure_reason = "worker_exit" if exited else (_attempt.failure_reason if _attempt else None)
        failure_at = time.time() if exited else (_attempt.failure_at if _attempt else None)
        status.update({
            "attempt_id": _attempt.id if _attempt else None,
            "attempt_model": _attempt.model if _attempt else None,
            "attempt_started_at": _attempt.started_at if _attempt else None,
            "attempt_state": ("failed" if failure_reason else _state.value) if _attempt else None,
            "failure_reason": failure_reason,
            "failure_at": failure_at,
        })
        if _last_failed_attempt:
            status["last_failed_attempt"] = {
                "attempt_id": _last_failed_attempt.id,
                "model": _last_failed_attempt.model,
                "reason": _last_failed_attempt.failure_reason,
                "failure_at": _last_failed_attempt.failure_at,
            }
    return status


def get_process_pid() -> int | None:
    """Return the PID of the active subprocess, or None."""
    return _process.pid if _process else None


def get_active_model_name() -> str | None:
    """Return the name of the currently active subprocess model."""
    return _active_model
