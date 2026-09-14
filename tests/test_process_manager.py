"""Tests for subprocess command construction, and for process_manager lifecycle
logic: failure cooldown and download-aware startup timeout.

The lifecycle tests exercise the pure decision helpers plus ensure_model/
_switch_model behavior, without spawning real MLX subprocesses.
"""

import importlib
import json
import os
import time
from pathlib import Path

import pytest
from fastapi import HTTPException

from mlx_serve import process_manager as pm
from mlx_serve.config import ModelConfig


def test_build_command_emits_generation_defaults_json(monkeypatch):
    """generation_defaults is forwarded as one opaque --generation-defaults JSON arg.

    mlx-serve does not enumerate the keys — it round-trips the whole dict, so a new
    generation param needs no mlx-serve change. enable_thinking now lives inside the
    block (no dedicated --enable-thinking flag)."""
    # Bypass the executable-exists guard so the test runs without mlx-vlm installed.
    monkeypatch.setattr(pm, "_MLX_VLM_SERVER", Path("/"))

    defaults = {"temperature": 0.3, "top_p": 0.95, "top_k": 20, "enable_thinking": True}
    cfg = ModelConfig(name="t", type="vision", hf_path="x", generation_defaults=defaults)
    cmd = pm._build_command(cfg)

    assert json.loads(cmd[cmd.index("--generation-defaults") + 1]) == defaults
    # enable_thinking is no longer its own flag — it travels inside the block.
    assert "--enable-thinking" not in cmd


def test_build_command_omits_generation_defaults_when_empty(monkeypatch):
    """No generation_defaults -> no --generation-defaults flag (and no --enable-thinking)."""
    monkeypatch.setattr(pm, "_MLX_VLM_SERVER", Path("/"))

    cmd = pm._build_command(ModelConfig(name="t", type="vision", hf_path="x"))
    assert "--generation-defaults" not in cmd
    assert "--enable-thinking" not in cmd


def test_build_command_emits_suffix_draft_flags(monkeypatch):
    """draft_kind + suffix knobs are threaded to the mlx-vlm server CLI."""
    monkeypatch.setattr(pm, "_MLX_VLM_SERVER", Path("/"))

    cfg = ModelConfig(
        name="t",
        type="vision",
        hf_path="x",
        draft_kind="suffix",
        draft_block_size=16,
        suffix_min_match=2,
        draft_cooldown=3,
    )
    cmd = pm._build_command(cfg)

    assert cmd[cmd.index("--draft-kind") + 1] == "suffix"
    assert cmd[cmd.index("--draft-block-size") + 1] == "16"
    assert cmd[cmd.index("--suffix-min-match") + 1] == "2"
    assert cmd[cmd.index("--draft-cooldown") + 1] == "3"


def test_build_command_omits_draft_flags_by_default(monkeypatch):
    """No draft_kind -> no draft flags (and cooldown=0 -> omitted)."""
    monkeypatch.setattr(pm, "_MLX_VLM_SERVER", Path("/"))

    cmd = pm._build_command(ModelConfig(name="t", type="vision", hf_path="x"))
    assert "--draft-kind" not in cmd
    assert "--draft-cooldown" not in cmd

    # cooldown is opt-in even when suffix is on
    cmd2 = pm._build_command(ModelConfig(name="t", type="vision", hf_path="x", draft_kind="suffix"))
    assert "--draft-kind" in cmd2
    assert "--draft-cooldown" not in cmd2


def test_kv_prealloc_tokens_emitted_for_vision(monkeypatch):
    """--kv-prealloc-tokens is emitted for vision workers when kv_prealloc_tokens > 0."""
    monkeypatch.setattr(pm, "_MLX_VLM_SERVER", Path("/"))

    cfg = ModelConfig(name="m", type="vision", hf_path="x",
                      max_kv_cache_size=262144, kv_prealloc_tokens=262144)
    cmd = pm._build_command(cfg)
    assert "--kv-prealloc-tokens" in cmd
    assert cmd[cmd.index("--kv-prealloc-tokens") + 1] == "262144"


def test_kv_prealloc_tokens_omitted_when_zero_or_non_vision(monkeypatch):
    """No --kv-prealloc-tokens when unset, and never for non-vision (text) workers."""
    monkeypatch.setattr(pm, "_MLX_VLM_SERVER", Path("/"))
    monkeypatch.setattr(pm, "_MLX_LM_SERVER", Path("/"))

    cmd = pm._build_command(ModelConfig(name="m", type="vision", hf_path="x"))
    assert "--kv-prealloc-tokens" not in cmd

    cmd2 = pm._build_command(
        ModelConfig(name="m", type="text", hf_path="x", kv_prealloc_tokens=8192)
    )
    assert "--kv-prealloc-tokens" not in cmd2


def test_build_command_emits_kv_quant_mode(monkeypatch):
    """--kv-quant-mode is emitted iff ModelConfig.kv_quant_mode is set."""
    monkeypatch.setattr(pm, "_MLX_VLM_SERVER", Path("/"))

    on = ModelConfig(name="t", type="vision", hf_path="x",
                     kv_bits=3, kv_quant_scheme="turboquant", kv_quant_mode="prod")
    off = ModelConfig(name="t", type="vision", hf_path="x",
                      kv_bits=3, kv_quant_scheme="turboquant")  # mode unset

    cmd_on = pm._build_command(on)
    assert cmd_on[cmd_on.index("--kv-quant-mode") + 1] == "prod"
    assert "--kv-quant-mode" not in pm._build_command(off)


def test_build_command_emits_moe_expand(monkeypatch):
    """moe_expand is forwarded as --moe-expand <str> for both text and vision workers."""
    monkeypatch.setattr(pm, "_MLX_VLM_SERVER", Path("/"))
    monkeypatch.setattr(pm, "_MLX_LM_SERVER", Path("/"))

    vision_cfg = ModelConfig(name="t", type="vision", hf_path="x", moe_expand="27-39:20:0.8:0.5")
    cmd = pm._build_command(vision_cfg)
    assert cmd[cmd.index("--moe-expand") + 1] == "27-39:20:0.8:0.5"

    text_cfg = ModelConfig(name="t", type="text", hf_path="x", moe_expand="27-39:20:0.8:0.5")
    cmd2 = pm._build_command(text_cfg)
    assert cmd2[cmd2.index("--moe-expand") + 1] == "27-39:20:0.8:0.5"


def test_build_command_omits_moe_expand_when_empty(monkeypatch):
    """No moe_expand -> no --moe-expand flag."""
    monkeypatch.setattr(pm, "_MLX_VLM_SERVER", Path("/"))
    monkeypatch.setattr(pm, "_MLX_LM_SERVER", Path("/"))

    assert "--moe-expand" not in pm._build_command(ModelConfig(name="t", type="vision", hf_path="x"))
    assert "--moe-expand" not in pm._build_command(ModelConfig(name="t", type="text", hf_path="x"))


@pytest.fixture()
def reloaded_pm(tmp_config, log_dir):
    """Reload config + process_manager against the temp config, fresh state."""
    import mlx_serve.config

    importlib.reload(mlx_serve.config)
    import mlx_serve.events as events

    events.configure(log_dir)

    import mlx_serve.process_manager as process_manager

    importlib.reload(process_manager)
    return process_manager


# ---------------------------------------------------------------------------
# Bug 2 — failure cooldown / circuit breaker
# ---------------------------------------------------------------------------


def test_failure_cooldown_blocks_recent_failure(reloaded_pm):
    reloaded_pm._record_failure("test-text-model", "startup_timeout")
    assert reloaded_pm._failure_cooldown_remaining("test-text-model") > 0


def test_failure_cooldown_expires_after_window(reloaded_pm):
    reloaded_pm._record_failure("test-text-model", "startup_timeout")
    # Pretend the failure happened a full cooldown ago.
    reloaded_pm._last_failure_at = time.monotonic() - (
        reloaded_pm.config.FAILURE_COOLDOWN + 1
    )
    assert reloaded_pm._failure_cooldown_remaining("test-text-model") == 0


def test_failure_cooldown_only_affects_failed_model(reloaded_pm):
    reloaded_pm._record_failure("test-text-model", "startup_timeout")
    assert reloaded_pm._failure_cooldown_remaining("test-vision-model") == 0


def test_clear_failure_resets_cooldown(reloaded_pm):
    reloaded_pm._record_failure("test-text-model", "startup_timeout")
    reloaded_pm._clear_failure()
    assert reloaded_pm._failure_cooldown_remaining("test-text-model") == 0


@pytest.mark.asyncio
async def test_ensure_model_short_circuits_during_cooldown(reloaded_pm, monkeypatch):
    """A request for a recently-failed model returns 503 immediately and
    does NOT spawn another subprocess (the infinite-respawn bug)."""
    spawned = False

    async def fake_switch(model_name):
        nonlocal spawned
        spawned = True

    monkeypatch.setattr(reloaded_pm, "_switch_model", fake_switch)
    reloaded_pm._record_failure("test-text-model", "startup_timeout")

    with pytest.raises(HTTPException) as exc:
        await reloaded_pm.ensure_model("test-text-model")

    assert exc.value.status_code == 503
    assert spawned is False


# ---------------------------------------------------------------------------
# Bug 1 — download-aware readiness timeout + downloading state/event
# ---------------------------------------------------------------------------


def test_readiness_timeout_uses_download_timeout_when_not_cached(
    reloaded_pm, monkeypatch
):
    monkeypatch.setattr(reloaded_pm, "_is_model_cached", lambda cfg: False)
    cfg = reloaded_pm.config.MODELS["test-text-model"]
    assert reloaded_pm._readiness_timeout(cfg) == reloaded_pm.config.DOWNLOAD_TIMEOUT


def test_readiness_timeout_uses_startup_timeout_when_cached(reloaded_pm, monkeypatch):
    monkeypatch.setattr(reloaded_pm, "_is_model_cached", lambda cfg: True)
    cfg = reloaded_pm.config.MODELS["test-text-model"]
    assert reloaded_pm._readiness_timeout(cfg) == reloaded_pm.config.STARTUP_TIMEOUT


def test_diagnose_failure_reports_actual_timeout(reloaded_pm):
    """On a startup/download timeout, the reported timeout_seconds reflects the
    deadline actually used, not the bare STARTUP_TIMEOUT."""

    class StillRunning:
        returncode = None

        def poll(self):
            return None  # process alive -> this is a timeout, not a crash

    reloaded_pm._process = StillRunning()
    detail = reloaded_pm._diagnose_failure(
        timeout_seconds=reloaded_pm.config.DOWNLOAD_TIMEOUT
    )
    assert detail["reason"] == "startup_timeout"
    assert detail["timeout_seconds"] == reloaded_pm.config.DOWNLOAD_TIMEOUT


@pytest.mark.asyncio
async def test_switch_emits_downloading_event_when_not_cached(
    reloaded_pm, monkeypatch
):
    """When a model isn't cached, the switch enters DOWNLOADING and emits a
    model.downloading event so the hang is visible to the user."""
    import mlx_serve.events as events

    monkeypatch.setattr(reloaded_pm, "_is_model_cached", lambda cfg: False)
    monkeypatch.setattr(reloaded_pm, "_MLX_LM_SERVER", Path("/"))

    class FakeProc:
        pid = 4321

        def poll(self):
            return None

    monkeypatch.setattr(reloaded_pm.subprocess, "Popen", lambda *a, **k: FakeProc())
    # Don't run the real health loop (would poll a nonexistent server).
    monkeypatch.setattr(reloaded_pm.asyncio, "create_task", lambda coro: coro.close())

    await reloaded_pm._switch_model("test-text-model")

    downloading = events.get_events(event_type="model.downloading")
    assert any(e["model"] == "test-text-model" for e in downloading)
    assert reloaded_pm._state == reloaded_pm.ModelState.DOWNLOADING


@pytest.mark.parametrize("setting,expected", [(True, "on"), (False, "off")])
def test_cache_session_shrink_forwards_explicit_policy(monkeypatch, setting, expected):
    monkeypatch.setattr(pm, "_MLX_VLM_SERVER", Path("/"))
    # The explicit per-model flag overrides the opposite inherited worker default.
    inherited = "off" if setting else "on"
    monkeypatch.setenv("MLX_VLM_SESSION_SHRINK_ON_RETIRE", inherited)
    model = ModelConfig(
        name="configured", type="vision", hf_path="local-model", cache_session_shrink=setting
    )
    command = pm._build_command(model)
    assert command.count("--cache-session-shrink") == 1
    assert command[command.index("--cache-session-shrink") + 1] == expected
    assert os.environ["MLX_VLM_SESSION_SHRINK_ON_RETIRE"] == inherited


def test_unset_cache_session_shrink_leaves_worker_default(monkeypatch):
    monkeypatch.setattr(pm, "_MLX_VLM_SERVER", Path("/"))
    monkeypatch.setenv("MLX_VLM_SESSION_SHRINK_ON_RETIRE", "on")
    model = ModelConfig(name="unchanged", type="vision", hf_path="local-model")
    assert "--cache-session-shrink" not in pm._build_command(model)
