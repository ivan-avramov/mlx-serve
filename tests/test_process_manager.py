"""Tests for subprocess command construction."""

import json
from pathlib import Path

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
