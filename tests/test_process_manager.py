"""Tests for subprocess command construction."""

from pathlib import Path

from mlx_serve import process_manager as pm
from mlx_serve.config import ModelConfig


def test_build_command_includes_enable_thinking(monkeypatch):
    """--enable-thinking is emitted iff ModelConfig.enable_thinking is True."""
    # Bypass the executable-exists guard so the test runs without mlx-vlm installed.
    monkeypatch.setattr(pm, "_MLX_VLM_SERVER", Path("/"))

    on = ModelConfig(name="t", type="vision", hf_path="x", enable_thinking=True)
    off = ModelConfig(name="t", type="vision", hf_path="x", enable_thinking=False)

    assert "--enable-thinking" in pm._build_command(on)
    assert "--enable-thinking" not in pm._build_command(off)
