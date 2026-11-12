"""M77: a ``tq4_attention`` entry with MTP OFF (no ``draft_kind``, no ``mtp_verify_scan``) validates and builds a worker argv
with ``--tq4-attention`` and without ``--mtp-verify-scan`` / ``--draft-*``; every other combination keeps M72's outcome and
message."""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest

from mlx_serve import process_manager as pm
from mlx_serve.config import ModelConfig

MODES = ("m72_v1", "m72_exact")
OFF = {
    "name": "xo",
    "type": "vision",
    "hf_path": "org/m",
    "kv_bits": 4,
    "kv_quant_scheme": "turboquant",
    "attention_policy": "fused_v1",
}

SCAN_MSG = "requires mtp_verify_scan 'joint_v1'"
JOINT_NEEDS_MTP = "mtp_verify_scan 'joint_v1' requires draft_kind 'mtp'"


@pytest.mark.parametrize("scan", ["", "per_query"])
@pytest.mark.parametrize("mode", MODES)
def test_mtp_off_entry_validates(mode, scan):
    cfg = ModelConfig(**OFF, tq4_attention=mode, mtp_verify_scan=scan)
    assert cfg.tq4_attention == mode and cfg.draft_kind == ""


@pytest.mark.parametrize("mode", MODES)
def test_mtp_off_argv_is_as_traced(monkeypatch, mode):
    monkeypatch.setattr(pm, "_MLX_VLM_SERVER", Path("/"))
    cmd = pm._build_command(ModelConfig(**OFF, tq4_attention=mode))
    for flag, value in (
        ("--kv-bits", "4"),
        ("--kv-quant-scheme", "turboquant"),
        ("--attention-policy", "fused_v1"),
        ("--tq4-attention", mode),
    ):
        assert cmd.count(flag) == 1 and cmd[cmd.index(flag) + 1] == value
    for flag in ("--mtp-verify-scan", "--mtp-verify-ab", "--draft-kind", "--draft-model"):
        assert flag not in cmd
    assert not [a for a in cmd if a.startswith("--draft")]


@pytest.mark.parametrize("mode", MODES)
def test_mtp_off_with_a_draft_model_path_alone_passes_no_drafter(monkeypatch, mode):
    # draft_model without draft_kind is not forwarded (process_manager gates --draft-* on draft_kind)
    monkeypatch.setattr(pm, "_MLX_VLM_SERVER", Path("/"))
    cmd = pm._build_command(ModelConfig(**OFF, tq4_attention=mode, draft_model="/d"))
    assert not [a for a in cmd if a.startswith("--draft")]


@pytest.mark.parametrize("mode", MODES)
def test_mode_with_mtp_without_joint_v1_is_refused(mode):
    with pytest.raises(ValueError, match=SCAN_MSG):
        ModelConfig(**OFF, tq4_attention=mode, draft_kind="mtp", draft_model="/d")


@pytest.mark.parametrize("mode", MODES)
def test_mode_without_drafter_with_joint_v1_is_refused_by_the_scan_rule(mode):
    with pytest.raises(ValueError, match=JOINT_NEEDS_MTP):
        ModelConfig(**OFF, tq4_attention=mode, mtp_verify_scan="joint_v1")


@pytest.mark.parametrize("scan", ["", "per_query"])
@pytest.mark.parametrize("mode", MODES)
def test_mode_without_drafter_with_ab_is_refused(mode, scan):
    with pytest.raises(ValueError, match=SCAN_MSG):
        ModelConfig(**OFF, tq4_attention=mode, mtp_verify_scan=scan, mtp_verify_ab=True)


@pytest.mark.parametrize("scan", ["", "per_query", "joint_v1"])
@pytest.mark.parametrize("kind", ["suffix", "dflash", "eagle3"])
@pytest.mark.parametrize("mode", MODES)
def test_mode_with_other_drafter_kinds_is_refused(mode, kind, scan):
    with pytest.raises(ValueError, match=JOINT_NEEDS_MTP if scan == "joint_v1" else SCAN_MSG):
        ModelConfig(
            **OFF, tq4_attention=mode, draft_kind=kind, draft_model="/d", mtp_verify_scan=scan
        )


def _m72_outcome(mode, kind, scan, ab):
    """M72's outcome for the drafter/scan grid (the other companions valid): None = accepted, else the message part."""
    if mode == "":
        return "attention_policy 'fused_v1' requires kv_bits 0"  # kv_bits 4 needs the mode
    if scan != "joint_v1":
        return SCAN_MSG
    if kind != "mtp":
        return JOINT_NEEDS_MTP
    return None


GRID = list(
    itertools.product(
        ("",) + MODES,
        ("", "mtp", "suffix", "dflash", "eagle3"),
        ("", "per_query", "joint_v1"),
        (False, True),
    )
)


@pytest.mark.parametrize("mode,kind,scan,ab", GRID)
def test_only_the_mtp_off_set_changes_outcome(mode, kind, scan, ab):
    new_accept = mode != "" and kind == "" and scan in ("", "per_query") and not ab
    expected = None if new_accept else _m72_outcome(mode, kind, scan, ab)
    kw = dict(OFF, tq4_attention=mode, draft_kind=kind, mtp_verify_scan=scan, mtp_verify_ab=ab)
    if kind:
        kw["draft_model"] = "/d"
    if expected is None:
        ModelConfig(**kw)
    else:
        with pytest.raises(ValueError, match=expected):
            ModelConfig(**kw)
