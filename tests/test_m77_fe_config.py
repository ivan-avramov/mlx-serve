"""M77 (serve `tq4_attention` with MTP OFF): registry validation and the worker argv in mlx-serve.

Spec: ``mlx-vlm/.fe-material/m77-tq4-mtp-off.md`` (Interfaces: ``ModelConfig._validate_tq4_attention``; acceptance criterion 1).
Entries are parsed the way mlx-serve parses a registry (``config._load``); the argv is ``process_manager._build_command``.

Every test is classed in its comment: (a) reproduces a failure the specification fixes, (b) preserves an invariant,
(c) new behaviour.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from mlx_serve import process_manager as pm
from mlx_serve.config import ModelConfig

MODES = ("m72_v1", "m72_exact")
OFF = dict(
    type="vision",
    hf_path="org/model",
    kv_bits=4,
    kv_quant_scheme="turboquant",
    attention_policy="fused_v1",
)
ON = dict(OFF, mtp_verify_scan="joint_v1", draft_kind="mtp")


def make(base, **over):
    return ModelConfig(name="configured", **{**base, **over})


def load(tmp_path, monkeypatch, **entry):
    """One registry entry through the YAML loader (what mlx-serve does at start-up)."""
    from mlx_serve import config

    doc = {"models": [{"name": "configured", "type": "vision", "hf_path": "local-model", **entry}]}
    path = tmp_path / "m77.yaml"
    path.write_text(yaml.safe_dump(doc))
    monkeypatch.setattr(config, "_CONFIG_PATH", path)
    return config._load()[0]["configured"]


def cmd_of(cfg, monkeypatch):
    monkeypatch.setattr(pm, "_MLX_VLM_SERVER", Path("/"))
    return pm._build_command(cfg)


# ============================================================ accepted: the MTP-OFF success entry
# (c) the traced success entry validates, both modes, built directly
@pytest.mark.parametrize("mode", MODES)
def test_off_entry_validates(mode):
    cfg = make(OFF, tq4_attention=mode)
    assert cfg.tq4_attention == mode
    assert cfg.draft_kind == "" and cfg.mtp_verify_scan in ("", "per_query")


# (c) ... and from a registry file (draft keys omitted, scan omitted)
@pytest.mark.parametrize("mode", MODES)
def test_off_entry_loads_from_registry(tmp_path, monkeypatch, mode):
    entry = {k: v for k, v in OFF.items() if k not in ("type", "hf_path")}
    cfg = load(tmp_path, monkeypatch, tq4_attention=mode, **entry)
    assert (cfg.tq4_attention, cfg.kv_bits, cfg.attention_policy) == (mode, 4, "fused_v1")
    assert cfg.draft_kind == ""


# (c) explicit `mtp_verify_scan: per_query` / "" with no drafter is also OFF
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("scan", ["", "per_query"])
def test_off_entry_accepts_explicit_no_joint_scan(mode, scan):
    assert make(OFF, tq4_attention=mode, mtp_verify_scan=scan, draft_kind="").tq4_attention == mode


# (c) kv_quant_mode "mse" stays accepted in the OFF shape (today's rule)
@pytest.mark.parametrize("mode", MODES)
def test_off_entry_accepts_kv_quant_mode_mse(mode):
    assert make(OFF, tq4_attention=mode, kv_quant_mode="mse").tq4_attention == mode


# ============================================================ argv
# (c) the exact argv fragment of the traced success path
@pytest.mark.parametrize("mode", MODES)
def test_off_entry_argv_has_mode_and_no_mtp_or_draft_flags(monkeypatch, mode):
    cmd = cmd_of(make(OFF, tq4_attention=mode), monkeypatch)
    assert cmd.count("--tq4-attention") == 1 and cmd[cmd.index("--tq4-attention") + 1] == mode
    assert cmd[cmd.index("--kv-bits") + 1] == "4"
    assert cmd[cmd.index("--kv-quant-scheme") + 1] == "turboquant"
    assert cmd[cmd.index("--attention-policy") + 1] == "fused_v1"
    assert "--mtp-verify-scan" not in cmd
    assert "--mtp-verify-ab" not in cmd
    assert not [a for a in cmd if a.startswith("--draft-")], cmd


# (c) the same through the registry loader
@pytest.mark.parametrize("mode", MODES)
def test_off_entry_argv_from_registry(tmp_path, monkeypatch, mode):
    entry = {k: v for k, v in OFF.items() if k not in ("type", "hf_path")}
    cmd = cmd_of(load(tmp_path, monkeypatch, tq4_attention=mode, **entry), monkeypatch)
    assert cmd[cmd.index("--tq4-attention") + 1] == mode
    assert (
        "--mtp-verify-scan" not in cmd and "--draft-kind" not in cmd and "--draft-model" not in cmd
    )


# (b) pins process_manager: with `mtp_verify_scan: per_query` the scan flag is still not emitted
@pytest.mark.parametrize("mode", MODES)
def test_off_entry_per_query_scan_emits_no_flag(monkeypatch, mode):
    cmd = cmd_of(make(OFF, tq4_attention=mode, mtp_verify_scan="per_query"), monkeypatch)
    assert "--mtp-verify-scan" not in cmd


# (b) MTP-ON argv unchanged: scan, draft kind and mode all present
@pytest.mark.parametrize("mode", MODES)
def test_on_entry_argv_unchanged(monkeypatch, mode):
    cmd = cmd_of(make(ON, tq4_attention=mode, draft_model="/d"), monkeypatch)
    assert cmd[cmd.index("--mtp-verify-scan") + 1] == "joint_v1"
    assert cmd[cmd.index("--draft-kind") + 1] == "mtp"
    assert cmd[cmd.index("--draft-model") + 1] == "/d"
    assert cmd[cmd.index("--tq4-attention") + 1] == mode


# ============================================================ MTP-ON accepted/refused exactly as today
# (b) full MTP-ON combination accepted
@pytest.mark.parametrize("mode", MODES)
def test_on_entry_still_validates(mode):
    assert make(ON, tq4_attention=mode).tq4_attention == mode


# (b) draft_kind mtp + mode without joint_v1 stays refused with today's message
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("scan", ["", "per_query"])
def test_mtp_with_mode_without_joint_v1_is_refused_with_todays_message(mode, scan):
    with pytest.raises(ValueError, match=r"tq4_attention .* requires mtp_verify_scan 'joint_v1'"):
        make(OFF, tq4_attention=mode, draft_kind="mtp", mtp_verify_scan=scan)


# (b) joint_v1 with draft_kind mtp and mtp_verify_ab stays accepted
@pytest.mark.parametrize("mode", MODES)
def test_on_entry_with_ab_still_accepted(mode):
    assert make(ON, tq4_attention=mode, mtp_verify_ab=True).mtp_verify_ab is True


# (b) every other companion still refuses with the MTP-ON shape
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    "over",
    [
        dict(kv_bits=0),
        dict(kv_bits=8),
        dict(kv_quant_scheme="uniform"),
        dict(attention_policy=""),
        dict(attention_policy="auto"),
        dict(type="text"),
        dict(kv_quant_mode="prod"),
    ],
    ids=lambda o: "-".join(f"{k}={v!r}" for k, v in o.items()),
)
def test_on_entry_companion_refusals_unchanged(mode, over):
    with pytest.raises(ValueError):
        make(ON, tq4_attention=mode, **over)


# ============================================================ OFF entry: the same companion rules still refuse
# (b) in the OFF shape each companion alone wrong refuses (today via the scan rule too; after, via the companion rule)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    "over",
    [
        dict(kv_bits=0),
        dict(kv_bits=2),
        dict(kv_bits=8),
        dict(kv_quant_scheme=""),
        dict(kv_quant_scheme="uniform"),
        dict(attention_policy=""),
        dict(attention_policy="auto"),
        dict(type="text"),
        dict(kv_quant_mode="prod"),
    ],
    ids=lambda o: "-".join(f"{k}={v!r}" for k, v in o.items()),
)
def test_off_entry_companion_refusals(mode, over):
    with pytest.raises(ValueError):
        make(OFF, tq4_attention=mode, **over)


# ============================================================ refused: what OFF must not open
# (b) mode + no drafter + joint_v1 stays refused (by _validate_mtp_verify_scan), with its message
@pytest.mark.parametrize("mode", MODES)
def test_joint_v1_without_a_drafter_is_refused(mode):
    with pytest.raises(ValueError, match=r"joint_v1.*requires draft_kind 'mtp'"):
        make(OFF, tq4_attention=mode, mtp_verify_scan="joint_v1")


# (b) mode + no drafter + mtp_verify_ab true: refused today (by the missing joint_v1); after the change the OFF rule must keep refusing it
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("scan", ["", "per_query"])
def test_ab_without_a_drafter_is_refused(mode, scan):
    with pytest.raises(ValueError):
        make(OFF, tq4_attention=mode, mtp_verify_scan=scan, mtp_verify_ab=True)


# (b) mode + other drafter kinds: refused today with every scan spelling (by the scan rule); the change must keep refusing
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("kind", ["suffix", "dflash", "eagle3"])
@pytest.mark.parametrize("scan", ["", "per_query", "joint_v1"])
def test_other_drafter_kinds_with_the_mode_are_refused(mode, kind, scan):
    with pytest.raises(ValueError):
        make(OFF, tq4_attention=mode, draft_kind=kind, mtp_verify_scan=scan)


# (b) an unknown draft kind with the mode is refused
@pytest.mark.parametrize("mode", MODES)
def test_unknown_draft_kind_is_refused(mode):
    with pytest.raises(ValueError):
        make(OFF, tq4_attention=mode, draft_kind="bogus")


# ============================================================ the sentinel: omitted draft_kind => the empty string, i.e. OFF
# (c) a registry entry without any draft key parses to draft_kind "" (the OFF discriminator)
def test_omitted_draft_kind_parses_as_empty(tmp_path, monkeypatch):
    assert load(tmp_path, monkeypatch).draft_kind == ""


# ============================================================ mode absent: today
# (b) no mode, no drafter, kv_bits 4 + fused_v1 stays refused (the relaxation is only with the mode)
def test_fused_v1_kv_bits_4_without_the_mode_stays_refused():
    with pytest.raises(ValueError, match="fused_v1.*kv_bits"):
        make(OFF)


# (b) no mode: plain TQ4 entry (no policies) stays accepted and its argv has no mode flag
def test_plain_tq4_without_the_mode_unchanged(monkeypatch):
    cfg = ModelConfig(name="p", type="vision", hf_path="x", kv_bits=4, kv_quant_scheme="turboquant")
    assert not cfg.tq4_attention
    assert "--tq4-attention" not in cmd_of(cfg, monkeypatch)


# (b) no mode: MTP shipped native combination stays accepted
def test_native_mtp_combination_without_the_mode_unchanged():
    cfg = make(ON, kv_bits=0)
    assert cfg.kv_bits == 0 and not cfg.tq4_attention


# (b) no mode: other drafter kinds without the mode stay accepted (the new refusal is mode-scoped)
@pytest.mark.parametrize("kind", ["suffix", "dflash", "eagle3"])
def test_other_drafter_kinds_without_the_mode_stay_accepted(kind):
    assert make(dict(type="vision", hf_path="x"), draft_kind=kind).draft_kind == kind


# (b) no mode: mtp_verify_ab without joint_v1 stays refused
def test_ab_without_joint_v1_and_without_the_mode_stays_refused():
    with pytest.raises(ValueError, match="mtp_verify_ab"):
        ModelConfig(name="x", type="vision", hf_path="x", mtp_verify_ab=True)


# (b) the mode on a native cache stays refused
@pytest.mark.parametrize("mode", MODES)
def test_mode_on_a_native_cache_stays_refused(mode):
    with pytest.raises(ValueError):
        make(OFF, tq4_attention=mode, kv_bits=0)


# (b) unknown mode value stays refused
@pytest.mark.parametrize("value", ["m72_v2", "on", "", None])
def test_unknown_mode_value_in_registry_is_refused_or_absent(tmp_path, monkeypatch, value):
    entry = {k: v for k, v in OFF.items() if k not in ("type", "hf_path")}
    with pytest.raises(ValueError, match="tq4_attention"):
        load(tmp_path, monkeypatch, tq4_attention=value, **entry)
