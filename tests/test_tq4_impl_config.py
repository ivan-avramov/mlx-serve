"""M72 1.3 (A8): registry key `tq4_attention` — the four-way validation, the kv_bits 4 relaxation of
`fused_v1` / `joint_v1` (only with a mode), loading from YAML and the worker flag pass-through."""

from pathlib import Path

import pytest
import yaml

from mlx_serve import config
from mlx_serve import process_manager as pm
from mlx_serve.config import ModelConfig

MODES = ("m72_v1", "m72_exact")


def _arm(**over):
    """The arm entry 1.4 runs: TQ4 KV + MTP + both shipped native16 policies + a mode."""
    fields = dict(
        name="arm",
        type="vision",
        hf_path="local-model",
        kv_bits=4,
        kv_quant_scheme="turboquant",
        draft_kind="mtp",
        attention_policy="fused_v1",
        mtp_verify_scan="joint_v1",
        tq4_attention="m72_v1",
    )
    fields.update(over)
    return ModelConfig(**fields)


@pytest.mark.parametrize("mode", MODES)
def test_valid_arm_with_each_mode(mode):
    assert _arm(tq4_attention=mode).tq4_attention == mode


def test_absent_is_the_default():
    assert ModelConfig(name="m", type="vision", hf_path="x").tq4_attention == ""


@pytest.mark.parametrize("value", ["off", "m72", "M72_V1", "joint_v1", "true"])
def test_unknown_value_refused(value):
    with pytest.raises(ValueError, match="tq4_attention"):
        _arm(tq4_attention=value)


@pytest.mark.parametrize(
    "over,needle",
    [
        ({"kv_bits": 0}, "kv_bits 4"),
        ({"kv_bits": 8}, "kv_bits 4"),
        ({"kv_quant_scheme": "uniform"}, "turboquant"),
        ({"kv_quant_scheme": ""}, "turboquant"),
        ({"attention_policy": ""}, "fused_v1"),
        ({"attention_policy": "auto"}, "fused_v1"),
        ({"mtp_verify_scan": ""}, "joint_v1"),
        ({"mtp_verify_scan": "per_query"}, "joint_v1"),
        ({"type": "text"}, "vision"),
    ],
)
@pytest.mark.parametrize("mode", MODES)
def test_each_mode_requires_all_four(mode, over, needle):
    with pytest.raises(ValueError, match=needle):
        _arm(tq4_attention=mode, **over)


def test_fused_v1_with_kv_bits_4_refused_without_the_mode():
    with pytest.raises(ValueError, match="attention_policy 'fused_v1' requires kv_bits 0"):
        ModelConfig(
            name="m",
            type="vision",
            hf_path="x",
            kv_bits=4,
            kv_quant_scheme="turboquant",
            attention_policy="fused_v1",
        )


def test_joint_v1_with_kv_bits_4_refused_without_the_mode():
    with pytest.raises(ValueError, match="mtp_verify_scan 'joint_v1' requires kv_bits 0"):
        ModelConfig(
            name="m",
            type="vision",
            hf_path="x",
            kv_bits=4,
            kv_quant_scheme="turboquant",
            draft_kind="mtp",
            mtp_verify_scan="joint_v1",
        )


@pytest.mark.parametrize("bits", [2, 3, 8])
def test_other_kv_bits_still_refused_for_the_policies(bits):
    with pytest.raises(ValueError, match="requires kv_bits 0"):
        ModelConfig(name="m", type="vision", hf_path="x", kv_bits=bits, attention_policy="fused_v1")


def test_native16_pick_unchanged():
    cfg = ModelConfig(
        name="m",
        type="vision",
        hf_path="x",
        kv_bits=0,
        kv_quant_scheme="turboquant",
        draft_kind="mtp",
        attention_policy="fused_v1",
        mtp_verify_scan="joint_v1",
    )
    assert cfg.tq4_attention == ""


def _load(tmp_path, monkeypatch, **entry):
    doc = {"name": "configured", "type": "vision", "hf_path": "local-model", **entry}
    path = tmp_path / "tq4.yaml"
    path.write_text(yaml.safe_dump({"models": [doc]}))
    monkeypatch.setattr(config, "_CONFIG_PATH", path)
    return config._load()[0]["configured"]


ARM_YAML = dict(
    kv_bits=4,
    kv_quant_scheme="turboquant",
    draft_kind="mtp",
    attention_policy="fused_v1",
    mtp_verify_scan="joint_v1",
)


@pytest.mark.parametrize("mode", MODES)
def test_yaml_key_loads(tmp_path, monkeypatch, mode):
    assert _load(tmp_path, monkeypatch, tq4_attention=mode, **ARM_YAML).tq4_attention == mode


def test_yaml_key_absent(tmp_path, monkeypatch):
    assert _load(tmp_path, monkeypatch).tq4_attention == ""


@pytest.mark.parametrize(
    "value", [False, 0, None, "", [], ["m72_v1"], "off", "M72_V1", True, 1, {}]
)
def test_yaml_explicit_non_mode_value_refused(tmp_path, monkeypatch, value):
    # an explicit key is never read as absent, with or without the companions
    with pytest.raises(ValueError, match="tq4_attention"):
        _load(tmp_path, monkeypatch, tq4_attention=value)
    with pytest.raises(ValueError, match="tq4_attention"):
        _load(tmp_path, monkeypatch, tq4_attention=value, **ARM_YAML)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("kv_quant_mode", ["", "mse"])
def test_mode_accepts_kv_quant_mode_absent_or_mse(mode, kv_quant_mode):
    assert _arm(tq4_attention=mode, kv_quant_mode=kv_quant_mode).tq4_attention == mode


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("kv_quant_mode", ["prod", "MSE", "qjl", None])
def test_mode_refuses_other_kv_quant_mode(mode, kv_quant_mode):
    with pytest.raises(ValueError, match="tq4_attention.*kv_quant_mode"):
        _arm(tq4_attention=mode, kv_quant_mode=kv_quant_mode)


def test_yaml_mode_with_prod_key_codec_refused(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="kv_quant_mode"):
        _load(
            tmp_path,
            monkeypatch,
            tq4_attention="m72_v1",
            kv_quant_mode="prod",
            **ARM_YAML,
        )
    loaded = _load(tmp_path, monkeypatch, tq4_attention="m72_v1", kv_quant_mode="mse", **ARM_YAML)
    assert loaded.kv_quant_mode == "mse"


def test_prod_key_codec_without_the_mode_unchanged():
    cfg = ModelConfig(
        name="m",
        type="vision",
        hf_path="x",
        kv_bits=4,
        kv_quant_scheme="turboquant",
        kv_quant_mode="prod",
    )
    assert cfg.kv_quant_mode == "prod" and cfg.tq4_attention == ""


def test_yaml_mode_without_policies_refused(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="fused_v1"):
        _load(
            tmp_path,
            monkeypatch,
            tq4_attention="m72_v1",
            kv_bits=4,
            kv_quant_scheme="turboquant",
        )


@pytest.mark.parametrize("mode", MODES)
def test_flag_passed_through(monkeypatch, mode):
    monkeypatch.setattr(pm, "_MLX_VLM_SERVER", Path("/"))
    cmd = pm._build_command(_arm(tq4_attention=mode))
    assert cmd[cmd.index("--tq4-attention") + 1] == mode
    assert cmd.count("--tq4-attention") == 1
    for flag, value in (
        ("--kv-bits", "4"),
        ("--kv-quant-scheme", "turboquant"),
        ("--attention-policy", "fused_v1"),
        ("--mtp-verify-scan", "joint_v1"),
    ):
        assert cmd[cmd.index(flag) + 1] == value


def test_flag_absent_by_default(monkeypatch):
    monkeypatch.setattr(pm, "_MLX_VLM_SERVER", Path("/"))
    plain = ModelConfig(name="t", type="vision", hf_path="x")
    assert "--tq4-attention" not in pm._build_command(plain)
    native = ModelConfig(
        name="t",
        type="vision",
        hf_path="x",
        draft_kind="mtp",
        attention_policy="fused_v1",
        mtp_verify_scan="joint_v1",
    )
    assert "--tq4-attention" not in pm._build_command(native)
