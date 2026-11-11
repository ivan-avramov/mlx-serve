"""M72 1.3: registry key ``tq4_attention`` (absent | ``m72_v1`` | ``m72_exact``) in mlx-serve: parsing, the four-way validation, the
two relaxed existing rules, and the ``--tq4-attention`` pass-through to the worker command line (spec
``docs/specs/m72-1.3-tq4-production.md`` v1.1, "Scope" item 2, A8).

Validation (spec): either value requires ``kv_bits: 4``, ``kv_quant_scheme: turboquant``, ``attention_policy: fused_v1`` and
``mtp_verify_scan: joint_v1``; ``fused_v1`` and ``joint_v1`` become valid with ``kv_bits: 4`` ONLY together with a
``tq4_attention`` value (today both are refused with ``kv_bits != 0``). Type must stay ``vision``, draft kind ``mtp``
(``joint_v1``'s own rule).

Every test is classed in its comment: [new] exercises new behaviour, [invariant] preserves an invariant of today's code,
[fix] reproduces a refusal / a pass-through the specification changes.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from mlx_serve import process_manager as pm
from mlx_serve.config import ModelConfig

MODES = ("m72_v1", "m72_exact")
FULL = dict(
    name="configured",
    type="vision",
    hf_path="org/model",
    kv_bits=4,
    kv_quant_scheme="turboquant",
    attention_policy="fused_v1",
    mtp_verify_scan="joint_v1",
    draft_kind="mtp",
)


def make(**over):
    return ModelConfig(**{**FULL, **over})


def load(tmp_path, monkeypatch, **entry):
    from mlx_serve import config

    doc = {"models": [{"name": "configured", "type": "vision", "hf_path": "local-model", **entry}]}
    path = tmp_path / "tq4.yaml"
    path.write_text(yaml.safe_dump(doc))
    monkeypatch.setattr(config, "_CONFIG_PATH", path)
    return config._load()[0]["configured"]


YAML_FULL = {k: v for k, v in FULL.items() if k not in ("name", "type", "hf_path")}


# ================================================================== parsing
# [new] absent -> a falsy default (nothing configured)
def test_tq4_attention_defaults_to_absent(tmp_path, monkeypatch):
    model = load(tmp_path, monkeypatch)
    assert not model.tq4_attention


# [new] both values load from the registry with their companions
@pytest.mark.parametrize("mode", MODES)
def test_tq4_attention_loads_from_the_registry(tmp_path, monkeypatch, mode):
    model = load(tmp_path, monkeypatch, tq4_attention=mode, **YAML_FULL)
    assert model.tq4_attention == mode
    assert (model.kv_bits, model.kv_quant_scheme) == (4, "turboquant")
    assert (model.attention_policy, model.mtp_verify_scan) == ("fused_v1", "joint_v1")


# [new] only the two values exist
@pytest.mark.parametrize("value", ["m72_v2", "M72_V1", "on", "true", "m72", "joint_v1", 1, True])
def test_tq4_attention_rejects_unknown_values(value):
    with pytest.raises(ValueError, match="tq4_attention"):
        make(tq4_attention=value)


# ================================================================== the four-way validation
# [new] each companion alone wrong refuses, for both values, and the error names the key
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    "over",
    [
        dict(kv_bits=0),
        dict(kv_bits=8),
        dict(kv_bits=2),
        dict(kv_quant_scheme=""),
        dict(kv_quant_scheme="uniform"),
        dict(attention_policy=""),
        dict(attention_policy="auto"),
        dict(mtp_verify_scan=""),
        dict(mtp_verify_scan="per_query"),
        dict(draft_kind=""),
        dict(draft_kind="suffix"),
        dict(type="text"),
    ],
    ids=lambda o: "-".join(f"{k}={v!r}" for k, v in o.items()),
)
def test_tq4_attention_requires_its_companions(mode, over):
    with pytest.raises(ValueError):
        make(tq4_attention=mode, **over)


# [new] the complete combination is accepted
@pytest.mark.parametrize("mode", MODES)
def test_tq4_attention_full_combination_is_accepted(mode):
    assert make(tq4_attention=mode).tq4_attention == mode


# [new] a refusal names the key so an operator can find the rule (kv_bits is the one a registry edit most often misses)
@pytest.mark.parametrize(
    "over", [dict(kv_bits=0), dict(attention_policy=""), dict(mtp_verify_scan="")]
)
def test_tq4_attention_refusal_message_names_the_key(over):
    with pytest.raises(ValueError, match="tq4_attention"):
        make(tq4_attention="m72_v1", **over)


# ================================================================== fused_v1 / joint_v1 with kv_bits 4
# [invariant] [fix] without the key both stay refused with a quantized KV (today's rules)
def test_fused_v1_with_kv_bits_4_is_refused_without_the_mode():
    with pytest.raises(ValueError, match="fused_v1.*kv_bits"):
        ModelConfig(
            name="configured", type="vision", hf_path="x", kv_bits=4, attention_policy="fused_v1"
        )


def test_joint_v1_with_kv_bits_4_is_refused_without_the_mode():
    with pytest.raises(ValueError, match="joint_v1.*kv_bits"):
        ModelConfig(
            name="configured",
            type="vision",
            hf_path="x",
            draft_kind="mtp",
            kv_bits=4,
            mtp_verify_scan="joint_v1",
        )


@pytest.mark.parametrize("kv_bits", (2, 3, 8))
def test_fused_and_joint_are_refused_with_any_other_quantized_kv_even_with_the_mode(kv_bits):
    with pytest.raises(ValueError):
        make(tq4_attention="m72_v1", kv_bits=kv_bits)


# [new] with the key, both are valid at kv_bits 4 (the relaxation is exactly that pairing)
@pytest.mark.parametrize("mode", MODES)
def test_fused_v1_and_joint_v1_are_valid_with_kv_bits_4_together_with_the_mode(mode):
    model = make(tq4_attention=mode)
    assert (
        model.attention_policy == "fused_v1"
        and model.mtp_verify_scan == "joint_v1"
        and model.kv_bits == 4
    )


# [invariant] the native shipped combination (kv_bits 0, TurboQuant declared, fused_v1 + joint_v1) stays valid with no mode
def test_shipped_native_combination_stays_valid_without_the_mode():
    model = make(kv_bits=0)
    assert not getattr(model, "tq4_attention", "") and model.kv_bits == 0


# [invariant] and the mode cannot be combined with it (kv_bits 0 + mode refuses: the mode is a TQ4 cache mode)
@pytest.mark.parametrize("mode", MODES)
def test_the_mode_cannot_ride_on_a_native_cache(mode):
    with pytest.raises(ValueError):
        make(tq4_attention=mode, kv_bits=0)


# [invariant] unrelated rules keep applying with the mode (mtp_verify_ab needs joint_v1, which the mode already requires)
@pytest.mark.parametrize("mode", MODES)
def test_other_rules_still_apply_with_the_mode(mode):
    assert make(tq4_attention=mode, mtp_verify_ab=True).mtp_verify_ab is True
    with pytest.raises(ValueError):
        make(tq4_attention=mode, cache_session_shrink="yes")


# ================================================================== pass-through
GOLDEN_KW = dict(
    name="g",
    type="vision",
    hf_path="org/m",
    context_length=262144,
    max_kv_cache_size=262144,
    kv_prealloc_tokens=262144,
    cache_session_shrink=True,
    draft_kind="mtp",
    draft_model="/d",
    prefill_step_size=512,
)
GOLDEN_CMD = [
    "/", "--model", "org/m", "--host", "127.0.0.1", "--port", "{port}",
    "--max-tokens", "262144", "--max-kv-size", "262144", "--kv-prealloc-tokens", "262144",
    "--cache-session-shrink", "on", "--prefill-step-size", "512",
    "--draft-kind", "mtp", "--draft-model", "/d",
    "--quantized-kv-start", "0", "--memory-limit-frac", "0.85",
]  # fmt: skip


def golden_cmd():
    return [str(pm.config.MLX_PORT) if a == "{port}" else a for a in GOLDEN_CMD]


# [invariant] with the key absent the worker command is byte-identical to today's golden (also with "" / unset spellings)
@pytest.mark.parametrize("extra", [{}, {"tq4_attention": ""}])
def test_command_is_byte_identical_without_the_key(monkeypatch, extra):
    monkeypatch.setattr(pm, "_MLX_VLM_SERVER", Path("/"))
    assert pm._build_command(ModelConfig(**GOLDEN_KW, **extra)) == golden_cmd()


# [new] [fix] both values are passed to the worker as `--tq4-attention <value>`, once, nothing else changes
@pytest.mark.parametrize("mode", MODES)
def test_command_passes_the_mode_through(monkeypatch, mode):
    monkeypatch.setattr(pm, "_MLX_VLM_SERVER", Path("/"))
    cfg = ModelConfig(
        **GOLDEN_KW,
        kv_bits=4,
        kv_quant_scheme="turboquant",
        attention_policy="fused_v1",
        mtp_verify_scan="joint_v1",
        tq4_attention=mode,
    )
    cmd = pm._build_command(cfg)
    assert cmd.count("--tq4-attention") == 1
    assert cmd[cmd.index("--tq4-attention") + 1] == mode
    # the companions travel too (the worker's own load-time rule needs them)
    assert cmd[cmd.index("--attention-policy") + 1] == "fused_v1"
    assert cmd[cmd.index("--mtp-verify-scan") + 1] == "joint_v1"
    assert (
        cmd[cmd.index("--kv-bits") + 1] == "4"
        and cmd[cmd.index("--kv-quant-scheme") + 1] == "turboquant"
    )
    # removing the pair leaves exactly what the same config without the key builds (validation bypassed on a copy)
    bare = copy.copy(cfg)
    bare.tq4_attention = ""
    rest = list(cmd)
    rest.remove("--tq4-attention")
    rest.remove(mode)
    assert rest == pm._build_command(bare)


# [new] the flag is never emitted for a non-vision model or with the key absent
def test_command_never_emits_the_flag_for_text_or_absent_key(monkeypatch):
    monkeypatch.setattr(pm, "_MLX_LM_SERVER", Path("/"))
    assert "--tq4-attention" not in pm._build_command(
        ModelConfig(name="t", type="text", hf_path="x")
    )
    monkeypatch.setattr(pm, "_MLX_VLM_SERVER", Path("/"))
    assert "--tq4-attention" not in pm._build_command(ModelConfig(**GOLDEN_KW))
