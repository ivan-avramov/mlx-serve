"""Tests for config discovery and parsing."""

import os
from pathlib import Path

import pytest


def test_config_discovery_cwd(tmp_path, monkeypatch):
    """Config is found in the current working directory."""
    config_file = tmp_path / "models.yaml"
    config_file.write_text("""
mlx_port: 9091
manager_port: 9095
inactivity_timeout_seconds: 300
startup_timeout_seconds: 60
models: []
""")
    monkeypatch.delenv("MLX_SERVE_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)

    from mlx_serve.config import _find_config

    found = _find_config()
    assert found == config_file


def test_config_discovery_env_var(tmp_path, monkeypatch):
    """MLX_SERVE_CONFIG env var takes priority."""
    config_file = tmp_path / "custom.yaml"
    config_file.write_text("mlx_port: 9091\nmanager_port: 9095\nmodels: []\n")
    monkeypatch.setenv("MLX_SERVE_CONFIG", str(config_file))

    from mlx_serve.config import _find_config

    assert _find_config() == config_file


def test_config_discovery_env_var_missing(tmp_path, monkeypatch):
    """MLX_SERVE_CONFIG pointing to missing file raises FileNotFoundError."""
    monkeypatch.setenv("MLX_SERVE_CONFIG", str(tmp_path / "nope.yaml"))

    from mlx_serve.config import _find_config

    with pytest.raises(FileNotFoundError, match="MLX_SERVE_CONFIG"):
        _find_config()


def test_config_discovery_user_dir(tmp_path, monkeypatch):
    """Config is found in ~/.mlx-serve/models.yaml."""
    fake_home = tmp_path / "home"
    config_dir = fake_home / ".mlx-serve"
    config_dir.mkdir(parents=True)
    config_file = config_dir / "models.yaml"
    config_file.write_text("mlx_port: 9091\nmanager_port: 9095\nmodels: []\n")

    monkeypatch.delenv("MLX_SERVE_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)  # no models.yaml in CWD
    monkeypatch.setenv("HOME", str(fake_home))

    from mlx_serve.config import _find_config

    assert _find_config() == config_file


def test_config_discovery_bundled_default(tmp_path, monkeypatch):
    """Falls back to bundled _default_models.yaml."""
    monkeypatch.delenv("MLX_SERVE_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)  # no models.yaml here
    monkeypatch.setenv("HOME", str(tmp_path / "empty_home"))  # no ~/.mlx-serve/

    from mlx_serve.config import _find_config

    found = _find_config()
    assert found.name == "_default_models.yaml"


def test_config_parses_models(tmp_path, monkeypatch):
    """Config correctly parses model entries."""
    config_file = tmp_path / "models.yaml"
    config_file.write_text("""
mlx_port: 8091
manager_port: 8095
inactivity_timeout_seconds: 600
startup_timeout_seconds: 120
models:
  - name: my-model
    type: text
    hf_path: mlx-community/test
    context_length: 4096
    max_kv_cache_size: 8192
""")
    monkeypatch.setenv("MLX_SERVE_CONFIG", str(config_file))

    import importlib
    import mlx_serve.config as cfg

    importlib.reload(cfg)

    assert "my-model" in cfg.MODELS
    assert cfg.MODELS["my-model"].type == "text"
    assert cfg.MODELS["my-model"].hf_path == "mlx-community/test"
    assert cfg.MODELS["my-model"].context_length == 4096
    assert cfg.MODELS["my-model"].max_kv_cache_size == 8192
    assert cfg.MODELS["my-model"].generation_defaults == {}  # default: empty
    assert cfg.MLX_PORT == 8091
    assert cfg.MANAGER_PORT == 8095


def test_config_parses_generation_defaults(tmp_path, monkeypatch):
    """generation_defaults is parsed verbatim as an opaque dict (default empty)."""
    config_file = tmp_path / "models.yaml"
    config_file.write_text("""
mlx_port: 8091
manager_port: 8095
inactivity_timeout_seconds: 600
startup_timeout_seconds: 120
models:
  - name: tuned
    type: vision
    hf_path: mlx-community/test
    generation_defaults:
      temperature: 0.3
      top_p: 0.95
      enable_thinking: true
  - name: plain
    type: vision
    hf_path: mlx-community/test2
""")
    monkeypatch.setenv("MLX_SERVE_CONFIG", str(config_file))

    import importlib
    import mlx_serve.config as cfg

    importlib.reload(cfg)

    assert cfg.MODELS["tuned"].generation_defaults == {
        "temperature": 0.3, "top_p": 0.95, "enable_thinking": True,
    }
    assert cfg.MODELS["plain"].generation_defaults == {}


def test_config_raises_on_top_level_enable_thinking(tmp_path, monkeypatch):
    """A stale top-level enable_thinking: key fails loud and fast (it moved into
    generation_defaults) so nothing silently disables thinking during migration."""
    config_file = tmp_path / "models.yaml"
    config_file.write_text("""
mlx_port: 8091
manager_port: 8095
models:
  - name: stale
    type: vision
    hf_path: mlx-community/test
    enable_thinking: true
""")
    monkeypatch.setenv("MLX_SERVE_CONFIG", str(config_file))

    import importlib
    import mlx_serve.config as cfg

    with pytest.raises(ValueError, match="enable_thinking"):
        importlib.reload(cfg)


def test_config_parses_kv_prealloc_tokens(tmp_path, monkeypatch):
    """kv_prealloc_tokens is parsed (default 0) and accepted when <= max_kv_cache_size."""
    config_file = tmp_path / "models.yaml"
    config_file.write_text("""
mlx_port: 8091
manager_port: 8095
models:
  - name: preallocated
    type: vision
    hf_path: mlx-community/test
    max_kv_cache_size: 262144
    kv_prealloc_tokens: 262144
  - name: plain
    type: vision
    hf_path: mlx-community/test2
""")
    monkeypatch.setenv("MLX_SERVE_CONFIG", str(config_file))

    import importlib
    import mlx_serve.config as cfg

    importlib.reload(cfg)

    assert cfg.MODELS["preallocated"].kv_prealloc_tokens == 262144
    assert cfg.MODELS["plain"].kv_prealloc_tokens == 0  # default


def test_config_parses_moe_expand(tmp_path, monkeypatch):
    """moe_expand is parsed verbatim as an opaque string (default empty)."""
    config_file = tmp_path / "models.yaml"
    config_file.write_text("""
mlx_port: 8091
manager_port: 8095
models:
  - name: expanded
    type: text
    hf_path: mlx-community/test
    moe_expand: "27-39:20:0.8:0.5"
  - name: plain
    type: text
    hf_path: mlx-community/test2
""")
    monkeypatch.setenv("MLX_SERVE_CONFIG", str(config_file))

    import importlib
    import mlx_serve.config as cfg

    importlib.reload(cfg)

    assert cfg.MODELS["expanded"].moe_expand == "27-39:20:0.8:0.5"
    assert cfg.MODELS["plain"].moe_expand == ""  # default


def test_kv_prealloc_over_cap_fails_loud(tmp_path, monkeypatch):
    """kv_prealloc_tokens > max_kv_cache_size fails loud at parse time."""
    config_file = tmp_path / "models.yaml"
    config_file.write_text("""
mlx_port: 8091
manager_port: 8095
models:
  - name: m
    type: vision
    hf_path: x
    max_kv_cache_size: 262144
    kv_prealloc_tokens: 300000
""")
    monkeypatch.setenv("MLX_SERVE_CONFIG", str(config_file))

    import importlib
    import mlx_serve.config as cfg

    with pytest.raises(ValueError, match="kv_prealloc_tokens"):
        importlib.reload(cfg)


def test_config_invalid_type(tmp_path, monkeypatch):
    """Invalid model type raises ValueError."""
    config_file = tmp_path / "models.yaml"
    config_file.write_text("""
models:
  - name: bad
    type: invalid
    hf_path: test/test
""")
    monkeypatch.setenv("MLX_SERVE_CONFIG", str(config_file))

    import importlib
    import mlx_serve.config as cfg

    with pytest.raises(ValueError, match="invalid type"):
        importlib.reload(cfg)


def test_config_monitoring_defaults(tmp_path, monkeypatch):
    """Monitoring config uses sensible defaults when not specified."""
    config_file = tmp_path / "models.yaml"
    config_file.write_text("models: []\n")
    monkeypatch.setenv("MLX_SERVE_CONFIG", str(config_file))

    import importlib
    import mlx_serve.config as cfg

    importlib.reload(cfg)

    assert cfg.MONITORING.metrics_history_size == 500
    assert cfg.MONITORING.events_history_size == 1000
    assert cfg.MONITORING.memory_sample_interval == 10


def _load_shrink_setting(tmp_path, monkeypatch, value="missing", model_type="vision"):
    import yaml
    from mlx_serve import config

    entry = {"name": "configured", "type": model_type, "hf_path": "local-model"}
    if value != "missing":
        entry["cache_session_shrink"] = value
    path = tmp_path / "session-policy.yaml"
    path.write_text(yaml.safe_dump({"models": [entry]}))
    monkeypatch.setattr(config, "_CONFIG_PATH", path)
    return config._load()[0]["configured"]


@pytest.mark.parametrize(
    "value,expected", [("missing", None), (None, None), (True, True), (False, False)]
)
def test_cache_session_shrink_is_optional_strict_boolean(tmp_path, monkeypatch, value, expected):
    model = _load_shrink_setting(tmp_path, monkeypatch, value)
    assert model.cache_session_shrink is expected


@pytest.mark.parametrize("value", ["on", "off", "true", "false", "", 0, 1, 0.0, 1.0, [], {}])
def test_cache_session_shrink_rejects_non_boolean_values(tmp_path, monkeypatch, value):
    with pytest.raises(ValueError, match="cache_session_shrink.*bool"):
        _load_shrink_setting(tmp_path, monkeypatch, value)


@pytest.mark.parametrize("model_type", ["text", "embedding", "tts", "stt"])
@pytest.mark.parametrize("value", [True, False])
def test_cache_session_shrink_rejects_unsupported_model_types(
    tmp_path, monkeypatch, model_type, value
):
    with pytest.raises(ValueError, match="cache_session_shrink.*vision"):
        _load_shrink_setting(tmp_path, monkeypatch, value, model_type)


@pytest.mark.parametrize("model_type", ["text", "embedding", "tts", "stt"])
def test_unset_cache_session_shrink_preserves_other_model_types(tmp_path, monkeypatch, model_type):
    assert _load_shrink_setting(tmp_path, monkeypatch, model_type=model_type).cache_session_shrink is None


def test_direct_model_config_validates_retirement_policy():
    from mlx_serve.config import ModelConfig

    with pytest.raises(ValueError, match="cache_session_shrink.*bool"):
        ModelConfig(name="invalid", type="vision", hf_path="local-model", cache_session_shrink=1)
    with pytest.raises(ValueError, match="cache_session_shrink.*vision"):
        ModelConfig(name="invalid", type="text", hf_path="local-model", cache_session_shrink=False)


def _load_attention_policy(tmp_path, monkeypatch, value="missing", **extra):
    import yaml
    from mlx_serve import config

    entry = {"name": "configured", "type": "vision", "hf_path": "local-model", **extra}
    if value != "missing":
        entry["attention_policy"] = value
    path = tmp_path / "attention-policy.yaml"
    path.write_text(yaml.safe_dump({"models": [entry]}))
    monkeypatch.setattr(config, "_CONFIG_PATH", path)
    return config._load()[0]["configured"]


@pytest.mark.parametrize(
    "value,expected", [("missing", ""), ("", ""), ("auto", "auto"), ("fused_v1", "fused_v1")]
)
def test_attention_policy_loaded_from_registry(tmp_path, monkeypatch, value, expected):
    assert _load_attention_policy(tmp_path, monkeypatch, value).attention_policy == expected


@pytest.mark.parametrize("value", ["fused_v2", "FUSED_V1", "off", "none"])
def test_attention_policy_rejects_unknown_value(tmp_path, monkeypatch, value):
    with pytest.raises(ValueError, match="configured.*attention_policy.*fused_v1"):
        _load_attention_policy(tmp_path, monkeypatch, value)


def test_attention_policy_fused_v1_rejects_quantized_kv():
    from mlx_serve.config import ModelConfig

    with pytest.raises(ValueError, match="configured.*fused_v1.*kv_bits"):
        ModelConfig(
            name="configured", type="vision", hf_path="x", kv_bits=4, attention_policy="fused_v1"
        )


@pytest.mark.parametrize("model_type", ["text", "embedding", "tts", "stt"])
def test_attention_policy_fused_v1_rejects_non_vision(model_type):
    from mlx_serve.config import ModelConfig

    with pytest.raises(ValueError, match="configured.*fused_v1.*vision"):
        ModelConfig(name="configured", type=model_type, hf_path="x", attention_policy="fused_v1")


def test_attention_policy_validation_runs_with_cache_session_shrink_set():
    from mlx_serve.config import ModelConfig

    with pytest.raises(ValueError, match="configured.*attention_policy"):
        ModelConfig(
            name="configured", type="vision", hf_path="x",
            cache_session_shrink=True, attention_policy="bogus",
        )  # fmt: skip


def _load_lazy_setting(tmp_path, monkeypatch, value="missing", model_type="vision"):
    import yaml
    from mlx_serve import config

    entry = {"name": "configured", "type": model_type, "hf_path": "local-model"}
    if value != "missing":
        entry["lazy_prompt_embeddings"] = value
    path = tmp_path / "lazy.yaml"
    path.write_text(yaml.safe_dump({"models": [entry]}))
    monkeypatch.setattr(config, "_CONFIG_PATH", path)
    return config._load()[0]["configured"]


@pytest.mark.parametrize(
    "value,expected", [("missing", None), (None, None), (True, True), (False, False)]
)
def test_lazy_prompt_embeddings_loaded_strict_bool(tmp_path, monkeypatch, value, expected):
    assert _load_lazy_setting(tmp_path, monkeypatch, value).lazy_prompt_embeddings is expected


@pytest.mark.parametrize("value", ["on", "true", "", 0, 1, 1.0, [], {}])
def test_lazy_prompt_embeddings_rejects_non_bool(tmp_path, monkeypatch, value):
    with pytest.raises(ValueError, match="configured.*lazy_prompt_embeddings.*bool"):
        _load_lazy_setting(tmp_path, monkeypatch, value)


@pytest.mark.parametrize("model_type", ["text", "embedding", "tts", "stt"])
@pytest.mark.parametrize("value", [True, False])
def test_lazy_prompt_embeddings_rejects_non_vision(tmp_path, monkeypatch, model_type, value):
    with pytest.raises(ValueError, match="configured.*lazy_prompt_embeddings.*vision"):
        _load_lazy_setting(tmp_path, monkeypatch, value, model_type)


# --- M58 mtp_verify_scan / mtp_verify_ab -------------------------------------------------------


def _load_mtp_verify(tmp_path, monkeypatch, **extra):
    import yaml
    from mlx_serve import config

    entry = {"name": "configured", "type": "vision", "hf_path": "local-model", **extra}
    path = tmp_path / "mtp-verify.yaml"
    path.write_text(yaml.safe_dump({"models": [entry]}))
    monkeypatch.setattr(config, "_CONFIG_PATH", path)
    return config._load()[0]["configured"]


def test_mtp_verify_defaults_when_absent(tmp_path, monkeypatch):
    model = _load_mtp_verify(tmp_path, monkeypatch)
    assert model.mtp_verify_scan == "" and model.mtp_verify_ab is False


@pytest.mark.parametrize("value", ["", "per_query"])
def test_mtp_verify_scan_default_values_need_no_companions(tmp_path, monkeypatch, value):
    assert _load_mtp_verify(tmp_path, monkeypatch, mtp_verify_scan=value).mtp_verify_scan == value


def test_mtp_verify_joint_v1_loaded_from_registry(tmp_path, monkeypatch):
    model = _load_mtp_verify(
        tmp_path, monkeypatch, mtp_verify_scan="joint_v1", draft_kind="mtp", mtp_verify_ab=True
    )
    assert (model.mtp_verify_scan, model.mtp_verify_ab) == ("joint_v1", True)


@pytest.mark.parametrize("value", ["joint_v2", "JOINT_V1", "off", "auto"])
def test_mtp_verify_scan_rejects_unknown_value(tmp_path, monkeypatch, value):
    with pytest.raises(ValueError, match="configured.*mtp_verify_scan.*joint_v1"):
        _load_mtp_verify(tmp_path, monkeypatch, mtp_verify_scan=value, draft_kind="mtp")


@pytest.mark.parametrize("kind", ["", "suffix", "dflash", "eagle3"])
def test_mtp_verify_joint_v1_requires_draft_kind_mtp(kind):
    from mlx_serve.config import ModelConfig

    with pytest.raises(ValueError, match="configured.*joint_v1.*draft_kind"):
        ModelConfig(
            name="configured", type="vision", hf_path="x", draft_kind=kind,
            mtp_verify_scan="joint_v1",
        )  # fmt: skip


def test_mtp_verify_joint_v1_rejects_quantized_kv():
    from mlx_serve.config import ModelConfig

    with pytest.raises(ValueError, match="configured.*joint_v1.*kv_bits"):
        ModelConfig(
            name="configured", type="vision", hf_path="x", draft_kind="mtp", kv_bits=4,
            mtp_verify_scan="joint_v1",
        )  # fmt: skip


@pytest.mark.parametrize("scan", ["", "per_query"])
def test_mtp_verify_ab_requires_joint_v1(scan):
    from mlx_serve.config import ModelConfig

    with pytest.raises(ValueError, match="configured.*mtp_verify_ab.*joint_v1"):
        ModelConfig(
            name="configured", type="vision", hf_path="x", draft_kind="mtp",
            mtp_verify_scan=scan, mtp_verify_ab=True,
        )  # fmt: skip


@pytest.mark.parametrize("value", [1, "yes", None])
def test_mtp_verify_ab_must_be_a_bool(value):
    from mlx_serve.config import ModelConfig

    with pytest.raises(ValueError, match="configured.*mtp_verify_ab.*bool"):
        ModelConfig(
            name="configured", type="vision", hf_path="x", draft_kind="mtp",
            mtp_verify_scan="joint_v1", mtp_verify_ab=value,
        )  # fmt: skip


def test_mtp_verify_joint_v1_rejects_non_vision():
    from mlx_serve.config import ModelConfig

    with pytest.raises(ValueError, match="configured.*joint_v1.*vision"):
        ModelConfig(
            name="configured", type="text", hf_path="x", draft_kind="mtp",
            mtp_verify_scan="joint_v1",
        )  # fmt: skip


# --- review R1: a KV quantization scheme is incompatible with the native-KV-only policies --------


@pytest.mark.parametrize("scheme", ["uniform", "turboquant"])
def test_native_kv_policies_reject_a_kv_quant_scheme(scheme):
    from mlx_serve.config import ModelConfig

    with pytest.raises(ValueError, match="configured.*joint_v1.*kv_quant_scheme"):
        ModelConfig(
            name="configured", type="vision", hf_path="x", draft_kind="mtp",
            kv_quant_scheme=scheme, mtp_verify_scan="joint_v1",
        )  # fmt: skip
    with pytest.raises(ValueError, match="configured.*fused_v1.*kv_quant_scheme"):
        ModelConfig(
            name="configured", type="vision", hf_path="x",
            kv_quant_scheme=scheme, attention_policy="fused_v1",
        )  # fmt: skip


def test_empty_kv_quant_scheme_still_accepted_by_both_policies():
    from mlx_serve.config import ModelConfig

    ModelConfig(name="c", type="vision", hf_path="x", draft_kind="mtp", mtp_verify_scan="joint_v1")
    ModelConfig(name="c", type="vision", hf_path="x", attention_policy="fused_v1")
