"""
config.py — reads models.yaml and exposes typed configuration.

Config discovery order:
  1. MLX_SERVE_CONFIG env var (explicit override)
  2. ./models.yaml  (current working directory)
  3. ~/.mlx-serve/models.yaml  (user config directory)
  4. Bundled _default_models.yaml inside the package
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


def _find_config() -> Path:
    """Locate models.yaml using a fallback chain."""
    # 1. Explicit env var
    env_path = os.environ.get("MLX_SERVE_CONFIG")
    if env_path:
        p = Path(env_path).expanduser()
        if p.exists():
            return p
        raise FileNotFoundError(f"MLX_SERVE_CONFIG points to {p} which does not exist")

    # 2. Current working directory
    cwd_path = Path.cwd() / "models.yaml"
    if cwd_path.exists():
        return cwd_path

    # 3. User config directory
    user_path = Path.home() / ".mlx-serve" / "models.yaml"
    if user_path.exists():
        return user_path

    # 4. Bundled default
    bundled = Path(__file__).parent / "_default_models.yaml"
    if bundled.exists():
        return bundled

    raise FileNotFoundError(
        "No models.yaml found. Searched:\n"
        f"  - {cwd_path}\n"
        f"  - {user_path}\n"
        "Run 'mlx-serve init' to generate one."
    )


_CONFIG_PATH = _find_config()


_ATTENTION_POLICIES = ("", "auto", "fused_v1")
_MTP_VERIFY_SCANS = ("", "per_query", "joint_v1")


@dataclass
class ModelConfig:
    name: str
    type: str  # "text", "vision", "embedding", "tts", or "stt"
    hf_path: str
    context_length: int = 0  # max output tokens per response (--max-tokens); 0 = server default
    max_kv_cache_size: int = (
        0  # KV cache token capacity for prompt caching (--max-kv-cache-size); 0 = model default
    )
    kv_bits: int = 0
    kv_quant_scheme: str = ""
    kv_quant_mode: str = ""  # "mse" (default) | "prod" (TQ key codec + 1-bit QJL residual)
    reasoning_parser: str = ""
    tool_call_parser: str = ""
    prefill_step_size: int = 2048
    quantized_kv_start: int = 0
    kv_prealloc_tokens: int = 0  # pre-alloc every KV cache to this token floor; must be <= max_kv_cache_size
    # MLX Metal buffer-pool cap (GB). 0 = auto-derive in mlx-vlm at startup from
    # n_heads x prefill_step x max_kv_cache_size (+2 GB margin), which bounds RSS to
    # ~= active_peak + cap without throttling intra-chunk score reuse. Set >0 to override.
    cache_limit_gb: float = 0.0
    # MLX total-memory backstop, as a fraction of physical RAM (capped at the Metal
    # recommended working set). MLX evicts the pool before the OS swaps. 0 = off.
    memory_limit_frac: float = 0.85
    # Per-model generation defaults (temperature/top_p/top_k/min_p/presence_penalty/
    # max_tokens/thinking_budget/enable_thinking/…). Opaque to mlx-serve: it forwards the
    # whole dict verbatim as one --generation-defaults JSON arg, so a new generation param
    # needs no mlx-serve change. mlx-vlm applies each entry as a default ONLY when the
    # request omits it (request params always win). enable_thinking lives here now (there is
    # no dedicated top-level enable_thinking field — a stale one fails loud in _load).
    generation_defaults: dict = field(default_factory=dict)
    # Speculative decoding (vision / mlx-vlm only). draft_kind="suffix" enables the
    # drafter-free n-gram speculator; the rest tune it. Empty/0 => not passed.
    draft_kind: str = ""  # "suffix" | "dflash" | "eagle3" | "mtp"
    draft_block_size: int = 0  # suffix: max draft (proposal) length
    suffix_min_match: int = 0  # suffix: minimum n-gram match length (default 2)
    draft_cooldown: int = 0  # suffix: consecutive 0-accept rounds before pausing (0=off)
    draft_model: str = ""  # mtp/dflash/eagle3: path to the split drafter folder (--draft-model)
    # Layer-scoped MoE expert-budget expansion (M34), CLI string "LS-LE:N:T:D".
    # Empty => not passed; forwarded verbatim as --moe-expand for text and vision types.
    moe_expand: str = ""
    # Per-model mlx-vlm session retirement policy; None preserves worker defaults.
    cache_session_shrink: bool | None = None
    # M57 attention dispatch policy (vision only): "" / "auto" => worker default, not passed.
    attention_policy: str = ""
    # Lazy per-chunk prompt embeddings (vision only); None/False => flag not passed.
    lazy_prompt_embeddings: bool | None = None
    # M58 MTP verification scan (vision only): "" / "per_query" => worker default, not passed.
    mtp_verify_scan: str = ""
    # M58 gate-1 instrument (requires joint_v1); False => flag not passed.
    mtp_verify_ab: bool = False

    def __post_init__(self) -> None:
        self._validate_cache_session_shrink()
        self._validate_lazy_prompt_embeddings()
        self._validate_attention_policy()
        self._validate_mtp_verify_scan()

    def _validate_cache_session_shrink(self) -> None:
        if self.cache_session_shrink is None:
            return
        if type(self.cache_session_shrink) is not bool:
            raise ValueError(f"Model '{self.name}': cache_session_shrink must be a bool or null.")
        if self.type != "vision":
            raise ValueError(
                f"Model '{self.name}': cache_session_shrink is only supported for type 'vision'."
            )

    def _validate_lazy_prompt_embeddings(self) -> None:
        if self.lazy_prompt_embeddings is None:
            return
        if type(self.lazy_prompt_embeddings) is not bool:
            raise ValueError(
                f"Model '{self.name}': lazy_prompt_embeddings must be a bool or null."
            )
        if self.type != "vision":
            raise ValueError(
                f"Model '{self.name}': lazy_prompt_embeddings is only supported for type 'vision'."
            )

    def _validate_attention_policy(self) -> None:
        if self.attention_policy not in _ATTENTION_POLICIES:
            allowed = ", ".join(repr(v) for v in _ATTENTION_POLICIES)
            raise ValueError(
                f"Model '{self.name}': attention_policy {self.attention_policy!r} is invalid; "
                f"allowed values: {allowed}."
            )
        if self.attention_policy == "fused_v1":
            if self.type != "vision":
                raise ValueError(
                    f"Model '{self.name}': attention_policy 'fused_v1' requires type 'vision'."
                )
            if self.kv_bits != 0:
                raise ValueError(
                    f"Model '{self.name}': attention_policy 'fused_v1' requires kv_bits 0 "
                    f"(got {self.kv_bits})."
                )

    def _validate_mtp_verify_scan(self) -> None:
        if self.mtp_verify_scan not in _MTP_VERIFY_SCANS:
            allowed = ", ".join(repr(v) for v in _MTP_VERIFY_SCANS)
            raise ValueError(
                f"Model '{self.name}': mtp_verify_scan {self.mtp_verify_scan!r} is invalid; "
                f"allowed values: {allowed}."
            )
        if type(self.mtp_verify_ab) is not bool:
            raise ValueError(f"Model '{self.name}': mtp_verify_ab must be a bool.")
        if self.mtp_verify_scan == "joint_v1":
            if self.type != "vision":
                raise ValueError(
                    f"Model '{self.name}': mtp_verify_scan 'joint_v1' requires type 'vision'."
                )
            if self.draft_kind != "mtp":
                raise ValueError(
                    f"Model '{self.name}': mtp_verify_scan 'joint_v1' requires draft_kind 'mtp' "
                    f"(got {self.draft_kind!r})."
                )
            if self.kv_bits != 0:
                raise ValueError(
                    f"Model '{self.name}': mtp_verify_scan 'joint_v1' requires kv_bits 0 "
                    f"(got {self.kv_bits})."
                )
        if self.mtp_verify_ab and self.mtp_verify_scan != "joint_v1":
            raise ValueError(
                f"Model '{self.name}': mtp_verify_ab requires mtp_verify_scan 'joint_v1'."
            )


@dataclass
class MonitoringConfig:
    log_dir: Path
    metrics_history_size: int = 500
    events_history_size: int = 1000
    memory_sample_interval: int = 10  # seconds, in-memory
    memory_log_interval: int = 60  # seconds, to disk
    log_retention_mb: int = 50  # per JSONL file


_VALID_TYPES = {"text", "vision", "embedding", "tts", "stt"}


def _load() -> tuple[dict[str, ModelConfig], int, int, int, int, MonitoringConfig]:
    with _CONFIG_PATH.open() as f:
        data = yaml.safe_load(f)

    models = {}
    for entry in data.get("models", []):
        if entry["type"] not in _VALID_TYPES:
            raise ValueError(
                f"Model '{entry['name']}' has invalid type '{entry['type']}'. "
                f"Must be one of: {sorted(_VALID_TYPES)}"
            )
        if "enable_thinking" in entry:
            raise ValueError(
                f"Model '{entry['name']}' has a top-level 'enable_thinking' key. It moved "
                f"into the 'generation_defaults' block — put 'enable_thinking: true' there "
                f"instead so all generation params live in one place."
            )
        models[entry["name"]] = ModelConfig(
            name=entry["name"],
            type=entry["type"],
            hf_path=entry["hf_path"],
            context_length=entry.get("context_length", 0),
            max_kv_cache_size=entry.get("max_kv_cache_size", 0),
            kv_bits=entry.get("kv_bits", 0),
            kv_quant_scheme=entry.get("kv_quant_scheme", ""),
            kv_quant_mode=entry.get("kv_quant_mode", ""),
            reasoning_parser=entry.get("reasoning_parser", ""),
            tool_call_parser=entry.get("tool_call_parser", ""),
            prefill_step_size=entry.get("prefill_step_size", ""),
            quantized_kv_start=entry.get("quantized_kv_start", 0),
            kv_prealloc_tokens=entry.get("kv_prealloc_tokens", 0),
            cache_session_shrink=entry.get("cache_session_shrink"),
            cache_limit_gb=entry.get("cache_limit_gb", 0.0),
            memory_limit_frac=entry.get("memory_limit_frac", 0.85),
            generation_defaults=entry.get("generation_defaults", {}),
            draft_kind=entry.get("draft_kind", ""),
            draft_block_size=entry.get("draft_block_size", 0),
            suffix_min_match=entry.get("suffix_min_match", 0),
            draft_cooldown=entry.get("draft_cooldown", 0),
            draft_model=entry.get("draft_model", ""),
            moe_expand=entry.get("moe_expand", ""),
            attention_policy=entry.get("attention_policy", ""),
            lazy_prompt_embeddings=entry.get("lazy_prompt_embeddings"),
            mtp_verify_scan=entry.get("mtp_verify_scan", ""),
            mtp_verify_ab=entry.get("mtp_verify_ab", False),
        )
        if (
            models[entry["name"]].kv_prealloc_tokens
            and models[entry["name"]].max_kv_cache_size
            and models[entry["name"]].kv_prealloc_tokens > models[entry["name"]].max_kv_cache_size
        ):
            raise ValueError(
                f"Model '{entry['name']}': kv_prealloc_tokens "
                f"({models[entry['name']].kv_prealloc_tokens}) exceeds max_kv_cache_size "
                f"({models[entry['name']].max_kv_cache_size})."
            )

    # Monitoring settings (optional section in models.yaml)
    mon_raw = data.get("monitoring", {})
    default_log_dir = Path.home() / ".mlx-serve" / "logs"
    log_dir_str = mon_raw.get("log_dir", str(default_log_dir))
    monitoring = MonitoringConfig(
        log_dir=Path(log_dir_str).expanduser(),
        metrics_history_size=mon_raw.get("metrics_history_size", 500),
        events_history_size=mon_raw.get("events_history_size", 1000),
        memory_sample_interval=mon_raw.get("memory_sample_interval", 10),
        memory_log_interval=mon_raw.get("memory_log_interval", 60),
        log_retention_mb=mon_raw.get("log_retention_mb", 50),
    )

    return (
        models,
        data.get("mlx_port", 8091),
        data.get("manager_port", 8095),
        data.get("inactivity_timeout_seconds", 600),
        data.get("startup_timeout_seconds", 120),
        # Max wait for a first-time model download (HF pull) to finish. Far
        # larger than startup_timeout because multi-GB weights can't arrive in
        # 120s. Only applied when the model is not yet in the HF cache.
        data.get("download_timeout_seconds", 1800),
        # After a model fails to load, reject further requests for it for this
        # many seconds instead of respawning on every retry (prevents the
        # infinite reload loop when a client auto-retries).
        data.get("failure_cooldown_seconds", 30),
        monitoring,
    )


(
    MODELS,
    MLX_PORT,
    MANAGER_PORT,
    INACTIVITY_TIMEOUT,
    STARTUP_TIMEOUT,
    DOWNLOAD_TIMEOUT,
    FAILURE_COOLDOWN,
    MONITORING,
) = _load()

# Optional bearer token auth. Set MLX_API_KEY env var to enable.
# When empty, all requests are accepted (safe for localhost-only deployments).
API_KEY: str = os.environ.get("MLX_API_KEY", "")
