"""Legacy /completions proxy: must mirror /chat/completions exactly except for
the backend target path.

Verifies that the new route reuses the same model-name -> hf_path rewrite and
the ensure_model load step, and forwards to the backend's /v1/completions text
endpoint (NOT /v1/chat/completions).
"""

import time
from types import SimpleNamespace

import pytest
from fastapi.responses import JSONResponse

from mlx_serve import config, router


class _FakeRequest:
    """Minimal stand-in for fastapi.Request for the proxy handlers."""

    def __init__(self, body: dict):
        self._body = body
        self.headers = {}

    async def json(self):
        return self._body


class _FakePostResponse:
    status_code = 200

    def json(self):
        return {"choices": [{"text": "ok"}], "usage": {"completion_tokens": 1}}


@pytest.fixture()
def text_model(monkeypatch):
    """Register a single text model in the config used by the router."""
    cfg = SimpleNamespace(
        name="test-text",
        type="text",
        hf_path="mlx-community/test-text-4bit",
        context_length=4096,
        max_kv_cache_size=None,
    )
    monkeypatch.setattr(config, "MODELS", {"test-text": cfg})
    return cfg


async def _capture_forward(monkeypatch):
    """Stub the load path + HTTP client; return a dict that captures the
    forwarded (url, body)."""
    captured: dict = {}

    async def fake_unload():
        return None

    async def fake_ensure_model(name):
        return False  # not a cold start

    async def fake_post(url, json=None, headers=None):
        captured["url"] = url
        captured["body"] = json
        return _FakePostResponse()

    monkeypatch.setattr(router.inline_manager, "unload", fake_unload)
    monkeypatch.setattr(router.process_manager, "ensure_model", fake_ensure_model)
    monkeypatch.setattr(router.process_manager, "set_keep_alive", lambda *_a, **_k: None)
    monkeypatch.setattr(router.metrics, "record_request", lambda *_a, **_k: None)
    monkeypatch.setattr(router._HTTP_CLIENT, "post", fake_post)
    return captured


async def test_completions_forwards_to_v1_completions(monkeypatch, text_model):
    captured = await _capture_forward(monkeypatch)

    resp = await router.completions(
        _FakeRequest({"model": "test-text", "prompt": "hello"})
    )

    assert isinstance(resp, JSONResponse)
    assert resp.status_code == 200
    # Forwarded to the legacy TEXT endpoint, not the chat endpoint.
    assert captured["url"].endswith("/v1/completions")
    assert not captured["url"].endswith("/v1/chat/completions")
    # The model name was rewritten to the HuggingFace path.
    assert captured["body"]["model"] == "mlx-community/test-text-4bit"
    assert captured["body"]["prompt"] == "hello"


async def test_chat_completions_still_forwards_to_chat(monkeypatch, text_model):
    """Regression guard: the chat route must remain unchanged."""
    captured = await _capture_forward(monkeypatch)

    resp = await router.chat_completions(
        _FakeRequest({"model": "test-text", "messages": []})
    )

    assert isinstance(resp, JSONResponse)
    assert captured["url"].endswith("/v1/chat/completions")
    assert captured["body"]["model"] == "mlx-community/test-text-4bit"


async def test_completions_model_not_found(monkeypatch):
    monkeypatch.setattr(config, "MODELS", {})
    with pytest.raises(router.HTTPException) as exc:
        await router.completions(_FakeRequest({"model": "nope", "prompt": "hi"}))
    assert exc.value.status_code == 404


async def test_completions_wrong_type_rejected(monkeypatch):
    cfg = SimpleNamespace(
        name="emb", type="embedding", hf_path="x", context_length=None,
        max_kv_cache_size=None,
    )
    monkeypatch.setattr(config, "MODELS", {"emb": cfg})
    with pytest.raises(router.HTTPException) as exc:
        await router.completions(_FakeRequest({"model": "emb", "prompt": "hi"}))
    assert exc.value.status_code == 404
    assert "use the correct endpoint" in exc.value.detail["error"]["message"]


# --------------------------------------------------------------------------- C102(a) 2026-09-27
# Session-identifying headers must reach the worker. Measured 2026-09-23 (stack M45): every
# request in the worker log was `session=anon:*` because this allowlist dropped
# X-MLX-VLM-Chat-Id; opencode sends x-session-id / x-session-affinity on every request
# (verified on the wire, opencode 1.18.30), Claude Code sends x-claude-code-session-id,
# Codex sends session-id, OpenWebUI sends X-OpenWebUI-Chat-Id when
# ENABLE_FORWARD_USER_INFO_HEADERS is on. Still an allowlist: nothing else passes.
def test_forward_headers_passes_session_identifiers_and_drops_the_rest():
    incoming = {
        "Content-Type": "application/json",
        "Authorization": "Bearer x",
        "X-MLX-VLM-Chat-Id": "pin-1",
        "x-session-id": "ses_1",
        "X-Session-Affinity": "ses_1",
        "x-parent-session-id": "ses_0",
        "x-claude-code-session-id": "cc-1",
        "X-OpenWebUI-Chat-Id": "owui-1",
        "session-id": "codex-1",
        "session_id": "pi-1",
        "x-switchyard-session-id": "sw-1",
        # must NOT pass: per-request ids and user PII headers
        "X-Request-Id": "req-1",
        "X-OpenWebUI-User-Email": "someone@example.com",
        "X-OpenWebUI-User-Name": "Someone",
        "X-OpenWebUI-User-Role": "admin",
        "Cookie": "a=b",
        "Host": "localhost:8000",
    }
    out = router._forward_headers(incoming)
    lowered = {k.lower(): v for k, v in out.items()}
    for name in ("x-mlx-vlm-chat-id", "x-session-id", "x-session-affinity", "x-parent-session-id",
                 "x-claude-code-session-id", "x-openwebui-chat-id", "session-id", "session_id",
                 "x-switchyard-session-id", "content-type", "authorization", "x-request-id"):
        assert name in lowered, f"{name} must be forwarded"
    for name in ("x-openwebui-user-email", "x-openwebui-user-name", "x-openwebui-user-role", "cookie", "host"):
        assert name not in lowered, f"{name} must NOT be forwarded"
    # values pass through untouched
    assert lowered["x-session-id"] == "ses_1" and lowered["x-openwebui-chat-id"] == "owui-1"


async def test_chat_completions_forwards_session_headers_to_the_worker(monkeypatch, text_model):
    captured = await _capture_forward(monkeypatch)
    seen = {}

    async def fake_post(url, json=None, headers=None):
        seen.update(headers or {})
        return _FakePostResponse()

    monkeypatch.setattr(router._HTTP_CLIENT, "post", fake_post)
    req = _FakeRequest({"model": "test-text", "messages": [{"role": "user", "content": "hi"}]})
    req.headers = {"x-session-id": "ses_abc", "X-Foo": "no"}
    await router.chat_completions(req)
    assert {k.lower(): v for k, v in seen.items()}.get("x-session-id") == "ses_abc"
    assert "x-foo" not in {k.lower() for k in seen}
