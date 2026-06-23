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
