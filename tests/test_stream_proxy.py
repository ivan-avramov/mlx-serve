"""_instrumented_stream_response: upstream errors must propagate, not vanish into a 200 stream."""

import json
import time
from types import SimpleNamespace

from fastapi.responses import JSONResponse, StreamingResponse

from mlx_serve import router


class FakeErrorResponse:
    status_code = 400

    def __init__(self):
        self.closed = False

    async def aread(self):
        return json.dumps(
            {"detail": "Request needs 131086 context tokens, but MAX_KV_SIZE is 131072."}
        ).encode()

    async def aclose(self):
        self.closed = True


class FakeStreamingResponse:
    status_code = 200

    def __init__(self, lines):
        self._lines = lines
        self.closed = False

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def aclose(self):
        self.closed = True


class FakeClient:
    def __init__(self, response):
        self._response = response

    def build_request(self, method, url, json=None, headers=None):
        return SimpleNamespace(method=method, url=url, json=json, headers=headers)

    async def send(self, request, stream=False):
        return self._response


async def test_upstream_error_propagates_status_and_body(monkeypatch):
    fake = FakeErrorResponse()
    monkeypatch.setattr(router, "_HTTP_CLIENT", FakeClient(fake))
    recorded = []
    monkeypatch.setattr(router.metrics, "record_request", recorded.append)

    resp = await router._instrumented_stream_response(
        "http://127.0.0.1:1/v1/chat/completions",
        {},
        {},
        "test-model",
        time.monotonic(),
        False,
    )

    assert isinstance(resp, JSONResponse)
    assert resp.status_code == 400
    assert b"MAX_KV_SIZE" in resp.body
    assert fake.closed
    assert len(recorded) == 1
    assert recorded[0].status_code == 400
    assert "MAX_KV_SIZE" in recorded[0].error


async def test_upstream_success_still_streams_chunks(monkeypatch):
    lines = [
        'data: {"choices":[{"delta":{"content":"Hello"}}]}',
        "data: [DONE]",
    ]
    fake = FakeStreamingResponse(lines)
    monkeypatch.setattr(router, "_HTTP_CLIENT", FakeClient(fake))
    recorded = []
    monkeypatch.setattr(router.metrics, "record_request", recorded.append)

    resp = await router._instrumented_stream_response(
        "http://127.0.0.1:1/v1/chat/completions",
        {},
        {},
        "test-model",
        time.monotonic(),
        False,
    )

    assert isinstance(resp, StreamingResponse)
    chunks = [c async for c in resp.body_iterator]
    assert any("Hello" in c for c in chunks)
    assert any("[DONE]" in c for c in chunks)
    assert fake.closed
    assert len(recorded) == 1
    assert recorded[0].status_code == 200


async def test_streaming_path_forwards_session_headers_through_the_same_allowlist(monkeypatch):
    """C102(a): the streaming proxy builds its upstream request with the same header allowlist
    as the non-streaming path — a session id must survive both, PII headers neither."""
    fake = FakeStreamingResponse(["data: [DONE]"])
    client = FakeClient(fake)
    sent = {}

    def build_request(method, url, json=None, headers=None):
        sent.update(headers or {})
        return SimpleNamespace(method=method, url=url, json=json, headers=headers)

    client.build_request = build_request
    monkeypatch.setattr(router, "_HTTP_CLIENT", client)
    monkeypatch.setattr(router.metrics, "record_request", lambda *_a, **_k: None)
    incoming = {"X-Session-Id": "ses_stream", "X-OpenWebUI-User-Role": "admin",
                "X-OpenWebUI-User-Email": "someone@example.com", "Cookie": "a=b"}
    resp = await router._instrumented_stream_response(
        "http://127.0.0.1:1/v1/chat/completions", {}, incoming, "test-model", time.monotonic(), False)
    [c async for c in resp.body_iterator]
    lowered = {k.lower(): v for k, v in sent.items()}
    assert lowered.get("x-session-id") == "ses_stream"  # value preserved, name case kept
    assert not {"x-openwebui-user-role", "x-openwebui-user-email", "cookie"} & set(lowered)
