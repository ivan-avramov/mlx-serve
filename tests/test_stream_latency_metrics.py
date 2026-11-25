"""CPU route tests. No model, sockets, wall-clock sleeps, or GPU access."""
import asyncio
import json
import sys
import time as wall_time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


@pytest.fixture
def run_stream(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "mlx", None)
    monkeypatch.setitem(sys.modules, "mlx.core", None)
    from mlx_serve import router

    monkeypatch.setattr(router.metrics, "_requests_file", tmp_path / "requests.jsonl")
    start = 18.0
    clock = SimpleNamespace(now=start)
    monkeypatch.setattr(router, "time", SimpleNamespace(
        monotonic=lambda: clock.now, gmtime=wall_time.gmtime, strftime=wall_time.strftime,
    ))

    def run(frames, *, cold=False):
        async def lines():
            for elapsed, data in frames:
                clock.now = start + elapsed
                yield "data: " + json.dumps(data)
            clock.now += 0.1
            yield "data: [DONE]"

        fake = SimpleNamespace(aiter_lines=lines, status_code=200, aclose=AsyncMock())
        fake_client = SimpleNamespace(build_request=lambda *args, **kwargs: object(),
                                      send=AsyncMock(return_value=fake))
        monkeypatch.setattr(router, "_HTTP_CLIENT", fake_client)

        async def consume():
            resp = await router._instrumented_stream_response(
                "http://127.0.0.1:1/v1/chat/completions",
                {"messages": [{"role": "user", "content": "private-prompt-canary"}]},
                {"X-OpenWebUI-Chat-Id": "private-session-canary"},
                "test-model", start, cold,
            )
            result = []
            async for chunk in resp.body_iterator:
                result.append(chunk)
                # A manager observes the upstream chunk before a downstream client handles it.
                clock.now += 0.0625
            return result

        wire = asyncio.run(consume())
        assert len(wire) == len(frames) + 1 and "[DONE]" in wire[-1]
        for idx, (_, frame) in enumerate(frames):
            assert json.loads(wire[idx].split(":", 1)[1]) == frame
        record = router.metrics.get_request_history(last_n=1)[0]
        disk = json.loads((tmp_path / "requests.jsonl").read_text().splitlines()[-1])
        for key in record:
            assert disk[key] == record[key], key
        for private in ("private-prompt-canary", "private-answer-canary",
                        "private-reason-canary", "private-session-canary"):
            assert private not in json.dumps(disk)
        return record
    return run


def event(**delta):
    return {"choices": [{"delta": delta}]}


def test_reasoning_content_and_delta_times_are_different_observations(run_stream):
    rec = run_stream([
        (0.0, event(role="assistant")), (0.1, event(content="")),
        (0.125, event(content="  ")),
        (0.25, event(tool_calls=[{"function": {"name": "fixture_tool"}}])),
        (0.5, event(reasoning_content="private-reason-canary")),
        (0.75, event(reasoning="several words in this single chunk")),
        (1.5, event(content="private-answer-canary")),
        (2.0, {"choices": [], "usage": {"prompt_tokens": 81, "completion_tokens": 11}}),
    ], cold=True)
    assert rec["first_meaningful_delta_ms"] == 250
    assert rec["first_text_delta_ms"] == rec["first_reasoning_delta_ms"] == 500
    assert rec["first_answer_delta_ms"] == 1500
    assert rec["next_text_delta_gap_ms"] == 250
    assert rec["text_delta_chunks"] == 3
    assert rec["completion_tokens"] == 11
    assert rec["prompt_tokens"] == 81
    assert rec["cold_start"] is True


def test_mixed_frame_counted_once_no_fabricated_token_count(run_stream):
    rec = run_stream([(0.0, event(content="many words with spaces", reasoning="three r tokens"))])
    assert rec["first_reasoning_delta_ms"] == rec["first_answer_delta_ms"] == 0
    assert rec["first_meaningful_delta_ms"] == rec["first_text_delta_ms"] == 0
    assert rec["text_delta_chunks"] == 1
    assert rec["next_text_delta_gap_ms"] is None
    assert rec["cold_start"] is False
    if rec["completion_tokens"] is not None:
        # Legacy callers may read an explicitly labeled estimate.
        assert "chunk" in rec["completion_tokens_source"].lower()


def test_no_text_and_reported_zero_are_distinct_from_absence(run_stream):
    rec = run_stream([
        (0.1, event(role="assistant")), (0.2, event(content="")),
        (0.3, {"choices": [], "usage": {"completion_tokens": 0, "prompt_tokens": 9}}),
    ])
    for k in ("first_meaningful_delta_ms", "first_reasoning_delta_ms",
              "first_answer_delta_ms", "first_text_delta_ms", "next_text_delta_gap_ms"):
        assert rec[k] is None, k
    assert rec["text_delta_chunks"] == 0
    assert rec["completion_tokens"] == 0


def test_answer_only_leaves_reasoning_unobserved(run_stream):
    rec = run_stream([(0.5, event(content="A")), (2.25, event(content="B"))])
    assert rec["first_reasoning_delta_ms"] is None
    assert rec["first_answer_delta_ms"] == rec["first_text_delta_ms"] == 500
    assert rec["next_text_delta_gap_ms"] == 1750
    assert rec["text_delta_chunks"] == 2
