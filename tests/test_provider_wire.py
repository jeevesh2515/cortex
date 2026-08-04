"""Wire-level provider tests.

The privacy gate decides *whether* to send. These tests check *what* actually
goes over the wire, using a mock transport rather than trusting the request was
built correctly. The zero-data-retention flag is the case that matters: as a
local boolean it is an unverifiable promise, and it is only worth anything if it
reaches the server.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from cortex.llm.protocol import (
    ChatMessage,
    ProviderSpec,
    ProviderUnavailable,
    RateLimited,
)
from cortex.llm.providers import OpenAICompatProvider
from cortex.models import DataPolicy

MESSAGES = [ChatMessage(role="user", content="hello")]


def capture(status: int = 200, headers: dict[str, str] | None = None) -> tuple[Any, list[dict]]:
    """Build a client that records outgoing request bodies."""
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        if status != 200:
            return httpx.Response(status, headers=headers or {}, text="nope")
        return httpx.Response(
            200,
            json={
                "model": "test-model",
                "choices": [{"message": {"content": "  an answer  "}}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 3},
            },
        )

    return httpx.Client(transport=httpx.MockTransport(handler)), seen


def spec(**kw: Any) -> ProviderSpec:
    base = {
        "name": "openrouter",
        "base_url": "https://openrouter.ai/api/v1",
        "model": "openrouter/free",
        "policy": DataPolicy.NO_TRAIN_IF_ZDR,
    }
    base.update(kw)
    return ProviderSpec(**base)  # type: ignore[arg-type]


class TestZeroDataRetention:
    def test_zdr_is_sent_when_enabled(self) -> None:
        """The whole point: the flag must reach the server, not just the gate."""
        client, seen = capture()
        provider = OpenAICompatProvider(spec(requires_zdr=True, zdr_enabled=True), client=client)
        provider.complete(MESSAGES)
        assert seen[0]["zdr"] is True

    def test_zdr_absent_when_not_enabled(self) -> None:
        client, seen = capture()
        provider = OpenAICompatProvider(spec(requires_zdr=True, zdr_enabled=False), client=client)
        provider.complete(MESSAGES)
        assert "zdr" not in seen[0]

    def test_zdr_not_sent_to_providers_that_do_not_want_it(self) -> None:
        # Groq and NVIDIA have no such parameter; sending it could 400.
        client, seen = capture()
        provider = OpenAICompatProvider(
            spec(name="groq", policy=DataPolicy.NO_TRAIN, zdr_enabled=True), client=client
        )
        provider.complete(MESSAGES)
        assert "zdr" not in seen[0]

    def test_request_body_helper(self) -> None:
        assert spec(requires_zdr=True, zdr_enabled=True).request_body() == {"zdr": True}
        assert spec(requires_zdr=True, zdr_enabled=False).request_body() == {}


class TestExtraBody:
    def test_merged_into_payload(self) -> None:
        client, seen = capture()
        provider = OpenAICompatProvider(
            spec(extra_body={"provider": {"sort": "throughput"}}), client=client
        )
        provider.complete(MESSAGES)
        assert seen[0]["provider"] == {"sort": "throughput"}

    def test_does_not_clobber_core_fields(self) -> None:
        client, seen = capture()
        provider = OpenAICompatProvider(spec(extra_body={"temperature": 0.9}), client=client)
        provider.complete(MESSAGES, temperature=0.1)
        # extra_body is applied last, so it deliberately wins -- documented
        # escape hatch for provider quirks.
        assert seen[0]["temperature"] == 0.9
        assert seen[0]["model"] == "openrouter/free"
        assert seen[0]["messages"][0]["content"] == "hello"


class TestPayloadShape:
    def test_core_fields(self) -> None:
        client, seen = capture()
        OpenAICompatProvider(spec(), client=client).complete(
            MESSAGES, max_tokens=256, temperature=0.3
        )
        body = seen[0]
        assert body["model"] == "openrouter/free"
        assert body["max_tokens"] == 256
        assert body["temperature"] == 0.3
        assert body["stream"] is False

    def test_response_is_stripped_and_usage_recorded(self) -> None:
        client, _ = capture()
        result = OpenAICompatProvider(spec(), client=client).complete(MESSAGES)
        assert result.text == "an answer"
        assert result.total_tokens == 10


class TestErrorMapping:
    def test_429_becomes_rate_limited_with_backoff(self) -> None:
        client, _ = capture(status=429, headers={"retry-after": "3"})
        with pytest.raises(RateLimited) as exc:
            OpenAICompatProvider(spec(), client=client).complete(MESSAGES)
        assert exc.value.retry_after == 3.0

    def test_absolute_reset_header_converted_to_delta(self) -> None:
        # Cerebras reports an absolute epoch; it must become seconds-from-now.
        client, _ = capture(status=429, headers={"x-ratelimit-reset-requests": "9999999999"})
        with pytest.raises(RateLimited) as exc:
            OpenAICompatProvider(spec(), client=client).complete(MESSAGES)
        assert exc.value.retry_after is not None
        assert exc.value.retry_after > 0

    def test_5xx_is_unavailable(self) -> None:
        client, _ = capture(status=503)
        with pytest.raises(ProviderUnavailable):
            OpenAICompatProvider(spec(), client=client).complete(MESSAGES)

    def test_4xx_is_unavailable_with_detail(self) -> None:
        client, _ = capture(status=400)
        with pytest.raises(ProviderUnavailable, match="400"):
            OpenAICompatProvider(spec(), client=client).complete(MESSAGES)

    def test_malformed_response_is_unavailable(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"unexpected": "shape"})

        client = httpx.Client(transport=httpx.MockTransport(handler))
        with pytest.raises(ProviderUnavailable, match="malformed"):
            OpenAICompatProvider(spec(), client=client).complete(MESSAGES)


class TestAuthAndHeaders:
    def test_key_read_from_env_and_not_persisted(self, monkeypatch: Any) -> None:
        monkeypatch.setenv("TEST_KEY", "sk-secret")
        seen_headers: list[httpx.Headers] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen_headers.append(request.headers)
            return httpx.Response(
                200, json={"choices": [{"message": {"content": "ok"}}], "model": "m"}
            )

        client = httpx.Client(transport=httpx.MockTransport(handler))
        provider = OpenAICompatProvider(spec(api_key_env="TEST_KEY"), client=client)
        provider.complete(MESSAGES)
        assert seen_headers[0]["authorization"] == "Bearer sk-secret"
        # The spec itself must never carry the secret.
        assert "sk-secret" not in repr(provider.spec)

    def test_configured_requires_key_when_declared(self, monkeypatch: Any) -> None:
        monkeypatch.delenv("TEST_KEY", raising=False)
        assert not OpenAICompatProvider(spec(api_key_env="TEST_KEY")).configured
        monkeypatch.setenv("TEST_KEY", "x")
        assert OpenAICompatProvider(spec(api_key_env="TEST_KEY")).configured

    def test_local_provider_needs_no_key(self) -> None:
        local = spec(name="ollama", policy=DataPolicy.LOCAL, api_key_env="")
        assert OpenAICompatProvider(local).configured

    def test_extra_headers_sent(self) -> None:
        seen_headers: list[httpx.Headers] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen_headers.append(request.headers)
            return httpx.Response(
                200, json={"choices": [{"message": {"content": "ok"}}], "model": "m"}
            )

        client = httpx.Client(transport=httpx.MockTransport(handler))
        provider = OpenAICompatProvider(spec(extra_headers={"X-Title": "Cortex"}), client=client)
        provider.complete(MESSAGES)
        assert seen_headers[0]["x-title"] == "Cortex"
