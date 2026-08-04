"""HTTP provider implementations.

Every endpoint Cortex talks to -- Ollama, mlx-lm, Groq, NVIDIA NIM, OpenRouter
-- speaks the OpenAI chat-completions shape, so one client covers all of them
and the differences collapse into configuration.

Keys are read from the environment at call time and never logged, never stored
in the catalog, and never written to the config file.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Sequence
from typing import Any

import httpx

from cortex.llm.protocol import (
    ChatMessage,
    CompletionResult,
    ProviderSpec,
    ProviderUnavailable,
    RateLimited,
)

logger = logging.getLogger(__name__)

__all__ = ["HashEmbedder", "OllamaEmbedder", "OpenAICompatProvider"]


def _retry_after(response: httpx.Response) -> float | None:
    """Extract a backoff hint, handling the several header spellings in use.

    Cerebras reports absolute reset timestamps; most others use a relative
    ``Retry-After``. Both are normalised to seconds-from-now.
    """
    raw = response.headers.get("retry-after")
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    for header in ("x-ratelimit-reset-requests", "x-ratelimit-reset-tokens"):
        raw = response.headers.get(header)
        if not raw:
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        # Heuristic: a large value is an absolute epoch, a small one is a delta.
        return max(0.0, value - time.time()) if value > 1e6 else value
    return None


class OpenAICompatProvider:
    """Chat provider for any OpenAI-compatible ``/chat/completions`` endpoint."""

    def __init__(self, spec: ProviderSpec, *, client: httpx.Client | None = None) -> None:
        self.spec = spec
        self._client = client
        self._owns_client = client is None

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", **self.spec.extra_headers}
        if self.spec.api_key_env:
            key = os.environ.get(self.spec.api_key_env, "")
            if key:
                headers["Authorization"] = f"Bearer {key}"
        return headers

    @property
    def configured(self) -> bool:
        """True when this provider has what it needs to be called.

        A local endpoint needs no key; a remote one without its key is simply
        skipped rather than attempted and failed.
        """
        if not self.spec.api_key_env:
            return True
        return bool(os.environ.get(self.spec.api_key_env))

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=60.0)
        return self._client

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        timeout: float = 60.0,
    ) -> CompletionResult:
        url = f"{self.spec.base_url.rstrip('/')}/chat/completions"
        payload: dict[str, Any] = {
            "model": self.spec.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        # Provider-specific fields, notably OpenRouter's per-request `zdr`.
        # Server-side enforcement beats a local flag we cannot verify.
        payload.update(self.spec.request_body())

        started = time.monotonic()
        try:
            response = self._get_client().post(
                url, json=payload, headers=self._headers(), timeout=timeout
            )
        except httpx.TimeoutException as exc:
            raise ProviderUnavailable(self.spec.name, f"timeout: {exc}") from exc
        except httpx.HTTPError as exc:
            raise ProviderUnavailable(self.spec.name, str(exc)) from exc

        if response.status_code == 429:
            raise RateLimited(self.spec.name, _retry_after(response))
        if response.status_code >= 500:
            raise ProviderUnavailable(self.spec.name, f"HTTP {response.status_code}")
        if response.status_code >= 400:
            # 4xx is a bad request, not a transient fault. Surface the detail
            # but truncate: some gateways echo the whole prompt back.
            raise ProviderUnavailable(
                self.spec.name, f"HTTP {response.status_code}: {response.text[:200]}"
            )

        try:
            data = response.json()
            text = data["choices"][0]["message"]["content"] or ""
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ProviderUnavailable(self.spec.name, f"malformed response: {exc}") from exc

        usage = data.get("usage") or {}
        return CompletionResult(
            text=text.strip(),
            provider=self.spec.name,
            model=data.get("model", self.spec.model),
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            elapsed_ms=(time.monotonic() - started) * 1000,
        )

    def close(self) -> None:
        if self._client is not None and self._owns_client:
            self._client.close()
            self._client = None


class OllamaEmbedder:
    """Embeddings via Ollama's native ``/api/embed`` endpoint.

    Default model is ``qwen3-embedding:0.6b``: 70.7 MTEB(eng,v2), 32K context,
    Apache-2.0, about 1.5 GB resident. On a 16 GB machine that is the best
    quality-per-gigabyte available in 2026 -- the 8B variant scores higher but
    wants 17 GB to itself.
    """

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:11434",
        model: str = "qwen3-embedding:0.6b",
        dimensions: int = 1024,
        client: httpx.Client | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.dimensions = dimensions
        self.timeout = timeout
        self._client = client
        self._owns_client = client is None

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    def embed(self, texts: list[str], *, is_query: bool = False) -> list[list[float]]:
        """Embed a batch.

        Asymmetric models want queries and documents prefixed differently. The
        Qwen3 family expects an instruction prefix on the *query* side only;
        applying it to documents measurably degrades retrieval.
        """
        if not texts:
            return []
        payload_input = (
            [f"Instruct: Retrieve relevant passages\nQuery: {t}" for t in texts]
            if is_query
            else texts
        )
        try:
            response = self._get_client().post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": payload_input},
                timeout=self.timeout,
            )
        except httpx.HTTPError as exc:
            raise ProviderUnavailable("ollama-embed", str(exc)) from exc

        if response.status_code >= 400:
            raise ProviderUnavailable(
                "ollama-embed", f"HTTP {response.status_code}: {response.text[:200]}"
            )
        data = response.json()
        vectors = data.get("embeddings") or []
        if not vectors:
            raise ProviderUnavailable("ollama-embed", "no embeddings returned")
        return [[float(x) for x in vec] for vec in vectors]

    def close(self) -> None:
        if self._client is not None and self._owns_client:
            self._client.close()
            self._client = None


class HashEmbedder:
    """Deterministic hashing embedder.

    Not a semantic model. It exists so the full pipeline -- index, search, fuse,
    cite -- can be exercised end to end in CI and on a fresh clone before the
    user has pulled a real model. Bag-of-words hashing does give genuine lexical
    overlap signal, so pipeline tests assert meaningful behaviour rather than
    noise, but it must never be used for real retrieval.
    """

    def __init__(self, dimensions: int = 256) -> None:
        self.dimensions = dimensions

    def embed(self, texts: list[str], *, is_query: bool = False) -> list[list[float]]:
        import hashlib
        import math
        import re

        vectors: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dimensions
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
                index = int.from_bytes(digest[:4], "big") % self.dimensions
                sign = 1.0 if digest[4] & 1 else -1.0
                vec[index] += sign
            norm = math.sqrt(sum(x * x for x in vec))
            vectors.append([x / norm for x in vec] if norm else vec)
        return vectors


def build_chat_providers(
    specs: Sequence[ProviderSpec], *, client: httpx.Client | None = None
) -> list[OpenAICompatProvider]:
    """Instantiate providers, dropping any whose key is absent."""
    providers = []
    for spec in specs:
        provider = OpenAICompatProvider(spec, client=client)
        if provider.configured:
            providers.append(provider)
        else:
            logger.debug("skipping %s: %s not set", spec.name, spec.api_key_env)
    return providers
