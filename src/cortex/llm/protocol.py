"""Provider interfaces.

Everything that can generate text or vectors sits behind one of these. Keeping
them structural (``Protocol``) rather than inheritance-based means the test
suite can substitute a fake without importing httpx, and swapping Ollama for
mlx-lm is a config change rather than a code change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from cortex.models import DataPolicy, Sensitivity

__all__ = [
    "ChatMessage",
    "ChatProvider",
    "CompletionResult",
    "EmbeddingProvider",
    "PolicyViolation",
    "ProviderError",
    "ProviderSpec",
    "ProviderUnavailable",
    "RateLimited",
    "RerankProvider",
]


class ProviderError(RuntimeError):
    """Base class for provider failures."""


class RateLimited(ProviderError):
    """Provider returned 429. Carries the server's requested backoff."""

    def __init__(self, provider: str, retry_after: float | None = None) -> None:
        super().__init__(f"{provider} rate limited")
        self.provider = provider
        self.retry_after = retry_after


class ProviderUnavailable(ProviderError):
    """Transport failure, 5xx, or timeout."""

    def __init__(self, provider: str, detail: str = "") -> None:
        super().__init__(f"{provider} unavailable: {detail}".rstrip(": "))
        self.provider = provider


class PolicyViolation(ProviderError):
    """Refused: routing this content to this provider would breach its policy.

    Raised rather than silently downgraded. A second brain that quietly sends
    private notes somewhere unexpected is worse than one that errors.
    """


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: str
    content: str


@dataclass(slots=True)
class CompletionResult:
    text: str
    provider: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    elapsed_ms: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True, slots=True)
class ProviderSpec:
    """Declarative description of a provider endpoint.

    This is what the privacy gate reads. ``policy`` is a claim about the
    *operator*, verified out-of-band by reading their terms -- see
    ``docs/adr/0004-privacy-tiered-routing.md`` for the audit behind each
    shipped default.
    """

    name: str
    base_url: str
    model: str
    policy: DataPolicy
    api_key_env: str = ""
    """Environment variable holding the key. Never the key itself."""

    max_context: int = 8192
    rpm: int | None = None
    rpd: int | None = None
    tpd: int | None = None
    """Daily token budget, where the provider publishes one."""

    requires_zdr: bool = False
    """True when the policy is only honoured with zero-data-retention enabled."""

    zdr_enabled: bool = False
    priority: int = 100
    """Lower sorts earlier in the fallback chain."""

    extra_headers: dict[str, str] = field(default_factory=dict)
    enabled: bool = True

    def accepts(self, sensitivity: Sensitivity) -> bool:
        """Whether this provider may receive content at the given sensitivity.

        The whole privacy design reduces to this one predicate, which is why it
        lives on the spec and is tested directly.
        """
        if sensitivity is Sensitivity.PUBLIC:
            return True
        if self.policy is DataPolicy.LOCAL:
            return True
        if self.policy is DataPolicy.NO_TRAIN:
            return True
        if self.policy is DataPolicy.NO_TRAIN_IF_ZDR:
            return self.zdr_enabled
        return False

    @property
    def is_local(self) -> bool:
        return self.policy is DataPolicy.LOCAL


@runtime_checkable
class ChatProvider(Protocol):
    """Generates a completion from a message list."""

    spec: ProviderSpec

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        timeout: float = 60.0,
    ) -> CompletionResult: ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Produces dense vectors. Always local in the shipped configuration."""

    dimensions: int

    def embed(self, texts: list[str], *, is_query: bool = False) -> list[list[float]]: ...


@runtime_checkable
class RerankProvider(Protocol):
    """Cross-encoder reranking of candidates against a query."""

    def rerank(self, query: str, documents: list[str], *, top_k: int) -> list[tuple[int, float]]:
        """Return ``(original_index, score)`` pairs, best first."""
        ...
