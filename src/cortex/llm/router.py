"""Privacy-tiered model router.

The problem this exists to solve: several free LLM tiers train on submitted
data by default. A second brain routed naively through "whatever free endpoint
is up" will eventually post a private journal entry into someone's training
corpus, silently and irreversibly.

No general-purpose router models this. LiteLLM will happily fail over from a
no-train provider to a training one, because to it they are interchangeable
OpenAI-compatible endpoints. So the gate is the reason this module exists
rather than a dependency:

    content sensitivity  x  provider data policy  ->  allowed / refused

On top of the gate sit the ordinary reliability concerns -- ordered fallback,
429 backoff honouring ``Retry-After``, per-provider rolling budgets, and a
circuit breaker so a provider that is down stops being retried on every query.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from cortex.llm.protocol import (
    ChatMessage,
    ChatProvider,
    CompletionResult,
    PolicyViolation,
    ProviderError,
    ProviderSpec,
    ProviderUnavailable,
    RateLimited,
)
from cortex.models import Sensitivity

logger = logging.getLogger(__name__)

__all__ = ["BudgetLedger", "CircuitBreaker", "Router", "RoutingDecision"]


@dataclass(slots=True)
class RoutingDecision:
    """Audit record of how a single request was served.

    Retained and surfaced through the CLI and MCP layer so a user can always
    answer "where did this answer actually go?" -- that question having a
    cheap, honest answer is the point of the whole module.
    """

    provider: str
    sensitivity: Sensitivity
    attempted: list[str] = field(default_factory=list)
    refused: dict[str, str] = field(default_factory=dict)
    escalated: bool = False
    elapsed_ms: float = 0.0


@dataclass(slots=True)
class CircuitBreaker:
    """Per-provider failure gate.

    Opens after ``threshold`` consecutive failures and stays open for
    ``cooldown`` seconds. Prevents a dead provider from adding its timeout to
    the latency of every subsequent query.
    """

    threshold: int = 3
    cooldown: float = 60.0
    _failures: dict[str, int] = field(default_factory=dict)
    _opened_at: dict[str, float] = field(default_factory=dict)

    def is_open(self, provider: str, *, now: float | None = None) -> bool:
        opened = self._opened_at.get(provider)
        if opened is None:
            return False
        clock = time.monotonic() if now is None else now
        if clock - opened >= self.cooldown:
            # Cooldown elapsed: half-open, allow one probe.
            self._opened_at.pop(provider, None)
            self._failures[provider] = 0
            return False
        return True

    def record_failure(self, provider: str, *, now: float | None = None) -> None:
        count = self._failures.get(provider, 0) + 1
        self._failures[provider] = count
        if count >= self.threshold:
            self._opened_at[provider] = time.monotonic() if now is None else now
            logger.warning("circuit opened for %s after %d failures", provider, count)

    def record_success(self, provider: str) -> None:
        self._failures.pop(provider, None)
        self._opened_at.pop(provider, None)


@dataclass(slots=True)
class BudgetLedger:
    """Rolling request and token counters per provider.

    Free tiers publish limits as requests/minute, requests/day and
    tokens/day. Tracking them locally means we skip a provider we know is
    exhausted rather than burning a request to discover it -- which matters on
    OpenRouter, where *failed* requests still count against the daily quota.
    """

    _requests: dict[str, deque[float]] = field(default_factory=dict)
    _tokens: dict[str, deque[tuple[float, int]]] = field(default_factory=dict)

    _MINUTE = 60.0
    _DAY = 86_400.0

    def _prune(self, provider: str, now: float) -> None:
        reqs = self._requests.setdefault(provider, deque())
        while reqs and now - reqs[0] > self._DAY:
            reqs.popleft()
        toks = self._tokens.setdefault(provider, deque())
        while toks and now - toks[0][0] > self._DAY:
            toks.popleft()

    def has_headroom(self, spec: ProviderSpec, *, now: float | None = None) -> bool:
        clock = time.time() if now is None else now
        self._prune(spec.name, clock)
        reqs = self._requests[spec.name]

        if spec.rpm is not None:
            recent = sum(1 for ts in reqs if clock - ts <= self._MINUTE)
            if recent >= spec.rpm:
                return False
        if spec.rpd is not None and len(reqs) >= spec.rpd:
            return False
        if spec.tpd is not None:
            used = sum(count for _, count in self._tokens[spec.name])
            if used >= spec.tpd:
                return False
        return True

    def record(self, provider: str, tokens: int = 0, *, now: float | None = None) -> None:
        clock = time.time() if now is None else now
        self._prune(provider, clock)
        self._requests[provider].append(clock)
        if tokens:
            self._tokens[provider].append((clock, tokens))

    def usage(self, provider: str, *, now: float | None = None) -> dict[str, int]:
        clock = time.time() if now is None else now
        self._prune(provider, clock)
        reqs = self._requests[provider]
        return {
            "requests_last_minute": sum(1 for ts in reqs if clock - ts <= self._MINUTE),
            "requests_today": len(reqs),
            "tokens_today": sum(count for _, count in self._tokens[provider]),
        }


class Router:
    """Routes a completion request through an ordered, policy-gated chain."""

    def __init__(
        self,
        providers: Sequence[ChatProvider],
        *,
        breaker: CircuitBreaker | None = None,
        ledger: BudgetLedger | None = None,
        max_attempts: int = 4,
        sleep: object = None,
    ) -> None:
        self._providers = sorted(providers, key=lambda p: p.spec.priority)
        self.breaker = breaker or CircuitBreaker()
        self.ledger = ledger or BudgetLedger()
        self.max_attempts = max_attempts
        # Injected so tests never actually sleep through a backoff.
        self._sleep = sleep if callable(sleep) else time.sleep
        self.history: list[RoutingDecision] = []

    @property
    def providers(self) -> list[ChatProvider]:
        return list(self._providers)

    def eligible(self, sensitivity: Sensitivity) -> list[ChatProvider]:
        """Providers permitted to see content at this sensitivity."""
        return [p for p in self._providers if p.spec.enabled and p.spec.accepts(sensitivity)]

    def explain(self, sensitivity: Sensitivity) -> dict[str, str]:
        """Why each provider is or is not eligible. Powers ``cortex providers``."""
        out: dict[str, str] = {}
        for provider in self._providers:
            spec = provider.spec
            if not spec.enabled:
                out[spec.name] = "disabled"
            elif spec.accepts(sensitivity):
                out[spec.name] = f"eligible ({spec.policy.value})"
            elif spec.requires_zdr and not spec.zdr_enabled:
                out[spec.name] = "refused: zero-data-retention not enabled"
            else:
                out[spec.name] = f"refused: policy is {spec.policy.value}"
        return out

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        sensitivity: Sensitivity = Sensitivity.PRIVATE,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        local_only: bool = False,
        timeout: float = 60.0,
    ) -> tuple[CompletionResult, RoutingDecision]:
        """Complete via the first provider that is eligible, healthy and funded.

        Raises :class:`PolicyViolation` when the chain contains no provider
        permitted to see this content -- deliberately an error rather than a
        silent downgrade.
        """
        started = time.monotonic()
        decision = RoutingDecision(provider="", sensitivity=sensitivity)

        candidates: list[ChatProvider] = []
        for provider in self._providers:
            spec = provider.spec
            if not spec.enabled:
                continue
            if local_only and not spec.is_local:
                decision.refused[spec.name] = "local-only mode"
                continue
            if not spec.accepts(sensitivity):
                reason = (
                    "zero-data-retention not enabled"
                    if spec.requires_zdr and not spec.zdr_enabled
                    else f"provider policy is {spec.policy.value}"
                )
                decision.refused[spec.name] = reason
                logger.info("refusing %s for %s content: %s", spec.name, sensitivity.value, reason)
                continue
            candidates.append(provider)

        if not candidates:
            decision.elapsed_ms = (time.monotonic() - started) * 1000
            self.history.append(decision)
            raise PolicyViolation(
                f"No provider may handle {sensitivity.value} content. "
                f"Refused: {decision.refused or 'none configured'}"
            )

        last_error: ProviderError | None = None
        attempts = 0

        for provider in candidates:
            if attempts >= self.max_attempts:
                break
            spec = provider.spec

            if self.breaker.is_open(spec.name):
                decision.refused[spec.name] = "circuit open"
                continue
            if not self.ledger.has_headroom(spec):
                decision.refused[spec.name] = "budget exhausted"
                continue

            attempts += 1
            decision.attempted.append(spec.name)
            try:
                result = provider.complete(
                    messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=timeout,
                )
            except RateLimited as exc:
                last_error = exc
                self.breaker.record_failure(spec.name)
                # Consume quota even on failure: OpenRouter counts failed
                # requests against the daily allowance.
                self.ledger.record(spec.name)
                if exc.retry_after and exc.retry_after <= 2.0:
                    self._sleep(exc.retry_after)
                continue
            except ProviderError as exc:
                last_error = exc
                self.breaker.record_failure(spec.name)
                continue
            except Exception as exc:
                last_error = ProviderUnavailable(spec.name, str(exc))
                self.breaker.record_failure(spec.name)
                continue

            self.breaker.record_success(spec.name)
            self.ledger.record(spec.name, result.total_tokens)
            decision.provider = spec.name
            decision.escalated = not spec.is_local
            decision.elapsed_ms = (time.monotonic() - started) * 1000
            self.history.append(decision)
            return result, decision

        decision.elapsed_ms = (time.monotonic() - started) * 1000
        self.history.append(decision)
        raise last_error or ProviderUnavailable("router", "no provider succeeded")


def build_chain(specs: Iterable[ProviderSpec]) -> list[ProviderSpec]:
    """Order specs into a fallback chain: local first, then by priority."""
    return sorted(specs, key=lambda s: (not s.is_local, s.priority, s.name))
