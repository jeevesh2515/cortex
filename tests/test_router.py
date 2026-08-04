"""Router tests.

The privacy gate is the security boundary of this project, so it is tested as
one: explicitly, including the failure modes where a naive implementation would
leak (fallback past a refused provider, ZDR disabled, local-only mode).
"""

from __future__ import annotations

import pytest

from cortex.llm.protocol import (
    ChatMessage,
    CompletionResult,
    PolicyViolation,
    ProviderSpec,
    ProviderUnavailable,
    RateLimited,
)
from cortex.llm.router import BudgetLedger, CircuitBreaker, Router, build_chain
from cortex.models import DataPolicy, Sensitivity

MESSAGES = [ChatMessage(role="user", content="what did I write about retrieval?")]


class FakeProvider:
    """Scriptable provider: succeeds, or raises a queued exception."""

    def __init__(self, spec: ProviderSpec, *, fail_with: Exception | None = None) -> None:
        self.spec = spec
        self.fail_with = fail_with
        self.calls = 0

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        timeout: float = 60.0,
    ) -> CompletionResult:
        self.calls += 1
        if self.fail_with is not None:
            raise self.fail_with
        return CompletionResult(
            text=f"answer from {self.spec.name}",
            provider=self.spec.name,
            model=self.spec.model,
            prompt_tokens=10,
            completion_tokens=20,
        )


def spec(
    name: str,
    policy: DataPolicy,
    *,
    priority: int = 100,
    zdr: bool = False,
    requires_zdr: bool = False,
    **kwargs: object,
) -> ProviderSpec:
    return ProviderSpec(
        name=name,
        base_url=f"https://{name}.example/v1",
        model=f"{name}-model",
        policy=policy,
        priority=priority,
        zdr_enabled=zdr,
        requires_zdr=requires_zdr,
        **kwargs,  # type: ignore[arg-type]
    )


LOCAL = spec("local", DataPolicy.LOCAL, priority=0)
GROQ = spec("groq", DataPolicy.NO_TRAIN, priority=10)
NVIDIA = spec("nvidia", DataPolicy.NO_TRAIN, priority=20)
OPENROUTER_ZDR = spec(
    "openrouter", DataPolicy.NO_TRAIN_IF_ZDR, priority=30, zdr=True, requires_zdr=True
)
OPENROUTER_NO_ZDR = spec(
    "openrouter", DataPolicy.NO_TRAIN_IF_ZDR, priority=30, zdr=False, requires_zdr=True
)
TRAINER = spec("zen", DataPolicy.TRAINS, priority=5)  # deliberately high priority


class TestPolicyGate:
    """The security boundary."""

    def test_private_content_never_reaches_a_training_provider(self) -> None:
        trainer = FakeProvider(TRAINER)
        groq = FakeProvider(GROQ)
        router = Router([trainer, groq])

        result, decision = router.complete(MESSAGES, sensitivity=Sensitivity.PRIVATE)

        assert trainer.calls == 0, "training provider must never be called"
        assert result.provider == "groq"
        assert "zen" in decision.refused
        assert "policy is trains" in decision.refused["zen"]

    def test_training_provider_allowed_for_public_content(self) -> None:
        trainer = FakeProvider(TRAINER)
        router = Router([trainer, FakeProvider(GROQ)])
        result, _ = router.complete(MESSAGES, sensitivity=Sensitivity.PUBLIC)
        # TRAINER has priority 5, ahead of groq's 10.
        assert result.provider == "zen"

    def test_zdr_required_and_disabled_is_refused(self) -> None:
        provider = FakeProvider(OPENROUTER_NO_ZDR)
        router = Router([provider])
        with pytest.raises(PolicyViolation):
            router.complete(MESSAGES, sensitivity=Sensitivity.PRIVATE)
        assert provider.calls == 0

    def test_zdr_required_and_enabled_is_allowed(self) -> None:
        router = Router([FakeProvider(OPENROUTER_ZDR)])
        result, _ = router.complete(MESSAGES, sensitivity=Sensitivity.PRIVATE)
        assert result.provider == "openrouter"

    def test_no_eligible_provider_raises_rather_than_downgrades(self) -> None:
        router = Router([FakeProvider(TRAINER)])
        with pytest.raises(PolicyViolation, match="No provider may handle private"):
            router.complete(MESSAGES, sensitivity=Sensitivity.PRIVATE)

    def test_failover_does_not_cross_the_policy_line(self) -> None:
        # The subtle leak: first choice fails, and a naive chain falls through
        # to whatever is next -- including a training provider.
        groq = FakeProvider(GROQ, fail_with=ProviderUnavailable("groq", "503"))
        trainer = FakeProvider(TRAINER)
        router = Router([groq, trainer])

        with pytest.raises(ProviderUnavailable):
            router.complete(MESSAGES, sensitivity=Sensitivity.PRIVATE)
        assert trainer.calls == 0

    def test_local_only_blocks_all_remote(self) -> None:
        local = FakeProvider(LOCAL)
        groq = FakeProvider(GROQ)
        router = Router([groq, local])
        result, decision = router.complete(MESSAGES, local_only=True)
        assert result.provider == "local"
        assert groq.calls == 0
        assert decision.refused["groq"] == "local-only mode"
        assert not decision.escalated

    def test_spec_accepts_matrix(self) -> None:
        assert LOCAL.accepts(Sensitivity.PRIVATE)
        assert GROQ.accepts(Sensitivity.PRIVATE)
        assert not TRAINER.accepts(Sensitivity.PRIVATE)
        assert TRAINER.accepts(Sensitivity.PUBLIC)
        assert not OPENROUTER_NO_ZDR.accepts(Sensitivity.PRIVATE)
        assert OPENROUTER_ZDR.accepts(Sensitivity.PRIVATE)


class TestFallbackChain:
    def test_prefers_lowest_priority_number(self) -> None:
        router = Router([FakeProvider(NVIDIA), FakeProvider(GROQ)])
        result, _ = router.complete(MESSAGES)
        assert result.provider == "groq"

    def test_falls_through_on_failure(self) -> None:
        groq = FakeProvider(GROQ, fail_with=ProviderUnavailable("groq", "timeout"))
        nvidia = FakeProvider(NVIDIA)
        router = Router([groq, nvidia])
        result, decision = router.complete(MESSAGES)
        assert result.provider == "nvidia"
        assert decision.attempted == ["groq", "nvidia"]

    def test_rate_limit_falls_through(self) -> None:
        groq = FakeProvider(GROQ, fail_with=RateLimited("groq", retry_after=0.01))
        nvidia = FakeProvider(NVIDIA)
        router = Router([groq, nvidia], sleep=lambda _: None)
        result, _ = router.complete(MESSAGES)
        assert result.provider == "nvidia"

    def test_unexpected_exception_is_normalised(self) -> None:
        broken = FakeProvider(GROQ, fail_with=ValueError("kaboom"))
        router = Router([broken, FakeProvider(NVIDIA)])
        result, _ = router.complete(MESSAGES)
        assert result.provider == "nvidia"

    def test_max_attempts_respected(self) -> None:
        providers = [
            FakeProvider(
                spec(f"p{i}", DataPolicy.NO_TRAIN, priority=i),
                fail_with=ProviderUnavailable(f"p{i}"),
            )
            for i in range(6)
        ]
        router = Router(providers, max_attempts=2)
        with pytest.raises(ProviderUnavailable):
            router.complete(MESSAGES)
        assert sum(p.calls for p in providers) == 2

    def test_escalation_flag(self) -> None:
        router = Router([FakeProvider(LOCAL)])
        _, decision = router.complete(MESSAGES)
        assert not decision.escalated

        router = Router([FakeProvider(GROQ)])
        _, decision = router.complete(MESSAGES)
        assert decision.escalated


class TestCircuitBreaker:
    def test_opens_after_threshold(self) -> None:
        breaker = CircuitBreaker(threshold=2, cooldown=30)
        assert not breaker.is_open("groq")
        breaker.record_failure("groq")
        assert not breaker.is_open("groq")
        breaker.record_failure("groq")
        assert breaker.is_open("groq")

    def test_success_resets(self) -> None:
        breaker = CircuitBreaker(threshold=2)
        breaker.record_failure("groq")
        breaker.record_success("groq")
        breaker.record_failure("groq")
        assert not breaker.is_open("groq")

    def test_cooldown_half_opens(self) -> None:
        breaker = CircuitBreaker(threshold=1, cooldown=10)
        breaker.record_failure("groq", now=1000.0)
        assert breaker.is_open("groq", now=1005.0)
        assert not breaker.is_open("groq", now=1011.0)

    def test_router_skips_open_circuit(self) -> None:
        breaker = CircuitBreaker(threshold=1)
        breaker.record_failure("groq")
        groq = FakeProvider(GROQ)
        router = Router([groq, FakeProvider(NVIDIA)], breaker=breaker)
        result, decision = router.complete(MESSAGES)
        assert result.provider == "nvidia"
        assert groq.calls == 0
        assert decision.refused["groq"] == "circuit open"


class TestBudgetLedger:
    def test_rpm_enforced(self) -> None:
        ledger = BudgetLedger()
        limited = spec("groq", DataPolicy.NO_TRAIN, rpm=2)
        assert ledger.has_headroom(limited, now=1000.0)
        ledger.record("groq", now=1000.0)
        ledger.record("groq", now=1001.0)
        assert not ledger.has_headroom(limited, now=1002.0)
        # A minute later the window has slid.
        assert ledger.has_headroom(limited, now=1065.0)

    def test_rpd_enforced(self) -> None:
        ledger = BudgetLedger()
        limited = spec("groq", DataPolicy.NO_TRAIN, rpd=2)
        ledger.record("groq", now=1000.0)
        ledger.record("groq", now=2000.0)
        assert not ledger.has_headroom(limited, now=3000.0)

    def test_tpd_enforced(self) -> None:
        ledger = BudgetLedger()
        limited = spec("cerebras", DataPolicy.NO_TRAIN, tpd=100)
        ledger.record("cerebras", tokens=60, now=1000.0)
        assert ledger.has_headroom(limited, now=1001.0)
        ledger.record("cerebras", tokens=50, now=1002.0)
        assert not ledger.has_headroom(limited, now=1003.0)

    def test_failed_request_still_consumes_quota(self) -> None:
        # OpenRouter counts failed requests against the daily allowance.
        limited = spec("groq", DataPolicy.NO_TRAIN, rpd=1, priority=10)
        groq = FakeProvider(limited, fail_with=RateLimited("groq"))
        router = Router([groq, FakeProvider(NVIDIA)], sleep=lambda _: None)
        router.complete(MESSAGES)
        assert not router.ledger.has_headroom(limited)

    def test_usage_report(self) -> None:
        ledger = BudgetLedger()
        ledger.record("groq", tokens=42, now=1000.0)
        usage = ledger.usage("groq", now=1010.0)
        assert usage["requests_today"] == 1
        assert usage["tokens_today"] == 42

    def test_router_skips_exhausted_provider(self) -> None:
        ledger = BudgetLedger()
        limited = spec("groq", DataPolicy.NO_TRAIN, rpd=1, priority=10)
        ledger.record("groq")
        groq = FakeProvider(limited)
        router = Router([groq, FakeProvider(NVIDIA)], ledger=ledger)
        result, decision = router.complete(MESSAGES)
        assert result.provider == "nvidia"
        assert decision.refused["groq"] == "budget exhausted"


class TestAudit:
    def test_history_records_every_decision(self) -> None:
        router = Router([FakeProvider(GROQ)])
        router.complete(MESSAGES)
        router.complete(MESSAGES)
        assert len(router.history) == 2
        assert all(d.provider == "groq" for d in router.history)

    def test_explain_lists_reasons(self) -> None:
        router = Router(
            [FakeProvider(TRAINER), FakeProvider(GROQ), FakeProvider(OPENROUTER_NO_ZDR)]
        )
        explained = router.explain(Sensitivity.PRIVATE)
        assert "eligible" in explained["groq"]
        assert "refused" in explained["zen"]
        assert "zero-data-retention" in explained["openrouter"]

    def test_disabled_provider_skipped(self) -> None:
        disabled = spec("groq", DataPolicy.NO_TRAIN, priority=1, enabled=False)
        router = Router([FakeProvider(disabled), FakeProvider(NVIDIA)])
        result, _ = router.complete(MESSAGES)
        assert result.provider == "nvidia"


class TestBuildChain:
    def test_local_sorts_first(self) -> None:
        ordered = build_chain([GROQ, LOCAL, NVIDIA])
        assert ordered[0].name == "local"

    def test_remote_sorted_by_priority(self) -> None:
        ordered = build_chain([NVIDIA, GROQ])
        assert [s.name for s in ordered] == ["groq", "nvidia"]
