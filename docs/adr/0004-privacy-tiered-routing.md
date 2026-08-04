# ADR 0004: Gate model routing on provider data policy

**Status:** Accepted · **Date:** 2026-08-04

## Context

Cortex runs over private notes. Cloud escalation is desirable — a 4B local
model cannot match a 70B on hard synthesis — but several free tiers use
submitted data for training.

Audit of the providers considered, August 2026:

| Provider | Trains on free-tier data? | Notes |
|---|---|---|
| Ollama / mlx-lm | No | Local. |
| Groq | No | 14,400 req/day on llama-3.1-8b — most generous free tier found. |
| NVIDIA NIM | No | ~40 RPM, no daily cap. Dev/test terms. |
| Cerebras | No | 1M tokens/day, but 8K context cap. |
| OpenRouter | Not itself | Downstream provider policy applies. Safe **only** with zero-data-retention enabled. |
| OpenCode Zen (free) | **Yes** | Docs state free-period data "may be used to improve the model". |
| Google AI Studio (free) | **Yes** | Outside EU/UK/EEA. Vertex AI and the paid tier are exempt. |
| Mistral Experiment | **Yes** | Opt-out available in console. |

Also noted: GitHub Models was retired 30 July 2026; Together AI has trial
credits, not a lasting free tier.

No general-purpose router models this distinction. LiteLLM will fail over from
a no-training provider to a training one, because to it they are
interchangeable OpenAI-compatible endpoints.

## Decision

Providers declare a `DataPolicy`; content carries a `Sensitivity`. Eligibility
is one predicate:

```python
def accepts(self, sensitivity) -> bool:
    if sensitivity is PUBLIC:              return True
    if self.policy in (LOCAL, NO_TRAIN):   return True
    if self.policy is NO_TRAIN_IF_ZDR:     return self.zdr_enabled
    return False
```

Vault content defaults to `PRIVATE`. Opting out requires explicit frontmatter.
When no eligible provider exists, Cortex raises `PolicyViolation` rather than
downgrading.

A thin (~250 LOC) router is written rather than adopting LiteLLM, because the
gate is the whole point and we would be wrapping LiteLLM anyway. It also avoids
a heavy dependency tree on a 16 GB machine.

## Consequences

- A mixed context is private. One private chunk makes the whole request
  private — a summary of private notes is itself private, so there is no
  partial redaction.
- Failure is loud. Tested explicitly: when a preferred no-train provider fails
  and a training provider sits next in the chain, the router raises instead of
  falling through. That is the precise leak a naive chain produces.
- Providers that train are shipped **disabled** with their policy labelled, so
  enabling one is deliberate and visible.
- Every decision is recorded in `Router.history` and surfaced by
  `cortex providers`. "Where did this answer actually go?" must have a cheap,
  honest answer.
- Retrieval never escalates. Only synthesis, and only the top-k chunks.
- The router re-implements circuit breaking and budget tracking. Failed
  requests consume quota, because OpenRouter counts failed requests against the
  daily allowance.
