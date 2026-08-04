# ADR 0003: Treat thermal headroom as a schedulable resource

**Status:** Accepted · **Date:** 2026-08-04

## Context

The target machine is a fanless MacBook Air M5 (16 GB, 153 GB/s). Community
thermal logging of sustained local inference shows:

| Elapsed | Throughput |
|---|---|
| 0–2 min | ~35 tok/s |
| 5–8 min | ~28 tok/s |
| 12–20 min | ~15 tok/s |
| 20+ min | ~13 tok/s |

A ~60% collapse. The MacBook Pro, with active cooling, holds peak indefinitely.
No software change reverses this; it is a chassis property.

Indexing at full tilt therefore means the machine is hot and slow for the whole
backfill, and — critically — the user's *interactive* queries land in the
throttled regime.

## Decision

A governor samples power and thermal state and classifies the machine into
`boost` / `nominal` / `throttled` / `critical`. Background indexing concurrency,
inter-batch pauses, and whether backfill runs at all derive from that state.

**Interactive queries are exempt.** `workers(interactive=True)` always returns
full concurrency regardless of state.

Signals come from `pmset -g therm` (`CPU_Speed_Limit`) and `pmset -g batt`.
`pmset` is used rather than `powermetrics` specifically because it does **not
require sudo** — a background daemon prompting for an admin password is
unshippable.

## Consequences

- Hysteresis is asymmetric: tightening applies on a single hot reading,
  loosening needs two consecutive agreeing samples. Symmetric hysteresis
  oscillates — the machine cools for one sample, resumes at full tilt, and
  immediately re-throttles.
- The first sample after start bypasses hysteresis. Hysteresis guards against
  flapping between states, which presupposes a prior state; on cold start there
  is nothing to flap from, and requiring two intervals to climb out of the
  default was a real bug caught in test.
- Backfill pauses below 30% battery regardless of temperature.
- Deletion reconciliation runs even when backfill is paused. Ghost chunks are
  cheap to remove and actively harmful to leave.
- The governor is observable via `cortex status` and the MCP `vault_status`
  tool. An invisible mechanism that makes a machine feel slow is a bug report
  waiting to happen.
- Probes are injected behind a protocol, so the policy is unit-tested on any
  platform.
