# Cortex OS dashboard contract

## Status

Cortex now has a read-only live analytics endpoint backed by the local Obsidian vault, the Graphify manifest, and the Cortex report-card state. The frontend intentionally exposes aggregate telemetry only. The Obsidian note bodies are never sent to the browser endpoint or to cloud agents.

The repository contains PAUL directory placeholders, but PAUL is not initialized: the expected workflow, rule, reference, and template files are absent. This contract is therefore the safe integration seam for PAUL once the framework is installed correctly.

## Data sources

| Source | Read | Write | Exposed |
|---|---:|---:|---|
| Obsidian vault | Yes | No | Note count and top-level folder counts only |
| Graphify output | Yes | No | Artifact names and availability only |
| `agent_os/state.json` | Yes | No | Agent task counts, success rates, and privilege state |
| Cloud Scout/Manager | Yes | No | Supervised roster and cycle health |

## PAUL handoff contract

A future PAUL workflow should consume `GET /api/analytics` as its dashboard data contract. It may propose plans from agent health and graph-shape changes, but it must not send vault note text to a cloud model, enable agent writes, or bypass the report-card gate. Any implementation task must remain supervised until the task type has at least 20 completed runs and a 95% success rate; privilege is revoked below 90%.

## Run locally

```bash
CORTEX_VAULT="/path/to/your/Obsidian/vault" python3 agent_os/dashboard.py
```

Then open `http://127.0.0.1:8765/`. The page refreshes aggregate telemetry every 15 seconds.
