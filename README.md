<div align="center">

# Cortex

**A local-first second brain for Obsidian, engineered for a fanless MacBook Air.**

Hybrid retrieval over your own notes — dense vectors, BM25, and the wikilink graph you already built by hand. Nothing leaves your machine unless you say so, and never to a provider that trains on your data.

[![CI](https://github.com/jeevesh2515/cortex/actions/workflows/ci.yml/badge.svg)](https://github.com/jeevesh2515/cortex/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

</div>

---

## Why this exists

Most local-RAG projects are a vector database with a chat box bolted on. Three things here are different, and each came out of a specific problem.

### 1. Your vault is already a knowledge graph

GraphRAG extracts entity relationships with an LLM. It is slow, it costs real money — roughly $34 per thousand chunks — and a controlled 2026 evaluation found it *loses* to flat hybrid retrieval on multi-hop prose queries.

None of that applies to Obsidian. You already built a link graph, one `[[wikilink]]` at a time. It is cleaner than anything an extractor would infer, it costs nothing, and it updates the moment you save. Cortex reads it directly and uses it as a third retrieval signal alongside vectors and keywords.

The practical effect: ask about fermentation and you get the note on pectin chains you linked six months ago, even though it shares not one word with your query.

### 2. The MacBook Air has no fan

Sustained inference on a fanless Air collapses about 60% — roughly 35 tokens/sec for the first two minutes, then a slide to 13 as the chassis saturates. Software cannot undo that.

So Cortex treats thermal headroom as a schedulable resource. A governor reads power and thermal state and adapts:

| State | When | Background indexing |
|---|---|---|
| `boost` | cool, on mains, idle | full concurrency |
| `nominal` | normal | standard |
| `throttled` | hot, or on battery | serialised |
| `critical` | heavily throttled, or battery < 30% | suspended |

**Interactive queries are never throttled.** Only backfill yields. A second brain that is slow to answer is a second brain nobody opens.

### 3. Some free model APIs train on what you send them

This is the one that matters most for a vault of private notes. Several free tiers — OpenCode Zen, Google AI Studio, Mistral's Experiment tier — use submitted data for model improvement by default. A naive fallback chain will eventually post a private journal entry into someone's training corpus, silently and irreversibly.

No general-purpose router models this. LiteLLM will happily fail over from a no-training provider to a training one, because to it they are interchangeable OpenAI-compatible endpoints.

Cortex refuses:

```
content sensitivity  ×  provider data policy  →  allowed / refused
```

Every note is **private by default**. Making one shareable takes an explicit `sensitivity: public` in frontmatter. If no eligible provider is available, Cortex raises an error rather than quietly downgrading — and one private chunk in the context makes the whole request private, because a summary of private notes is itself private.

```console
$ cortex providers
                Provider routing for PRIVATE content
┌────────────┬─────────────────────────────────┬─────────────────┬───────────────────────┐
│ Provider   │ Model                           │ Policy          │ Status                │
├────────────┼─────────────────────────────────┼─────────────────┼───────────────────────┤
│ ollama     │ qwen3:4b                        │ local           │ eligible (local)      │
│ groq       │ llama-3.3-70b-versatile         │ no_train        │ eligible (no_train)   │
│ nvidia     │ meta/llama-3.3-70b-instruct     │ no_train        │ eligible (no_train)   │
│ openrouter │ ...llama-3.3-70b-instruct:free  │ no_train_if_zdr │ refused: zero-data-   │
│            │                                 │                 │ retention not enabled │
└────────────┴─────────────────────────────────┴─────────────────┴───────────────────────┘
```

---

## How it works

```
 Obsidian vault
      │
      ▼
 watcher ──► hash gate ──► parse ──► chunk ──► embed ──► LanceDB
             (skip         (frontmatter,  (heading-      │      + SQLite catalog
              unchanged)    wikilinks,     first)        │        (hashes, lineage)
                            tags)                        │
                        ┌─── thermal governor gates concurrency ───┘

 query ──► embed ──┬─► dense search ─┐
                   ├─► BM25 search   ├──► RRF fusion ──► rerank ──► synthesis ──► cited answer
                   └─► graph expand ─┘                                 │
                                                             privacy gate decides
                                                             local vs. cloud
```

**Retrieval is always local.** Embedding and reranking never touch the network. Only synthesis can escalate, and then only the handful of chunks needed to answer — never the vault.

---

## Install

Requires Python 3.11+ and [Ollama](https://ollama.com) for local models.

```bash
git clone https://github.com/jeevesh2515/cortex.git
cd cortex
./scripts/bootstrap.sh
```

The bootstrap script creates a virtualenv, installs Cortex, pulls the embedding model, and writes a starter config. To do it by hand:

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[all]"
ollama pull qwen3-embedding:0.6b     # ~1.5 GB, the embedder
ollama pull qwen3:4b                 # ~3 GB, local synthesis
```

### Model choices for 16 GB

Sized so no two large models are resident at once, leaving headroom for everything else:

| Role | Model | Resident | Why |
|---|---|---|---|
| Embeddings | `qwen3-embedding:0.6b` | ~1.5 GB | 70.7 MTEB(eng,v2), 32K context, Apache-2.0. Best quality-per-GB in 2026; the 8B scores higher but wants 17 GB to itself. |
| Synthesis | `qwen3:4b` | ~3 GB | Fits comfortably; escalate to a free cloud tier for hard questions. |
| Reranking | `bge-reranker-v2-m3` | ~1.2 GB | 84 ms median. Qwen3-Reranker-4B scores better but needs 6.8 GB. |

Peak resident is about 11 GB including macOS. On 32 GB, raise these in `cortex.toml`.

---

## Use

```bash
cortex index                          # incremental; unchanged notes are skipped
cortex ask "what did I conclude about chunking?"
cortex search "hybrid retrieval"      # raw passages, no synthesis
cortex status                         # index size, thermal state
cortex providers                      # who may see private content, and why
cortex graph --note "Retrieval.md"    # inspect wikilinks
cortex watch                          # run continuously
```

Everything works without a model configured — pass `--offline` to use a deterministic hashing embedder. Useful for trying the pipeline before pulling gigabytes.

### Connect it to Antigravity

Cortex speaks MCP, so Antigravity can query your notes directly. Add to `~/.gemini/antigravity-cli/mcp_config.json`:

```json
{
  "mcpServers": {
    "cortex": {
      "command": "/absolute/path/to/cortex/.venv/bin/python",
      "args": ["-m", "cortex.cli", "serve-mcp"]
    }
  }
}
```

Use the venv's interpreter directly — MCP clients launch the process without a login shell, so anything relying on `PATH` will not be found.

Five tools become available: `search_notes`, `ask_notes`, `list_links`, `vault_status`, `reindex`. All read-only except `reindex`; nothing can rewrite a note.

### Run it in the background

```bash
./scripts/install-daemon.sh
```

Installs a LaunchDaemon that indexes continuously. It is a *Daemon* rather than an Agent deliberately: since macOS 26 Tahoe, Homebrew Python running as a LaunchAgent is subject to TCC checks and fails local network calls with `errno 65`. A LaunchDaemon with a `UserName` key avoids this without prompting for an admin password at runtime.

---

## Configure

`~/.config/cortex/cortex.toml`:

```toml
vault_path = "~/Documents/MyVault"

[retrieval]
top_k = 8
graph_hops = 1
fusion_weights = { dense = 1.0, fts = 0.8, graph = 0.5 }

[thermal]
boost_workers = 4
throttled_workers = 1
min_battery_for_backfill = 30

# Patch a shipped provider without restating it
[[providers]]
name = "openrouter"
zdr_enabled = true    # only after enabling ZDR in your OpenRouter account
```

API keys come from the environment, never the config file:

```bash
export GROQ_API_KEY=...
export NVIDIA_API_KEY=...
export OPENROUTER_API_KEY=...
```

Set `local_only = true` to hard-disable all network egress regardless of policy.

---

## Development

```bash
uv pip install -e ".[dev,all]"
make check      # ruff + mypy --strict + pytest
```

247 tests, mypy strict, zero lint warnings. The privacy gate is tested as a security boundary — including the subtle leak where a preferred provider fails and a naive chain falls through to a training one.

Architecture decisions and their tradeoffs are recorded in [`docs/adr/`](docs/adr/).

---

## License

MIT
