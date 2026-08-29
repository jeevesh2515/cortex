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

## ⚡ Quickstart (Zero-Setup Demo)

Try Cortex immediately with the included sample knowledge vault:

```bash
# 1. Clone and enter repo
git clone https://github.com/jeevesh2515/cortex.git
cd cortex

# 2. Install editable package
pip install -e .

# 3. Query sample vault using offline deterministic embedder
cortex query "What are the notes on pectin and fermentation?" --offline
```

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
cortex serve                          # local web app + voice + chat UI
```

Everything works without a model configured — pass `--offline` to use a deterministic hashing embedder. Useful for trying the pipeline before pulling gigabytes.

### Talk to it in the browser — `cortex serve`

A Vite-built React app runs alongside the Python backend on one port. Streaming
chat, citations, voice input (browser Web Speech API), TTS read-aloud, a vault
search sidebar and a memory-folder browser — all in one window, all local.

```bash
uv pip install -e ".[server]"            # adds fastapi + uvicorn
cd frontend && npm install && npm run build  # one-time, builds the SPA
cd .. && cortex serve                    # opens http://127.0.0.1:7331
```

The mobile layout collapses the left rail into a slide-in drawer and pins a
tab bar to the bottom of the viewport; on desktop the sidebar stays put.

**Voice.** Hold the mic button (or type) and the browser's Web Speech API
transcribes in real time. A "Read aloud" button on each assistant message
plays it back via the browser's `SpeechSynthesis` with the voice you pick in
Settings. Both rely on browser-native engines; for fully local STT/TTS, route
audio to a local Whisper / Piper daemon on the server and we'll hook them up.

**Privacy, in the UI.** Every assistant message carries a provider chip
(`groq · no-train`, `nvidia · no-train`, `openrouter · zdr`, etc.) so you can
see at a glance where each answer came from. The "Local only" toggle on the
composer forces the request to stay on-device.

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

### Ask about time

Daily notes are only useful if you can query them by date, and "last week"
carries almost no semantic signal — so Cortex resolves the window and filters on
it instead of hoping the embedder understands calendars:

```bash
cortex ask "what did I work on last week?"
cortex search "meetings in March"
cortex ask "what happened on 2026-07-14?"
```

Dates come from frontmatter first, then a daily-note filename, then mtime.
Crucially, a dated query returns **full coverage of the window** rather than the
top 8 — a truncated answer to "what did I do last week" is confidently
incomplete, which is worse than a slow one. The CLI shows how your phrasing was
interpreted, so there's no guessing.

A bare month name does *not* trigger a filter: "March release planning" is a
note about March, not a date query. Only "in March" is.

### Clickable citations

Every citation is an `obsidian://` link, so a source is one click away rather
than a path to go hunting for:

```
Sources:
  [1] daily/2026-07-14.md > Standup      ← opens in Obsidian
```

### It remembers

Ask with `--remember` and Cortex writes the exchange as a Markdown note in
`Memory/`, wikilinked to the notes it drew on:

```bash
cortex ask --remember "what chunk size did I settle on?"
cortex memory                     # list what it has kept
```

Because it's an ordinary note, it shows up in Obsidian's graph view, you can
edit or delete it, and it participates in future retrieval automatically — no
second database, nothing hidden.

There's a trap here worth knowing about. Memory notes *summarise* your primary
notes, so they compete with their own sources. Left uncapped they slowly
displace the real content and answers drift towards summarising previous
answers — more confident each round, less grounded. `memory_max_results`
(default 2) caps how much memory can occupy any result set.

### PDFs and web clippings

```toml
ingest_documents = true   # needs: pip install 'cortex-brain[documents]'
```

PDFs, EPUBs, saved HTML and `.txt` become markdown and then follow the identical
path as a hand-written note — same chunker, same citations. Neither PDF backend
does OCR, so a scanned PDF says so explicitly rather than quietly indexing
nothing.

### Measure instead of guessing

Every default here came from published findings on someone else's corpus.
Whether the graph or the reranker earns its latency on *your* vault is an
empirical question:

```bash
cortex bench --cases my-questions.yaml --reindex
```

```yaml
# my-questions.yaml — ~20 questions you actually asked
- query: what did I decide about chunking?
  expect: [Chunking Strategy.md]
```

You get recall@5/10, MRR, MAP, nDCG@10 and p50/p95 latency — then an **ablation
table** showing what each component contributes. A negative delta means that
component is hurting you and should be off. The project should make it easy to
reach that conclusion about its own features.

### Reranking

Reranking is the single highest-leverage addition to hybrid retrieval — roughly
+18% recall@5. It needs a cross-encoder, which pulls torch, so it's opt-in:

```bash
uv pip install -e ".[rerank]"
```

Weights load lazily on first query and unload after 10 minutes idle, because the
embedder, reranker and chat model don't co-reside comfortably in 16GB. If the
extra isn't installed, `cortex status` says so plainly rather than pretending.

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

API keys come from the environment, never the config file. A project `.env` is
picked up automatically by walking up from the working directory:

```bash
# .env  (gitignored)
OPENROUTER_API_KEY=sk-or-...
GROQ_API_KEY=gsk_...
NVIDIA_API_KEY=nvapi-...
```

Anything already exported in your shell **wins over the file**, so a stale
`.env` can never shadow a key you set deliberately.

Set `local_only = true` to hard-disable all network egress regardless of policy.

---

## Development

```bash
uv pip install -e ".[dev,all]"
make check      # ruff + mypy --strict + pytest
```

The privacy gate is tested as a security boundary — including the subtle leak where a preferred provider fails and a naive chain falls through to a training one. The HTTP surface (`src/cortex/server/`) is covered by `tests/test_server.py` using httpx against the ASGI app directly, so the chat SSE event sequence is asserted in CI.

Architecture decisions and their tradeoffs are recorded in [`docs/adr/`](docs/adr/).

---

## License

MIT
