"""Configuration.

Layered, in increasing precedence: shipped defaults, ``cortex.toml``,
``CORTEX_*`` environment variables. Secrets are never part of this -- config
stores the *name* of the environment variable holding a key, never the key.

The shipped provider chain reflects an audit of each operator's published data
policy as of August 2026, recorded in ``docs/adr/0004-privacy-tiered-routing.md``.
Providers that train on submitted data are present but disabled, so that
enabling one is a deliberate act with a visible policy label attached.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from cortex.env import load_dotenv
from cortex.ingest.chunker import ChunkConfig
from cortex.llm.protocol import ProviderSpec
from cortex.models import DataPolicy
from cortex.thermal.governor import GovernorConfig

__all__ = ["DEFAULT_CONFIG_PATH", "Settings", "default_providers", "load_settings"]

DEFAULT_CONFIG_PATH = Path.home() / ".config" / "cortex" / "cortex.toml"


def default_providers() -> list[ProviderSpec]:
    """The shipped fallback chain.

    Ordering is local-first, then by free-tier generosity. Groq leads the remote
    tier because 14,400 requests/day on llama-3.1-8b is the most generous free
    allowance available and it does not train on submitted data.
    """
    return [
        ProviderSpec(
            name="ollama",
            base_url="http://localhost:11434/v1",
            model="qwen3:4b",
            policy=DataPolicy.LOCAL,
            max_context=32_768,
            priority=0,
        ),
        ProviderSpec(
            name="groq",
            base_url="https://api.groq.com/openai/v1",
            model="llama-3.3-70b-versatile",
            policy=DataPolicy.NO_TRAIN,
            api_key_env="GROQ_API_KEY",
            max_context=128_000,
            rpm=30,
            rpd=1_000,
            priority=10,
        ),
        ProviderSpec(
            name="nvidia",
            base_url="https://integrate.api.nvidia.com/v1",
            model="meta/llama-3.3-70b-instruct",
            policy=DataPolicy.NO_TRAIN,
            api_key_env="NVIDIA_API_KEY",
            max_context=128_000,
            rpm=40,
            priority=20,
        ),
        ProviderSpec(
            name="openrouter",
            base_url="https://openrouter.ai/api/v1",
            # The free roster rotates weekly, so pin the auto-router rather
            # than a slug that will 404 in a fortnight. openrouter/free selects
            # from whatever free models are currently available.
            model="openrouter/free",
            policy=DataPolicy.NO_TRAIN_IF_ZDR,
            api_key_env="OPENROUTER_API_KEY",
            max_context=128_000,
            # 20 RPM / 50 RPD on a free account; 1,000 RPD once any credit has
            # ever been purchased. The conservative figure is the default.
            rpm=20,
            rpd=50,
            requires_zdr=True,
            zdr_enabled=False,
            priority=30,
            extra_headers={
                "HTTP-Referer": "https://github.com/jeevesh2515/cortex",
                "X-Title": "Cortex",
            },
        ),
    ]


@dataclass(slots=True)
class Settings:
    """Resolved runtime configuration."""

    vault_path: Path = field(default_factory=lambda: Path.home() / "Obsidian")
    vault_name: str = ""
    """Obsidian vault name, for obsidian:// citation links. Defaults to the
    vault directory's own name, which is what Obsidian uses unless the vault was
    explicitly renamed in the app."""

    data_dir: Path = field(default_factory=lambda: Path.home() / ".local" / "share" / "cortex")

    embed_model: str = "qwen3-embedding:0.6b"
    embed_dimensions: int = 1024
    ollama_url: str = "http://localhost:11434"

    rerank_model: str = "bge-reranker-v2-m3"
    rerank_enabled: bool = True
    rerank_candidates: int = 30
    rerank_idle_ttl: float = 600.0
    """Seconds of inactivity before the reranker releases its weights. Matters
    on 16 GB: embedder + reranker + chat model do not co-reside comfortably."""

    expansion_enabled: bool = True
    """Extract salient terms and run extra lexical variants. A question is
    mostly stopwords, which dilutes BM25."""

    ingest_documents: bool = False
    """Also index PDFs, HTML clippings, EPUBs and text files. Off by default:
    extraction needs an optional dependency and a vault of attachments can be
    far larger than its markdown."""

    temporal_enabled: bool = True
    """Resolve date expressions in queries and return full coverage of the
    window rather than the top k."""

    memory_enabled: bool = True
    memory_folder: str = "Memory"
    memory_auto: bool = False
    """Write a memory note after every ``ask``. Off by default: automatic
    capture on every question would fill the vault with noise, so it is opt-in
    per query via ``--remember`` until you decide you want all of them."""

    memory_max_results: int = 2
    """Ceiling on memory notes per result set. Memory summarises primary notes,
    so uncapped it would gradually displace the sources it came from."""

    top_k: int = 8
    dense_k: int = 30
    fts_k: int = 30
    graph_hops: int = 1
    graph_enabled: bool = True
    fusion_weights: dict[str, float] = field(
        default_factory=lambda: {"dense": 1.0, "fts": 0.8, "graph": 0.5}
    )

    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    governor: GovernorConfig = field(default_factory=GovernorConfig)
    providers: list[ProviderSpec] = field(default_factory=default_providers)

    local_only: bool = False
    """Hard switch. When true nothing leaves the machine, regardless of policy."""

    exclude_globs: list[str] = field(
        default_factory=lambda: [
            ".obsidian/**",
            ".trash/**",
            ".git/**",
            "**/node_modules/**",
            "**/.DS_Store",
        ]
    )

    @property
    def display_vault_name(self) -> str:
        return self.vault_name or self.vault_path.name

    @property
    def db_path(self) -> Path:
        return self.data_dir / "catalog.db"

    @property
    def index_path(self) -> Path:
        return self.data_dir / "index"

    @property
    def log_path(self) -> Path:
        return self.data_dir / "logs"

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.index_path, self.log_path):
            path.mkdir(parents=True, exist_ok=True)

    def provider(self, name: str) -> ProviderSpec | None:
        return next((p for p in self.providers if p.name == name), None)


def _coerce_path(value: Any) -> Path:
    return Path(str(value)).expanduser()


def _providers_from_toml(raw: Any, defaults: list[ProviderSpec]) -> list[ProviderSpec]:
    """Merge a ``[[providers]]`` array over the shipped defaults.

    A table naming an existing provider patches it; a new name appends. This
    means a user can flip ``zdr_enabled`` on OpenRouter without restating the
    whole spec, which is the common case.
    """
    if not isinstance(raw, list):
        return defaults

    by_name = {spec.name: spec for spec in defaults}
    order = [spec.name for spec in defaults]

    for entry in raw:
        if not isinstance(entry, dict) or "name" not in entry:
            continue
        name = str(entry["name"])
        patch: dict[str, Any] = {}
        for key, value in entry.items():
            if key == "name":
                continue
            if key == "policy":
                try:
                    patch["policy"] = DataPolicy(str(value))
                except ValueError:
                    continue
            else:
                patch[key] = value

        if name in by_name:
            valid = {k: v for k, v in patch.items() if k in ProviderSpec.__dataclass_fields__}
            by_name[name] = replace(by_name[name], **valid)
        else:
            required = {"base_url", "model"}
            if not required <= set(patch):
                continue
            patch.setdefault("policy", DataPolicy.TRAINS)
            valid = {k: v for k, v in patch.items() if k in ProviderSpec.__dataclass_fields__}
            by_name[name] = ProviderSpec(name=name, **valid)
            order.append(name)

    return [by_name[name] for name in order]


def load_settings(
    config_path: Path | None = None, *, env: dict[str, str] | None = None
) -> Settings:
    """Load settings from file and environment.

    When ``env`` is omitted -- the live path, as opposed to a test passing an
    explicit mapping -- a ``.env`` file is discovered by walking up from the
    working directory and merged into the process environment first. Keys
    already exported in the shell take precedence over the file.
    """
    if env is None:
        # Only on the live path: an explicit `env` mapping means a test is
        # controlling the environment and must not have a stray .env leak in.
        load_dotenv()
    env = dict(os.environ if env is None else env)
    settings = Settings()

    path = config_path or Path(env.get("CORTEX_CONFIG", DEFAULT_CONFIG_PATH))
    if path.exists():
        with path.open("rb") as handle:
            data = tomllib.load(handle)

        if "vault_path" in data:
            settings.vault_path = _coerce_path(data["vault_path"])
        if "vault_name" in data:
            settings.vault_name = str(data["vault_name"])
        if "data_dir" in data:
            settings.data_dir = _coerce_path(data["data_dir"])
        if "local_only" in data:
            settings.local_only = bool(data["local_only"])
        if "ingest_documents" in data:
            settings.ingest_documents = bool(data["ingest_documents"])
        if isinstance(data.get("exclude_globs"), list):
            settings.exclude_globs = [str(g) for g in data["exclude_globs"]]

        embed = data.get("embedding", {})
        if isinstance(embed, dict):
            settings.embed_model = str(embed.get("model", settings.embed_model))
            settings.embed_dimensions = int(embed.get("dimensions", settings.embed_dimensions))
            settings.ollama_url = str(embed.get("ollama_url", settings.ollama_url))

        retrieval = data.get("retrieval", {})
        if isinstance(retrieval, dict):
            settings.top_k = int(retrieval.get("top_k", settings.top_k))
            settings.dense_k = int(retrieval.get("dense_k", settings.dense_k))
            settings.fts_k = int(retrieval.get("fts_k", settings.fts_k))
            settings.graph_hops = int(retrieval.get("graph_hops", settings.graph_hops))
            settings.graph_enabled = bool(retrieval.get("graph_enabled", settings.graph_enabled))
            settings.rerank_enabled = bool(retrieval.get("rerank_enabled", settings.rerank_enabled))
            weights = retrieval.get("fusion_weights")
            if isinstance(weights, dict):
                settings.fusion_weights = {str(k): float(v) for k, v in weights.items()}

        mem = data.get("memory", {})
        if isinstance(mem, dict):
            settings.memory_enabled = bool(mem.get("enabled", settings.memory_enabled))
            settings.memory_folder = str(mem.get("folder", settings.memory_folder))
            settings.memory_auto = bool(mem.get("auto", settings.memory_auto))

        chunking = data.get("chunking", {})
        if isinstance(chunking, dict):
            settings.chunk = ChunkConfig(
                target_tokens=int(chunking.get("target_tokens", 512)),
                overlap_ratio=float(chunking.get("overlap_ratio", 0.10)),
                min_tokens=int(chunking.get("min_tokens", 64)),
                max_tokens=int(chunking.get("max_tokens", 900)),
            )

        thermal = data.get("thermal", {})
        if isinstance(thermal, dict):
            base = GovernorConfig()
            settings.governor = GovernorConfig(
                boost_workers=int(thermal.get("boost_workers", base.boost_workers)),
                nominal_workers=int(thermal.get("nominal_workers", base.nominal_workers)),
                throttled_workers=int(thermal.get("throttled_workers", base.throttled_workers)),
                critical_workers=int(thermal.get("critical_workers", base.critical_workers)),
                throttle_speed_limit=int(
                    thermal.get("throttle_speed_limit", base.throttle_speed_limit)
                ),
                critical_speed_limit=int(
                    thermal.get("critical_speed_limit", base.critical_speed_limit)
                ),
                min_battery_for_backfill=int(
                    thermal.get("min_battery_for_backfill", base.min_battery_for_backfill)
                ),
                sample_interval=float(thermal.get("sample_interval", base.sample_interval)),
            )

        settings.providers = _providers_from_toml(data.get("providers"), default_providers())

    # Environment overrides win.
    if "CORTEX_VAULT" in env:
        settings.vault_path = _coerce_path(env["CORTEX_VAULT"])
    if "CORTEX_DATA_DIR" in env:
        settings.data_dir = _coerce_path(env["CORTEX_DATA_DIR"])
    if "CORTEX_LOCAL_ONLY" in env:
        settings.local_only = env["CORTEX_LOCAL_ONLY"].strip().lower() in {"1", "true", "yes"}
    if "OLLAMA_HOST" in env:
        settings.ollama_url = env["OLLAMA_HOST"]

    return settings
