"""CLI and MCP tool tests.

The MCP tool layer is tested through ``CortexTools`` directly rather than over
the wire, so the behaviour is covered even where the optional ``mcp`` package
is absent. The privacy guarantee is re-asserted at this boundary because it is
the surface an autonomous agent actually touches.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from cortex.cli import app
from cortex.config import Settings, load_settings
from cortex.mcp_server import TOOL_DEFINITIONS, CortexTools
from cortex.models import DataPolicy
from cortex.runtime import build_runtime

runner = CliRunner()

VAULT = {
    "Retrieval.md": (
        "---\ntags: [rag]\n---\n\n# Retrieval\n\n"
        "Hybrid search fuses dense vectors with bm25 keyword matching.\n"
        "See [[Embeddings]].\n"
    ),
    "Embeddings.md": "# Embeddings\n\nDense vectors encode semantic similarity.\n",
    "Sourdough.md": "# Sourdough\n\nFeed the starter and bulk ferment the dough.\n",
}


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    root = tmp_path / "vault"
    root.mkdir()
    for name, body in VAULT.items():
        (root / name).write_text(body, encoding="utf-8")
    return root


@pytest.fixture
def settings(vault: Path, tmp_path: Path) -> Settings:
    s = Settings()
    s.vault_path = vault
    s.data_dir = tmp_path / "data"
    s.embed_dimensions = 128
    return s


@pytest.fixture
def tools(settings: Settings) -> CortexTools:
    rt = build_runtime(settings=settings, offline=True, in_memory=True)
    rt.pipeline.run()
    return CortexTools(rt)


class TestToolDefinitions:
    def test_every_tool_has_a_handler(self, tools: CortexTools) -> None:
        for spec in TOOL_DEFINITIONS:
            result = tools.dispatch(spec["name"], {})  # type: ignore[index]
            assert result.get("error") != "unknown_tool"

    def test_schemas_are_wellformed(self) -> None:
        for spec in TOOL_DEFINITIONS:
            assert spec["name"]
            assert len(str(spec["description"])) > 40, "agents need real descriptions"
            schema = spec["input_schema"]
            assert schema["type"] == "object"  # type: ignore[index]
            for prop in schema.get("properties", {}).values():  # type: ignore[union-attr]
                assert "type" in prop

    def test_unknown_tool_is_reported(self, tools: CortexTools) -> None:
        assert tools.dispatch("nope", {})["error"] == "unknown_tool"

    def test_bad_arguments_are_reported(self, tools: CortexTools) -> None:
        assert tools.dispatch("search_notes", {"wrong": 1})["error"] == "bad_arguments"


class TestSearchTool:
    def test_returns_results(self, tools: CortexTools) -> None:
        out = tools.search_notes("bulk ferment starter")
        assert out["results"]
        assert out["results"][0]["note_id"] == "Sourdough.md"

    def test_includes_provenance(self, tools: CortexTools) -> None:
        out = tools.search_notes("hybrid search bm25")
        top = out["results"][0]
        assert "source" in top
        assert top["found_via"]

    def test_top_k_respected(self, tools: CortexTools) -> None:
        assert len(tools.search_notes("dense vectors", top_k=1)["results"]) <= 1


class TestLinksTool:
    def test_forward_and_back(self, tools: CortexTools) -> None:
        out = tools.list_links("Retrieval.md")
        assert "Embeddings.md" in out["links_to"]
        back = tools.list_links("Embeddings.md")
        assert "Retrieval.md" in back["linked_from"]

    def test_unknown_note_is_empty_not_an_error(self, tools: CortexTools) -> None:
        out = tools.list_links("Nope.md")
        assert out["links_to"] == []


class TestStatusTool:
    def test_reports_counts_and_thermal(self, tools: CortexTools) -> None:
        out = tools.vault_status()
        assert out["notes_indexed"] == 3
        assert "state" in out["thermal"]
        assert out["provider_routing_private"] is not None


class TestReindexTool:
    def test_incremental_skips_unchanged(self, tools: CortexTools) -> None:
        out = tools.reindex()
        assert out["indexed"] == 0
        assert out["skipped"] == 3

    def test_detects_new_note(self, tools: CortexTools, vault: Path) -> None:
        (vault / "New.md").write_text("# New\n\nFresh content.\n", encoding="utf-8")
        assert tools.reindex()["indexed"] == 1


class TestAskToolPrivacy:
    def test_refuses_when_only_a_training_provider_exists(self, settings: Settings) -> None:
        from cortex.llm.protocol import ProviderSpec

        settings.providers = [
            ProviderSpec(
                name="zen",
                base_url="https://example.invalid/v1",
                model="free-model",
                policy=DataPolicy.TRAINS,
            )
        ]
        rt = build_runtime(settings=settings, offline=True, in_memory=True)
        rt.pipeline.run()
        out = CortexTools(rt).ask_notes("what did I write about sourdough?")
        assert out["error"] == "privacy_policy"
        assert "hint" in out

    def test_reports_provider_failure_distinctly(self, settings: Settings) -> None:
        from cortex.llm.protocol import ProviderSpec

        settings.providers = [
            ProviderSpec(
                name="ollama",
                base_url="http://127.0.0.1:1/v1",  # guaranteed refused
                model="m",
                policy=DataPolicy.LOCAL,
            )
        ]
        rt = build_runtime(settings=settings, offline=True, in_memory=True)
        rt.pipeline.run()
        out = CortexTools(rt).ask_notes("sourdough")
        assert out["error"] == "provider_unavailable"


class TestCLI:
    def test_version(self) -> None:
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "cortex" in result.stdout

    def test_help_lists_commands(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        for command in ("index", "ask", "search", "status", "providers", "graph"):
            assert command in result.stdout

    def test_missing_vault_exits_cleanly(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["status", "--vault", str(tmp_path / "nope"), "--offline"])
        assert result.exit_code == 2

    def test_index_then_search(self, vault: Path, tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        monkeypatch.setenv("CORTEX_DATA_DIR", str(tmp_path / "data"))
        indexed = runner.invoke(app, ["index", "--vault", str(vault), "--offline"])
        assert indexed.exit_code == 0, indexed.stdout
        assert "indexed" in indexed.stdout

        found = runner.invoke(app, ["search", "bulk ferment", "--vault", str(vault), "--offline"])
        assert found.exit_code == 0
        assert "Sourdough" in found.stdout

    def test_status(self, vault: Path, tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        monkeypatch.setenv("CORTEX_DATA_DIR", str(tmp_path / "data"))
        result = runner.invoke(app, ["status", "--vault", str(vault), "--offline"])
        assert result.exit_code == 0
        assert "Thermal" in result.stdout

    def test_providers_shows_policy(self, vault: Path, tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        monkeypatch.setenv("CORTEX_DATA_DIR", str(tmp_path / "data"))
        result = runner.invoke(app, ["providers", "--vault", str(vault), "--offline"])
        assert result.exit_code == 0
        assert "ollama" in result.stdout

    def test_graph_stats(self, vault: Path, tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        monkeypatch.setenv("CORTEX_DATA_DIR", str(tmp_path / "data"))
        result = runner.invoke(app, ["graph", "--vault", str(vault), "--offline"])
        assert result.exit_code == 0
        assert "Edges" in result.stdout


class TestSettingsLoading:
    def test_toml_roundtrip(self, tmp_path: Path) -> None:
        config = tmp_path / "cortex.toml"
        config.write_text(
            "vault_path = '/tmp/myvault'\n"
            "local_only = true\n\n"
            "[retrieval]\n"
            "top_k = 15\n"
            "graph_enabled = false\n\n"
            "[chunking]\n"
            "target_tokens = 256\n",
            encoding="utf-8",
        )
        loaded = load_settings(config, env={})
        assert loaded.vault_path == Path("/tmp/myvault")
        assert loaded.local_only is True
        assert loaded.top_k == 15
        assert loaded.graph_enabled is False
        assert loaded.chunk.target_tokens == 256

    def test_provider_patch_merges_over_defaults(self, tmp_path: Path) -> None:
        # The common case: flip ZDR on OpenRouter without restating the spec.
        config = tmp_path / "cortex.toml"
        config.write_text(
            "[[providers]]\nname = 'openrouter'\nzdr_enabled = true\n", encoding="utf-8"
        )
        loaded = load_settings(config, env={})
        openrouter = loaded.provider("openrouter")
        assert openrouter is not None
        assert openrouter.zdr_enabled is True
        assert openrouter.base_url.startswith("https://openrouter.ai")
        assert loaded.provider("groq") is not None, "other defaults must survive"

    def test_env_overrides_file(self, tmp_path: Path) -> None:
        config = tmp_path / "cortex.toml"
        config.write_text("vault_path = '/tmp/fromfile'\n", encoding="utf-8")
        loaded = load_settings(config, env={"CORTEX_VAULT": "/tmp/fromenv"})
        assert loaded.vault_path == Path("/tmp/fromenv")

    def test_defaults_when_no_file(self, tmp_path: Path) -> None:
        loaded = load_settings(tmp_path / "absent.toml", env={})
        assert loaded.top_k == 8
        assert [p.name for p in loaded.providers][:2] == ["ollama", "groq"]

    def test_shipped_chain_is_privacy_safe_by_default(self) -> None:
        # Regression guard: no shipped provider may accept private content
        # unless it is local or contractually no-training.
        from cortex.config import default_providers
        from cortex.models import Sensitivity

        for spec in default_providers():
            if spec.accepts(Sensitivity.PRIVATE):
                assert spec.policy in (DataPolicy.LOCAL, DataPolicy.NO_TRAIN), spec.name
