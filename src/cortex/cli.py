"""Command-line interface.

cortex index          incremental index of the vault
cortex ask "..."      grounded answer with citations
cortex search "..."   raw retrieval, no synthesis
cortex status         index, thermal and routing state
cortex providers      which providers may see private content, and why
cortex graph          wikilink graph statistics
cortex watch          run the incremental indexer continuously
cortex serve-mcp      MCP stdio server for Antigravity
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from cortex import __version__
from cortex.llm.protocol import PolicyViolation, ProviderError
from cortex.models import Sensitivity, obsidian_uri
from cortex.runtime import build_runtime

app = typer.Typer(
    name="cortex",
    help="Local-first second brain for Obsidian.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()
err_console = Console(stderr=True)

ConfigOpt = Annotated[Path | None, typer.Option("--config", "-c", help="Path to cortex.toml.")]
VaultOpt = Annotated[Path | None, typer.Option("--vault", "-v", help="Override the vault path.")]
OfflineOpt = Annotated[
    bool,
    typer.Option("--offline", help="Use the deterministic hashing embedder (no model needed)."),
]


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def _runtime(config: Path | None, vault: Path | None, offline: bool):  # type: ignore[no-untyped-def]
    from cortex.config import load_settings

    settings = load_settings(config)
    if vault is not None:
        settings.vault_path = vault.expanduser()
    if not settings.vault_path.exists():
        err_console.print(
            f"[red]Vault not found:[/red] {settings.vault_path}\n"
            "Set it with --vault, or in cortex.toml, or via CORTEX_VAULT."
        )
        raise typer.Exit(code=2)
    return build_runtime(settings=settings, offline=offline)


@app.command()
def version() -> None:
    """Print the version."""
    console.print(f"cortex {__version__}")


@app.command()
def index(
    config: ConfigOpt = None,
    vault: VaultOpt = None,
    offline: OfflineOpt = False,
    full: Annotated[bool, typer.Option("--full", help="Rebuild from scratch.")] = False,
    ignore_thermal: Annotated[
        bool, typer.Option("--ignore-thermal", help="Index at full speed regardless of heat.")
    ] = False,
    verbose: Annotated[bool, typer.Option("--verbose")] = False,
) -> None:
    """Index the vault incrementally."""
    _setup_logging(verbose)
    with _runtime(config, vault, offline) as rt:
        console.print(f"Vault: [cyan]{rt.settings.vault_path}[/cyan]")
        rt.governor.sample(force=True)
        console.print(f"Thermal: [yellow]{rt.governor.state.value}[/yellow]")

        with console.status("Indexing..."):
            report = rt.pipeline.run(full=full, respect_thermal=not ignore_thermal)

        console.print(f"[green]{report.summary()}[/green]")
        if report.paused_for_thermal:
            console.print(
                "[yellow]Backfill paused to let the machine cool. "
                "Re-run later, or pass --ignore-thermal.[/yellow]"
            )
        for note_id, error in report.errors[:10]:
            err_console.print(f"[red]error[/red] {note_id}: {error}")

        graph = rt.refresh_graph()
        stats = graph.stats
        console.print(f"Link graph: {stats['edges']} edges across {stats['linked_notes']} notes")


@app.command()
def search(
    query: Annotated[str, typer.Argument(help="What to look for.")],
    config: ConfigOpt = None,
    vault: VaultOpt = None,
    offline: OfflineOpt = False,
    top_k: Annotated[int, typer.Option("--top-k", "-k")] = 8,
    no_graph: Annotated[bool, typer.Option("--no-graph")] = False,
    verbose: Annotated[bool, typer.Option("--verbose")] = False,
) -> None:
    """Retrieve matching notes without synthesising an answer."""
    _setup_logging(verbose)
    with _runtime(config, vault, offline) as rt:
        rt.refresh_graph()
        result = rt.engine().retrieve(query, top_k=top_k, use_graph=not no_graph)

        if not result.chunks:
            console.print("[yellow]No matches.[/yellow]")
            raise typer.Exit()

        meta = ", ".join(f"{k}:{v}" for k, v in result.per_retriever.items())
        console.print(f"[dim]{result.elapsed_ms:.0f}ms - {meta}[/dim]")
        if result.date_range is not None:
            mode = "full coverage" if not result.truncated else "no dated matches"
            console.print(
                f"[cyan]date filter:[/cyan] {result.date_range} "
                f"[dim]({result.date_range.days}d, {mode})[/dim]"
            )
        console.print()
        for i, scored in enumerate(result.chunks, start=1):
            snippet = scored.chunk.text.strip().replace("\n", " ")
            if len(snippet) > 240:
                snippet = snippet[:240] + "..."
            via = "+".join(sorted(scored.components)) or scored.source
            console.print(
                Panel(
                    snippet,
                    title=f"[{i}] {scored.chunk.citation}",
                    subtitle=f"[dim]{via} - {scored.score:.4f}[/dim]",
                    title_align="left",
                    subtitle_align="right",
                )
            )


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="Your question.")],
    config: ConfigOpt = None,
    vault: VaultOpt = None,
    offline: OfflineOpt = False,
    top_k: Annotated[int, typer.Option("--top-k", "-k")] = 8,
    local_only: Annotated[
        bool, typer.Option("--local-only", help="Never leave this machine.")
    ] = False,
    remember: Annotated[
        bool, typer.Option("--remember", help="Save this exchange as a note in your vault.")
    ] = False,
    verbose: Annotated[bool, typer.Option("--verbose")] = False,
) -> None:
    """Answer a question from your notes, with citations."""
    _setup_logging(verbose)
    with _runtime(config, vault, offline) as rt:
        rt.refresh_graph()
        engine = rt.engine()
        try:
            with console.status("Thinking..."):
                answer = engine.ask(
                    question,
                    top_k=top_k,
                    local_only=local_only or rt.settings.local_only,
                )
        except PolicyViolation as exc:
            err_console.print(
                Panel(
                    f"{exc}\n\n[dim]Cortex refused rather than sending private notes to a "
                    "provider that trains on submitted data. Run 'cortex providers' to see "
                    "the policy for each configured provider.[/dim]",
                    title="[red]Refused on privacy grounds[/red]",
                    title_align="left",
                )
            )
            raise typer.Exit(code=3) from exc
        except ProviderError as exc:
            err_console.print(f"[red]All providers failed:[/red] {exc}")
            raise typer.Exit(code=4) from exc

        console.print()
        console.print(answer.text)
        console.print()

        if answer.citations:
            vault_name = rt.settings.display_vault_name
            console.print("[dim]Sources:[/dim]")
            for i, chunk in enumerate(answer.citations, start=1):
                # Rich renders this as a hyperlink; obsidian:// opens the note
                # in the app, so a citation is one click rather than a path to
                # go hunting for.
                uri = obsidian_uri(chunk.note_id, vault_name)
                console.print(f"  [dim][{i}][/dim] [link={uri}]{chunk.citation}[/link]")

        marker = "[yellow]escalated[/yellow]" if answer.escalated else "[green]local[/green]"
        console.print(f"\n[dim]{answer.provider} ({marker}) - {answer.elapsed_ms:.0f}ms[/dim]")

        if (remember or rt.settings.memory_auto) and rt.memory is not None:
            try:
                written = rt.memory.from_answer(answer)
            except OSError as exc:
                err_console.print(f"[yellow]could not save memory note: {exc}[/yellow]")
            else:
                if written is not None:
                    rel = written.relative_to(rt.settings.vault_path)
                    console.print(f"[dim]remembered -> {rel}[/dim]")


@app.command()
def status(
    config: ConfigOpt = None,
    vault: VaultOpt = None,
    offline: OfflineOpt = False,
) -> None:
    """Show index, thermal and configuration state."""
    with _runtime(config, vault, offline) as rt:
        rt.governor.sample(force=True)
        counts = rt.catalog.stats()

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_row("Vault", str(rt.settings.vault_path))
        table.add_row("Data", str(rt.settings.data_dir))
        table.add_row("Notes indexed", str(counts["notes"]))
        table.add_row("Chunks", str(counts["chunks"]))
        table.add_row("Store", type(rt.store).__name__)
        table.add_row("Embedder", f"{rt.settings.embed_model} ({type(rt.embedder).__name__})")
        if rt.reranker is not None:
            loaded = getattr(rt.reranker, "loaded", False)
            state = "loaded" if loaded else "lazy"
            table.add_row(
                "Reranker", f"[green]{getattr(rt.reranker, 'name', '?')}[/green] ({state})"
            )
        elif rt.settings.rerank_enabled:
            table.add_row(
                "Reranker",
                "[yellow]enabled but unavailable[/yellow] - pip install 'cortex-brain[rerank]'",
            )
        else:
            table.add_row("Reranker", "[dim]disabled[/dim]")
        table.add_row("Vault name", rt.settings.display_vault_name)
        if rt.memory is not None and rt.memory.enabled:
            mode = "auto" if rt.settings.memory_auto else "on request"
            table.add_row(
                "Memory", f"{rt.memory.count()} notes in {rt.settings.memory_folder}/ ({mode})"
            )
        else:
            table.add_row("Memory", "[dim]disabled[/dim]")
        table.add_row("", "")

        described = rt.governor.describe()
        state = str(described["state"])
        colour = {"boost": "green", "nominal": "cyan", "throttled": "yellow"}.get(state, "red")
        table.add_row("Thermal", f"[{colour}]{state}[/{colour}]")
        table.add_row("Power", str(described["power"]))
        if described["cpu_speed_limit"] is not None:
            table.add_row("CPU speed limit", f"{described['cpu_speed_limit']}%")
        if described["battery_percent"] is not None:
            table.add_row("Battery", f"{described['battery_percent']}%")
        table.add_row("Index workers", str(described["workers"]))
        table.add_row("Backfill allowed", "yes" if described["may_backfill"] else "no")

        console.print(Panel(table, title="cortex status", title_align="left"))


@app.command()
def providers(
    config: ConfigOpt = None,
    vault: VaultOpt = None,
    offline: OfflineOpt = False,
) -> None:
    """Show which providers may see private content, and why."""
    with _runtime(config, vault, offline) as rt:
        explained = rt.router.explain(Sensitivity.PRIVATE)

        table = Table(title="Provider routing for PRIVATE content")
        table.add_column("Provider")
        table.add_column("Model", overflow="fold")
        table.add_column("Policy")
        table.add_column("Status")

        for provider in rt.router.providers:
            spec = provider.spec
            verdict = explained.get(spec.name, "unknown")
            colour = "green" if verdict.startswith("eligible") else "red"
            table.add_row(
                spec.name,
                spec.model,
                spec.policy.value,
                f"[{colour}]{verdict}[/{colour}]",
            )

        console.print(table)

        configured = {p.spec.name for p in rt.router.providers}
        missing = [s for s in rt.settings.providers if s.name not in configured and s.api_key_env]
        if missing:
            console.print("\n[dim]Not configured (API key not set):[/dim]")
            for spec in missing:
                console.print(f"  {spec.name} - set [cyan]{spec.api_key_env}[/cyan]")


@app.command()
def graph(
    config: ConfigOpt = None,
    vault: VaultOpt = None,
    offline: OfflineOpt = False,
    note: Annotated[str | None, typer.Option("--note", help="Show links for one note.")] = None,
) -> None:
    """Inspect the wikilink graph."""
    with _runtime(config, vault, offline) as rt:
        link_graph = rt.refresh_graph()

        if note:
            forward = sorted(link_graph.forward.get(note, set()))
            backward = sorted(link_graph.backward.get(note, set()))
            if not forward and not backward:
                console.print(f"[yellow]No links found for {note}[/yellow]")
                raise typer.Exit()
            console.print(f"[bold]{note}[/bold]")
            for target in forward:
                console.print(f"  -> {target}")
            for source in backward:
                console.print(f"  <- {source}")
            raise typer.Exit()

        stats = link_graph.stats
        table = Table(show_header=False, box=None, padding=(0, 2))
        for key, value in stats.items():
            table.add_row(key.replace("_", " ").title(), str(value))
        console.print(Panel(table, title="wikilink graph", title_align="left"))


@app.command()
def memory(
    config: ConfigOpt = None,
    vault: VaultOpt = None,
    offline: OfflineOpt = False,
    recent: Annotated[int, typer.Option("--recent", help="How many to list.")] = 10,
) -> None:
    """List saved memory notes.

    Deliberately read-only. Memory notes are ordinary Markdown in your vault, so
    editing and deleting them belongs in Obsidian -- a delete command here would
    be a second, worse file manager.
    """
    with _runtime(config, vault, offline) as rt:
        if rt.memory is None or not rt.memory.enabled:
            console.print("[yellow]Memory is disabled in config.[/yellow]")
            raise typer.Exit()

        root = rt.memory.root
        console.print(f"Memory folder: [cyan]{root}[/cyan]")
        console.print(f"Notes saved:   {rt.memory.count()}")

        if not root.exists():
            console.print('[dim]No memory notes yet. Use: cortex ask --remember "..."[/dim]')
            raise typer.Exit()

        notes = sorted(root.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
        if notes:
            console.print()
            for path in notes[:recent]:
                uri = obsidian_uri(
                    path.relative_to(rt.settings.vault_path).as_posix(),
                    rt.settings.display_vault_name,
                )
                console.print(f"  [link={uri}]{path.stem}[/link]")


@app.command()
def bench(
    config: ConfigOpt = None,
    vault: VaultOpt = None,
    offline: OfflineOpt = False,
    cases: Annotated[
        Path | None,
        typer.Option("--cases", help="Ground-truth YAML/JSON of query -> expected notes."),
    ] = None,
    top_k: Annotated[int, typer.Option("--top-k", "-k")] = 10,
    reindex: Annotated[
        bool, typer.Option("--reindex", help="Time a full cold index first.")
    ] = False,
) -> None:
    """Measure retrieval quality and indexing throughput on your own vault.

    Every default in Cortex was set from published findings on someone else's
    corpus. Whether the graph, the reranker and query expansion actually earn
    their latency on *your* notes is an empirical question, and this answers it.
    """
    import time as _time

    from cortex.bench import IndexBenchmark, ablate, load_cases

    with _runtime(config, vault, offline) as rt:
        if reindex:
            with console.status("Cold index..."):
                started = _time.perf_counter()
                report = rt.pipeline.run(full=True)
                cold = _time.perf_counter() - started
                warm_started = _time.perf_counter()
                rt.pipeline.run()
                warm = _time.perf_counter() - warm_started

            timing = IndexBenchmark(
                notes=report.indexed,
                chunks=report.chunks_written,
                elapsed_s=cold,
                reindex_elapsed_s=warm,
            )
            table = Table(title="Indexing", show_header=False, box=None, padding=(0, 2))
            table.add_row("Notes", f"{timing.notes}")
            table.add_row("Chunks", f"{timing.chunks}")
            table.add_row("Cold index", f"{timing.elapsed_s:.2f}s")
            table.add_row("Notes/sec", f"{timing.notes_per_second:.1f}")
            table.add_row("Chunks/sec", f"{timing.chunks_per_second:.1f}")
            table.add_row("No-op rescan", f"{timing.reindex_elapsed_s:.3f}s")
            # Proves the content-hash gate is working. A low figure means
            # something is defeating it and the vault is being re-embedded.
            table.add_row("Hash-gate speedup", f"{timing.speedup:.0f}x")
            console.print(table)
            console.print()
        else:
            rt.pipeline.run()

        rt.refresh_graph()
        engine = rt.engine()

        if cases is None:
            console.print(
                "[yellow]No --cases file given, so quality cannot be measured.[/yellow]\n"
                "[dim]Write ~20 real questions with the notes that should answer them:\n\n"
                "  - query: what did I decide about chunking?\n"
                "    expect: [Chunking Strategy.md]\n[/dim]"
            )
            raise typer.Exit()

        loaded = load_cases(cases)
        if not loaded:
            err_console.print(f"[red]No usable cases in {cases}[/red]")
            raise typer.Exit(code=2)

        with console.status(f"Evaluating {len(loaded)} cases..."):
            result = ablate(engine, loaded, top_k=top_k)

        base = result.baseline
        summary = Table(title=f"Retrieval quality ({base.cases} cases)")
        summary.add_column("Metric")
        summary.add_column("Value", justify="right")
        for key, value in (
            ("recall@5", f"{base.recall_at_5:.3f}"),
            ("recall@10", f"{base.recall_at_10:.3f}"),
            ("MRR", f"{base.mrr:.3f}"),
            ("MAP", f"{base.map_score:.3f}"),
            ("nDCG@10", f"{base.ndcg_at_10:.3f}"),
            ("p50 latency", f"{base.p50_ms:.0f}ms"),
            ("p95 latency", f"{base.p95_ms:.0f}ms"),
        ):
            summary.add_row(key, value)
        console.print(summary)

        rows = result.deltas()
        if rows:
            ablation = Table(title="Ablation - what each component contributes")
            ablation.add_column("Disabled")
            ablation.add_column("recall@5", justify="right")
            ablation.add_column("Δ recall", justify="right")
            ablation.add_column("Δ nDCG", justify="right")
            ablation.add_column("ms saved", justify="right")
            for row in rows:
                delta = float(row["recall_delta"])
                colour = "green" if delta > 0.001 else "red" if delta < -0.001 else "dim"
                ablation.add_row(
                    str(row["disabled"]),
                    f"{row['recall@5']:.3f}",
                    f"[{colour}]{delta:+.3f}[/{colour}]",
                    f"{row['ndcg_delta']:+.3f}",
                    f"{row['p50_saved_ms']:+.0f}",
                )
            console.print(ablation)
            console.print(
                "[dim]Positive Δ means the component helps: disabling it lost that "
                "much recall. Negative means it is hurting you -- turn it off.[/dim]"
            )

        if base.misses:
            console.print(
                f"\n[yellow]{len(base.misses)} queries retrieved nothing expected:[/yellow]"
            )
            for miss in base.misses[:10]:
                console.print(f"  [dim]-[/dim] {miss}")


@app.command()
def watch(
    config: ConfigOpt = None,
    vault: VaultOpt = None,
    offline: OfflineOpt = False,
    interval: Annotated[
        float, typer.Option("--interval", help="Poll interval when not using events.")
    ] = 30.0,
    debounce: Annotated[
        float, typer.Option("--debounce", help="Seconds to settle after an edit.")
    ] = 2.0,
    poll: Annotated[
        bool, typer.Option("--poll", help="Force polling instead of filesystem events.")
    ] = False,
    verbose: Annotated[bool, typer.Option("--verbose")] = False,
) -> None:
    """Continuously index the vault as it changes.

    Event-driven via FSEvents where watchdog is available, so a saved note is
    indexed in about a second rather than whenever the next poll happens to
    land. Falls back to polling if watchdog is missing.
    """
    _setup_logging(verbose)
    import time

    from cortex.ingest.watcher import VaultWatcher, watchdog_available

    with _runtime(config, vault, offline) as rt:
        # Index once up front, so `watch` leaves the vault consistent even if
        # nothing changes while it runs.
        initial = rt.pipeline.run()
        if initial.indexed or initial.deleted:
            console.print(f"[dim]startup: {initial.summary()}[/dim]")
        rt.refresh_graph()

        def reindex(changed: set[Path]) -> None:
            rt.governor.sample()
            report = rt.pipeline.run()
            if report.indexed or report.deleted:
                console.print(f"[dim]{report.summary()}[/dim]")
                rt.refresh_graph()

        if not poll and watchdog_available():
            watcher = VaultWatcher(rt.settings.vault_path, reindex, debounce=debounce)
            try:
                watcher.start()
            except Exception as exc:
                err_console.print(f"[yellow]watcher unavailable ({exc}); polling[/yellow]")
            else:
                console.print(
                    f"Watching [cyan]{rt.settings.vault_path}[/cyan] "
                    f"[dim](events, {debounce:.0f}s debounce - Ctrl-C to stop)[/dim]"
                )
                try:
                    while True:
                        time.sleep(1.0)
                        # Release reranker weights while the machine is idle.
                        if rt.reranker is not None:
                            maybe = getattr(rt.reranker, "maybe_unload", None)
                            if callable(maybe):
                                maybe()
                except KeyboardInterrupt:
                    console.print("\nStopping...")
                finally:
                    watcher.stop()
                return

        console.print(
            f"Polling [cyan]{rt.settings.vault_path}[/cyan] "
            f"[dim](every {interval:.0f}s - Ctrl-C to stop)[/dim]"
        )
        try:
            while True:
                time.sleep(interval)
                reindex(set())
        except KeyboardInterrupt:
            console.print("\nStopped.")


@app.command("serve-mcp")
def serve_mcp(
    config: ConfigOpt = None,
    vault: VaultOpt = None,
    offline: OfflineOpt = False,
) -> None:
    """Run the MCP stdio server (for Antigravity and other MCP clients)."""
    from cortex.mcp_server import run_stdio

    run_stdio(config_path=config, vault=vault, offline=offline)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
