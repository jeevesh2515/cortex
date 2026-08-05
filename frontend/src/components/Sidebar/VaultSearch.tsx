import { useState } from 'react';
import { Search, ExternalLink, FileText } from 'lucide-react';

import { searchCitations } from '../../api/client';
import type { Citation } from '../../types';

export function VaultSearch() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<Citation[] | null>(null);
  const [busy, setBusy] = useState(false);
  const [elapsed, setElapsed] = useState<number | null>(null);

  const run = async () => {
    const q = query.trim();
    if (!q) {
      setResults(null);
      return;
    }
    setBusy(true);
    try {
      const r = await searchCitations(q, 6);
      setResults(r.results);
      setElapsed(r.elapsed_ms);
    } catch {
      setResults([]);
      setElapsed(null);
    } finally {
      setBusy(false);
    }
  };

  return (
    <section
      aria-label="Vault search"
      className="rounded-2xl border border-ink-800/80 bg-ink-900/55 p-3"
    >
      <header className="mb-2 flex items-center justify-between">
        <h3 className="text-[11px] font-medium uppercase tracking-[0.18em] text-ink-400">
          Vault · search
        </h3>
      </header>
      <form
        onSubmit={(e) => {
          e.preventDefault();
          void run();
        }}
        className="relative"
      >
        <Search size={13} className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-ink-500" />
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onBlur={() => {
            if (query.trim()) void run();
          }}
          placeholder="Find a note…"
          className="w-full rounded-full border border-ink-800 bg-ink-800/50 py-1.5 pl-8 pr-3 text-[13px] text-ink-100 placeholder-ink-500 focus:border-violet-500/60 focus:outline-none"
          aria-label="Search the vault"
        />
      </form>
      {busy && (
        <p className="mt-2 text-center font-mono text-[10px] uppercase tracking-[0.2em] text-ink-500">
          searching…
        </p>
      )}
      {!busy && results && (
        <div className="mt-2">
          {elapsed != null && (
            <p className="font-mono text-[10px] uppercase tracking-wide text-ink-500">
              {results.length} results · {elapsed.toFixed(0)} ms
            </p>
          )}
          <ul className="mt-1 flex flex-col gap-1">
            {results.map((c) => (
              <li key={c.note_id}>
                <a
                  href={c.obsidian_uri}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="group flex items-start gap-2 rounded-lg px-2 py-1.5 text-[12px] text-ink-200 transition hover:bg-ink-800/70 hover:text-ink-50 focus-ring"
                >
                  <FileText size={11} className="mt-0.5 text-ink-400 group-hover:text-violet-300" />
                  <span className="min-w-0 flex-1">
                    <span className="block truncate font-medium">{c.title}</span>
                    <span className="line-clamp-1 text-[11px] text-ink-500">
                      {c.snippet}
                    </span>
                  </span>
                  <ExternalLink size={10} className="mt-0.5 text-ink-500 group-hover:text-violet-300" />
                </a>
              </li>
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}
