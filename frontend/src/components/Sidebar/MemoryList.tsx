import { useEffect, useState } from 'react';
import { BookOpen, ExternalLink } from 'lucide-react';

import { listMemoryNotes } from '../../api/client';
import type { MemoryNoteOut } from '../../types';

export function MemoryList() {
  const [notes, setNotes] = useState<MemoryNoteOut[] | null>(null);

  useEffect(() => {
    let cancelled = false;
    listMemoryNotes(8).then((rows) => {
      if (!cancelled) setNotes(rows);
    }).catch(() => {
      if (!cancelled) setNotes([]);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <section
      aria-label="Recent memory notes"
      className="rounded-2xl border border-ink-800/80 bg-ink-900/55 p-3"
    >
      <header className="mb-2 flex items-center justify-between">
        <h3 className="text-[11px] font-medium uppercase tracking-[0.18em] text-ink-400">
          Memory · recent
        </h3>
        <span className="font-mono text-[10px] text-ink-500">
          {notes === null ? '…' : `${notes.length} saved`}
        </span>
      </header>
      <ul className="flex flex-col gap-1">
        {notes === null ? (
          <li className="rounded-lg border border-dashed border-ink-700 p-3 text-center text-[11px] text-ink-500">
            Loading…
          </li>
        ) : notes.length === 0 ? (
          <li className="rounded-lg border border-dashed border-ink-700 p-3 text-center text-[11px] text-ink-500">
            No memory notes yet. Use “remember” on an answer.
          </li>
        ) : (
          notes.map((n) => (
            <li key={n.path}>
              <a
                href={n.obsidian_uri}
                target="_blank"
                rel="noopener noreferrer"
                className="group flex items-start gap-2 rounded-lg px-2.5 py-2 text-[12px] text-ink-200 transition hover:bg-ink-800/70 hover:text-ink-50 focus-ring"
              >
                <BookOpen size={12} className="mt-0.5 text-violet-300" />
                <span className="min-w-0 flex-1">
                  <span className="block truncate font-medium">{n.title}</span>
                  <span className="block truncate text-[11px] text-ink-500">
                    {n.question}
                  </span>
                </span>
                <ExternalLink
                  size={11}
                  className="mt-0.5 text-ink-500 transition group-hover:text-violet-300"
                />
              </a>
            </li>
          ))
        )}
      </ul>
    </section>
  );
}
