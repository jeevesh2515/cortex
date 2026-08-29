import { useCallback, useEffect, useState } from 'react';

import { AppShell } from './components/Layout/AppShell';
import { Sidebar } from './components/Layout/Sidebar';
import { ChatPane } from './components/Chat/ChatPane';
import { MobileNav } from './components/Layout/MobileNav';
import { SettingsPane } from './components/Sidebar/SettingsPane';

import { getStatus, getWhoAmI } from './api/client';
import type { StatusSnapshot, WhoAmI } from './types';

type Surface = 'chat' | 'memory' | 'status' | 'settings';

export default function App() {
  const [status, setStatus] = useState<StatusSnapshot | null>(null);
  const [whoami, setWhoami] = useState<WhoAmI | null>(null);
  const [surface, setSurface] = useState<Surface>('chat');
  const [drawerOpen, setDrawerOpen] = useState(false);
  // Monotonically-increasing epoch bumped whenever the user reindexes. The
  // sidebar's graph view watches this so it refetches immediately rather
  // than waiting for its next 60-s poll tick.
  const [reindexEpoch, setReindexEpoch] = useState(0);

  const bumpReindex = useCallback(() => {
    setReindexEpoch((n) => n + 1);
  }, []);

  // Initial snapshots, then poll status every 30s so the sidebar's "vault
  // count" / "thermal" stay current without the user touching anything.
  useEffect(() => {
    let cancelled = false;
    const refresh = async () => {
      try {
        const [w, s] = await Promise.all([getWhoAmI(), getStatus()]);
        if (!cancelled) {
          setWhoami(w);
          setStatus(s);
        }
      } catch {
        // network blip or first-request before backend is up -- ignore so
        // the UI does not flap.
      }
    };
    void refresh();
    const handle = window.setInterval(refresh, 30_000);
    return () => {
      cancelled = true;
      window.clearInterval(handle);
    };
  }, []);

  return (
    <AppShell
      status={status}
      whoami={whoami}
      drawerOpen={drawerOpen}
      onToggleDrawer={() => setDrawerOpen((v) => !v)}
    >
      <Sidebar
        status={status}
        whoami={whoami}
        surface={surface}
        reindexEpoch={reindexEpoch}
        drawerOpen={drawerOpen}
        onSelectSurface={setSurface}
        onClose={() => setDrawerOpen(false)}
      />
      <div className="flex min-h-0 flex-1 flex-col bg-ink-950">
        {surface === 'chat' && <ChatPane />}
        {surface === 'memory' && <MemoryView />}
        {surface === 'status' && status && <StatusView status={status} whoami={whoami} />}
        {surface === 'settings' && <SettingsPane onAfterReindex={bumpReindex} />}
        <MobileNav surface={surface} onSelect={setSurface} />
      </div>
    </AppShell>
  );
}

function MemoryView() {
  return (
    <div className="mx-auto max-w-3xl px-4 py-12 text-ink-300">
      <h1 className="mb-2 text-2xl font-semibold text-ink-100">Memory</h1>
      <p className="text-sm text-ink-400">
        Memory notes are written into the <code className="font-mono">Memory/</code> folder of your
        vault by the &ldquo;remember&rdquo; button on each assistant message. Open{' '}
        <span className="font-mono text-violet-300">Memory</span> in Obsidian or the sidebar&rsquo;s
        Memory tab to read or edit them.
      </p>
    </div>
  );
}

function StatusView({ status, whoami }: { status: StatusSnapshot; whoami: WhoAmI | null }) {
  return (
    <div className="mx-auto max-w-3xl px-4 py-12 text-ink-200">
      <h1 className="mb-6 text-2xl font-semibold text-ink-100">Status</h1>
      <dl className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <Stat label="Vault" value={status.vault_path} mono />
        <Stat label="Vault name" value={status.vault_name} />
        <Stat label="Notes indexed" value={String(status.notes_indexed)} />
        <Stat label="Chunks" value={String(status.chunks)} />
        <Stat label="Memory notes" value={String(status.memory_count)} />
        <Stat label="Thermal state" value={status.thermal.state} />
        <Stat
          label="Workers"
          value={`${String(status.thermal.workers)}${status.thermal.may_backfill ? ' (backfill ok)' : ' (no backfill)'}`}
        />
        <Stat label="Local-only" value={whoami?.local_only ? 'on' : 'off'} />
      </dl>
    </div>
  );
}

function Stat({ label, value, mono = false }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="rounded-xl border border-ink-700/60 bg-ink-900/55 p-4">
      <dt className="text-[11px] uppercase tracking-wide text-ink-400">{label}</dt>
      <dd className={`mt-1 text-base text-ink-100 ${mono ? 'font-mono text-sm' : ''}`}>{value}</dd>
    </div>
  );
}
