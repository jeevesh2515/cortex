import { ReactNode } from 'react';
import {
  Activity,
  Brain,
  Database,
  Settings,
  Sparkles,
} from 'lucide-react';

import type { StatusSnapshot, WhoAmI } from '../../types';

import { ProviderMatrix } from '../Sidebar/ProviderMatrix';
import { MemoryList } from '../Sidebar/MemoryList';
import { VaultSearch } from '../Sidebar/VaultSearch';
import { GraphView } from '../Sidebar/GraphView';

export type Surface = 'chat' | 'memory' | 'status' | 'settings';

export interface SidebarProps {
  status: StatusSnapshot | null;
  whoami: WhoAmI | null;
  surface: Surface;
  drawerOpen: boolean;
  reindexEpoch?: number;
  onSelectSurface: (s: Surface) => void;
  onClose: () => void;
}

const SURFACES: Array<{ id: Surface; label: string; icon: ReactNode }> = [
  { id: 'chat', label: 'Chat', icon: <Sparkles size={16} /> },
  { id: 'memory', label: 'Memory', icon: <Brain size={16} /> },
  { id: 'status', label: 'Status', icon: <Activity size={16} /> },
  { id: 'settings', label: 'Settings', icon: <Settings size={16} /> },
];

export function Sidebar({
  status,
  whoami,
  surface,
  drawerOpen,
  reindexEpoch,
  onSelectSurface,
  onClose,
}: SidebarProps) {
  return (
    <aside
      className={`fixed inset-y-0 left-0 z-40 flex w-[min(20rem,90vw)] flex-col border-r border-ink-800 bg-ink-900/90 backdrop-blur transition-transform duration-200 ease-out md:relative md:z-auto md:flex md:w-[20rem] md:translate-x-0 ${drawerOpen ? 'translate-x-0' : '-translate-x-full'}`}
    >
      <SidebarHeader whoami={whoami} onClose={onClose} />
      <SurfaceTabs surface={surface} onSelect={(s) => { onSelectSurface(s); onClose(); }} />
      <div className="flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto px-4 pb-6">
        <VaultSearch key={surface /* re-mount on tab switch */} />
        {whoami && (
          <GraphView
            vaultName={whoami.vault_name}
            reindexEpoch={reindexEpoch}
          />
        )}
        <ProviderMatrix status={status} />
        <MemoryList />
      </div>
      <SidebarFooter status={status} />
    </aside>
  );
}

function SidebarHeader({ whoami, onClose }: { whoami: WhoAmI | null; onClose: () => void }) {
  return (
    <header className="flex items-start justify-between px-4 pt-5 pb-3">
      <div>
        <div className="flex items-center gap-2">
          <span className="inline-flex h-7 w-7 items-center justify-center rounded-lg bg-gradient-to-br from-violet-500/80 to-violet-900/80 text-violet-50 shadow-glow">
            <span className="text-sm font-semibold">C</span>
          </span>
          <h2 className="text-lg font-semibold tracking-tight text-ink-100">
            {whoami?.vault_name ?? 'Cortex'}
          </h2>
        </div>
        <p className="mt-1 font-mono text-[10px] uppercase tracking-[0.18em] text-ink-500">
          Second Brain · local
        </p>
      </div>
      <button
        type="button"
        onClick={onClose}
        className="rounded-full p-1 text-ink-400 transition hover:text-ink-100 focus-ring md:hidden"
        aria-label="Close sidebar"
      >
        <span aria-hidden>×</span>
      </button>
    </header>
  );
}

function SurfaceTabs({
  surface,
  onSelect,
}: {
  surface: Surface;
  onSelect: (s: Surface) => void;
}) {
  return (
    <nav aria-label="Sections" className="px-2 pb-2">
      <ul className="flex flex-wrap gap-1">
        {SURFACES.map(({ id, label, icon }) => {
          const active = id === surface;
          return (
            <li key={id}>
              <button
                type="button"
                onClick={() => onSelect(id)}
                className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1.5 text-xs font-medium transition focus-ring ${
                  active
                    ? 'bg-violet-500/20 text-violet-100 ring-1 ring-violet-500/40'
                    : 'text-ink-300 hover:bg-ink-800/60'
                }`}
                aria-current={active ? 'page' : undefined}
              >
                {icon}
                {label}
              </button>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}

function SidebarFooter({ status }: { status: StatusSnapshot | null }) {
  const notes = status?.notes_indexed ?? 0;
  const chunks = status?.chunks ?? 0;
  const memories = status?.memory_count ?? 0;
  return (
    <footer className="border-t border-ink-800/80 px-4 py-3 text-[11px] text-ink-400">
      <div className="flex items-center justify-between gap-3 font-mono">
        <span className="inline-flex items-center gap-1.5">
          <Database size={11} /> {notes.toLocaleString()} notes / {chunks.toLocaleString()} chunks
        </span>
        <span className="inline-flex items-center gap-1.5 text-violet-300/80">
          <Brain size={11} /> {memories} memory
        </span>
      </div>
    </footer>
  );
}
