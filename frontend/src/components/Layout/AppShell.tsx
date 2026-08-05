import { ReactNode } from 'react';

import type { StatusSnapshot, WhoAmI } from '../../types';

export interface AppShellProps {
  children: ReactNode;
  status: StatusSnapshot | null;
  whoami: WhoAmI | null;
  drawerOpen: boolean;
  onToggleDrawer: () => void;
}

export function AppShell({
  children,
  status,
  whoami,
  drawerOpen,
  onToggleDrawer,
}: AppShellProps) {
  return (
    <div className="relative isolate min-h-svh w-full overflow-hidden bg-ink-950 text-ink-100">
      <div className="pointer-events-none absolute inset-0 bg-radial-fade" aria-hidden />
      <div className="relative flex min-h-svh w-full">
        <MobileSidebarToggle onClick={onToggleDrawer} />
        {drawerOpen && (
          <div
            className="fixed inset-0 z-30 bg-ink-950/70 backdrop-blur-sm md:hidden"
            onClick={onToggleDrawer}
            aria-label="close drawer"
          />
        )}
        {children}
      </div>
      <BottomHalo status={status} whoami={whoami} />
    </div>
  );
}

function MobileSidebarToggle({ onClick }: { onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="fixed left-3 top-3 z-40 inline-flex h-9 w-9 items-center justify-center rounded-full border border-ink-700/70 bg-ink-900/70 text-ink-200 shadow-sm backdrop-blur transition hover:border-violet-500/60 hover:text-violet-200 focus-ring md:hidden"
      aria-label="Open sidebar"
    >
      <span className="block h-0.5 w-4 bg-current [box-shadow:0_-5px_0_currentColor,0_5px_0_currentColor]" />
    </button>
  );
}

function BottomHalo({ status, whoami }: { status: StatusSnapshot | null; whoami: WhoAmI | null }) {
  const vaultLabel = whoami?.vault_name ?? 'Cortex';
  const thermal = status?.thermal?.state ?? 'unknown';

  return (
    <div
      className="pointer-events-none fixed bottom-3 right-4 z-10 hidden text-right font-mono text-[10px] uppercase tracking-[0.18em] text-ink-500 sm:block"
      aria-hidden
    >
      {vaultLabel} · {thermal}
    </div>
  );
}
