import { Check, ShieldAlert, ShieldCheck } from 'lucide-react';

import type { StatusSnapshot } from '../../types';

interface ProviderMatrixProps {
  status: StatusSnapshot | null;
}

const POLICY_CHIP: Record<string, { label: string; chipClass: string }> = {
  local: { label: 'local', chipClass: 'chip-sage' },
  no_train: { label: 'no-train', chipClass: 'chip-sage' },
  no_train_if_zdr: { label: 'zdr', chipClass: 'chip-violet' },
  trains: { label: 'trains', chipClass: 'chip-rose' },
};

export function ProviderMatrix({ status }: ProviderMatrixProps) {
  const providers = status?.providers ?? [];
  return (
    <section
      aria-label="Provider routing"
      className="rounded-2xl border border-ink-800/80 bg-ink-900/55 p-3"
    >
      <header className="mb-2 flex items-center justify-between">
        <h3 className="text-[11px] font-medium uppercase tracking-[0.18em] text-ink-400">
          Providers
        </h3>
        <span className="font-mono text-[10px] text-ink-500">
          {providers.length} configured
        </span>
      </header>
      <ul className="flex flex-col gap-1.5">
        {providers.length === 0 ? (
          <li className="rounded-lg border border-dashed border-ink-700 p-3 text-center text-[11px] text-ink-500">
            Loading…
          </li>
        ) : (
          providers.map((p) => {
            const chip = POLICY_CHIP[p.policy] ?? { label: p.policy, chipClass: 'chip' };
            return (
              <li
                key={p.name}
                className="flex items-center justify-between gap-2 rounded-lg bg-ink-800/40 px-2.5 py-1.5 text-[12px] text-ink-200"
                title={p.reason ?? (p.eligible_private ? 'eligible for private traffic' : 'refused for private traffic')}
              >
                <span className="flex min-w-0 items-center gap-1.5">
                  <span
                    className={`relative inline-flex h-4 w-4 items-center justify-center rounded-full ${
                      p.eligible_private ? 'text-sage-400' : 'text-rose-400'
                    }`}
                  >
                    {p.eligible_private ? <ShieldCheck size={12} /> : <ShieldAlert size={12} />}
                  </span>
                  <span className="truncate font-medium">{p.name}</span>
                  <span className="font-mono text-[10px] text-ink-500 truncate">· {p.model}</span>
                </span>
                <span className={`chip ${chip.chipClass} shrink-0`}>{chip.label}</span>
              </li>
            );
          })
        )}
      </ul>
      <p className="mt-2 font-mono text-[10px] uppercase tracking-wide text-ink-500">
        <Check size={9} className="inline" /> eligible for private traffic
      </p>
    </section>
  );
}
