import { Brain, Activity, Settings, Sparkles } from 'lucide-react';

import type { Surface } from './Sidebar';

const ITEMS: Array<{ id: Surface; label: string; Icon: typeof Sparkles }> = [
  { id: 'chat', label: 'Chat', Icon: Sparkles },
  { id: 'memory', label: 'Memory', Icon: Brain },
  { id: 'status', label: 'Status', Icon: Activity },
  { id: 'settings', label: 'Settings', Icon: Settings },
];

export function MobileNav({
  surface,
  onSelect,
}: {
  surface: Surface;
  onSelect: (s: Surface) => void;
}) {
  return (
    <nav
      aria-label="Primary"
      className="sticky bottom-0 z-20 mt-auto flex justify-around border-t border-ink-800/80 bg-ink-900/85 px-2 pb-[env(safe-area-inset-bottom)] pt-1 backdrop-blur md:hidden"
    >
      {ITEMS.map(({ id, label, Icon }) => {
        const active = id === surface;
        return (
          <button
            key={id}
            type="button"
            onClick={() => onSelect(id)}
            aria-current={active ? 'page' : undefined}
            className={`flex flex-1 flex-col items-center gap-0.5 py-2 text-[10px] font-medium transition focus-ring ${
              active ? 'text-violet-200' : 'text-ink-400 hover:text-ink-100'
            }`}
          >
            <Icon size={18} />
            {label}
          </button>
        );
      })}
    </nav>
  );
}
