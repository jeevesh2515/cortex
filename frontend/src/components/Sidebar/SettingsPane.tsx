import { useEffect, useState } from 'react';
import { Mic, Volume2, ShieldOff, RefreshCw, Cloud, Smartphone } from 'lucide-react';

import { getStatus, reindex } from '../../api/client';
import type { StatusSnapshot } from '../../types';

import { useSpeechSynthesis } from '../../hooks/useSpeechSynthesis';

interface SettingsPaneProps {
  /**
   * Called after a successful reindex. Used to bump the App-level
   * ``reindexEpoch`` so other panels (the live graph view in the sidebar)
   * refetch without having to wait for their next polling tick.
   */
  onAfterReindex?: () => void;
}

export function SettingsPane({ onAfterReindex }: SettingsPaneProps = {}) {
  const [status, setStatus] = useState<StatusSnapshot | null>(null);
  const [reindexMsg, setReindexMsg] = useState<string | null>(null);
  const [reindexing, setReindexing] = useState(false);
  const tts = useSpeechSynthesis();

  // We ship a chat controller-less settings view so this can render without
  // the chat pane being mounted; in practice Settings and Chat coexist via the
  // top-level App layout switch.
  // ``useChatController`` would read here except that would force the user to
  // instantiate a controller when they click "Settings" without first opening
  // the chat. Instead, the local-only toggle stays in the ChatInput until the
  // user wants a global override.

  useEffect(() => {
    let cancelled = false;
    void getStatus().then((s) => {
      if (!cancelled) setStatus(s);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className="mx-auto w-full max-w-3xl px-4 py-12 text-ink-200">
      <h1 className="mb-2 text-2xl font-semibold text-ink-100">Settings</h1>
      <p className="mb-8 text-sm text-ink-400">
        Local-first defaults; turn clouds on only where you need them.
      </p>

      <Section icon={<ShieldOff size={16} />} title="Privacy">
        <Row
          label="Local-only"
          hint="When on, nothing leaves the machine — Cortex uses the local Ollama endpoint only, even if Groq / OpenRouter / NVIDIA NIM are configured."
        >
          <Coming />
        </Row>
        <Row
          label="OpenRouter ZDR"
          hint="The OpenRouter provider is gated on zero-data-retention. Enable ZDR on your account, then flip this — otherwise private traffic is refused."
        >
          <button
            type="button"
            disabled
            className="rounded-full border border-ink-700 bg-ink-800/50 px-3 py-1 text-[11px] text-ink-500"
            title="Configure providers via §providers in cortex.toml"
          >
            Set in cortex.toml
          </button>
        </Row>
      </Section>

      <Section icon={<Mic size={16} />} title="Voice">
        <Row
          label="Browser STT"
          hint="Speech‑to‑text uses the browser Web Speech API. On Chrome this is Google's cloud STT (not local) — for fully offline voice, route audio through a local Whisper daemon on the server."
        >
          <MicGateHint />
        </Row>
        <Row label="TTS voice" hint="Pick a SpeechSynthesis voice installed in your OS for 'Read aloud' on each assistant message.">
          <select
            value={tts.voices[0]?.voiceURI ?? ''}
            onChange={(e) => {
              const target = tts.voices.find((v) => v.voiceURI === e.target.value);
              if (target) {
                // Speak a tiny test so the user immediately hears the change.
                tts.speak('Cortex voice test.', { voiceURI: target.voiceURI });
              }
            }}
            className="rounded-md border border-ink-800 bg-ink-800/60 px-2 py-1 text-[12px] text-ink-100"
          >
            {tts.voices.map((v) => (
              <option key={v.voiceURI} value={v.voiceURI}>
                {v.name} · {v.lang}
              </option>
            ))}
          </select>
        </Row>
      </Section>

      <Section icon={<Cloud size={16} />} title="Indexing">
        <Row
          label="Reindex vault"
          hint={`Notes indexed: ${status?.notes_indexed ?? '…'}. Reindex picks up new / changed notes; unchanged notes are skipped via content hashing.`}
        >
          <button
            type="button"
            onClick={async () => {
              setReindexing(true);
              setReindexMsg('Running…');
              try {
                const out = (await reindex({})) as { summary?: string };
                setReindexMsg(out?.summary ?? 'Done.');
                const s = await getStatus();
                setStatus(s);
                onAfterReindex?.();
              } catch (err) {
                setReindexMsg(`Failed: ${(err as Error).message}`);
              } finally {
                setReindexing(false);
              }
            }}
            disabled={reindexing}
            className="inline-flex items-center gap-1.5 rounded-full bg-violet-500/20 px-3 py-1 text-[12px] font-medium text-violet-100 ring-1 ring-violet-500/40 transition hover:bg-violet-500/30 disabled:opacity-50 focus-ring"
          >
            <RefreshCw size={12} className={reindexing ? 'animate-spin' : ''} /> Reindex
          </button>
        </Row>
        {reindexMsg && (
          <p className="pl-7 font-mono text-[11px] text-ink-400">{reindexMsg}</p>
        )}
      </Section>

      <Section icon={<Smartphone size={16} />} title="Mobile">
        <Row
          label="Layout"
          hint="Mobile shows a bottom tab bar and a slide-in sidebar; desktop gets a fixed left rail."
        />
      </Section>
    </div>
  );
}

function Section({
  icon,
  title,
  children,
}: {
  icon: React.ReactNode;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section className="mb-7 rounded-2xl border border-ink-800/80 bg-ink-900/45 p-4">
      <header className="mb-3 flex items-center gap-2 text-[11px] font-medium uppercase tracking-[0.18em] text-violet-300">
        {icon}
        {title}
      </header>
      <div className="flex flex-col gap-3">{children}</div>
    </section>
  );
}

function Row({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children?: React.ReactNode;
}) {
  return (
    <div className="flex items-center justify-between gap-4 border-b border-ink-800/50 py-2 last:border-b-0">
      <div className="flex-1">
        <p className="text-sm font-medium text-ink-100">{label}</p>
        {hint ? <p className="mt-0.5 text-[12px] text-ink-500">{hint}</p> : null}
      </div>
      <div className="shrink-0">{children}</div>
    </div>
  );
}

function MicGateHint() {
  const available =
    typeof window !== 'undefined' &&
    (Boolean(window.SpeechRecognition) || Boolean(window.webkitSpeechRecognition));
  return available ? (
    <span className="chip chip-sage">
      <Volume2 size={10} /> available
    </span>
  ) : (
    <span className="chip chip-rose">unavailable in this browser</span>
  );
}

function Coming() {
  return (
    <span className="chip">
      per-turn toggle in chat
    </span>
  );
}
