import { useEffect, useRef, useState } from 'react';
import { ArrowUp, Loader2, Mic, Square, Wand2 } from 'lucide-react';

import { useSpeechRecognition } from '../../hooks/useSpeechRecognition';

interface ChatController {
  send: (q: string, opts?: { remember?: boolean }) => Promise<void>;
  stop: () => void;
  clear: () => void;
  setLocalOnly: (v: boolean) => void;
  remember: (turn: import('../../hooks/useChatController').ChatTurn) => Promise<void>;
  state: import('../../hooks/useChatController').ChatState;
  lastCitations: import('../../types').Citation[];
}

interface ChatInputProps {
  chat: ChatController;
}

export function ChatInput({ chat }: ChatInputProps) {
  const [draft, setDraft] = useState('');
  const taRef = useRef<HTMLTextAreaElement>(null);
  const [rememberNext, setRememberNext] = useState(false);

  // Auto-grow the textarea up to ~8 lines, then keep it scrollable inside.
  useEffect(() => {
    const ta = taRef.current;
    if (!ta) return;
    ta.style.height = 'auto';
    ta.style.height = `${Math.min(ta.scrollHeight, 220)}px`;
  }, [draft]);

  // Push-to-talk via Web Speech API. ``interim`` updates the draft while the
  // user is still speaking; on a final result we replace the interim span.
  const speaking = useSpeechRecognition({
    onFinal: (text) => {
      setDraft((prev) => {
        const base = prev.replace(/\s*\[listening\\u2026\]?\s*$/, '');
        return base ? `${base} ${text}` : text;
      });
    },
  });

  const startListening = () => speaking.start();
  const stopListening = () => speaking.stop();

  // Build the placeholder dynamically so the user always knows what is happening.
  const placeholder =
    chat.state.streaming
      ? 'Cortex is thinking…'
      : chat.state.localOnly
        ? 'Ask — local-only (no network egress)'
        : 'Ask Cortex. Citations appear as the answer streams.';

  // Submit handler. Stops any in-flight stream before sending a new prompt.
  const submit = async () => {
    const text = draft.trim();
    if (!text || chat.state.streaming) return;
    setDraft('');
    if (speaking.listening) speaking.abort();
    chat.send(text, { remember: rememberNext });
    setRememberNext(false);
  };

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      void submit();
    } else if (e.key === 'Escape') {
      if (speaking.listening) speaking.abort();
      if (chat.state.streaming) chat.stop();
      chat.clear();
      setDraft('');
    }
  };

  // Stream intermediate transcript visibly inside the textarea.
  const displayValue = speaking.listening && speaking.interim
    ? `${draft.replace(/\s*\[listening\\u2026\]?\s*$/, '').trim()} ${speaking.interim}`.trim() +
      ' [listening…]'
    : draft;

  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        void submit();
      }}
      className="pointer-events-none sticky bottom-0 z-10 flex w-full justify-center bg-gradient-to-t from-ink-950 via-ink-950/95 to-transparent pb-6 pt-10"
    >
      <div className="pointer-events-auto w-full max-w-3xl px-3">
        <div className="glass relative flex w-full flex-col rounded-3xl shadow-glow">
          <textarea
            ref={taRef}
            value={displayValue}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder={placeholder}
            rows={1}
            aria-label="Type your question"
            className="w-full resize-none bg-transparent px-5 pb-2 pt-4 text-[15px] leading-relaxed text-ink-50 placeholder-ink-500 focus:outline-none"
            disabled={false}
          />
          <div className="flex items-center justify-between gap-2 px-3 pb-3 pt-1">
            <div className="flex items-center gap-2">
              {speaking.available ? (
                <button
                  type="button"
                  onClick={speaking.listening ? stopListening : startListening}
                  aria-label={speaking.listening ? 'Stop dictation' : 'Start dictation'}
                  className={`group inline-flex h-10 min-h-[44px] items-center gap-1.5 rounded-full px-3 text-xs font-medium transition focus-ring ${
                    speaking.listening
                      ? 'bg-rose-500/20 text-rose-200 ring-1 ring-rose-400/40'
                      : 'bg-ink-800/70 text-ink-200 hover:bg-violet-900/30 hover:text-violet-200'
                  }`}
                >
                  {speaking.listening ? (
                    <>
                      <span className="relative inline-flex h-2 w-2">
                        <span className="absolute inset-0 animate-pulse_ring rounded-full bg-rose-300" />
                        <span className="relative inline-flex h-2 w-2 rounded-full bg-rose-400" />
                      </span>
                      Listening…
                    </>
                  ) : (
                    <>
                      <Mic size={14} /> Mic
                    </>
                  )}
                </button>
              ) : (
                <button
                  type="button"
                  disabled
                  className="inline-flex h-9 cursor-not-allowed items-center gap-1.5 rounded-full bg-ink-800/40 px-3 text-xs text-ink-500"
                  title="Browser doesn't support Web Speech API"
                >
                  <Mic size={14} /> No mic
                </button>
              )}

              <Toggle
                checked={chat.state.localOnly}
                onChange={chat.setLocalOnly}
                label="Local only"
                color="sage"
              />
              <Toggle
                checked={rememberNext}
                onChange={setRememberNext}
                label="Remember"
                color="amber"
              />
            </div>
            <div className="flex items-center gap-2">
              {chat.state.streaming ? (
                <button
                  type="button"
                  onClick={chat.stop}
                  className="inline-flex h-9 items-center gap-1.5 rounded-full bg-rose-500/15 px-3 text-xs font-medium text-rose-200 transition hover:bg-rose-500/25 focus-ring"
                >
                  <Square size={12} /> Stop
                </button>
              ) : null}
              <button
                type="submit"
                disabled={!draft.trim() || chat.state.streaming}
                className="inline-flex h-9 items-center gap-1.5 rounded-full bg-gradient-to-br from-violet-500 to-violet-700 px-4 text-sm font-medium text-violet-50 transition hover:from-violet-400 hover:to-violet-600 disabled:cursor-not-allowed disabled:from-ink-700 disabled:to-ink-800 disabled:text-ink-500 focus-ring shadow-glow"
                aria-label="Send"
              >
                {chat.state.streaming ? <Loader2 size={14} className="animate-spin" /> : <ArrowUp size={14} />}
                <Wand2 size={12} className="opacity-70" />
              </button>
            </div>
          </div>
          {speaking.error ? (
            <div className="px-4 pb-3 text-[11px] text-rose-300">Mic error: {speaking.error}</div>
          ) : null}
        </div>
        <p className="mt-2 text-center font-mono text-[10px] uppercase tracking-[0.2em] text-ink-500">
          ⏎ send · ⇧⏎ newline · esc stop
        </p>
      </div>
    </form>
  );
}

function Toggle({
  checked,
  onChange,
  label,
  color,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  label: string;
  color: 'sage' | 'amber' | 'violet';
}) {
  const dotColor = {
    sage: 'bg-sage-500',
    amber: 'bg-amber-400',
    violet: 'bg-violet-400',
  }[color];
  return (
    <button
      type="button"
      onClick={() => onChange(!checked)}
      aria-pressed={checked}
      className={`inline-flex h-7 items-center gap-2 rounded-full px-2.5 text-[11px] font-medium uppercase tracking-wide transition focus-ring ${
        checked
          ? `${color === 'sage' ? 'bg-sage-500/15 text-sage-300 ring-1 ring-sage-500/40' : color === 'amber' ? 'bg-amber-500/15 text-amber-300 ring-1 ring-amber-500/40' : 'bg-violet-500/15 text-violet-200 ring-1 ring-violet-500/40'}`
          : 'bg-ink-800/50 text-ink-400 hover:text-ink-200'
      }`}
    >
      <span
        className={`inline-block h-3 w-3 rounded-full transition ${
          checked ? dotColor : 'bg-ink-600'
        }`}
      />
      {label}
    </button>
  );
}
