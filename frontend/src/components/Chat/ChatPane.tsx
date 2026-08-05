import { useEffect, useMemo, useRef } from 'react';
import { AnimatePresence, motion } from 'framer-motion';

import { ChatInput } from './ChatInput';
import { MessageBubble } from './MessageBubble';
import { ScrollToBottomButton } from './ScrollToBottomButton';

import { useChatController } from '../../hooks/useChatController';
import { useChatScroll } from '../../hooks/useChatScroll';

export function ChatPane() {
  const chat = useChatController();
  const threadRef = useRef<HTMLDivElement>(null);
  const scroll = useChatScroll(threadRef, chat.state.messages.length);

  const last = chat.state.messages[chat.state.messages.length - 1] ?? null;

  // Whenever a streaming turn is active we want to stay pinned to the bottom.
  // ``useChatScroll`` already handles that, but a re-render-triggering guard
  // here prevents a one-frame desync while the assistant is mid-word.
  useEffect(() => {
    if (last && !last.done) scroll.scrollToBottom('auto');
  }, [last?.answer, last?.done, scroll]);

  const isEmpty = chat.state.messages.length === 0;

  return (
    <main className="relative flex min-h-0 flex-1 flex-col bg-ink-950">
      <div
        ref={threadRef}
        className="relative flex-1 overflow-y-auto px-1 pb-32 pt-6 sm:px-4"
        role="log"
        aria-live="polite"
      >
        {isEmpty ? (
          <WelcomePanel onSend={(q) => chat.send(q)} disabled={chat.state.streaming} />
        ) : (
          <Thread messages={chat.state.messages} remember={chat.remember} />
        )}
        <AnimatePresence>{!scroll.atBottom && chat.state.streaming && <ScrollToBottomButton onClick={() => scroll.scrollToBottom('smooth')} />}</AnimatePresence>
      </div>
      <ChatInput chat={chat} />
    </main>
  );
}

function Thread({
  messages,
  remember,
}: {
  messages: ReturnType<typeof useChatController>['state']['messages'];
  remember: (turn: ReturnType<typeof useChatController>['state']['messages'][number]) => Promise<void>;
}) {
  return (
    <ul className="mx-auto flex w-full max-w-3xl flex-col gap-6">
      {messages.map((turn, idx) => (
        <motion.li
          key={turn.id}
          initial={{ opacity: 0, y: 4 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.18 }}
        >
          <MessageBubble turn={turn} isLast={idx === messages.length - 1} onRemember={remember} />
        </motion.li>
      ))}
    </ul>
  );
}

function WelcomePanel({
  onSend,
  disabled,
}: {
  onSend: (q: string) => void;
  disabled: boolean;
}) {
  const suggestions = useMemo(
    () => [
      'What did I learn last week?',
      'Summarise today\u2019s daily note',
      'Where did I keep the chunking rationale?',
      'What decisions do I regret, by project?',
    ],
    [],
  );
  return (
    <section className="mx-auto flex h-full w-full max-w-3xl flex-col items-center justify-center px-4 text-center">
      <div
        className="pointer-events-none absolute inset-x-0 top-12 h-40 bg-gradient-radial from-violet-500/25 to-transparent blur-3xl"
        aria-hidden
      />
      <p className="font-mono text-[10px] uppercase tracking-[0.4em] text-violet-300/80">
        Cortex online
      </p>
      <h1 className="mt-3 text-balance text-3xl font-semibold tracking-tight text-ink-50 sm:text-4xl">
        Talk to your second brain.
      </h1>
      <p className="mt-3 max-w-xl text-pretty text-sm text-ink-300 sm:text-base">
        Ask in plain language; Cortex reads your Obsidian vault, finds the sources, writes an
        answer with citations you can one-click into Obsidian. Nothing leaves the machine unless
        you turn that off in Settings.
      </p>
      <div className="mt-7 grid w-full max-w-2xl grid-cols-1 gap-2 sm:grid-cols-2">
        {suggestions.map((s) => (
          <button
            key={s}
            type="button"
            onClick={() => onSend(s)}
            disabled={disabled}
            className="group relative overflow-hidden rounded-2xl border border-ink-800/80 bg-ink-900/60 p-4 text-left text-sm text-ink-200 transition hover:border-violet-500/50 hover:bg-violet-900/20 disabled:cursor-not-allowed disabled:opacity-50 focus-ring"
          >
            <span className="absolute -right-6 -top-6 h-20 w-20 rounded-full bg-violet-500/30 opacity-0 blur-2xl transition group-hover:opacity-100" aria-hidden />
            <span className="relative">{s}</span>
          </button>
        ))}
      </div>
    </section>
  );
}
