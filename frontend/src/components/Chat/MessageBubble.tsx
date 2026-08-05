import { useMemo, useState } from 'react';
import { Bot, PauseCircle, PlayCircle, Save, ShieldCheck, ShieldOff, User2 } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

import type { ChatTurn } from '../../hooks/useChatController';
import { useSpeechSynthesis } from '../../hooks/useSpeechSynthesis';

const POLICY_LABEL: Record<string, { label: string; color: 'violet' | 'sage' | 'amber' | 'rose' }> = {
  local: { label: 'local', color: 'sage' },
  no_train: { label: 'no-train', color: 'sage' },
  no_train_if_zdr: { label: 'zdr', color: 'violet' },
  trains: { label: 'trains', color: 'rose' },
};

interface MessageBubbleProps {
  turn: ChatTurn;
  isLast: boolean;
  onRemember?: (turn: ChatTurn) => Promise<void>;
}

export function MessageBubble({ turn, isLast, onRemember }: MessageBubbleProps) {
  const isUser = turn.answer.length === 0 && !turn.provider && !turn.error;
  const tts = useSpeechSynthesis();

  // Markdown rendering of the streamed answer. react-markdown is light and
  // handles GFM tables / strikethrough out of the box.
  const renderedAnswer = useMemo(() => {
    if (isUser) return null;
    return (
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a: ({ href, children, ...rest }) => (
            <a
              href={href}
              target="_blank"
              rel="noopener noreferrer"
              className="text-violet-300 underline-offset-2 hover:underline"
              {...rest}
            >
              {children}
            </a>
          ),
          p: (props) => <p className="my-1.5 leading-relaxed" {...props} />,
          ul: (props) => <ul className="my-1.5 list-disc pl-5" {...props} />,
          ol: (props) => <ol className="my-1.5 list-decimal pl-5" {...props} />,
          code: (props) => (
            <code className="rounded bg-ink-800/80 px-1 py-0.5 font-mono text-[90%]" {...props} />
          ),
          pre: (props) => (
            <pre className="my-2 overflow-x-auto rounded-lg bg-ink-900/80 p-3 text-[12px]" {...props} />
          ),
          h1: (props) => <h1 className="mt-3 text-xl font-semibold" {...props} />,
          h2: (props) => <h2 className="mt-3 text-lg font-semibold" {...props} />,
          h3: (props) => <h3 className="mt-2 text-base font-semibold" {...props} />,
        }}
      >
        {turn.answer || (isLast ? '' : '(no answer)')}
      </ReactMarkdown>
    );
  }, [turn.answer, isLast, isUser]);

  const streaming = isLast && !turn.done;
  const policyChip = turn.policy ? POLICY_LABEL[turn.policy] ?? null : null;

  return (
    <article className={`flex gap-3 ${isUser ? 'justify-end' : 'justify-start'}`}>
      {!isUser && (
        <Avatar kind="bot" />
      )}
      <div className={`flex max-w-[88%] flex-col gap-2 ${isUser ? 'items-end' : 'items-start'}`}>
        {isUser ? (
          <Bubble tone="user">
            <p className="whitespace-pre-wrap leading-relaxed text-ink-50">{turn.question}</p>
          </Bubble>
        ) : (
          <Bubble tone="assistant" streaming={streaming}>
            <div className={`prose prose-invert max-w-none text-ink-100 ${streaming ? 'streaming-caret' : ''}`}>
              {renderedAnswer}
            </div>
            <CitationsStrip citations={turn.citations} />
            {turn.error ? (
              <p className="mt-2 text-sm text-rose-300">
                <strong className="font-semibold">{turn.error.type ?? 'error'}:</strong> {turn.error.message}
              </p>
            ) : null}
            <BubbleMeta
              turn={turn}
              policy={policyChip}
              onPlay={() => tts.speak(turn.answer)}
              onStop={() => tts.cancel()}
              onRemember={onRemember ? () => onRemember(turn) : undefined}
              ttsBusy={tts.speaking}
            />
          </Bubble>
        )}
      </div>
      {isUser && <Avatar kind="user" />}
    </article>
  );
}

function Avatar({ kind }: { kind: 'user' | 'bot' }) {
  if (kind === 'user') {
    return (
      <div className="hidden h-8 w-8 shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-ink-700 to-ink-900 text-ink-200 ring-1 ring-ink-700 sm:flex">
        <User2 size={14} />
      </div>
    );
  }
  return (
    <div className="relative h-8 w-8 shrink-0">
      <div
        className="absolute inset-0 animate-breathe rounded-full bg-gradient-to-br from-violet-500 to-violet-900 shadow-glow"
        aria-hidden
      />
      <div className="relative flex h-full w-full items-center justify-center text-violet-50">
        <Bot size={15} />
      </div>
    </div>
  );
}

function Bubble({
  tone,
  streaming,
  children,
}: {
  tone: 'user' | 'assistant';
  streaming?: boolean;
  children: React.ReactNode;
}) {
  if (tone === 'user') {
    return (
      <div className="rounded-2xl rounded-br-md bg-gradient-to-br from-violet-500/85 to-violet-700/90 px-4 py-2.5 text-[15px] text-violet-50 shadow-glow">
        {children}
      </div>
    );
  }
  return (
    <div className="rounded-2xl rounded-tl-md border border-ink-800/80 bg-ink-900/65 px-4 py-3 shadow-[0_1px_0_0_rgba(255,255,255,0.02)]">
      {children}
      {streaming ? <span className="mt-2 block text-[11px] uppercase tracking-wide text-violet-300/80">streaming…</span> : null}
    </div>
  );
}

function CitationsStrip({ citations }: { citations: ChatTurn['citations'] }) {
  if (!citations?.length) return null;
  return (
    <div className="mt-3 flex flex-wrap gap-1.5">
      {citations.map((c) => (
        <CitationChip key={c.note_id + c.index} citation={c} />
      ))}
    </div>
  );
}

function CitationChip({ citation }: { citation: ChatTurn['citations'][number] }) {
  return (
    <a
      href={citation.obsidian_uri}
      target="_blank"
      rel="noopener noreferrer"
      className="group inline-flex max-w-[14rem] items-center gap-1.5 rounded-full border border-violet-700/50 bg-violet-900/30 px-2.5 py-1 text-[11px] text-violet-200 transition hover:border-violet-500 hover:bg-violet-900/50 hover:text-violet-50 focus-ring"
      title={citation.title}
    >
      <span className="font-mono text-[10px] opacity-80">[{citation.index}]</span>
      <span className="truncate">{citation.title.split(' > ').pop() ?? citation.title}</span>
    </a>
  );
}

function BubbleMeta({
  turn,
  policy,
  onPlay,
  onStop,
  onRemember,
  ttsBusy,
}: {
  turn: ChatTurn;
  policy: { label: string; color: 'violet' | 'sage' | 'amber' | 'rose' } | null;
  onPlay: () => void;
  onStop: () => void;
  onRemember?: () => void;
  ttsBusy: boolean;
}) {
  const [remembering, setRemembering] = useState(false);
  const handleRemember = async () => {
    if (!onRemember) return;
    setRemembering(true);
    try {
      await onRemember();
    } finally {
      setRemembering(false);
    }
  }
  if (!turn.provider && !turn.error) {
    // Stream is still mid-flight and has not told us who answered yet; render
    // a careful placeholder so the bubble still has a recognisable shape.
    return (
      <div className="mt-2 text-[11px] text-ink-500">retrieving and routing…</div>
    );
  }
  return (
    <div className="mt-2 flex flex-wrap items-center gap-2 text-[11px] text-ink-400">
      <span className={`chip ${policy?.color === 'sage' ? 'chip-sage' : policy?.color === 'amber' ? 'chip-amber' : policy?.color === 'rose' ? 'chip-rose' : 'chip-violet'}`}>
        {turn.escalated ? <ShieldOff size={10} /> : <ShieldCheck size={10} />}
        {turn.provider} {policy ? `· ${policy.label}` : ''}
      </span>
      {turn.retrievalMatched != null ? (
        <span className="font-mono text-[10px] uppercase tracking-wide text-ink-500">
          {turn.retrievalMatched} matches · {turn.retrievalMs?.toFixed(0)}ms
          {turn.reranked ? ' · reranked' : ''}
        </span>
      ) : null}
      <button
        type="button"
        onClick={ttsBusy ? onStop : onPlay}
        disabled={!turn.answer}
        className="inline-flex items-center gap-1 rounded-full border border-ink-700/70 bg-ink-800/50 px-2 py-0.5 text-[11px] text-ink-200 transition hover:border-violet-500/60 hover:text-violet-200 disabled:opacity-50 focus-ring"
        title={ttsBusy ? 'Stop reading' : 'Read aloud'}
      >
        {ttsBusy ? <PauseCircle size={11} /> : <PlayCircle size={11} />}
        {ttsBusy ? 'Stop' : 'Read'}
      </button>
      <button
        type="button"
        onClick={handleRemember}
        disabled={remembering || !!turn.memorySaved || !turn.answer || !onRemember}
        className="inline-flex items-center gap-1 rounded-full border border-ink-700/70 bg-ink-800/50 px-2 py-0.5 text-[11px] text-ink-200 transition hover:border-amber-500/60 hover:text-amber-200 disabled:opacity-50 focus-ring"
        title={turn.memorySaved ? 'Saved to Memory/' : 'Save this answer to Memory/'}
      >
        <Save size={11} /> {turn.memorySaved ? 'Saved' : remembering ? 'Saving…' : 'Remember'}
      </button>
    </div>
  );
}

export type { MessageBubbleProps };
