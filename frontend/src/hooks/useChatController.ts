/**
 * Chat controller hook -- the brain of the chat pane.
 * Owns the message list, the in-flight SSE stream, and the local-only toggle
 * for each turn. Components mount one ``useChatController`` and read the
 * returned state.
 *
 * The hook intentionally does NOT import React state machinery like
 * useReducer -- the messages are append-only during a running turn, so a
 * rolling bag of small setState calls is clearer than a reducer.
 */

import { useCallback, useRef, useState } from 'react';

import { openChatStream, rememberExchange } from '../api/client';
import type { ChatEvent, ChatMessage, Citation } from '../types';

export interface ChatTurn {
  id: string;
  question: string;
  answer: string;
  citations: Citation[];
  provider?: string;
  policy?: string;
  escalated?: boolean;
  retrievalMs?: number;
  retrievalMatched?: number;
  reranked?: boolean;
  dateWindow?: string | null;
  error?: { type?: string; message: string };
  memorySaved?: string;
  done: boolean;
}

export interface ChatState {
  messages: ChatTurn[];
  streaming: boolean;
  aborted: boolean;
  localOnly: boolean;
  voiceAutoSend: boolean;
  canSpeak: boolean;
}

export interface ChatController {
  state: ChatState;
  lastCitations: Citation[];
  send: (prompt: string, opts?: { remember?: boolean }) => Promise<void>;
  stop: () => void;
  clear: () => void;
  setLocalOnly: (v: boolean) => void;
  setVoiceAutoSend: (v: boolean) => void;
  remember: (turn: ChatTurn) => Promise<void>;
}

const STARTER: ChatState = {
  messages: [],
  streaming: false,
  aborted: false,
  localOnly: false,
  voiceAutoSend: false,
  canSpeak: typeof window !== 'undefined' && Boolean(window.speechSynthesis),
};

export function useChatController(): ChatController {
  const [state, setState] = useState<ChatState>(STARTER);
  const controllerRef = useRef<AbortController | null>(null);

  const patch = useCallback((p: Partial<ChatState>) => {
    setState((prev) => ({ ...prev, ...p }));
  }, []);

  const stop = useCallback(() => {
    controllerRef.current?.abort();
    controllerRef.current = null;
    patch({ streaming: false, aborted: true });
    // Anchor the last turn as ``done`` so its stream text stays where it is
    // rather than appearing to disappear mid-assist.
    setState((s) => ({
      ...s,
      messages: s.messages.map((m, i) =>
        i === s.messages.length - 1 ? { ...m, done: true } : m,
      ),
    }));
  }, [patch]);

  const send = useCallback(
    async (prompt: string, opts?: { remember?: boolean }) => {
      const text = prompt.trim();
      if (!text) return;
      const id = `t-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
      const placeholder: ChatTurn = {
        id,
        question: text,
        answer: '',
        citations: [],
        done: false,
      };

      // Build the wire-format history (last 8 turns is plenty; older context
      // belongs on disk in a memory note if it's still relevant).
      const history: ChatMessage[] = state.messages.flatMap((turn) => [
        { role: 'user' as const, content: turn.question },
        { role: 'assistant' as const, content: turn.answer },
      ]);
      history.push({ role: 'user', content: text });

      setState((s) => ({
        ...s,
        messages: [...s.messages, placeholder],
        streaming: true,
        aborted: false,
      }));

      const ac = new AbortController();
      controllerRef.current = ac;

      const apply = (mut: (t: ChatTurn) => ChatTurn) =>
        setState((s) => ({
          ...s,
          messages: s.messages.map((m) => (m.id === id ? mut(m) : m)),
        }));

      try {
        for await (const event of openChatStream(
          {
            messages: history,
            local_only: state.localOnly,
            remember: opts?.remember ?? false,
            top_k: 8,
          },
          ac.signal,
        )) {
          applyEvent(event, apply);
          if (event.type === 'done' || event.type === 'error') break;
        }
      } catch (err) {
        if ((err as Error)?.name === 'AbortError') {
          // ``stop()`` already set state; this branch is the catch for an
          // observer that does not call stop().
        } else {
          apply((t) => ({
            ...t,
            error: {
              type: 'network',
              message: (err as Error)?.message ?? 'Stream failed',
            },
            done: true,
          }));
        }
      } finally {
        controllerRef.current = null;
        setState((s) => ({
          ...s,
          streaming: false,
          messages: s.messages.map((m) => (m.id === id ? { ...m, done: true } : m)),
        }));
      }
    },
    [state.localOnly, state.messages],
  );

  const clear = useCallback(() => {
    controllerRef.current?.abort();
    controllerRef.current = null;
    setState((s) => ({ ...s, messages: [], streaming: false, aborted: false }));
  }, []);

  const setLocalOnly = useCallback((v: boolean) => {
    setState((s) => ({ ...s, localOnly: v }));
  }, []);

  const setVoiceAutoSend = useCallback((v: boolean) => {
    setState((s) => ({ ...s, voiceAutoSend: v }));
  }, []);

  const remember = useCallback(async (turn: ChatTurn) => {
    try {
      const saved = await rememberExchange({
        question: turn.question,
        answer: turn.answer,
        sources: turn.citations.map((c) => c.note_id),
        provider: turn.provider,
      });
      setState((s) => ({
        ...s,
        messages: s.messages.map((m) =>
          m.id === turn.id ? { ...m, memorySaved: saved.saved } : m,
        ),
      }));
    } catch (err) {
      setState((s) => ({
        ...s,
        messages: s.messages.map((m) =>
          m.id === turn.id
            ? { ...m, error: { type: 'memory', message: (err as Error)?.message ?? 'remember failed' } }
            : m,
        ),
      }));
    }
  }, []);

  const lastCitations = state.messages[state.messages.length - 1]?.citations ?? [];

  return { state, lastCitations, send, stop, clear, setLocalOnly, setVoiceAutoSend, remember };
}

function applyEvent(
  event: ChatEvent,
  apply: (mut: (t: ChatTurn) => ChatTurn) => void,
): void {
  switch (event.type) {
    case 'provider':
      if (event.data.name) {
        apply((t) => ({
          ...t,
          provider: event.data.name,
          policy: event.data.policy ?? t.policy,
          escalated: event.data.escalated ?? t.escalated,
        }));
      }
      break;
    case 'retrieval':
      apply((t) => ({
        ...t,
        retrievalMs: event.data.elapsed_ms,
        retrievalMatched: event.data.matched,
        reranked: event.data.reranked,
        dateWindow: event.data.date_filter,
      }));
      break;
    case 'citation':
      apply((t) =>
        t.citations.find((c) => c.note_id === event.data.note_id)
          ? t
          : {
              ...t,
              citations: [
                ...t.citations,
                {
                  index: event.data.index,
                  note_id: event.data.note_id,
                  obsidian_uri: event.data.obsidian_uri,
                  title: event.data.title,
                  snippet: event.data.snippet,
                  score: event.data.score,
                  tags: event.data.tags ?? [],
                },
              ].sort((a, b) => a.index - b.index),
            },
      );
      break;
    case 'text':
      apply((t) => ({ ...t, answer: t.answer + event.data.delta }));
      break;
    case 'memory':
      apply((t) => ({ ...t, memorySaved: event.data.saved }));
      break;
    case 'done':
      apply((t) => ({
        ...t,
        answer: event.data.answer || t.answer,
        provider: event.data.provider || t.provider,
        policy: event.data.policy ?? t.policy,
        escalated: event.data.escalated ?? t.escalated,
        done: true,
      }));
      break;
    case 'error':
      apply((t) => ({
        ...t,
        error: {
          type: event.data.type,
          message: event.data.message,
        },
        done: true,
      }));
      break;
  }
}
