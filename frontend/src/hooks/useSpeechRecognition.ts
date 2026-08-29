/**
 * Push-to-talk speech recognition via the browser Web Speech API.
 *
 * Why push-to-talk and not always-on listening:
 *  - Always-on requires fine-tuning VAD and permissions; a simple hold-to-
 *    record button works first time and is what most chat apps use.
 *  - Always-on STT must keep a microphone open even when the user is not
 *    speaking, which costs battery and creates a real privacy surface.
 *
 * Caveats the caller needs to know:
 *  - Chromium and Safari ship SpeechRecognition; Firefox does not. The hook
 *    reports ``available: false`` in that case so the UI can hide the mic.
 *  - Where it does work, the actual recognition may be cloud STT (Chrome
 *    uses Google's; Safari uses Apple's). The component surfaces this fact
 *    in the settings panel so the user is not confused about what "local" means.
 */

import { useCallback, useEffect, useRef, useState } from 'react';

interface SpeechRecognitionResult {
  isFinal: boolean;
  0: { transcript: string };
}

interface SpeechRecognitionEventLike {
  results: ArrayLike<SpeechRecognitionResult>;
  resultIndex: number;
}

interface SpeechRecognitionLike extends EventTarget {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  start(): void;
  stop(): void;
  abort(): void;
  onresult: ((ev: SpeechRecognitionEventLike) => void) | null;
  onerror: ((ev: { error: string; message?: string }) => void) | null;
  onend: (() => void) | null;
  onstart: (() => void) | null;
}

type SpeechRecognitionCtor = new () => SpeechRecognitionLike;

declare global {
  interface Window {
    SpeechRecognition?: SpeechRecognitionCtor;
    webkitSpeechRecognition?: SpeechRecognitionCtor;
  }
}

export interface UseSpeechRecognition {
  available: boolean;
  listening: boolean;
  interim: string;
  error: string | null;
  start: () => void;
  stop: () => void;
  abort: () => void;
}

export function useSpeechRecognition(opts?: {
  lang?: string;
  onFinal?: (text: string) => void;
}): UseSpeechRecognition {
  const lang = opts?.lang ?? (typeof navigator !== 'undefined' ? navigator.language : 'en-US');
  const onFinal = opts?.onFinal;
  const [available, setAvailable] = useState(false);
  const [listening, setListening] = useState(false);
  const [interim, setInterim] = useState('');
  const [error, setError] = useState<string | null>(null);
  const recognitionRef = useRef<SpeechRecognitionLike | null>(null);
  const onFinalRef = useRef(onFinal);
  onFinalRef.current = onFinal;

  useEffect(() => {
    const Ctor = typeof window !== 'undefined'
      ? (window.SpeechRecognition ?? window.webkitSpeechRecognition)
      : undefined;
    setAvailable(Boolean(Ctor));
    return () => {
      try {
        recognitionRef.current?.abort();
      } catch {
        // ignore
      }
    };
  }, []);

  const start = useCallback(() => {
    if (!available) return;
    if (listening) return;
    const Ctor = window.SpeechRecognition ?? window.webkitSpeechRecognition;
    if (!Ctor) return;
    setError(null);
    setInterim('');
    const rec = new Ctor();
    rec.continuous = false;
    rec.interimResults = true;
    rec.lang = lang;
    rec.onstart = () => setListening(true);
    rec.onresult = (ev) => {
      let interimText = '';
      let finalText = '';
      for (let i = ev.resultIndex; i < ev.results.length; i++) {
        const r = ev.results[i];
        if (r.isFinal) finalText += r[0].transcript;
        else interimText += r[0].transcript;
      }
      if (interimText) setInterim(interimText);
      if (finalText) {
        setInterim('');
        onFinalRef.current?.(finalText.trim());
      }
    };
    rec.onerror = (ev) => {
      setError(ev.error + (ev.message ? `: ${ev.message}` : ''));
      setListening(false);
    };
    rec.onend = () => {
      setListening(false);
      setInterim('');
    };
    recognitionRef.current = rec;
    rec.start();
  }, [available, lang, listening]);

  const stop = useCallback(() => {
    try {
      recognitionRef.current?.stop();
    } catch {
      // ignore
    }
  }, []);

  const abort = useCallback(() => {
    try {
      recognitionRef.current?.abort();
    } catch {
      // ignore
    }
    setListening(false);
    setInterim('');
  }, []);

  return { available, listening, interim, error, start, stop, abort };
}
