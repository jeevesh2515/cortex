/**
 * Browser SpeechSynthesis wrapper for the assistant "TTS play" button on
 * each message.
 *
 * Browser TTS is free, local-ish (some browsers do fetch voice data), and
 * zero install. Where the chrome is poor (older voices, robotic defaults)
 * we pick the best match from the available list rather than accepting
 * whatever the OS hands us. The actual quality lives at the OS level; this
 * hook just keeps the surface predictable.
 */

import { useCallback, useEffect, useRef, useState } from 'react';

export interface UseSpeechSynthesis {
  available: boolean;
  speaking: boolean;
  voices: Array<SpeechSynthesisVoice>;
  speak: (text: string, opts?: { voiceURI?: string; rate?: number }) => void;
  cancel: () => void;
}

export function useSpeechSynthesis(): UseSpeechSynthesis {
  const [available, setAvailable] = useState(false);
  const [speaking, setSpeaking] = useState(false);
  const [voices, setVoices] = useState<Array<SpeechSynthesisVoice>>([]);
  const utteranceRef = useRef<SpeechSynthesisUtterance | null>(null);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    if (!window.speechSynthesis || !window.SpeechSynthesisUtterance) {
      setAvailable(false);
      return;
    }
    setAvailable(true);
    const refresh = () => {
      const list = window.speechSynthesis?.getVoices() ?? [];
      setVoices([...list]);
    };
    refresh();
    // Some browsers (Chromium) populate ``getVoices`` asynchronously. The
    // ``voiceschanged`` event fires once that initial fetch resolves.
    window.speechSynthesis.onvoiceschanged = refresh;
    return () => {
      window.speechSynthesis?.cancel();
      if (window.speechSynthesis) {
        window.speechSynthesis.onvoiceschanged = null;
      }
    };
  }, []);

  const speak = useCallback(
    (text: string, opts?: { voiceURI?: string; rate?: number }) => {
      if (!available || typeof window === 'undefined' || !window.speechSynthesis) return;
      const trimmed = text.trim();
      if (!trimmed) return;
      const synth = window.speechSynthesis;
      const UtteranceCtor = window.SpeechSynthesisUtterance;
      if (!UtteranceCtor) return;
      synth.cancel();
      const u = new UtteranceCtor(trimmed);
      let voice = voices.find((v) => v.voiceURI === opts?.voiceURI) ?? null;
      if (!voice) {
        const userLang = navigator.language || 'en-US';
        voice =
          voices.find((v) => v.lang === userLang) ??
          voices.find((v) => v.lang.startsWith(userLang.split('-')[0])) ??
          voices[0] ??
          null;
      }
      if (voice) u.voice = voice;
      u.lang = voice?.lang ?? navigator.language ?? 'en-US';
      u.rate = opts?.rate ?? 1.0;
      u.pitch = 1.0;
      u.volume = 1.0;
      u.onstart = () => setSpeaking(true);
      u.onend = () => setSpeaking(false);
      u.onerror = () => setSpeaking(false);
      utteranceRef.current = u;
      synth.speak(u);
    },
    [available, voices],
  );

  const cancel = useCallback(() => {
    if (!available || typeof window === 'undefined') return;
    window.speechSynthesis?.cancel();
    setSpeaking(false);
  }, [available]);

  return { available, speaking, voices, speak, cancel };
}
