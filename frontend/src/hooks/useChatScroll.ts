import { useCallback, useEffect, useState } from 'react';
import type { RefObject } from 'react';

/**
 * Pin-to-bottom behaviour for a chat thread. Tracks whether the user has
 * scrolled *away* from the bottom; if they have, we leave their position
 * alone. When they are near the bottom (within ~120 px) we automatically
 * follow new tokens.
 */
export function useChatScroll(
  ref: RefObject<HTMLDivElement | null>,
  messageCount: number,
) {
  const [atBottom, setAtBottom] = useState(true);

  const onScroll = useCallback(() => {
    const el = ref.current;
    if (!el) return;
    const distance = el.scrollHeight - el.scrollTop - el.clientHeight;
    setAtBottom(distance < 120);
  }, [ref]);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.addEventListener('scroll', onScroll, { passive: true });
    return () => el.removeEventListener('scroll', onScroll);
  }, [onScroll, ref]);

  const scrollToBottom = useCallback(
    (behavior: ScrollBehavior = 'smooth') => {
      const el = ref.current;
      if (!el) return;
      el.scrollTo({ top: el.scrollHeight, behavior });
    },
    [ref],
  );

  // Whenever a new message is appended we scroll once -- only if the user
  // was already at the bottom, so their position is respected if they had
  // scrolled up to read older context.
  useEffect(() => {
    if (atBottom) scrollToBottom('auto');
    // We depend on messageCount rather than the messages list to avoid
    // triggering on every text-delta re-render; one scroll per turn.
  }, [messageCount, atBottom, scrollToBottom]);

  return { atBottom, scrollToBottom };
}
