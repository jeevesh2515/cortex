import { motion } from 'framer-motion';
import { ChevronDown } from 'lucide-react';

interface ScrollToBottomButtonProps {
  onClick: () => void;
}

export function ScrollToBottomButton({ onClick }: ScrollToBottomButtonProps) {
  return (
    <motion.button
      type="button"
      onClick={onClick}
      initial={{ opacity: 0, scale: 0.92 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.9 }}
      className="sticky bottom-32 left-1/2 -translate-x-1/2 rounded-full border border-ink-700/70 bg-ink-900/90 px-3 py-1.5 text-xs text-ink-200 shadow-glow backdrop-blur transition hover:border-violet-500/60 hover:text-violet-200 focus-ring"
      aria-label="Scroll to bottom"
    >
      <span className="inline-flex items-center gap-1.5">
        <ChevronDown size={12} /> Latest
      </span>
    </motion.button>
  );
}
