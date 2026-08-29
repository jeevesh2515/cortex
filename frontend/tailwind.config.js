/** @type {import('tailwindcss').Config} */

// Cortex palette: deep neutrals for a "console" feel that does not fatigue
// during long sessions, with a small set of accent hues that surface state
// (provider badge, citation, error, memory) without shouting.
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        ink: {
          50: '#f6f5f7',
          100: '#e9e7ec',
          200: '#c8c4d2',
          300: '#9c95ad',
          400: '#6f6783',
          500: '#524a66',
          600: '#3d3650',
          700: '#2c2640',
          800: '#1d1930',
          900: '#100d20',
          950: '#0a0816',
        },
        violet: {
          50: '#f3f0ff',
          100: '#e8e2ff',
          200: '#d2c5ff',
          300: '#b39eff',
          400: '#8e6dff',
          500: '#6f3cff',
          600: '#5a25e8',
          700: '#4a1dbf',
          800: '#3c1899',
          900: '#2c126e',
        },
        sage: {
          400: '#7ad0a8',
          500: '#4fbf87',
          600: '#3aa66e',
        },
        amber: {
          400: '#f4b967',
          500: '#e89e3a',
        },
        rose: {
          400: '#f06b94',
          500: '#dc467a',
        },
      },
      fontFamily: {
        sans: [
          'Inter',
          '-apple-system',
          'BlinkMacSystemFont',
          'SF Pro Text',
          'Helvetica Neue',
          'sans-serif',
        ],
        mono: ['JetBrains Mono', 'SF Mono', 'Menlo', 'monospace'],
      },
      boxShadow: {
        glow: '0 0 32px -8px rgba(143, 110, 255, 0.45)',
        ring: '0 0 0 1px rgba(143, 110, 255, 0.25), 0 0 24px -4px rgba(143, 110, 255, 0.35)',
      },
      backgroundImage: {
        'radial-fade':
          'radial-gradient(ellipse at top, rgba(143, 110, 255, 0.18), transparent 60%), radial-gradient(ellipse at bottom, rgba(74, 29, 191, 0.12), transparent 50%)',
      },
      keyframes: {
        pulse_ring: {
          '0%': { transform: 'scale(0.8)', opacity: '0.7' },
          '70%': { transform: 'scale(1.4)', opacity: '0' },
          '100%': { transform: 'scale(0.8)', opacity: '0' },
        },
        breathe: {
          '0%, 100%': { opacity: '0.5' },
          '50%': { opacity: '1' },
        },
      },
      animation: {
        pulse_ring: 'pulse_ring 1.6s ease-out infinite',
        breathe: 'breathe 2.4s ease-in-out infinite',
      },
    },
  },
  plugins: [],
};
