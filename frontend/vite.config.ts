import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Vite dev server proxies API calls to the FastAPI backend at 127.0.0.1:7331.
// In production ``cortex serve`` mounts the built dist/ on / so the proxy is
// unused -- but having it here makes ``npm run dev`` immediately useful even
// before the build is wired up.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:7331',
        changeOrigin: true,
        // SSE needs raw passthrough -- the default stream buffering breaks
        // EventSource if there's any handle on the response.
        ws: false,
        configure: (proxy) => {
          proxy.on('proxyReq', (proxyReq) => {
            proxyReq.setHeader('Connection', 'keep-alive');
          });
        },
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
    target: 'es2022',
  },
});
