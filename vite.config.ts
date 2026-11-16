import { defineConfig } from 'vite';

// Cross-origin isolation enables high-resolution performance.now() (5µs vs 100µs)
// for the profiler and SharedArrayBuffer if we ever move input to shared memory.
const isolation = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
};

export default defineConfig({
  server: { headers: isolation },
  preview: { headers: isolation },
  worker: { format: 'es' },
  build: {
    target: 'es2022',
    modulePreload: { polyfill: false },
    assetsInlineLimit: 0,
  },
  optimizeDeps: { exclude: ['@dimforge/rapier3d-compat'] },
});
