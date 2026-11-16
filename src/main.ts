// Main thread: tier detection, render worker (or main-thread fallback), input forwarding, resize, HUD, CSS fallback.
import type { DeviceTier, MainToWorker, WorkerToMain } from './engine/types';
import { mountHud, offerReport } from './ui/hud';

const canvas = document.getElementById('gfx') as HTMLCanvasElement;
const body = document.body;
const params = new URLSearchParams(location.search);
const bench: string | null = params.has('bench') ? params.get('bench') ?? '' : null;
const hud = bench !== null ? mountHud() : null;
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;

let send: ((m: MainToWorker) => void) | null = null;
let ready = false;

function detectTier(): DeviceTier {
  const uaMobile = (navigator as Navigator & { userAgentData?: { mobile: boolean } }).userAgentData?.mobile;
  if (uaMobile || matchMedia('(pointer: coarse)').matches) return 'mobile';
  const mem = (navigator as Navigator & { deviceMemory?: number }).deviceMemory;
  return mem !== undefined && mem <= 8 ? 'laptop' : 'desktop';
}

function onEngineMessage(m: WorkerToMain): void {
  switch (m.type) {
    case 'ready':
      ready = true;
      body.classList.add('ready');
      break;
    case 'stats':
      hud?.(m.stats);
      break;
    case 'bench-report':
      console.log('[bench]', m.report);
      offerReport(m.report);
      break;
    case 'error':
      console.error(m.message);
      // Before ready, any failure means no engine: show the CSS fallback. After ready, just log.
      if (!ready) body.classList.add('nogpu');
      break;
  }
}

function post(m: MainToWorker): void {
  send?.(m);
}

function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

function pointerAt(e: PointerEvent): { x: number; y: number } {
  const r = canvas.getBoundingClientRect();
  return { x: clamp01((e.clientX - r.left) / r.width), y: clamp01((e.clientY - r.top) / r.height) };
}

let down = false;
function sendPointer(e: PointerEvent, click: boolean): void {
  const { x, y } = pointerAt(e);
  post({ type: 'pointer', x, y, down, click, t: performance.timeOrigin + e.timeStamp });
}

function wireInput(): void {
  window.addEventListener('pointermove', (e) => sendPointer(e, false), { passive: true });
  window.addEventListener('pointerdown', (e) => { down = true; sendPointer(e, true); }, { passive: true });
  window.addEventListener('pointerup', (e) => { down = false; sendPointer(e, false); }, { passive: true });
  window.addEventListener('pointercancel', (e) => { down = false; sendPointer(e, false); }, { passive: true });

  new ResizeObserver(() => {
    const r = canvas.getBoundingClientRect();
    post({ type: 'resize', width: r.width, height: r.height, dpr: devicePixelRatio });
  }).observe(canvas);

  document.addEventListener('visibilitychange', () => post({ type: 'visibility', hidden: document.hidden }));
}

async function start(): Promise<void> {
  const r = canvas.getBoundingClientRect();
  const common = { width: r.width, height: r.height, dpr: devicePixelRatio, tier: detectTier(), bench, reducedMotion };

  // Preferred: engine in a render worker with the canvas transferred off the main thread.
  if ('transferControlToOffscreen' in canvas && typeof Worker !== 'undefined') {
    let worker: Worker | null = null;
    try {
      // Create the worker before transferring the canvas, so a failure here keeps the canvas usable.
      worker = new Worker(new URL('./engine/worker.ts', import.meta.url), { type: 'module' });
    } catch (e) {
      console.warn('Render worker unavailable, running on main thread:', e);
    }
    if (worker) {
      const w = worker;
      w.onmessage = (e: MessageEvent<WorkerToMain>) => onEngineMessage(e.data);
      w.onerror = (e) => { console.error(e.message); if (!ready) body.classList.add('nogpu'); };
      const offscreen = canvas.transferControlToOffscreen();
      const init: MainToWorker = { type: 'init', canvas: offscreen, ...common };
      w.postMessage(init, [offscreen]);
      send = (m) => w.postMessage(m);
      return;
    }
  }

  // Fallback: same host, same messages, on the main thread with the HTMLCanvasElement.
  const { createHost } = await import('./engine/host');
  const host = createHost(onEngineMessage);
  send = (m) => host(m);
  host({ type: 'init', canvas, ...common });
}

wireInput();
start().catch((e) => {
  console.error(e);
  body.classList.add('nogpu');
});
