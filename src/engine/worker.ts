// Render worker entry. All engine logic lives in host.ts; this file only wires messages through.
import { createHost } from './host';
import type { MainToWorker } from './types';

const h = createHost((m) => postMessage(m));
onmessage = (e: MessageEvent<MainToWorker>) => h(e.data);
