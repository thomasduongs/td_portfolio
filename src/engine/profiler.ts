// GPU pass timing (timestamp queries, 3-deep readback ring, never stalls) and frame-interval statistics.
import type { Caps, QualitySettings, Stats, TS } from './types';

const RING = 3;
const INTERVALS = 240;
const REFRESH_LIST = [60, 75, 90, 120, 144, 165, 240];

export class GpuProfiler {
  /** -1 when timestamp queries are unsupported. */
  gpuMs = -1;
  passMs: Record<string, number> = {};

  private querySet: GPUQuerySet | null = null;
  private resolveBuf: GPUBuffer | null = null;
  private free: GPUBuffer[] = [];
  private inFlight: GPUBuffer | null = null;
  private readonly bytes: number;

  constructor(device: GPUDevice, caps: Caps, private passNames: string[]) {
    this.bytes = passNames.length * 2 * 8;
    for (const n of passNames) this.passMs[n] = 0;
    if (!caps.timestamps) return;
    this.gpuMs = 0;
    this.querySet = device.createQuerySet({ type: 'timestamp', count: passNames.length * 2 });
    this.resolveBuf = device.createBuffer({ label: 'ts-resolve', size: this.bytes, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    for (let i = 0; i < RING; i++) {
      this.free.push(device.createBuffer({ label: 'ts-readback-' + i, size: this.bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }));
    }
  }

  ts(name: string): TS {
    if (!this.querySet) return undefined;
    const i = this.passNames.indexOf(name);
    if (i < 0) return undefined;
    return { querySet: this.querySet, beginningOfPassWriteIndex: 2 * i, endOfPassWriteIndex: 2 * i + 1 };
  }

  /** Call once per frame before submit. Skips (never waits) when all readback buffers are busy. */
  resolve(enc: GPUCommandEncoder): void {
    if (!this.querySet || !this.resolveBuf || this.inFlight) return;
    const buf = this.free.pop();
    if (!buf) return;
    enc.resolveQuerySet(this.querySet, 0, this.passNames.length * 2, this.resolveBuf, 0);
    enc.copyBufferToBuffer(this.resolveBuf, 0, buf, 0, this.bytes);
    this.inFlight = buf;
  }

  /** Call right after queue.submit. Maps the buffer resolved this frame and reads it asynchronously. */
  afterSubmit(): void {
    const buf = this.inFlight;
    if (!buf) return;
    this.inFlight = null;
    buf.mapAsync(GPUMapMode.READ).then(() => {
      const t = new BigInt64Array(buf.getMappedRange().slice(0));
      buf.unmap();
      this.free.push(buf);
      this.readTimes(t);
    }).catch(() => { this.free.push(buf); });
  }

  private readTimes(t: BigInt64Array): void {
    for (let i = 0; i < this.passNames.length; i++) {
      const b = t[2 * i], e = t[2 * i + 1];
      if (b === 0n && e === 0n) continue; // pass not written this frame
      const d = Number(e - b) / 1e6; // ns -> ms
      if (d < 0 || d > 1000) continue; // garbage from the driver
      const name = this.passNames[i];
      this.passMs[name] = this.passMs[name] * 0.9 + d * 0.1;
    }
    let sum = 0;
    for (const n of this.passNames) sum += this.passMs[n];
    this.gpuMs = sum;
  }
}

export class FrameStats {
  refreshHz = 60;
  lastInterval = 16.7;

  private ring = new Float64Array(INTERVALS);
  private n = 0; private head = 0; private count = 0;
  private cpuEma = 0; private lastLatency = 0;

  /** Frame-to-frame interval in ms. */
  interval(ms: number): void {
    this.lastInterval = ms;
    this.ring[this.head] = ms;
    this.head = (this.head + 1) % INTERVALS;
    if (this.n < INTERVALS) this.n++;
    if (++this.count % 30 === 0) this.refreshHz = this.estimateHz();
  }

  cpu(ms: number): void { this.cpuEma = this.cpuEma === 0 ? ms : this.cpuEma * 0.9 + ms * 0.1; }

  inputLatency(ms: number): void { this.lastLatency = ms; }

  snapshot(prof: GpuProfiler, q: QualitySettings, backend: 'webgpu' | 'webgl2'): Stats {
    const v = this.sorted();
    const mean = v.length ? v.reduce((a, b) => a + b, 0) / v.length : 16.7;
    const budget = 1000 / this.refreshHz;
    return {
      fps: 1000 / mean,
      cpuMs: this.cpuEma,
      gpuMs: prof.gpuMs,
      passes: { ...prof.passMs },
      p50: pct(v, 0.5), p95: pct(v, 0.95), p99: pct(v, 0.99),
      dropped: v.filter((x) => x > 1.5 * budget).length,
      inputLatencyMs: this.lastLatency,
      level: q.level,
      renderScale: 1, // v3: everything renders at output resolution
      refreshHz: this.refreshHz,
      backend,
    };
  }

  /** Sorted copy of the ring (only used at snapshot / hz-estimate time). */
  private sorted(): number[] {
    const out: number[] = [];
    for (let i = 0; i < this.n; i++) out.push(this.ring[i]);
    return out.sort((a, b) => a - b);
  }

  private estimateHz(): number {
    const v = this.sorted();
    if (v.length < 10) return this.refreshHz;
    const hz = 1000 / pct(v, 0.5);
    let best = REFRESH_LIST[0];
    for (const r of REFRESH_LIST) if (Math.abs(r - hz) < Math.abs(best - hz)) best = r;
    return best;
  }
}

function pct(sorted: number[], p: number): number {
  if (!sorted.length) return 0;
  return sorted[Math.min(sorted.length - 1, Math.floor(p * sorted.length))];
}
