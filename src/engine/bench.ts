// Optional benchmark modes (enabled with ?bench=...). Mode '' or '1' is HUD-only adaptive quality.
//  'auto'    : force each level 0..4, warm 1.5 s, record 4 s, report, then resume adaptive.
//  'sustain' : adaptive on; sample fps/p95/level every 10 s; report after 10 min.
import type { FrameStats, GpuProfiler } from './profiler';
import type { QualityController } from './quality';

const AUTO_WARM_MS = 1500;
const AUTO_RECORD_MS = 4000;
const SUSTAIN_PUSH_MS = 10_000;
const SUSTAIN_TOTAL_MS = 600_000;
const MAX_LEVEL = 4;

interface LevelResult {
  level: number;
  p50: number; p95: number; p99: number;
  gpuMs: number;
  passes: Record<string, number>;
  dropped: number;
}

interface SustainSample { t: number; fps: number; p95: number; level: number; gpuMs: number }

export class BenchRunner {
  private mode: 'auto' | 'sustain' | 'hud';

  // auto state
  private autoLevel = 0;
  private autoDone = false;
  private forcePending = true;   // next frame forces autoLevel
  private phaseStart = 0;        // when autoLevel was forced (warm-up starts here)
  private recStart = -1;         // when recording started (-1 = still warming)
  private intervals: number[] = [];
  private gpuSamples: number[] = [];
  private passSamples: Record<string, number[]> = {};
  private results: LevelResult[] = [];

  // sustain state
  private t0 = -1;
  private nextPush = 0;
  private samples: SustainSample[] = [];
  private sustainDone = false;

  constructor(
    mode: string,
    private qc: QualityController,
    private report: (r: unknown) => void,
  ) {
    this.mode = mode === 'auto' || mode === 'sustain' ? mode : 'hud';
  }

  /** Called once per frame. Returns true when the quality settings changed this frame. */
  onFrame(now: number, stats: FrameStats, prof: GpuProfiler): boolean {
    if (this.mode === 'auto' && !this.autoDone) return this.autoFrame(now, stats, prof);
    if (this.mode === 'sustain' && !this.sustainDone) return this.sustainFrame(now, stats, prof);
    return this.qc.update(stats.lastInterval, prof.gpuMs, stats.refreshHz);
  }

  private autoFrame(now: number, stats: FrameStats, prof: GpuProfiler): boolean {
    if (this.forcePending) {
      this.forcePending = false;
      this.qc.force(this.autoLevel);
      this.phaseStart = now;
      this.recStart = -1;
      this.intervals = []; this.gpuSamples = []; this.passSamples = {};
      return true;
    }
    if (this.recStart < 0) {
      if (now - this.phaseStart >= AUTO_WARM_MS) this.recStart = now;
      return false;
    }
    if (now - this.recStart < AUTO_RECORD_MS) {
      this.intervals.push(stats.lastInterval);
      this.gpuSamples.push(prof.gpuMs);
      for (const [k, v] of Object.entries(prof.passMs)) (this.passSamples[k] ??= []).push(v);
      return false;
    }
    this.finishLevel(stats);
    if (this.autoLevel >= MAX_LEVEL) {
      this.autoDone = true;
      this.report({ levels: this.results, ua: navigator.userAgent, refreshHz: stats.refreshHz });
      console.table(this.results.map((r) => ({ level: r.level, p50: r.p50, p95: r.p95, p99: r.p99, gpuMs: r.gpuMs, dropped: r.dropped })));
      return false;
    }
    this.autoLevel++;
    this.forcePending = true;
    return false;
  }

  private finishLevel(stats: FrameStats): void {
    const v = this.intervals.slice().sort((a, b) => a - b);
    const budget = 1000 / stats.refreshHz;
    const passes: Record<string, number> = {};
    for (const [k, arr] of Object.entries(this.passSamples)) passes[k] = mean(arr);
    this.results.push({
      level: this.autoLevel,
      p50: pct(v, 0.5), p95: pct(v, 0.95), p99: pct(v, 0.99),
      gpuMs: mean(this.gpuSamples),
      passes,
      dropped: v.filter((x) => x > 1.5 * budget).length,
    });
  }

  private sustainFrame(now: number, stats: FrameStats, prof: GpuProfiler): boolean {
    if (this.t0 < 0) { this.t0 = now; this.nextPush = now + SUSTAIN_PUSH_MS; }
    const changed = this.qc.update(stats.lastInterval, prof.gpuMs, stats.refreshHz);
    if (now >= this.nextPush) {
      this.nextPush = now + SUSTAIN_PUSH_MS;
      const snap = stats.snapshot(prof, this.qc.settings, 'webgpu');
      this.samples.push({ t: Math.round((now - this.t0) / 1000), fps: snap.fps, p95: snap.p95, level: snap.level, gpuMs: snap.gpuMs });
    }
    if (now - this.t0 >= SUSTAIN_TOTAL_MS) {
      this.sustainDone = true;
      this.report({ samples: this.samples, ua: navigator.userAgent });
    }
    return changed;
  }
}

function mean(a: number[]): number {
  return a.length ? a.reduce((s, x) => s + x, 0) / a.length : 0;
}

function pct(sorted: number[], p: number): number {
  if (!sorted.length) return 0;
  return sorted[Math.min(sorted.length - 1, Math.floor(p * sorted.length))];
}
