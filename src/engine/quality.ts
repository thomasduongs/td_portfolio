// Adaptive quality: picks a preset level from measured frame cost. Decided on the CPU, never stalls on GPU.
import type { DeviceTier, QualitySettings } from './types';

type Preset = Omit<QualitySettings, 'level'>;

// Index = level. v3 laser-optics presets from ENGINE_SPEC.md (fields: bloomMips, reflections, glowScale, glassDispersion).
const PRESETS: Preset[] = [
  { bloomMips: 3, reflections: false, glowScale: 0.6, glassDispersion: false },
  { bloomMips: 4, reflections: true,  glowScale: 0.8, glassDispersion: false },
  { bloomMips: 5, reflections: true,  glowScale: 1.0, glassDispersion: true },
  { bloomMips: 6, reflections: true,  glowScale: 1.0, glassDispersion: true },
  { bloomMips: 6, reflections: true,  glowScale: 1.2, glassDispersion: true },
];

const MAX_LEVEL = 4;
const WINDOW_MIN = 10;

export class QualityController {
  settings: QualitySettings;

  private level: number;
  private readonly maxLevel: number;
  private metrics: number[] = [];
  private windowLen = 60;
  private settleLeft = 0;
  private downCount = 0;
  private upCount = 0;
  private stableWindows = 0;

  constructor(tier: DeviceTier) {
    this.level = tier === 'desktop' ? 3 : tier === 'laptop' ? 2 : 1;
    this.maxLevel = tier === 'mobile' ? 3 : MAX_LEVEL;
    this.settings = this.build(this.level);
  }

  /** Feed one frame. Returns true when the level (and therefore `settings`) changed. */
  update(intervalMs: number, gpuMs: number, refreshHz: number): boolean {
    this.windowLen = Math.max(WINDOW_MIN, Math.round(refreshHz));
    const budget = 1000 / refreshHz;
    const hasTimers = gpuMs >= 0;
    this.metrics.push(hasTimers ? gpuMs : intervalMs);
    if (this.metrics.length < this.windowLen) return false;

    const p90 = percentile90(this.metrics);
    this.metrics.length = 0;
    if (this.settleLeft > 0) { this.settleLeft--; return false; }

    // Without GPU timestamps the interval is vsync-quantized: step down only when clearly over budget,
    // step up only when clearly under and stable for a while.
    const downT = (hasTimers ? 0.92 : 1.3) * budget;
    const upT = (hasTimers ? 0.65 : 1.05) * budget;
    const upNeeded = hasTimers ? 4 : 6;

    this.stableWindows++;
    if (p90 > downT) { this.downCount++; this.upCount = 0; } else if (p90 < upT) { this.upCount++; this.downCount = 0; } else { this.downCount = 0; this.upCount = 0; }

    if (this.downCount >= (hasTimers ? 2 : 1) && this.level > 0) return this.setLevel(this.level - 1);
    if (this.upCount >= upNeeded && this.stableWindows >= upNeeded && this.level < this.maxLevel) return this.setLevel(this.level + 1);
    return false;
  }

  /** Force a level (benchmark). Resets adaptive state and skips the next evaluation window. */
  force(level: number): void {
    this.setLevel(Math.max(0, Math.min(this.maxLevel, level)));
  }

  private setLevel(level: number): boolean {
    this.level = level;
    this.settings = this.build(level);
    this.metrics.length = 0;
    this.downCount = 0; this.upCount = 0; this.stableWindows = 0;
    this.settleLeft = 1;
    return true;
  }

  private build(level: number): QualitySettings {
    return { level, ...PRESETS[level] };
  }
}

function percentile90(values: number[]): number {
  const v = values.slice().sort((a, b) => a - b);
  return v[Math.min(v.length - 1, Math.floor(0.9 * v.length))];
}
