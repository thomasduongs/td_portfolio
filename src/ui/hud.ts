// ?bench HUD overlay: fixed top-left monospace panel with live engine stats, plus a JSON download for bench reports.
import type { Stats } from '../engine/types';

const PANEL_ID = 'hud';

interface Panel { stats: HTMLPreElement; report: HTMLDivElement }

function ensurePanel(): Panel {
  let el = document.getElementById(PANEL_ID);
  if (!(el instanceof HTMLDivElement)) {
    el = document.createElement('div');
    el.id = PANEL_ID;
    el.setAttribute('aria-hidden', 'true');
    document.body.appendChild(el);
  }
  let stats = el.querySelector<HTMLPreElement>('pre.hud-stats');
  if (!stats) {
    stats = document.createElement('pre');
    stats.className = 'hud-stats';
    el.appendChild(stats);
  }
  let report = el.querySelector<HTMLDivElement>('div.hud-report');
  if (!report) {
    report = document.createElement('div');
    report.className = 'hud-report';
    el.appendChild(report);
  }
  return { stats, report };
}

const fmt = (v: number, digits = 2): string => (v < 0 ? 'n/a' : v.toFixed(digits));

function formatStats(s: Stats): string {
  const passes = Object.entries(s.passes).map(([k, v]) => `  ${k.padEnd(11)} ${fmt(v)} ms`);
  return [
    `fps        ${s.fps.toFixed(1)}`,
    `cpu        ${fmt(s.cpuMs)} ms`,
    `gpu        ${fmt(s.gpuMs)} ms`,
    ...passes,
    `p50/95/99  ${fmt(s.p50)} / ${fmt(s.p95)} / ${fmt(s.p99)} ms`,
    `dropped    ${s.dropped}`,
    `latency    ${fmt(s.inputLatencyMs)} ms`,
    `level      ${s.level}  scale ${s.renderScale.toFixed(2)}`,
    `refresh    ${s.refreshHz} Hz`,
    `backend    ${s.backend}`,
  ].join('\n');
}

/** Returns an updater to call with each `stats` message (4 Hz). */
export function mountHud(): (s: Stats) => void {
  const { stats } = ensurePanel();
  return (s) => { stats.textContent = formatStats(s); };
}

/** Adds a download link for a bench report to the HUD panel. */
export function offerReport(report: unknown): void {
  const { report: container } = ensurePanel();
  const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'prism-bench.json';
  a.textContent = 'download bench report (JSON)';
  container.replaceChildren(a);
}
