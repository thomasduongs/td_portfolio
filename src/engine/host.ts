// Engine host (v3, laser optics): owns the frame loop. Runs inside the render worker (OffscreenCanvas)
// or, when transferControlToOffscreen is unavailable, on the main thread with an HTMLCanvasElement.
// Per frame: LaserPass (beams, mirrors, glints, floor reflection) → copy → GlassPass (refraction) → Bloom → Composite.
import type { MainToWorker, WorkerToMain, EngineCtx, FrameTargets, Size, QualitySettings, OpticsBuffers } from './types';
import { FrameData, FRAME_UBO_SIZE, O, frameBindGroupLayout } from './frame';
import { m4, perspective, lookAt, mul, invert } from './math';
import { initDevice } from './device';
import { ResourcePool } from './resources';
import { GpuProfiler, FrameStats } from './profiler';
import { QualityController } from './quality';
import { BenchRunner } from './bench';
import { rasterizeText, type TextRaster } from './sdfbake';
import { extractGlyphs } from './optics/contours';
import { buildOptics } from './optics/route';
import { packOptics, type OpticsScene } from './optics/types';
import { LaserPass } from './passes/lasers';
import { GlassPass } from './passes/glass';
import { BloomPass } from './passes/bloom';
import { CompositePass } from './passes/composite';

type Post = (m: WorkerToMain) => void;
type InitMsg = Extract<MainToWorker, { type: 'init' }>;

const FOV_Y = (32 * Math.PI) / 180;
const MAX_OUT_PIXELS = 3_700_000;
const REVEAL_SPEED = 9;    // path units per second while the lasers trace the name
const PULSE_SPEED = 14;    // click pulse speed
const REVEAL_DELAY = 0.4;

export function createHost(post: Post): (msg: MainToWorker) => void {
  let engine: Engine | null = null;
  let fallback: { handle(m: MainToWorker): void } | null = null;
  const queued: MainToWorker[] = [];

  return (msg) => {
    if (msg.type === 'init') {
      start(msg).catch((e) => post({ type: 'error', message: String(e?.stack ?? e) }));
      return;
    }
    if (engine) engine.handle(msg);
    else if (fallback) fallback.handle(msg);
    else queued.push(msg);
  };

  async function start(init: InitMsg) {
    const gpu = init.forceWebGL2 ? null : await initDevice(init.canvas, init.tier).catch(() => null);
    if (gpu) {
      engine = await Engine.create(gpu, init, post);
      for (const m of queued) engine.handle(m);
      post({ type: 'ready', backend: 'webgpu' });
      return;
    }
    const { WebGL2Engine } = await import('../fallback/webgl2');
    fallback = await WebGL2Engine.create(init.canvas, init, post);
    if (!fallback) { post({ type: 'error', message: 'nogpu' }); return; }
    for (const m of queued) fallback.handle(m);
    post({ type: 'ready', backend: 'webgl2' });
  }
}

/** Build the optics scene for a text layout (shared with the WebGL2 fallback). */
export function layoutOptics(lines: string[]): { raster: TextRaster; scene: OpticsScene } {
  const raster = rasterizeText(lines);
  const glyphs = extractGlyphs(raster);
  const scene = buildOptics(glyphs, { glyphHalfW: raster.glyphHalfW, glyphHalfH: raster.glyphHalfH, floorY: raster.floorY, lines: lines.length });
  return { raster, scene };
}

export function linesFor(cssW: number, cssH: number): string[] { return cssW / cssH < 0.9 ? ['THOMAS', 'DUONG'] : ['THOMAS DUONG']; }

type Gpu = NonNullable<Awaited<ReturnType<typeof initDevice>>>;

class Engine {
  private frame = new FrameData();
  private lasers!: LaserPass;
  private glassPass!: GlassPass;
  private bloom!: BloomPass;
  private composite!: CompositePass;
  private targets!: FrameTargets;
  private size!: Size;
  private q!: QualitySettings;
  private frameIndex = 0;
  private hidden = false;
  private cssW: number; private cssH: number; private dpr: number;
  private lastT = 0; private time = 0;
  private pointer = { x: 0.5, y: 0.5, activity: 0, lastMove: -1e9, pendingT: 0 };
  private pulseStart = -1;
  private proj = m4(); private view = m4(); private vp = m4(); private inv = m4();
  private eye = [0, 0.5, 8]; private parallax = [0, 0];
  private stats = new FrameStats();
  private statsTimer = 0;
  private bench: BenchRunner | null;
  private device: GPUDevice;
  private ctx!: EngineCtx;
  private pool: ResourcePool;
  private prof: GpuProfiler;
  private qc: QualityController;
  private raster!: TextRaster;
  private scene!: OpticsScene;
  private lines: string[] = [];

  static async create(gpu: Gpu, init: InitMsg, post: Post): Promise<Engine> {
    const e = new Engine(gpu, init, post);
    await e.setup();
    return e;
  }

  private constructor(private gpu: Gpu, private init: InitMsg, private post: Post) {
    this.device = gpu.device;
    this.cssW = init.width; this.cssH = init.height; this.dpr = init.dpr;
    this.pool = new ResourcePool(this.device);
    this.qc = new QualityController(gpu.caps.tier);
    this.prof = new GpuProfiler(this.device, gpu.caps, ['lasers', 'glass', 'bloom', 'composite']);
    this.bench = init.bench ? new BenchRunner(init.bench, this.qc, (r) => post({ type: 'bench-report', report: r })) : null;
  }

  private async setup() {
    const d = this.device, caps = this.gpu.caps;
    const frameUBO = d.createBuffer({ label: 'frame-ubo', size: FRAME_UBO_SIZE, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const frameBGL = frameBindGroupLayout(d);
    const frameBG = d.createBindGroup({ layout: frameBGL, entries: [{ binding: 0, resource: { buffer: frameUBO } }] });
    const linearClamp = d.createSampler({ magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge' });
    this.lines = linesFor(this.cssW, this.cssH);
    this.ctx = { device: d, caps, frameUBO, frameBGL, frameBG, linearClamp, pool: this.pool, optics: this.buildOpticsBuffers(this.lines) };

    this.lasers = new LaserPass(this.ctx);
    this.glassPass = new GlassPass(this.ctx);
    this.bloom = new BloomPass(this.ctx);
    this.composite = new CompositePass(this.ctx, this.gpu.context);
    await Promise.all([this.lasers, this.glassPass, this.bloom, this.composite].map((p) => p.init()));
    if (this.init.forceLevel !== undefined) this.qc.force(this.init.forceLevel);
    this.q = this.qc.settings;
    this.reconfigure();
    this.loop();
  }

  /** Solve the optics for a layout and upload it (storage buffers, written once). */
  private buildOpticsBuffers(lines: string[]): OpticsBuffers {
    const { raster, scene } = layoutOptics(lines);
    this.raster = raster; this.scene = scene;
    const packed = packOptics(scene);
    const mk = (label: string, data: Float32Array) => {
      const b = this.device.createBuffer({ label, size: data.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      this.device.queue.writeBuffer(b, 0, data as Float32Array<ArrayBuffer>);
      return b;
    };
    return {
      segments: mk('optics-segments', packed.segments), mirrors: mk('optics-mirrors', packed.mirrors), glass: mk('optics-glass', packed.glass),
      numSegments: scene.segments.length, numMirrors: scene.mirrors.length, numGlass: scene.glass.length,
    };
  }

  handle(m: MainToWorker) {
    switch (m.type) {
      case 'resize': {
        this.cssW = m.width; this.cssH = m.height; this.dpr = m.dpr;
        const lines = linesFor(this.cssW, this.cssH);
        if (lines.length !== this.lines.length) {
          const old = this.ctx.optics;
          this.lines = lines;
          this.ctx.optics = this.buildOpticsBuffers(lines);
          old.segments.destroy(); old.mirrors.destroy(); old.glass.destroy();
        }
        this.reconfigure();
        break;
      }
      case 'pointer': {
        const p = this.pointer;
        if (Math.hypot(m.x - p.x, m.y - p.y) > 1e-4) p.lastMove = this.time;
        p.x = m.x; p.y = m.y; p.pendingT = m.t;
        if (m.click) this.pulseStart = this.time;
        break;
      }
      case 'visibility': this.hidden = m.hidden; if (!m.hidden) this.lastT = 0; break;
    }
  }

  private reconfigure() {
    let outW = Math.max(1, Math.round(this.cssW * Math.min(this.dpr, 2)));
    let outH = Math.max(1, Math.round(this.cssH * Math.min(this.dpr, 2)));
    const px = outW * outH;
    if (px > MAX_OUT_PIXELS) { const k = Math.sqrt(MAX_OUT_PIXELS / px); outW = Math.round(outW * k); outH = Math.round(outH * k); }
    const c = this.init.canvas;
    if (c.width !== outW || c.height !== outH) { c.width = outW; c.height = outH; }
    this.size = { outW, outH };
    const caps = this.gpu.caps, P = this.pool;
    const RA = GPUTextureUsage.RENDER_ATTACHMENT, TB = GPUTextureUsage.TEXTURE_BINDING;
    this.targets = {
      hdr: P.texture('hdr', { size: [outW, outH], format: caps.hdrFormat, usage: RA | TB | GPUTextureUsage.COPY_SRC }),
      hdrCopy: P.texture('hdrCopy', { size: [outW, outH], format: caps.hdrFormat, usage: TB | GPUTextureUsage.COPY_DST }),
    };
    for (const p of [this.lasers, this.glassPass, this.bloom, this.composite]) p.configure(this.size, this.q, this.targets);
  }

  private loop = () => {
    const raf = (self as unknown as { requestAnimationFrame?: (cb: (t: number) => void) => number }).requestAnimationFrame;
    const tick = (now: number) => { this.schedule(raf, tick); this.render(now); };
    this.schedule(raf, tick);
  };
  private schedule(raf: ((cb: (t: number) => void) => number) | undefined, cb: (t: number) => void) {
    if (raf) raf.call(self, cb); else setTimeout(() => cb(performance.now()), 1000 / Math.max(60, this.stats.refreshHz));
  }

  private render(now: number) {
    if (this.hidden) return;
    const cpu0 = performance.now();
    const dt = this.lastT ? Math.min((now - this.lastT) / 1000, 0.1) : 1 / 60;
    if (this.lastT) this.stats.interval(now - this.lastT);
    this.lastT = now;
    this.time += dt;

    const changed = this.init.forceLevel !== undefined ? false
      : this.bench ? this.bench.onFrame(now, this.stats, this.prof)
      : this.qc.update(this.stats.lastInterval, this.prof.gpuMs, this.stats.refreshHz);
    if (changed) { this.q = this.qc.settings; this.reconfigure(); }

    this.updateCamera();
    const F = this.frame, sz = this.size, t = this.time, p = this.pointer;
    p.activity += ((t - p.lastMove < 0.8 ? 1 : 0) - p.activity) * Math.min(1, dt * 3);
    const pw = this.rayPlane(p.x, p.y, 0);
    const reveal = this.init.reducedMotion ? 1e6 : Math.max(0, t - REVEAL_DELAY) * REVEAL_SPEED;
    const pulse = this.pulseStart >= 0 ? (t - this.pulseStart) * PULSE_SPEED : -1;
    const pulseFront = pulse > this.scene.pathLength + 3 ? -1 : pulse;
    if (pulseFront < 0) this.pulseStart = -1;
    const k = sz.outH / 1000;

    F.setMat(O.viewProj, this.vp); F.setMat(O.invViewProj, this.inv); F.setMat(O.view, this.view);
    F.set4(O.camPos, this.eye[0], this.eye[1], this.eye[2], Math.tan(FOV_Y / 2));
    F.set4(O.resolution, sz.outW, sz.outH, 1 / sz.outW, 1 / sz.outH);
    F.set4(O.time, t, dt, this.frameIndex, this.q.level);
    // glint scale: on-screen cap height (px) relative to a 95 px reference, clamped
    const capPx = sz.outH / (2 * this.eye[2] * Math.tan(FOV_Y / 2));
    F.set4(O.pointer, pw[0], pw[1], Math.min(1.6, Math.max(0.45, capPx / 95)), p.activity);
    F.set4(O.misc, 1.0, 0.025, reveal, pulseFront);
    F.set4(O.beamStyle, Math.max(1.0, 1.2 * k), 4.5 * k, 0.07, this.raster.floorY - 0.06); // crisp core, tight halo
    F.set4(O.counts, this.ctx.optics.numSegments, this.ctx.optics.numMirrors, this.ctx.optics.numGlass, this.q.reflections ? 0.12 : 0);
    F.upload(this.device.queue, this.ctx.frameUBO);

    const enc = this.device.createCommandEncoder();
    this.composite.setOutput(this.gpu.context.getCurrentTexture().createView());
    this.lasers.encode(enc, this.prof.ts('lasers'));
    enc.copyTextureToTexture({ texture: this.targets.hdr }, { texture: this.targets.hdrCopy }, [sz.outW, sz.outH]);
    this.glassPass.encode(enc, this.prof.ts('glass'));
    this.bloom.encode(enc, this.prof.ts('bloom'));
    this.composite.encode(enc, this.prof.ts('composite'));
    this.prof.resolve(enc);
    this.device.queue.submit([enc.finish()]);
    this.prof.afterSubmit();

    if (p.pendingT) {
      const t0 = p.pendingT; p.pendingT = 0;
      this.device.queue.onSubmittedWorkDone().then(() => this.stats.inputLatency(performance.timeOrigin + performance.now() - t0));
    }
    this.frameIndex++;
    this.stats.cpu(performance.now() - cpu0);
    if (now - this.statsTimer > 250) {
      this.statsTimer = now;
      this.post({ type: 'stats', stats: this.stats.snapshot(this.prof, this.q, 'webgpu') });
    }
  }

  private updateCamera() {
    const r = this.raster, aspect = this.size.outW / this.size.outH, tanH = Math.tan(FOV_Y / 2);
    // Fit the text plus the prisms beside it and the exit glass above it.
    const dist = Math.max(((r.glyphHalfW + 1.4) * 1.08) / (tanH * aspect), ((r.glyphHalfH + 1.6) * 1.15) / tanH);
    const k = this.init.reducedMotion ? 0.02 : 0.1;
    this.parallax[0] += ((this.pointer.x - 0.5) - this.parallax[0]) * k;
    this.parallax[1] += ((this.pointer.y - 0.5) - this.parallax[1]) * k;
    this.eye[0] = this.parallax[0] * dist * 0.12;
    this.eye[1] = 0.25 + r.glyphHalfH * 0.15 - this.parallax[1] * dist * 0.06;
    this.eye[2] = dist;
    perspective(this.proj, FOV_Y, aspect, 0.1, dist + 60);
    lookAt(this.view, this.eye, [0, 0.15, 0], [0, 1, 0]);
    mul(this.vp, this.proj, this.view);
    invert(this.inv, this.vp);
  }

  /** World point on plane z under normalized screen coords (top-left origin). */
  private rayPlane(x: number, y: number, z: number): number[] {
    const nx = x * 2 - 1, ny = 1 - y * 2, m = this.inv;
    const w = m[3] * nx + m[7] * ny + m[11] + m[15];
    const fx = (m[0] * nx + m[4] * ny + m[8] + m[12]) / w, fy = (m[1] * nx + m[5] * ny + m[9] + m[13]) / w, fz = (m[2] * nx + m[6] * ny + m[10] + m[14]) / w;
    const dx = fx - this.eye[0], dy = fy - this.eye[1], dz = fz - this.eye[2];
    const s = (z - this.eye[2]) / dz;
    return [this.eye[0] + dx * s, this.eye[1] + dy * s, z];
  }
}
