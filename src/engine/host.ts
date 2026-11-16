// Engine host: owns the frame loop. Runs inside the render worker (OffscreenCanvas) or, when
// transferControlToOffscreen is unavailable, on the main thread with an HTMLCanvasElement.
// The DOM never touches this code path except via postMessage-shaped messages.
import type { MainToWorker, WorkerToMain, EngineCtx, FrameTargets, Pass, Size, QualitySettings } from './types';
import { FrameData, FRAME_UBO_SIZE, O, frameBindGroupLayout } from './frame';
import { m4, perspective, lookAt, mul, invert, halton } from './math';
import { initDevice } from './device';
import { ResourcePool } from './resources';
import { GpuProfiler, FrameStats } from './profiler';
import { QualityController } from './quality';
import { BenchRunner } from './bench';
import { rasterizeText, bakeSDFGPU, type TextRaster } from './sdfbake';
import { ParticlePass } from './passes/particles';
import { VolumetricPass } from './passes/volumetric';
import { ScenePass } from './passes/scene';
import { TaaPass } from './passes/taa';
import { BloomPass } from './passes/bloom';
import { CompositePass } from './passes/composite';
import { Physics } from '../physics/physics';

type Post = (m: WorkerToMain) => void;
type InitMsg = Extract<MainToWorker, { type: 'init' }>;

const FOV_Y = (32 * Math.PI) / 180;
const PHYS_DT = 1 / 120;
const MAX_OUT_PIXELS = 3_700_000; // ~2560x1440: beyond this, TAA upsampling + CSS scaling is cheaper than native.

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
    const gpu = await initDevice(init.canvas, init.tier).catch(() => null);
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

class Engine {
  private frame = new FrameData();
  private passes: Pass[] = [];
  private composite!: CompositePass;
  private targets!: FrameTargets;
  private size!: Size;
  private q!: QualitySettings;
  private frameIndex = 0;
  private hidden = false;
  private cssW: number; private cssH: number; private dpr: number;
  private lastT = 0; private time = 0; private physAcc = 0;
  private pointer = { x: 0.5, y: 0.5, down: false, intensity: 0, lastMove: -1e9, pendingT: 0 };
  private clicks: number[][] = [];
  private proj = m4(); private view = m4(); private vp = m4(); private vpJ = m4(); private inv = m4(); private prevVP = m4(); private jm = m4();
  private eye = [0, 0.5, 8]; private parallax = [0, 0];
  private stats = new FrameStats();
  private statsTimer = 0;
  private bench: BenchRunner | null;

  static async create(gpu: Awaited<ReturnType<typeof initDevice>> & {}, init: InitMsg, post: Post): Promise<Engine> {
    const e = new Engine(gpu, init, post);
    await e.setup();
    return e;
  }

  private device: GPUDevice;
  private ctx!: EngineCtx;
  private pool: ResourcePool;
  private prof: GpuProfiler;
  private qc: QualityController;
  private physics!: Physics;
  private raster!: TextRaster;
  private sdfTex: GPUTexture | null = null;

  private constructor(private gpu: NonNullable<Awaited<ReturnType<typeof initDevice>>>, private init: InitMsg, private post: Post) {
    this.device = gpu.device;
    this.cssW = init.width; this.cssH = init.height; this.dpr = init.dpr;
    this.pool = new ResourcePool(this.device);
    this.qc = new QualityController(gpu.caps.tier);
    this.prof = new GpuProfiler(this.device, gpu.caps, ['particles', 'volumetric', 'scene', 'taa', 'bloom', 'composite']);
    this.bench = init.bench ? new BenchRunner(init.bench, this.qc, (r) => post({ type: 'bench-report', report: r })) : null;
  }

  private async setup() {
    const d = this.device, caps = this.gpu.caps;
    const frameUBO = d.createBuffer({ label: 'frame-ubo', size: FRAME_UBO_SIZE, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const frameBGL = frameBindGroupLayout(d);
    const frameBG = d.createBindGroup({ layout: frameBGL, entries: [{ binding: 0, resource: { buffer: frameUBO } }] });
    const linearClamp = d.createSampler({ magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge', addressModeW: 'clamp-to-edge' });

    this.raster = rasterizeText(this.lines());
    this.sdfTex = await bakeSDFGPU(d, this.raster);
    this.ctx = { device: d, caps, frameUBO, frameBGL, frameBG, sdfView: this.sdfTex.createView(), linearClamp, pool: this.pool };

    this.composite = new CompositePass(this.ctx, this.gpu.context);
    const particles = new ParticlePass(this.ctx), scene = new ScenePass(this.ctx);
    scene.setOverlay(particles); // particle draw is recorded right after the scene render pass (needs scene depth)
    this.passes = [
      particles, new VolumetricPass(this.ctx), scene,
      new TaaPass(this.ctx), new BloomPass(this.ctx), this.composite,
    ];
    const [physics] = await Promise.all([Physics.create(this.raster), ...this.passes.map((p) => p.init())]);
    this.physics = physics;
    this.q = this.qc.settings;
    this.reconfigure();
    this.loop();
  }

  private lines(): string[] { return this.cssW / this.cssH < 0.9 ? ['THOMAS', 'DUONG'] : ['THOMAS DUONG']; }

  handle(m: MainToWorker) {
    switch (m.type) {
      case 'resize': {
        this.cssW = m.width; this.cssH = m.height; this.dpr = m.dpr;
        const lines = this.lines();
        if (lines.length !== this.raster.lines.length) this.rebake(lines); else this.reconfigure();
        break;
      }
      case 'pointer': {
        const p = this.pointer;
        if (Math.abs(m.x - p.x) + Math.abs(m.y - p.y) > 1e-4) p.lastMove = this.time;
        p.x = m.x; p.y = m.y; p.down = m.down; p.pendingT = m.t;
        if (m.click) this.clicks.push([m.x, m.y]);
        break;
      }
      case 'visibility': this.hidden = m.hidden; if (!m.hidden) this.lastT = 0; break;
    }
  }

  private async rebake(lines: string[]) {
    this.raster = rasterizeText(lines);
    const old = this.sdfTex;
    this.sdfTex = await bakeSDFGPU(this.device, this.raster);
    this.ctx.sdfView = this.sdfTex.createView();
    this.physics.rebuildText(this.raster);
    this.reconfigure();
    old?.destroy();
  }

  /** Resize swapchain + targets and push size/quality to all passes (bind groups rebuilt here, never per frame). */
  private reconfigure() {
    let outW = Math.max(1, Math.round(this.cssW * Math.min(this.dpr, 2)));
    let outH = Math.max(1, Math.round(this.cssH * Math.min(this.dpr, 2)));
    const px = outW * outH;
    if (px > MAX_OUT_PIXELS) { const k = Math.sqrt(MAX_OUT_PIXELS / px); outW = Math.round(outW * k); outH = Math.round(outH * k); }
    const c = this.init.canvas;
    if (c.width !== outW || c.height !== outH) { c.width = outW; c.height = outH; }
    const s = this.q.renderScale;
    this.size = { outW, outH, inW: Math.max(1, Math.round(outW * s)), inH: Math.max(1, Math.round(outH * s)) };
    const caps = this.gpu.caps, P = this.pool, sz = this.size;
    const RA = GPUTextureUsage.RENDER_ATTACHMENT, TB = GPUTextureUsage.TEXTURE_BINDING, ST = GPUTextureUsage.STORAGE_BINDING;
    this.targets = {
      sceneColor: P.texture('sceneColor', { size: [sz.inW, sz.inH], format: caps.hdrFormat, usage: RA | TB }),
      sceneDepth: P.texture('sceneDepth', { size: [sz.inW, sz.inH], format: caps.depthFormat, usage: RA | TB }),
      taa: [
        P.texture('taa0', { size: [sz.outW, sz.outH], format: caps.hdrFormat, usage: RA | TB | ST }),
        P.texture('taa1', { size: [sz.outW, sz.outH], format: caps.hdrFormat, usage: RA | TB | ST }),
      ],
      froxelIntegrated: P.texture('froxelInt', { size: this.q.froxel, dimension: '3d', format: 'rgba16float', usage: TB | ST }),
    };
    for (const p of this.passes) p.configure(this.size, this.q, this.targets);
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
    this.time += this.init.reducedMotion ? dt * 0.25 : dt;

    // ---- adaptive quality (decided on CPU from last-known GPU timings; never stalls) ----
    const changed = this.bench ? this.bench.onFrame(now, this.stats, this.prof) : this.qc.update(this.stats.lastInterval, this.prof.gpuMs, this.stats.refreshHz);
    if (changed) { this.q = this.qc.settings; this.reconfigure(); }

    this.updateCamera();
    const ray = this.pointerRay();

    // ---- physics: fixed 120 Hz with render interpolation ----
    for (const c of this.clicks) this.physics.spawn(this.unproject(c[0], c[1]), this.eye);
    this.clicks.length = 0;
    this.physAcc += dt;
    const p = this.pointer;
    while (this.physAcc >= PHYS_DT) { this.physics.step(PHYS_DT, ray); this.physAcc -= PHYS_DT; }
    const count = this.physics.writeMarbles(this.frame.f32, O.marbles, this.physAcc / PHYS_DT);

    // ---- frame uniforms ----
    const F = this.frame, sz = this.size, q = this.q, r = this.raster;
    p.intensity += ((this.time - p.lastMove < 1.2 ? 1 : 0.15) - p.intensity) * Math.min(1, dt * 4);
    const t = this.time;
    F.setMat(O.viewProj, this.vpJ); F.setMat(O.invViewProj, this.inv); F.setMat(O.prevViewProj, this.prevVP);
    F.set4(O.camPos, this.eye[0], this.eye[1], this.eye[2], Math.tan(FOV_Y / 2));
    F.set4(O.resolution, sz.inW, sz.inH, 1 / sz.inW, 1 / sz.inH);
    F.set4(O.outRes, sz.outW, sz.outH, 1 / sz.outW, 1 / sz.outH);
    F.set4(O.time, t, dt, this.frameIndex, q.level);
    F.set4(O.pointer, ray.light[0], ray.light[1], ray.light[2], p.intensity * 6);
    const sx = 3.2 * Math.sin(t * 0.11), sp = [sx, 3.6, -5.2], tg = [-sx * 0.25, r.floorY, 2.5];
    const dl = Math.hypot(tg[0] - sp[0], tg[1] - sp[1], tg[2] - sp[2]);
    F.set4(O.spotPos, sp[0], sp[1], sp[2], Math.cos((26 * Math.PI) / 180));
    F.set4(O.spotDir, (tg[0] - sp[0]) / dl, (tg[1] - sp[1]) / dl, (tg[2] - sp[2]) / dl, Math.cos((15 * Math.PI) / 180));
    F.set4(O.spotColor, 1.0 * 90, 0.9 * 90, 0.78 * 90, r.floorY);
    F.set4(O.textParams, r.halfW, r.halfH, 0.16, 0.035);
    F.set4(O.quality, q.primarySteps, q.interiorSteps, q.reflSteps, q.shadowSteps);
    F.set4(O.froxel, q.froxel[0], q.froxel[1], q.froxel[2], 22);
    F.set4(O.misc, 1.0, 0.05, 0.035, count);
    F.upload(this.device.queue, this.ctx.frameUBO);

    // ---- encode: one command buffer, one submit ----
    const enc = this.device.createCommandEncoder();
    this.composite.setOutput(this.gpu.context.getCurrentTexture().createView());
    for (const pass of this.passes) pass.encode(enc, this.prof.ts(pass.name), this.frameIndex);
    this.prof.resolve(enc);
    this.device.queue.submit([enc.finish()]);
    this.prof.afterSubmit();

    if (p.pendingT) {
      const t0 = p.pendingT; p.pendingT = 0;
      this.device.queue.onSubmittedWorkDone().then(() => this.stats.inputLatency(performance.timeOrigin + performance.now() - t0));
    }

    this.prevVP.set(this.vp);
    this.frameIndex++;
    this.stats.cpu(performance.now() - cpu0);
    if (now - this.statsTimer > 250) {
      this.statsTimer = now;
      this.post({ type: 'stats', stats: this.stats.snapshot(this.prof, this.q, 'webgpu') });
    }
  }

  private updateCamera() {
    const r = this.raster, aspect = this.size.outW / this.size.outH, tanH = Math.tan(FOV_Y / 2);
    // Fit text: 1.3x glyph half-width horizontally, 2.4x glyph half-height vertically.
    const dist = Math.max((r.glyphHalfW * 1.3) / (tanH * aspect), (r.glyphHalfH * 2.4) / tanH);
    const k = this.init.reducedMotion ? 0.02 : 0.12;
    this.parallax[0] += ((this.pointer.x - 0.5) - this.parallax[0]) * k;
    this.parallax[1] += ((this.pointer.y - 0.5) - this.parallax[1]) * k;
    this.eye[0] = this.parallax[0] * dist * 0.08;
    this.eye[1] = 0.35 + r.glyphHalfH * 0.2 - this.parallax[1] * dist * 0.04;
    this.eye[2] = dist;
    perspective(this.proj, FOV_Y, aspect, 0.1, 40);
    lookAt(this.view, this.eye, [0, 0.05, 0], [0, 1, 0]);
    mul(this.vp, this.proj, this.view);
    // Sub-pixel jitter (Halton 2,3; 8-cycle) in NDC at INTERNAL resolution.
    const i = (this.frameIndex % 8) + 1;
    const jx = this.q.taa ? ((halton(i, 2) - 0.5) * 2) / this.size.inW : 0;
    const jy = this.q.taa ? ((halton(i, 3) - 0.5) * 2) / this.size.inH : 0;
    const prevJ = [this.frame.f32[O.jitter], this.frame.f32[O.jitter + 1]];
    this.jm.fill(0); this.jm[0] = 1; this.jm[5] = 1; this.jm[10] = 1; this.jm[15] = 1; this.jm[12] = jx; this.jm[13] = jy;
    mul(this.vpJ, this.jm, this.vp);
    invert(this.inv, this.vpJ);
    this.frame.set4(O.jitter, jx, jy, prevJ[0], prevJ[1]);
  }

  /** World point on plane z=0 under normalized screen coords (top-left origin). */
  private unproject(x: number, y: number): number[] { return this.rayPlane(x, y, 0); }
  private rayPlane(x: number, y: number, z: number): number[] {
    const nx = x * 2 - 1, ny = 1 - y * 2, m = this.inv;
    const w = m[3] * nx + m[7] * ny + m[11] + m[15];
    const fx = (m[0] * nx + m[4] * ny + m[8] + m[12]) / w, fy = (m[1] * nx + m[5] * ny + m[9] + m[13]) / w, fz = (m[2] * nx + m[6] * ny + m[10] + m[14]) / w;
    const dx = fx - this.eye[0], dy = fy - this.eye[1], dz = fz - this.eye[2];
    const s = (z - this.eye[2]) / dz;
    return [this.eye[0] + dx * s, this.eye[1] + dy * s, z];
  }
  private pointerRay() {
    const p = this.pointer;
    return { light: this.rayPlane(p.x, p.y, 0.9), pusher: this.rayPlane(p.x, p.y, 0.25), active: this.time - p.lastMove < 1.2 };
  }
}
