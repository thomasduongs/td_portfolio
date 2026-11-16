// Engine host: owns the frame loop. Runs inside the render worker (OffscreenCanvas) or, when
// transferControlToOffscreen is unavailable, on the main thread with an HTMLCanvasElement.
// The DOM never touches this code path except via postMessage-shaped messages.
import type { MainToWorker, WorkerToMain, EngineCtx, FrameTargets, Pass, Size, QualitySettings } from './types';
import { FrameData, FRAME_UBO_SIZE, O, BEAM_STRIDE, NUM_BEAMS, frameBindGroupLayout } from './frame';
import { m4, perspective, lookAt, mul, invert, halton } from './math';
import { initDevice } from './device';
import { ResourcePool } from './resources';
import { GpuProfiler, FrameStats } from './profiler';
import { QualityController } from './quality';
import { BenchRunner } from './bench';
import { rasterizeText, bakeSDFGPU, type TextRaster } from './sdfbake';
import { ParticlePass } from './passes/particles';
import { bakeNoise } from './noise';
import { ScenePass } from './passes/scene';
import { TaaPass } from './passes/taa';
import { BloomPass } from './passes/bloom';
import { CompositePass } from './passes/composite';
import type { Physics } from '../physics/physics'; // lazy-loaded: Rapier WASM (~2 MB) must not block first frame

type Post = (m: WorkerToMain) => void;
type InitMsg = Extract<MainToWorker, { type: 'init' }>;

const FOV_Y = (32 * Math.PI) / 180;
const PHYS_DT = 1 / 120;

// Projector rig: rest position, orbit amplitude, orbit speed, scatter radiance (rgb), intro scatter offset.
const RIG = [
  { pos: [-7.5, 4.0, -4.5], orbit: [0.9, 0.8, 1.0], freq: 0.19, color: [9.0, 1.1, 0.55], scatter: [-1.4, 0.5] },
  { pos: [7.5, 4.0, -4.5], orbit: [0.9, 0.8, 1.0], freq: 0.17, color: [0.8, 8.0, 1.6], scatter: [1.3, -0.6] },
  { pos: [0.0, 7.6, -5.5], orbit: [1.8, 0.6, 0.8], freq: 0.13, color: [1.2, 3.2, 11.0], scatter: [0.2, 1.2] },
] as const;
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
  private agitation = 0;
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
  private physics: Physics | null = null;
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
    const [sdfTex, noiseTex] = await Promise.all([bakeSDFGPU(d, this.raster), bakeNoise(d)]);
    this.sdfTex = sdfTex;
    this.ctx = { device: d, caps, frameUBO, frameBGL, frameBG, sdfView: sdfTex.createView(), noiseView: noiseTex.createView({ dimension: '3d' }), linearClamp, pool: this.pool };

    this.composite = new CompositePass(this.ctx, this.gpu.context);
    const particles = new ParticlePass(this.ctx), scene = new ScenePass(this.ctx);
    scene.setOverlay(particles); // particle draw is recorded right after the scene render pass (needs scene depth)
    this.passes = [
      particles, scene,
      new TaaPass(this.ctx), new BloomPass(this.ctx), this.composite,
    ];
    await Promise.all(this.passes.map((p) => p.init()));
    if (this.init.forceLevel !== undefined) this.qc.force(this.init.forceLevel);
    this.q = this.qc.settings;
    this.reconfigure();
    this.loop();
    // Physics streams in after the first frames are on screen; clicks before then stay queued.
    import('../physics/physics')
      .then((m) => m.Physics.create(this.raster))
      .then((ph) => { this.physics = ph; ph.rebuildText(this.raster); })
      .catch((e) => console.warn('physics unavailable', e));
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
        const moved = Math.hypot(m.x - p.x, m.y - p.y);
        if (moved > 1e-4) p.lastMove = this.time;
        // Fast pointer motion knocks the projectors out of alignment; the letters split into RGB ghosts.
        this.agitation = Math.min(1.6, this.agitation + moved * 3.5);
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
    this.physics?.rebuildText(this.raster);
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
    const changed = this.init.forceLevel !== undefined ? false : this.bench ? this.bench.onFrame(now, this.stats, this.prof) : this.qc.update(this.stats.lastInterval, this.prof.gpuMs, this.stats.refreshHz);
    if (changed) { this.q = this.qc.settings; this.reconfigure(); }

    this.updateCamera();
    const ray = this.pointerRay();

    // ---- physics: fixed 120 Hz with render interpolation ----
    const p = this.pointer, ph = this.physics;
    let count = 0;
    if (ph) {
      for (const c of this.clicks) ph.spawn(this.unproject(c[0], c[1]), this.eye);
      this.clicks.length = 0;
      this.physAcc += dt;
      while (this.physAcc >= PHYS_DT) { ph.step(PHYS_DT, ray); this.physAcc -= PHYS_DT; }
      count = ph.writeMarbles(this.frame.f32, O.marbles, this.physAcc / PHYS_DT);
    }

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
    this.writeBeams(t, dt);
    F.set4(O.textParams, r.halfW, r.halfH, r.floorY, 0.01);
    F.set4(O.quality, q.hazeSteps, q.letterSteps, q.reflSteps, this.eye[2] + 16);
    F.set4(O.misc, 1.0, 0.06, 0.05, count);
    F.set4(O.beamParams, 0.06, 14.0, 0.15, 0.14); // slab ±0.15: past this, letters of different beams falsely coincide
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

  /**
   * Three gobo projectors (R, G, B). Their silhouettes coincide only on the plane z = 0, so the letters
   * exist only where all three beams cross. Lamps orbit slowly (streaks swing, letters stay formed);
   * misalignment (intro convergence + pointer agitation) shifts each gobo so the letters split apart.
   */
  private writeBeams(t: number, dt: number) {
    const F = this.frame, r = this.raster, still = this.init.reducedMotion;
    this.agitation *= Math.exp(-dt * 2.2);
    // Intro: beams start scattered and spring into alignment over ~3 s.
    const intro = still ? 0 : Math.exp(-1.35 * t) * Math.cos(2.1 * t);
    const reach = Math.hypot(r.glyphHalfW, r.glyphHalfH) * 1.12;
    for (let j = 0; j < NUM_BEAMS; j++) {
      const rig = RIG[j], o = O.beams + j * BEAM_STRIDE;
      const w = still ? 0 : 1;
      const px = rig.pos[0] + w * rig.orbit[0] * Math.sin(t * rig.freq + j * 2.1);
      const py = rig.pos[1] + w * rig.orbit[1] * Math.sin(t * rig.freq * 1.3 + j);
      const pz = rig.pos[2] + w * rig.orbit[2] * Math.cos(t * rig.freq * 0.8 + j * 1.7);
      const d0 = Math.hypot(px, py, pz);
      const outer = Math.atan(reach / d0) + (3 * Math.PI) / 180, inner = outer - (4 * Math.PI) / 180;
      F.set4(o, px, py, pz, Math.cos(outer));
      F.set4(o + 4, -px / d0, -py / d0, -pz / d0, Math.cos(inner));
      F.set4(o + 8, rig.color[0], rig.color[1], rig.color[2], 0.06);
      const ang = t * 0.9 + j * 2.094;
      const ax = rig.scatter[0] * intro + Math.cos(ang) * this.agitation * 0.35;
      const ay = rig.scatter[1] * intro + Math.sin(ang) * this.agitation * 0.35;
      F.set4(o + 12, ax, ay, 0, d0 * d0);
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
    perspective(this.proj, FOV_Y, aspect, 0.1, dist + 30);
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
