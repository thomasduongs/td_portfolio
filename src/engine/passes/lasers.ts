// Laser pass: clears hdr, then draws mirrors (alpha), beam quads (+ floor reflection) and glints (additive).
import type { EngineCtx, FrameTargets, Pass, QualitySettings, Size, TS } from '../types';
import { FRAME_WGSL } from '../frame';
import SRC from '../shaders/lasers.wgsl?raw';

export class LaserPass implements Pass {
  readonly name = 'lasers';
  private pMirror!: GPURenderPipeline;
  private pBeam!: GPURenderPipeline;
  private pGlint!: GPURenderPipeline;
  private readonly bgl: GPUBindGroupLayout;
  private bg!: GPUBindGroup;
  private view!: GPUTextureView;
  private ubo: GPUBuffer | null = null;
  private reflections = true;

  constructor(private readonly ctx: EngineCtx) {
    const vis = GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT;
    this.bgl = ctx.device.createBindGroupLayout({
      label: 'lasers-bgl',
      entries: [
        { binding: 0, visibility: vis, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: vis, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: vis, buffer: { type: 'uniform' } },
      ],
    });
  }

  async init(): Promise<void> {
    const { device, caps } = this.ctx;
    const module = device.createShaderModule({ label: 'lasers', code: FRAME_WGSL + SRC });
    const layout = device.createPipelineLayout({ bindGroupLayouts: [this.ctx.frameBGL, this.bgl] });
    const add: GPUBlendComponent = { srcFactor: 'one', dstFactor: 'one', operation: 'add' };
    const alpha: GPUBlendState = {
      color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
      alpha: add,
    };
    const mk = (vs: string, fs: string, blend: GPUBlendState) => device.createRenderPipelineAsync({
      label: fs,
      layout,
      vertex: { module, entryPoint: vs },
      fragment: { module, entryPoint: fs, targets: [{ format: caps.hdrFormat, blend }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
    });
    [this.pMirror, this.pBeam, this.pGlint] = await Promise.all([
      mk('vs_mirror', 'fs_mirror', alpha),
      mk('vs_beam', 'fs_beam', { color: add, alpha: add }),
      mk('vs_glint', 'fs_glint', { color: add, alpha: add }),
    ]);
  }

  configure(_size: Size, q: QualitySettings, t: FrameTargets): void {
    const { device, optics } = this.ctx;
    this.view = t.hdr.createView();
    this.reflections = q.reflections;
    this.ubo?.destroy();
    this.ubo = device.createBuffer({ label: 'lasers-ubo', size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(this.ubo, 0, new Float32Array([q.glowScale, 0, 0, 0]));
    this.bg = device.createBindGroup({
      label: 'lasers-bg',
      layout: this.bgl,
      entries: [
        { binding: 0, resource: { buffer: optics.segments } },
        { binding: 1, resource: { buffer: optics.mirrors } },
        { binding: 2, resource: { buffer: this.ubo } },
      ],
    });
  }

  encode(enc: GPUCommandEncoder, ts: TS): void {
    const { numSegments, numMirrors } = this.ctx.optics;
    const pass = enc.beginRenderPass({
      label: 'lasers',
      colorAttachments: [{ view: this.view, loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
      timestampWrites: ts as GPURenderPassTimestampWrites | undefined,
    });
    pass.setBindGroup(0, this.ctx.frameBG);
    pass.setBindGroup(1, this.bg);
    if (numMirrors > 0) {
      pass.setPipeline(this.pMirror);
      pass.draw(6, numMirrors);
    }
    if (numSegments > 0) {
      pass.setPipeline(this.pBeam);
      pass.draw(6, numSegments * (this.reflections ? 2 : 1));
    }
    if (numMirrors > 0) {
      pass.setPipeline(this.pGlint);
      pass.draw(6, numMirrors);
    }
    pass.end();
  }
}
