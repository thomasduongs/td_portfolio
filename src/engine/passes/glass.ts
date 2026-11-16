// Glass pass: loads hdr, redraws each glass object's pixels with screen-space refraction of hdrCopy.
import type { EngineCtx, FrameTargets, Pass, QualitySettings, Size, TS } from '../types';
import { FRAME_WGSL } from '../frame';
import SRC from '../shaders/glass.wgsl?raw';

export class GlassPass implements Pass {
  readonly name = 'glass';
  private pipeline!: GPURenderPipeline;
  private readonly bgl: GPUBindGroupLayout;
  private bg!: GPUBindGroup;
  private view!: GPUTextureView;
  private ubo: GPUBuffer | null = null;

  constructor(private readonly ctx: EngineCtx) {
    this.bgl = ctx.device.createBindGroupLayout({
      label: 'glass-bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ],
    });
  }

  async init(): Promise<void> {
    const { device, caps } = this.ctx;
    const module = device.createShaderModule({ label: 'glass', code: FRAME_WGSL + SRC });
    this.pipeline = await device.createRenderPipelineAsync({
      label: 'glass',
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.ctx.frameBGL, this.bgl] }),
      vertex: { module, entryPoint: 'vs' },
      fragment: { module, entryPoint: 'fs', targets: [{ format: caps.hdrFormat }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
    });
  }

  configure(_size: Size, q: QualitySettings, t: FrameTargets): void {
    const { device, optics, linearClamp } = this.ctx;
    this.view = t.hdr.createView();
    this.ubo?.destroy();
    this.ubo = device.createBuffer({ label: 'glass-ubo', size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(this.ubo, 0, new Float32Array([q.glassDispersion ? 1 : 0, 0, 0, 0]));
    this.bg = device.createBindGroup({
      label: 'glass-bg',
      layout: this.bgl,
      entries: [
        { binding: 0, resource: { buffer: optics.glass } },
        { binding: 1, resource: t.hdrCopy.createView() },
        { binding: 2, resource: linearClamp },
        { binding: 3, resource: { buffer: this.ubo } },
      ],
    });
  }

  encode(enc: GPUCommandEncoder, ts: TS): void {
    const n = this.ctx.optics.numGlass;
    if (n <= 0) return;
    const pass = enc.beginRenderPass({
      label: 'glass',
      colorAttachments: [{ view: this.view, loadOp: 'load', storeOp: 'store' }],
      timestampWrites: ts as GPURenderPassTimestampWrites | undefined,
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.ctx.frameBG);
    pass.setBindGroup(1, this.bg);
    pass.draw(6, n);
    pass.end();
  }
}
