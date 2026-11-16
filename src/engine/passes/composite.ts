// Composite: TAA output + bloom -> exposure -> AgX -> swapchain. host calls setOutput(view) each frame.
import type { EngineCtx, FrameTargets, Pass, QualitySettings, Size, TS } from '../types';
import { FRAME_WGSL } from '../frame';
import { bloomDesc } from './bloom';
import SRC from '../shaders/composite.wgsl?raw';

export class CompositePass implements Pass {
  readonly name = 'composite';
  private pipeline!: GPURenderPipeline;
  private readonly bgl: GPUBindGroupLayout;
  private bgs: GPUBindGroup[] = [];
  private output: GPUTextureView | null = null;

  constructor(private readonly ctx: EngineCtx, _context: GPUCanvasContext) {
    this.bgl = ctx.device.createBindGroupLayout({
      label: 'composite-bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
      ],
    });
  }

  setOutput(view: GPUTextureView): void { this.output = view; }

  async init(): Promise<void> {
    const { device, caps } = this.ctx;
    const module = device.createShaderModule({ label: 'composite', code: FRAME_WGSL + SRC });
    this.pipeline = await device.createRenderPipelineAsync({
      label: 'composite',
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.ctx.frameBGL, this.bgl] }),
      vertex: { module, entryPoint: 'vs' },
      fragment: {
        module,
        entryPoint: caps.presentFormat.endsWith('-srgb') ? 'fs_srgb' : 'fs',
        targets: [{ format: caps.presentFormat }],
      },
      primitive: { topology: 'triangle-list' },
    });
  }

  configure(size: Size, q: QualitySettings, t: FrameTargets): void {
    const { device, caps, pool, linearClamp } = this.ctx;
    // Same descriptor as BloomPass -> same pooled texture regardless of configure order.
    const bloom = pool.texture('bloom', bloomDesc(size, q, caps));
    const bloomView = bloom.createView({ baseMipLevel: 0, mipLevelCount: 1 });
    this.bgs = [0, 1].map((p) => device.createBindGroup({
      label: 'composite-bg' + p,
      layout: this.bgl,
      entries: [
        { binding: 0, resource: t.taa[p].createView() },
        { binding: 1, resource: bloomView },
        { binding: 2, resource: linearClamp },
      ],
    }));
  }

  encode(enc: GPUCommandEncoder, ts: TS, frameIndex: number): void {
    if (!this.output) return;
    const pass = enc.beginRenderPass({
      label: 'composite',
      colorAttachments: [{ view: this.output, loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
      timestampWrites: ts as GPURenderPassTimestampWrites | undefined,
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.ctx.frameBG);
    pass.setBindGroup(1, this.bgs[frameIndex & 1]);
    pass.draw(3);
    pass.end();
  }
}
