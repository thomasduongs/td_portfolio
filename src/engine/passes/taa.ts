// TAA upsample pass: internal-res jittered scene -> output-res taa[n&1], history taa[(n+1)&1].
import type { EngineCtx, FrameTargets, Pass, QualitySettings, Size, TS } from '../types';
import { FRAME_WGSL } from '../frame';
import SRC from '../shaders/taa.wgsl?raw';

export class TaaPass implements Pass {
  readonly name = 'taa';
  private pipeTaa!: GPURenderPipeline;
  private pipeUp!: GPURenderPipeline;
  private readonly bgl: GPUBindGroupLayout;
  private bgs: GPUBindGroup[] = [];
  private views: GPUTextureView[] = [];
  private useTaa = true;

  constructor(private readonly ctx: EngineCtx) {
    this.bgl = ctx.device.createBindGroupLayout({
      label: 'taa-bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
      ],
    });
  }

  async init(): Promise<void> {
    const { device, caps } = this.ctx;
    const module = device.createShaderModule({ label: 'taa', code: FRAME_WGSL + SRC });
    const layout = device.createPipelineLayout({ bindGroupLayouts: [this.ctx.frameBGL, this.bgl] });
    const mk = (entry: string) => device.createRenderPipelineAsync({
      label: 'taa-' + entry,
      layout,
      vertex: { module, entryPoint: 'vs' },
      fragment: { module, entryPoint: entry, targets: [{ format: caps.hdrFormat }] },
      primitive: { topology: 'triangle-list' },
    });
    [this.pipeTaa, this.pipeUp] = await Promise.all([mk('fs'), mk('fs_upsample')]);
  }

  configure(_size: Size, q: QualitySettings, t: FrameTargets): void {
    this.useTaa = q.taa;
    const { device } = this.ctx;
    this.views = [t.taa[0].createView(), t.taa[1].createView()];
    const sceneView = t.sceneColor.createView();
    const depthView = t.sceneDepth.createView({ aspect: 'depth-only' });
    // parity p: writes taa[p], reads history taa[p^1]
    this.bgs = [0, 1].map((p) => device.createBindGroup({
      label: 'taa-bg' + p,
      layout: this.bgl,
      entries: [
        { binding: 0, resource: sceneView },
        { binding: 1, resource: depthView },
        { binding: 2, resource: this.views[p ^ 1] },
        { binding: 3, resource: this.ctx.linearClamp },
      ],
    }));
  }

  encode(enc: GPUCommandEncoder, ts: TS, frameIndex: number): void {
    const p = frameIndex & 1;
    const pass = enc.beginRenderPass({
      label: 'taa',
      colorAttachments: [{ view: this.views[p], loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
      timestampWrites: ts as GPURenderPassTimestampWrites | undefined,
    });
    pass.setPipeline(this.useTaa ? this.pipeTaa : this.pipeUp);
    pass.setBindGroup(0, this.ctx.frameBG);
    pass.setBindGroup(1, this.bgs[p]);
    pass.draw(3);
    pass.end();
  }
}
