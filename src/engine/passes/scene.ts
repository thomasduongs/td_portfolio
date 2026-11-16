// Scene pass v2: per-pixel light-beam march (haze scatter + letter excitation) + floor + marbles -> sceneColor/sceneDepth.
// Cost scales with internal pixels x (hazeSteps + letterSteps), both owned by the quality controller.
import type { EngineCtx, FrameTargets, Pass, QualitySettings, Size, TS } from '../types';
import { FRAME_WGSL, BEAMS_WGSL } from '../frame';
import SRC from '../shaders/scene.wgsl?raw';

export interface Overlay { encodeDraw(enc: GPUCommandEncoder): void }

export class ScenePass implements Pass {
  readonly name = 'scene';
  private pipeline!: GPURenderPipeline;
  private readonly bgl: GPUBindGroupLayout;
  private bg!: GPUBindGroup;
  private colorView!: GPUTextureView;
  private depthView!: GPUTextureView;
  private overlay: Overlay | null = null;

  constructor(private readonly ctx: EngineCtx) {
    this.bgl = ctx.device.createBindGroupLayout({
      label: 'scene-bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '3d' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
      ],
    });
    this.repeat = ctx.device.createSampler({ label: 'noise-repeat', magFilter: 'linear', minFilter: 'linear', addressModeU: 'repeat', addressModeV: 'repeat', addressModeW: 'repeat' });
  }
  private readonly repeat: GPUSampler;

  setOverlay(o: Overlay): void { this.overlay = o; }

  async init(): Promise<void> {
    const { device, caps } = this.ctx;
    const module = device.createShaderModule({ label: 'scene', code: FRAME_WGSL + BEAMS_WGSL + SRC });
    this.pipeline = await device.createRenderPipelineAsync({
      label: 'scene',
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.ctx.frameBGL, this.bgl] }),
      vertex: { module, entryPoint: 'vs' },
      fragment: { module, entryPoint: 'fs', targets: [{ format: caps.hdrFormat }] },
      primitive: { topology: 'triangle-list' },
      // frag_depth is written for every pixel: 'always' + write (no early-z is possible with frag_depth anyway).
      depthStencil: { format: caps.depthFormat, depthWriteEnabled: true, depthCompare: 'always' },
    });
  }

  configure(_size: Size, _q: QualitySettings, targets: FrameTargets): void {
    this.colorView = targets.sceneColor.createView();
    this.depthView = targets.sceneDepth.createView();
    // sdfView may have changed (re-bake on layout change) — host always calls configure afterwards.
    this.bg = this.ctx.device.createBindGroup({
      label: 'scene-bg',
      layout: this.bgl,
      entries: [
        { binding: 0, resource: this.ctx.sdfView },
        { binding: 1, resource: this.ctx.linearClamp },
        { binding: 2, resource: this.ctx.noiseView },
        { binding: 3, resource: this.repeat },
      ],
    });
  }

  encode(enc: GPUCommandEncoder, ts: TS): void {
    const pass = enc.beginRenderPass({
      label: 'scene',
      colorAttachments: [{ view: this.colorView, loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
      depthStencilAttachment: { view: this.depthView, depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' },
      timestampWrites: ts as GPURenderPassTimestampWrites | undefined,
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.ctx.frameBG);
    pass.setBindGroup(1, this.bg);
    pass.draw(3);
    pass.end();
    this.overlay?.encodeDraw(enc);
  }
}
