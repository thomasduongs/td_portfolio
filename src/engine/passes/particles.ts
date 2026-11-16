import type { EngineCtx, FrameTargets, Pass, QualitySettings, Size, TS } from '../types';
import { FRAME_WGSL } from '../frame';
import { QualityController } from '../quality';
import SOURCE from '../shaders/particles.wgsl?raw';
import DRAW_SOURCE from '../shaders/particles-render.wgsl?raw';

export class ParticlePass implements Pass {
  readonly name = 'particles';
  private computeBGL!: GPUBindGroupLayout;
  private drawBGL!: GPUBindGroupLayout;
  private updatePipeline!: GPUComputePipeline;
  private drawPipeline!: GPURenderPipeline;
  private computeBG!: GPUBindGroup;
  private drawBG!: GPUBindGroup;
  private pos!: GPUBuffer;
  private vel!: GPUBuffer;
  private light!: GPUBuffer;
  private params!: GPUBuffer;
  private indirect!: GPUBuffer;
  private drawIndirect!: GPUBuffer;
  private colorView!: GPUTextureView;
  private depthView!: GPUTextureView;

  constructor(private ctx: EngineCtx) {}

  async init(): Promise<void> {
    const { device, caps, frameBGL } = this.ctx;
    const storage = (label: string, size: number) => device.createBuffer({ label, size, usage: GPUBufferUsage.STORAGE });
    this.pos = storage('Particle positions', QualityController.MAX_PARTICLES * 16);
    this.vel = storage('Particle velocities', QualityController.MAX_PARTICLES * 16);
    this.light = storage('Particle radiance', QualityController.MAX_PARTICLES * 4);
    this.params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.indirect = device.createBuffer({ size: 12, usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST });
    this.drawIndirect = device.createBuffer({ size: 16, usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST });
    this.computeBGL = device.createBindGroupLayout({ entries: [
      ...[0, 1, 2].map(binding => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' as const } })),
      { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, sampler: { type: 'filtering' } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ] });
    this.drawBGL = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
    ] });
    // Separate modules give the vertex stage read-only declarations for shared buffers.
    const computeModule = device.createShaderModule({ code: FRAME_WGSL + SOURCE });
    const drawModule = device.createShaderModule({ code: FRAME_WGSL + DRAW_SOURCE });
    const computeLayout = device.createPipelineLayout({ bindGroupLayouts: [frameBGL, this.computeBGL] });
    const [initialize, update, draw] = await Promise.all([
      device.createComputePipelineAsync({ layout: computeLayout, compute: { module: computeModule, entryPoint: 'initialize' } }),
      device.createComputePipelineAsync({ layout: computeLayout, compute: { module: computeModule, entryPoint: 'update' } }),
      device.createRenderPipelineAsync({
        layout: device.createPipelineLayout({ bindGroupLayouts: [frameBGL, this.drawBGL] }),
        vertex: { module: drawModule, entryPoint: 'vs' },
        fragment: { module: drawModule, entryPoint: 'fs', targets: [{ format: caps.hdrFormat,
          blend: { color: { srcFactor: 'one', dstFactor: 'one' }, alpha: { srcFactor: 'one', dstFactor: 'one' } } }] },
        primitive: { topology: 'triangle-list' }, depthStencil: { format: caps.depthFormat, depthWriteEnabled: false, depthCompare: 'less' },
      }),
    ]);
    this.updatePipeline = update; this.drawPipeline = draw;
    // Mark every slot unspawned. World extents become available in the first frame UBO.
    const initialBG = this.makeComputeBG();
    const enc = device.createCommandEncoder(); const pass = enc.beginComputePass();
    pass.setPipeline(initialize); pass.setBindGroup(0, this.ctx.frameBG); pass.setBindGroup(1, initialBG);
    pass.dispatchWorkgroups(QualityController.MAX_PARTICLES / 256); pass.end(); device.queue.submit([enc.finish()]);
  }

  private makeComputeBG(): GPUBindGroup {
    return this.ctx.device.createBindGroup({ layout: this.computeBGL, entries: [
      { binding: 0, resource: { buffer: this.pos } }, { binding: 1, resource: { buffer: this.vel } },
      { binding: 2, resource: { buffer: this.light } }, { binding: 3, resource: this.ctx.sdfView },
      { binding: 4, resource: this.ctx.linearClamp }, { binding: 5, resource: { buffer: this.params } },
    ] });
  }

  configure(_size: Size, q: QualitySettings, targets: FrameTargets): void {
    this.colorView = targets.sceneColor.createView();
    this.depthView = targets.sceneDepth.createView();
    this.computeBG = this.makeComputeBG();
    this.drawBG = this.ctx.device.createBindGroup({ layout: this.drawBGL, entries: [
      { binding: 0, resource: { buffer: this.pos } }, { binding: 2, resource: { buffer: this.light } },
    ] });
    const { queue } = this.ctx.device;
    queue.writeBuffer(this.params, 0, new Uint32Array([q.particles, 0, 0, 0]));
    queue.writeBuffer(this.indirect, 0, new Uint32Array([Math.ceil(q.particles / 256), 1, 1]));
    queue.writeBuffer(this.drawIndirect, 0, new Uint32Array([6, q.particles, 0, 0]));
  }

  encode(enc: GPUCommandEncoder, ts: TS, _frameIndex: number): void {
    const pass = enc.beginComputePass({ label: this.name, timestampWrites: ts });
    pass.setPipeline(this.updatePipeline); pass.setBindGroup(0, this.ctx.frameBG); pass.setBindGroup(1, this.computeBG);
    pass.dispatchWorkgroupsIndirect(this.indirect, 0); pass.end();
  }

  encodeDraw(enc: GPUCommandEncoder): void {
    const pass = enc.beginRenderPass({ label: 'Particle overlay', colorAttachments: [
      { view: this.colorView, loadOp: 'load', storeOp: 'store' },
    ], depthStencilAttachment: { view: this.depthView, depthLoadOp: 'load', depthStoreOp: 'store' } });
    pass.setPipeline(this.drawPipeline); pass.setBindGroup(0, this.ctx.frameBG); pass.setBindGroup(1, this.drawBG);
    pass.drawIndirect(this.drawIndirect, 0); pass.end();
  }
}
