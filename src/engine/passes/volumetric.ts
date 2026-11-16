import type { EngineCtx, FrameTargets, Pass, QualitySettings, Size, TS } from '../types';
import { FRAME_WGSL } from '../frame';
import SOURCE from '../shaders/froxel.wgsl?raw';

export class VolumetricPass implements Pass {
  readonly name = 'volumetric';
  private injectBGL!: GPUBindGroupLayout;
  private integrateBGL!: GPUBindGroupLayout;
  private inject!: GPUComputePipeline;
  private injectFresh!: GPUComputePipeline;
  private integrate!: GPUComputePipeline;
  private injectBG!: GPUBindGroup[];
  private integrateBG!: GPUBindGroup[];
  private grid: [number, number, number] = [1, 1, 1];
  private historyValid = false;

  constructor(private ctx: EngineCtx) {}

  async init(): Promise<void> {
    const { device, frameBGL } = this.ctx;
    this.injectBGL = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, sampler: { type: 'filtering' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float', viewDimension: '3d' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '3d' } },
    ] });
    this.integrateBGL = device.createBindGroupLayout({ entries: [
      { binding: 4, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float', viewDimension: '3d' } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '3d' } },
    ] });
    const module = device.createShaderModule({ label: 'Froxel fog', code: FRAME_WGSL + SOURCE });
    const injectionLayout = device.createPipelineLayout({ bindGroupLayouts: [frameBGL, this.injectBGL] });
    [this.inject, this.injectFresh, this.integrate] = await Promise.all([
      device.createComputePipelineAsync({ layout: injectionLayout, compute: { module, entryPoint: 'inject' } }),
      device.createComputePipelineAsync({ layout: injectionLayout, compute: { module, entryPoint: 'injectFresh' } }),
      device.createComputePipelineAsync({ layout: device.createPipelineLayout({ bindGroupLayouts: [frameBGL, this.integrateBGL] }), compute: { module, entryPoint: 'integrate' } }),
    ]);
  }

  configure(_size: Size, q: QualitySettings, targets: FrameTargets): void {
    this.grid = [...q.froxel]; this.historyValid = false;
    const views = ['froxelA', 'froxelB'].map(key => this.ctx.pool.texture(key, {
      size: this.grid, dimension: '3d', format: 'rgba16float', usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    }).createView());
    const output = targets.froxelIntegrated.createView();
    this.injectBG = views.map((view, i) => this.ctx.device.createBindGroup({ layout: this.injectBGL, entries: [
      { binding: 0, resource: this.ctx.sdfView }, { binding: 1, resource: this.ctx.linearClamp },
      { binding: 2, resource: views[1 - i] }, { binding: 3, resource: view },
    ] }));
    this.integrateBG = views.map(view => this.ctx.device.createBindGroup({ layout: this.integrateBGL, entries: [
      { binding: 4, resource: view }, { binding: 5, resource: output },
    ] }));
  }

  encode(enc: GPUCommandEncoder, ts: TS, frameIndex: number): void {
    const parity = frameIndex & 1;
    const inject = enc.beginComputePass({ label: 'Fog injection', timestampWrites: ts });
    inject.setPipeline(this.historyValid ? this.inject : this.injectFresh);
    inject.setBindGroup(0, this.ctx.frameBG); inject.setBindGroup(1, this.injectBG[parity]);
    inject.dispatchWorkgroups(Math.ceil(this.grid[0] / 4), Math.ceil(this.grid[1] / 4), Math.ceil(this.grid[2] / 4)); inject.end();
    const integrate = enc.beginComputePass({ label: 'Fog integration' });
    integrate.setPipeline(this.integrate); integrate.setBindGroup(0, this.ctx.frameBG); integrate.setBindGroup(1, this.integrateBG[parity]);
    integrate.dispatchWorkgroups(Math.ceil(this.grid[0] / 8), Math.ceil(this.grid[1] / 8)); integrate.end();
    this.historyValid = true;
  }
}
