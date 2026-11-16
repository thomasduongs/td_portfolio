// One-time bake of the 64^3 tileable haze noise (rgba8unorm). Runs once at startup, then the texture is read-only.
import NOISE_WGSL from './shaders/noise.wgsl?raw';

const SIZE = 64;
const WORKGROUP = 4;

export async function bakeNoise(device: GPUDevice): Promise<GPUTexture> {
  const tex = device.createTexture({
    label: 'haze-noise-3d',
    size: [SIZE, SIZE, SIZE],
    dimension: '3d',
    format: 'rgba8unorm',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
  });

  // Group 0 is unused by the shader; keep an empty layout there so the output sits at group 1.
  const empty = device.createBindGroupLayout({ entries: [] });
  const outBGL = device.createBindGroupLayout({
    entries: [{
      binding: 0,
      visibility: GPUShaderStage.COMPUTE,
      storageTexture: { access: 'write-only', format: 'rgba8unorm', viewDimension: '3d' },
    }],
  });
  const pipeline = await device.createComputePipelineAsync({
    label: 'haze-noise',
    layout: device.createPipelineLayout({ bindGroupLayouts: [empty, outBGL] }),
    compute: { module: device.createShaderModule({ label: 'noise.wgsl', code: NOISE_WGSL }), entryPoint: 'main' },
  });
  const bg = device.createBindGroup({ layout: outBGL, entries: [{ binding: 0, resource: tex.createView({ dimension: '3d' }) }] });

  const enc = device.createCommandEncoder({ label: 'haze-noise' });
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(1, bg);
  const n = SIZE / WORKGROUP;
  pass.dispatchWorkgroups(n, n, n);
  pass.end();
  device.queue.submit([enc.finish()]);
  await device.queue.onSubmittedWorkDone();
  return tex;
}
