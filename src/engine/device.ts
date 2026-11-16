// WebGPU device + canvas context setup. Returns null when WebGPU is unavailable (host falls back to WebGL2).
import type { Caps, DeviceTier } from './types';

export interface GpuDeviceInit {
  device: GPUDevice;
  context: GPUCanvasContext;
  caps: Caps;
}

const OPTIONAL_FEATURES = [
  'timestamp-query',
  'shader-f16',
  'rg11b10ufloat-renderable',
  'float32-filterable',
  'subgroups',
] as const;

export async function initDevice(canvas: OffscreenCanvas | HTMLCanvasElement, tier: DeviceTier): Promise<GpuDeviceInit | null> {
  if (!navigator.gpu) return null;
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: tier === 'mobile' ? 'low-power' : 'high-performance',
  });
  if (!adapter) return null;

  const requiredFeatures = OPTIONAL_FEATURES.filter((f) => adapter.features.has(f as GPUFeatureName)) as GPUFeatureName[];
  const device = await adapter.requestDevice({ requiredFeatures });

  const context = (canvas as unknown as { getContext(id: 'webgpu'): GPUCanvasContext | null }).getContext('webgpu');
  if (!context) return null;
  const presentFormat = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format: presentFormat, alphaMode: 'opaque' });

  const has = (f: string) => device.features.has(f as GPUFeatureName);
  const rg11b10Renderable = has('rg11b10ufloat-renderable');
  const caps: Caps = {
    timestamps: has('timestamp-query'),
    f16: has('shader-f16'),
    rg11b10Renderable,
    float32Filterable: has('float32-filterable'),
    subgroups: has('subgroups'),
    presentFormat,
    hdrFormat: 'rgba16float',
    bloomFormat: rg11b10Renderable ? 'rg11b10ufloat' : 'rgba16float',
    depthFormat: 'depth32float',
    tier,
    maxWorkgroupInvocations: device.limits.maxComputeInvocationsPerWorkgroup,
  };

  device.lost.then((info) => console.error('WebGPU device lost:', info.reason, info.message));
  device.addEventListener('uncapturederror', (e) => console.error(e.error.message));

  return { device, context, caps };
}
