// Keyed GPU resource cache. Same key + same descriptor -> same object; changed descriptor -> old one destroyed.
import type { ResourcePoolLike } from './types';

interface Entry<T> { obj: T; descHash: string }

export class ResourcePool implements ResourcePoolLike {
  private textures = new Map<string, Entry<GPUTexture>>();
  private buffers = new Map<string, Entry<GPUBuffer>>();

  constructor(private device: GPUDevice) {}

  texture(key: string, desc: GPUTextureDescriptor): GPUTexture {
    const hash = JSON.stringify([desc.size, desc.format, desc.usage, desc.dimension ?? '2d', desc.mipLevelCount ?? 1]);
    const hit = this.textures.get(key);
    if (hit && hit.descHash === hash) return hit.obj;
    hit?.obj.destroy();
    const obj = this.device.createTexture({ ...desc, label: key });
    this.textures.set(key, { obj, descHash: hash });
    return obj;
  }

  buffer(key: string, desc: GPUBufferDescriptor): GPUBuffer {
    const hash = JSON.stringify([desc.size, desc.usage]);
    const hit = this.buffers.get(key);
    if (hit && hit.descHash === hash) return hit.obj;
    hit?.obj.destroy();
    const obj = this.device.createBuffer({ ...desc, label: key });
    this.buffers.set(key, { obj, descHash: hash });
    return obj;
  }
}
