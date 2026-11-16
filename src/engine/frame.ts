// CPU mirror of shaders/frame.wgsl (v3). One Float32Array, written in place, one writeBuffer per frame.
import FRAME_WGSL_SRC from './shaders/frame.wgsl?raw';

export const FRAME_WGSL: string = FRAME_WGSL_SRC;
export const FRAME_UBO_SIZE = 304;

// float offsets (byte offset / 4)
export const O = {
  viewProj: 0, invViewProj: 16, view: 32,
  camPos: 48, resolution: 52, time: 56, pointer: 60, misc: 64, beamStyle: 68, counts: 72,
} as const;

export class FrameData {
  readonly f32 = new Float32Array(FRAME_UBO_SIZE / 4);
  set4(off: number, x: number, y: number, z: number, w: number): void {
    const a = this.f32; a[off] = x; a[off + 1] = y; a[off + 2] = z; a[off + 3] = w;
  }
  setMat(off: number, m: Float32Array): void { this.f32.set(m, off); }
  upload(queue: GPUQueue, buf: GPUBuffer): void { queue.writeBuffer(buf, 0, this.f32); }
}

/** Bind group layout used as @group(0) by every pipeline. */
export function frameBindGroupLayout(device: GPUDevice): GPUBindGroupLayout {
  return device.createBindGroupLayout({
    label: 'frame-bgl',
    entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }],
  });
}
