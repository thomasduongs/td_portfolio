// CPU mirror of shaders/frame.wgsl. One Float32Array, written in place every frame, uploaded with a
// single queue.writeBuffer (656 B) -> no per-frame allocation.
import FRAME_WGSL_SRC from './shaders/frame.wgsl?raw';
import BEAMS_WGSL_SRC from './shaders/beams.wgsl?raw';

export const FRAME_WGSL: string = FRAME_WGSL_SRC;
/** Beam library; consumers compile FRAME_WGSL + BEAMS_WGSL + source and declare `sdfTex` + `samp`. */
export const BEAMS_WGSL: string = BEAMS_WGSL_SRC;
export const FRAME_UBO_SIZE = 800;
export const MAX_MARBLES = 16;
export const NUM_BEAMS = 3;

// float offsets (byte offset / 4)
export const O = {
  viewProj: 0, invViewProj: 16, prevViewProj: 32,
  camPos: 48, resolution: 52, outRes: 56, jitter: 60, time: 64, pointer: 68,
  textParams: 72, quality: 76, misc: 80, beamParams: 84,
  beams: 88,   // + 16 * j : pos +0, dir +4, color +8, align +12
  marbles: 136,
} as const;
export const BEAM_STRIDE = 16;

export class FrameData {
  readonly f32 = new Float32Array(FRAME_UBO_SIZE / 4);
  set4(off: number, x: number, y: number, z: number, w: number): void {
    const a = this.f32; a[off] = x; a[off + 1] = y; a[off + 2] = z; a[off + 3] = w;
  }
  setMat(off: number, m: Float32Array): void { this.f32.set(m, off); }
  upload(queue: GPUQueue, buf: GPUBuffer): void { queue.writeBuffer(buf, 0, this.f32); }
}

/** Create the bind group layout used as @group(0) by every pipeline. */
export function frameBindGroupLayout(device: GPUDevice): GPUBindGroupLayout {
  return device.createBindGroupLayout({
    label: 'frame-bgl',
    entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }],
  });
}
