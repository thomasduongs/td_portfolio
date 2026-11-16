// Bloom chain: hdr -> downsample mips (Karis first) -> additive tent upsample back to mip0.
import type { Caps, EngineCtx, FrameTargets, Pass, QualitySettings, Size, TS } from '../types';
import { FRAME_WGSL } from '../frame';
import SRC from '../shaders/bloom.wgsl?raw';

/** Descriptor of the shared bloom mip chain (pool key 'bloom'); CompositePass uses the same helper to fetch mip0. */
export function bloomDesc(size: Size, q: QualitySettings, caps: Caps): GPUTextureDescriptor {
  return {
    size: [Math.max(1, size.outW >> 1), Math.max(1, size.outH >> 1)],
    format: caps.bloomFormat,
    mipLevelCount: q.bloomMips,
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  };
}

interface Step { view: GPUTextureView; bg: GPUBindGroup; pipe: 0 | 1 | 2 }

export class BloomPass implements Pass {
  readonly name = 'bloom';
  private pipes: GPURenderPipeline[] = []; // [down_first, down, up]
  private readonly bgl: GPUBindGroupLayout;
  private downs: Step[] = [];
  private ups: Step[] = [];
  private ubos: GPUBuffer[] = [];

  constructor(private readonly ctx: EngineCtx) {
    this.bgl = ctx.device.createBindGroupLayout({
      label: 'bloom-bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ],
    });
  }

  async init(): Promise<void> {
    const { device, caps } = this.ctx;
    const module = device.createShaderModule({ label: 'bloom', code: FRAME_WGSL + SRC });
    const layout = device.createPipelineLayout({ bindGroupLayouts: [this.ctx.frameBGL, this.bgl] });
    const additive: GPUBlendComponent = { srcFactor: 'one', dstFactor: 'one', operation: 'add' };
    const mk = (entry: string, blend?: GPUBlendState) => device.createRenderPipelineAsync({
      label: 'bloom-' + entry,
      layout,
      vertex: { module, entryPoint: 'vs' },
      fragment: { module, entryPoint: entry, targets: [{ format: caps.bloomFormat, blend }] },
      primitive: { topology: 'triangle-list' },
    });
    this.pipes = await Promise.all([
      mk('fs_down_first'), mk('fs_down'), mk('fs_up', { color: additive, alpha: additive }),
    ]);
  }

  configure(size: Size, q: QualitySettings, t: FrameTargets): void {
    const { device, caps, pool, linearClamp } = this.ctx;
    for (const b of this.ubos) b.destroy();
    this.ubos = [];
    const tex = pool.texture('bloom', bloomDesc(size, q, caps));
    const n = q.bloomMips;
    const dims: [number, number][] = [];
    for (let i = 0; i < n; i++) dims.push([Math.max(1, tex.width >> i), Math.max(1, tex.height >> i)]);
    const mipView = (i: number) => tex.createView({ baseMipLevel: i, mipLevelCount: 1 });
    const views = Array.from({ length: n }, (_, i) => mipView(i));
    const ubo = (w: number, h: number) => {
      const b = device.createBuffer({ label: 'bloom-ubo', size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(b, 0, new Float32Array([w, h, 0, 0]));
      this.ubos.push(b);
      return b;
    };
    const bg = (src: GPUTextureView, u: GPUBuffer) => device.createBindGroup({
      layout: this.bgl,
      entries: [
        { binding: 0, resource: src }, { binding: 1, resource: linearClamp }, { binding: 2, resource: { buffer: u } },
      ],
    });

    this.downs = [];
    for (let i = 0; i < n; i++) {
      const u = ubo(dims[i][0], dims[i][1]);
      const src = bg(i === 0 ? t.hdr.createView() : views[i - 1], u);
      this.downs.push({ view: views[i], bg: src, pipe: i === 0 ? 0 : 1 });
    }
    // Upsample mip i+1 -> mip i, from the lowest mip upward so each level accumulates the ones below it.
    this.ups = [];
    for (let i = n - 2; i >= 0; i--) {
      const u = ubo(dims[i][0], dims[i][1]);
      this.ups.push({ view: views[i], bg: bg(views[i + 1], u), pipe: 2 });
    }
  }

  encode(enc: GPUCommandEncoder, ts: TS): void {
    const run = (s: Step, load: GPULoadOp, timestampWrites?: GPURenderPassTimestampWrites) => {
      const pass = enc.beginRenderPass({
        label: 'bloom',
        colorAttachments: [{ view: s.view, loadOp: load, storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
        timestampWrites,
      });
      pass.setPipeline(this.pipes[s.pipe]);
      pass.setBindGroup(0, this.ctx.frameBG);
      pass.setBindGroup(1, s.bg);
      pass.draw(3);
      pass.end();
    };
    this.downs.forEach((s, i) => run(s, 'clear', i === 0 ? (ts as GPURenderPassTimestampWrites | undefined) : undefined));
    for (const s of this.ups) run(s, 'load');
  }
}
