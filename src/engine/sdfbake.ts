import JFA_WGSL from './shaders/jfa.wgsl?raw';

export interface TextRaster {
  lines: string[]; w: number; h: number; mask: Uint8Array;
  pxPerUnit: number; halfW: number; halfH: number;
  glyphHalfW: number; glyphHalfH: number; floorY: number;
}

/** Rows ascend in world Y, matching the SDF texture's world-space UVs. */
export function rasterizeText(lines: string[]): TextRaster {
  lines = lines.length ? lines : [''];
  const canvas = new OffscreenCanvas(1, 1);
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) throw new Error('Cannot rasterize text: 2D canvas unavailable');
  const setFont = (px: number) => { ctx.font = `800 ${px}px "Helvetica Neue", Helvetica, Arial, sans-serif`; };
  setFont(100);
  const capRatio = (ctx.measureText('H').actualBoundingBoxAscent || 72) / 100;
  const advance = (line: string, px: number) => Array.from(line).reduce((n, c) => n + ctx.measureText(c).width, 0)
    + Math.max(0, Array.from(line).length - 1) * px * 0.06;
  const widthRatio = Math.max(1, ...lines.map(line => advance(line, 100))) / 100;
  const w = lines.length === 1 ? 2048 : 1536;
  const fontPx = w / (widthRatio + 0.6 * capRatio);
  setFont(fontPx);
  const pxPerUnit = ctx.measureText('H').actualBoundingBoxAscent || fontPx * capRatio;
  const glyphHalfH = (1 + (lines.length - 1) * 1.45) / 2;
  const h = Math.ceil((glyphHalfH * 2 + 0.6) * pxPerUnit);
  canvas.width = w; canvas.height = h;
  setFont(fontPx); ctx.fillStyle = '#fff'; ctx.textBaseline = 'alphabetic';
  let maxWidth = 0;
  lines.forEach((line, i) => {
    const width = advance(line, fontPx); maxWidth = Math.max(maxWidth, width);
    let x = (w - width) / 2;
    const baselineWorld = glyphHalfH - 1 - i * 1.45;
    const y = h / 2 - baselineWorld * pxPerUnit;
    for (const c of line) { ctx.fillText(c, x, y); x += ctx.measureText(c).width + fontPx * 0.06; }
  });
  const pixels = ctx.getImageData(0, 0, w, h).data;
  const mask = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    mask[y * w + x] = pixels[((h - 1 - y) * w + x) * 4 + 3] > 127 ? 255 : 0;
  }
  return { lines, w, h, mask, pxPerUnit, halfW: w / pxPerUnit / 2, halfH: h / pxPerUnit / 2,
    glyphHalfW: maxWidth / pxPerUnit / 2, glyphHalfH, floorY: -glyphHalfH };
}

/** Exact squared Euclidean distance transform, separable lower envelope of parabolas. */
function edt(mask: Uint8Array, w: number, h: number, inside: boolean): Float64Array {
  const result = new Float64Array(w * h);
  const n = Math.max(w, h), f = new Float64Array(n), d = new Float64Array(n);
  const v = new Int32Array(n), z = new Float64Array(n + 1);
  const far = (w * w + h * h) * 4;
  const transform = (length: number) => {
    let k = 0; v[0] = 0; z[0] = -Infinity; z[1] = Infinity;
    for (let q = 1; q < length; q++) {
      let s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * (q - v[k]));
      while (s <= z[k]) { k--; s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * (q - v[k])); }
      k++; v[k] = q; z[k] = s; z[k + 1] = Infinity;
    }
    k = 0;
    for (let q = 0; q < length; q++) { while (z[k + 1] < q) k++; d[q] = (q - v[k]) ** 2 + f[v[k]]; }
  };
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) f[x] = (mask[y * w + x] > 0) === inside ? 0 : far;
    transform(w); for (let x = 0; x < w; x++) result[y * w + x] = d[x];
  }
  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) f[y] = result[y * w + x];
    transform(h); for (let y = 0; y < h; y++) result[y * w + x] = d[y];
  }
  return result;
}

export function bakeSDFCPU(r: TextRaster): Float32Array {
  const toInside = edt(r.mask, r.w, r.h, true), toOutside = edt(r.mask, r.w, r.h, false);
  const sdf = new Float32Array(r.w * r.h);
  for (let i = 0; i < sdf.length; i++) {
    sdf[i] = (Math.max(0, Math.sqrt(toInside[i]) - 0.5) - Math.max(0, Math.sqrt(toOutside[i]) - 0.5)) / r.pxPerUnit;
  }
  return sdf;
}

export async function bakeSDFGPU(device: GPUDevice, r: TextRaster): Promise<GPUTexture> {
  const temporary: GPUBuffer[] = [];
  const buffer = (label: string, size: number, usage: GPUBufferUsageFlags) => {
    const b = device.createBuffer({ label, size, usage }); temporary.push(b); return b;
  };
  const seeds = [0, 1].map(i => buffer(`JFA seeds ${i}`, r.w * r.h * 16, GPUBufferUsage.STORAGE));
  const mask = buffer('Text mask', r.w * r.h * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  device.queue.writeBuffer(mask, 0, Uint32Array.from(r.mask));
  const steps: number[] = [];
  for (let step = 2 ** Math.ceil(Math.log2(Math.max(r.w, r.h))) / 2; step >= 1; step /= 2) steps.push(step);
  steps.push(1);
  const params = buffer('JFA parameters', (steps.length + 1) * 256, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const data = new ArrayBuffer((steps.length + 1) * 256);
  [0, ...steps].forEach((step, i) => {
    new Uint32Array(data, i * 256, 3).set([r.w, r.h, step]); new Float32Array(data, i * 256 + 12, 1)[0] = r.pxPerUnit;
  });
  device.queue.writeBuffer(params, 0, data);
  const texture = device.createTexture({ label: 'Text SDF', size: [r.w, r.h], format: 'rgba16float',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING });
  const bgl = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: 16 } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 4, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } },
  ] });
  const module = device.createShaderModule({ label: 'JFA', code: JFA_WGSL });
  const layout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
  try {
    const [init, jump, finish] = await Promise.all(['seed', 'jump', 'finish'].map(entryPoint =>
      device.createComputePipelineAsync({ layout, compute: { module, entryPoint } })));
    const groups = seeds.map((_, i) => device.createBindGroup({ layout: bgl, entries: [
      { binding: 0, resource: { buffer: params, size: 16 } }, { binding: 1, resource: { buffer: mask } },
      { binding: 2, resource: { buffer: seeds[i] } }, { binding: 3, resource: { buffer: seeds[1 - i] } },
      { binding: 4, resource: texture.createView() },
    ] }));
    const enc = device.createCommandEncoder({ label: 'Bake text SDF' });
    const dispatch = (pipeline: GPUComputePipeline, group: number, offset: number) => {
      const pass = enc.beginComputePass(); pass.setPipeline(pipeline); pass.setBindGroup(0, groups[group], [offset]);
      pass.dispatchWorkgroups(Math.ceil(r.w / 8), Math.ceil(r.h / 8)); pass.end();
    };
    dispatch(init, 1, 0); // Seed writes buffer 0.
    let parity = 0;
    steps.forEach((_, i) => { dispatch(jump, parity, (i + 1) * 256); parity = 1 - parity; });
    dispatch(finish, parity, 0);
    device.queue.submit([enc.finish()]); await device.queue.onSubmittedWorkDone();
    return texture;
  } catch (error) { texture.destroy(); throw error; }
  finally { temporary.forEach(b => b.destroy()); }
}
