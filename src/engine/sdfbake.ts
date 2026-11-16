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

