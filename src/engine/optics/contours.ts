import type { TextRaster } from '../sdfbake';
import type { Glyph, Loop } from './types';

type Point = [number, number];
const distance2 = (a: Point, b: Point): number => (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2;

/** Douglas–Peucker for an open polyline; endpoints are retained. */
export function simplifyDP(pts: Point[], tol: number): Point[] {
  if (pts.length <= 2) return pts.slice();
  const keep = new Uint8Array(pts.length); keep[0] = keep[pts.length - 1] = 1;
  const stack: [number, number][] = [[0, pts.length - 1]];
  const tolerance2 = Math.max(0, tol) ** 2;
  while (stack.length) {
    const [start, end] = stack.pop()!;
    const a = pts[start], b = pts[end], dx = b[0] - a[0], dy = b[1] - a[1], length2 = dx * dx + dy * dy;
    let furthest = -1, maxDistance2 = tolerance2;
    for (let i = start + 1; i < end; i++) {
      const p = pts[i];
      const t = length2 ? Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / length2)) : 0;
      const d2 = (p[0] - a[0] - t * dx) ** 2 + (p[1] - a[1] - t * dy) ** 2;
      if (d2 > maxDistance2) { maxDistance2 = d2; furthest = i; }
    }
    if (furthest >= 0) { keep[furthest] = 1; stack.push([start, furthest], [furthest, end]); }
  }
  return pts.filter((_, i) => keep[i]);
}

function area(pts: Point[]): number {
  let sum = 0;
  for (let i = 0; i < pts.length; i++) { const a = pts[i], b = pts[(i + 1) % pts.length]; sum += a[0] * b[1] - b[0] * a[1]; }
  return sum / 2;
}

function contains(pts: Point[], p: Point): boolean {
  let inside = false;
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
    const a = pts[i], b = pts[j];
    if ((a[1] > p[1]) !== (b[1] > p[1]) && p[0] < (b[0] - a[0]) * (p[1] - a[1]) / (b[1] - a[1]) + a[0]) inside = !inside;
  }
  return inside;
}

function simplifyClosed(pts: Point[], tolerance: number): Point[] {
  // Split at distant vertices so DP never sees coincident endpoints of a closed loop.
  let first = 0, second = 0;
  for (let i = 1; i < pts.length; i++) if (distance2(pts[i], pts[0]) > distance2(pts[first], pts[0])) first = i;
  for (let i = 0; i < pts.length; i++) if (distance2(pts[i], pts[first]) > distance2(pts[second], pts[first])) second = i;
  const chain = (a: number, b: number): Point[] => {
    const out: Point[] = [pts[a]];
    while (a !== b) { a = (a + 1) % pts.length; out.push(pts[a]); }
    return out;
  };
  const a = simplifyDP(chain(first, second), tolerance), b = simplifyDP(chain(second, first), tolerance);
  return a.slice(0, -1).concat(b.slice(0, -1));
}

function mergeClose(pts: Point[]): Point[] {
  const out = pts.slice();
  let changed = true;
  while (changed && out.length > 3) {
    changed = false;
    for (let i = 0; i < out.length && out.length > 3; i++) {
      if (distance2(out[i], out[(i + 1) % out.length]) < 0.045 ** 2) {
        out.splice((i + 1) % out.length, 1); changed = true;
      }
    }
  }
  return out;
}

function offset(pts: Point[]): Point[] {
  return pts.map((p, i) => {
    const prev = pts[(i + pts.length - 1) % pts.length], next = pts[(i + 1) % pts.length];
    const al = Math.hypot(p[0] - prev[0], p[1] - prev[1]), bl = Math.hypot(next[0] - p[0], next[1] - p[1]);
    let nx = (p[1] - prev[1]) / al + (next[1] - p[1]) / bl;
    let ny = -(p[0] - prev[0]) / al - (next[0] - p[0]) / bl;
    let length = Math.hypot(nx, ny);
    if (length < 1e-8) { nx = (next[1] - p[1]) / bl; ny = -(next[0] - p[0]) / bl; length = 1; }
    return [p[0] + 0.025 * nx / length, p[1] + 0.025 * ny / length];
  });
}

/** Binary marching squares, including an empty border to close glyphs touching the raster boundary. */
function trace(r: TextRaster): Point[][] {
  const { w, h, mask } = r;
  const inside = (x: number, y: number) => x >= 0 && x < w && y >= 0 && y < h && mask[y * w + x] > 127;
  // Edge order: bottom, right, top, left. Saddles preserve disconnected foreground pixels.
  const cases: number[][] = [[], [3, 0], [0, 1], [3, 1], [1, 2], [3, 0, 1, 2], [0, 2], [3, 2],
    [2, 3], [2, 0], [0, 1, 2, 3], [2, 1], [1, 3], [1, 0], [0, 3], []];
  const points = new Map<number, Point>(), neighbors = new Map<number, number[]>();
  const stride = 2 * w + 3;
  const node = (x: number, y: number): number => {
    const key = (y + 1) * stride + x + 1;
    if (!points.has(key)) points.set(key, [x / 2 / r.pxPerUnit - r.halfW, y / 2 / r.pxPerUnit - r.halfH]);
    return key;
  };
  for (let y = -1; y < h; y++) for (let x = -1; x < w; x++) {
    const code = Number(inside(x, y)) | (Number(inside(x + 1, y)) << 1) |
      (Number(inside(x + 1, y + 1)) << 2) | (Number(inside(x, y + 1)) << 3);
    const edges = cases[code]; if (!edges.length) continue;
    const coords: Point[] = [[2 * x + 2, 2 * y + 1], [2 * x + 3, 2 * y + 2], [2 * x + 2, 2 * y + 3], [2 * x + 1, 2 * y + 2]];
    for (let i = 0; i < edges.length; i += 2) {
      const a = node(...coords[edges[i]]), b = node(...coords[edges[i + 1]]);
      if (!neighbors.has(a)) neighbors.set(a, []); if (!neighbors.has(b)) neighbors.set(b, []);
      neighbors.get(a)!.push(b); neighbors.get(b)!.push(a);
    }
  }
  const visited = new Set<number>(), loops: Point[][] = [];
  for (const first of neighbors.keys()) {
    if (visited.has(first)) continue;
    let previous = -1, current = first;
    const loop: Point[] = [];
    do {
      visited.add(current); loop.push(points.get(current)!);
      const next = neighbors.get(current)!.find(n => n !== previous);
      if (next === undefined) break;
      previous = current; current = next;
    } while (current !== first && !visited.has(current));
    if (current === first && loop.length >= 3) loops.push(loop);
  }
  return loops;
}

export function extractGlyphs(r: TextRaster): Glyph[] {
  const raw = trace(r).filter(pts => {
    let perimeter = 0;
    for (let i = 0; i < pts.length; i++) perimeter += Math.sqrt(distance2(pts[i], pts[(i + 1) % pts.length]));
    return perimeter >= 0.3 && Math.abs(area(pts)) > 1e-8;
  });
  const areas = raw.map(pts => Math.abs(area(pts)));
  const parents = raw.map((pts, i) => {
    let parent = -1;
    for (let j = 0; j < raw.length; j++) {
      if (areas[j] > areas[i] && (parent < 0 || areas[j] < areas[parent]) && contains(raw[j], pts[0])) parent = j;
    }
    return parent;
  });
  const holes = raw.map((_, i) => { let depth = 0; for (let p = parents[i]; p >= 0; p = parents[p]) depth++; return depth % 2 === 1; });
  const loops: Loop[] = raw.map((pts, i) => {
    let tolerance = 0.012, simplified = mergeClose(simplifyClosed(pts, tolerance));
    while (simplified.length > 28) { tolerance *= 1.4; simplified = mergeClose(simplifyClosed(pts, tolerance)); }
    // Extremely thin components can collapse to a line at a high simplification tolerance.
    if (simplified.length < 3) simplified = [pts[0], pts[Math.floor(pts.length / 3)], pts[Math.floor(pts.length * 2 / 3)]];
    if ((area(simplified) > 0) === holes[i]) simplified.reverse();
    const shifted = offset(simplified);
    let top = 0;
    for (let j = 1; j < shifted.length; j++) {
      if (shifted[j][1] > shifted[top][1] || (shifted[j][1] === shifted[top][1] && shifted[j][0] < shifted[top][0])) top = j;
    }
    return { pts: shifted.slice(top).concat(shifted.slice(0, top)), hole: holes[i] };
  });
  const glyphs: Glyph[] = [];
  for (let i = 0; i < loops.length; i++) {
    if (holes[i]) continue;
    const inner = loops.filter((_, j) => holes[j] && parents[j] === i);
    inner.sort((a, b) => Math.min(...a.pts.map(p => p[0])) - Math.min(...b.pts.map(p => p[0])));
    const all = [loops[i], ...inner], points = all.flatMap(loop => loop.pts);
    glyphs.push({ loops: all, line: 0, minX: Math.min(...points.map(p => p[0])), maxX: Math.max(...points.map(p => p[0])),
      minY: Math.min(...points.map(p => p[1])), maxY: Math.max(...points.map(p => p[1])) });
  }
  glyphs.sort((a, b) => b.maxY - a.maxY);
  let line = -1, bandTop = Infinity;
  for (const glyph of glyphs) { if (bandTop - glyph.maxY > 0.7) { line++; bandTop = glyph.maxY; } glyph.line = line; }
  return glyphs.sort((a, b) => a.line - b.line || a.minX - b.minX);
}
