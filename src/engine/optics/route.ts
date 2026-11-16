// Optics solver: builds the laser installation that draws the name. Everything is geometrically exact:
//   laser source → glass prism (Snell refraction in/out, dispersion fan) → router mirror → glyph loop
//   (outline mirror at every vertex, normal = bisector of in/out edges) → loop closes on the entry
//   splitter, whose reflection fixes the exit ray → router mirror on that ray → next loop → … →
//   final router → glass box (refraction in/out, lateral shift) → off frame.
// The only free parameter per hop is WHERE the router sits along the incoming ray; we search it so the
// resulting exit ray is tidy (out of the text plane, heading toward the next glyph, inside the frame).
import type { Glyph, Loop, OpticsScene, Segment, Mirror, Glass, V3 } from './types';
import { SegKind, MirrorKind, GlassKind } from './types';

export interface Layout { glyphHalfW: number; glyphHalfH: number; floorY: number; lines: number }

// ---------- vec3 ----------
const add = (a: V3, b: V3): V3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const mul = (a: V3, s: number): V3 => [a[0] * s, a[1] * s, a[2] * s];
const dot = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const len = (a: V3) => Math.hypot(a[0], a[1], a[2]);
const norm = (a: V3): V3 => mul(a, 1 / (len(a) || 1));
const reflect = (d: V3, n: V3): V3 => sub(d, mul(n, 2 * dot(d, n)));
/** Snell refraction; n faces against d; eta = n1/n2. Returns null on total internal reflection. */
function refract(d: V3, n: V3, eta: number): V3 | null {
  const c = -dot(n, d), k = 1 - eta * eta * (1 - c * c);
  return k < 0 ? null : norm(add(mul(d, eta), mul(n, eta * c - Math.sqrt(k))));
}
/** Mirror normal that turns direction dIn into dOut. */
const bisector = (dIn: V3, dOut: V3): V3 => norm(sub(dOut, dIn));

// ---------- convex glass ----------
interface Plane { n: V3; d: number } // inside: dot(n, p) <= d
function glassPlanes(g: Glass): Plane[] {
  const ps: Plane[] = [];
  const c = g.center;
  if (g.kind === GlassKind.Box) {
    for (const [ax, h] of [[g.ax, g.half[0]], [g.ay, g.half[1]], [g.az, g.half[2]]] as [V3, number][]) {
      ps.push({ n: ax, d: dot(ax, c) + h }, { n: mul(ax, -1), d: -dot(ax, c) + h });
    }
  } else {
    // equilateral triangle in local XY (circumradius R), side normals at 270°, 30°, 150°, inradius R/2
    const R = g.half[0];
    for (const deg of [270, 30, 150]) {
      const a = (deg * Math.PI) / 180;
      const n = norm(add(mul(g.ax, Math.cos(a)), mul(g.ay, Math.sin(a))));
      ps.push({ n, d: dot(n, c) + R / 2 });
    }
    ps.push({ n: g.az, d: dot(g.az, c) + g.half[2] }, { n: mul(g.az, -1), d: -dot(g.az, c) + g.half[2] });
  }
  return ps;
}
/** Entry of an outside ray into a convex solid: t and outward normal. */
function enter(o: V3, d: V3, ps: Plane[]): { t: number; n: V3 } | null {
  let tIn = -Infinity, tOut = Infinity, nIn: V3 = [0, 0, 0];
  for (const p of ps) {
    const den = dot(p.n, d), num = p.d - dot(p.n, o);
    if (Math.abs(den) < 1e-9) { if (num < 0) return null; continue; }
    const t = num / den;
    if (den < 0) { if (t > tIn) { tIn = t; nIn = p.n; } } else if (t < tOut) tOut = t;
  }
  return tIn < tOut && tIn > 1e-6 ? { t: tIn, n: nIn } : null;
}
/** Exit of an inside ray: t and outward normal. */
function leave(o: V3, d: V3, ps: Plane[]): { t: number; n: V3 } {
  let best = Infinity, nb: V3 = ps[0].n;
  for (const p of ps) {
    const den = dot(p.n, d);
    if (den > 1e-9) { const t = (p.d - dot(p.n, o)) / den; if (t > 1e-6 && t < best) { best = t; nb = p.n; } }
  }
  return { t: best, n: nb };
}
/** Trace through glass; returns interior points and outgoing ray, or null on miss / trapped light. */
function traceGlass(o: V3, d: V3, g: Glass, ior: number): { pIn: V3; inner: V3[]; pOut: V3; dOut: V3 } | null {
  const ps = glassPlanes(g);
  const e = enter(o, d, ps);
  if (!e) return null;
  const pIn = add(o, mul(d, e.t));
  let dir = refract(d, e.n, 1 / ior);
  if (!dir) return null;
  let p = pIn;
  const inner: V3[] = [];
  for (let bounce = 0; bounce < 4; bounce++) {
    const x = leave(p, dir, ps);
    const q = add(p, mul(dir, x.t));
    const out = refract(dir, mul(x.n, -1), ior);
    if (out) return { pIn, inner, pOut: q, dOut: out };
    inner.push(q);                       // total internal reflection: bounce inside and continue
    dir = reflect(dir, mul(x.n, -1));
    p = q;
  }
  return null;
}

// ---------- scene builder ----------
const IOR = 1.5;
const DISPERSION: { ior: number; color: V3 }[] = [
  { ior: 1.468, color: [1.0, 0.12, 0.06] }, { ior: 1.484, color: [1.0, 0.55, 0.05] },
  { ior: 1.516, color: [0.1, 0.85, 1.0] }, { ior: 1.532, color: [0.45, 0.15, 1.0] },
];
const LASER: V3[] = [[0.85, 0.95, 1.0], [1.0, 0.93, 0.82]]; // cool white, warm white
const GAIN = 2.4;
const OUTLINE_MIRROR = 0.03, SPLITTER = 0.045, ROUTER = 0.085;

class Builder {
  segs: Segment[] = []; mirrors: Mirror[] = []; glass: Glass[] = []; sources: V3[] = [];
  s = 0; color: V3 = [1, 1, 1];
  seg(a: V3, b: V3, kind = SegKind.Beam, color = this.color) {
    const l = len(sub(b, a));
    if (l < 1e-5) return;
    this.segs.push({ a, b, sA: this.s, sB: this.s + l, color: mul(color, GAIN), kind });
    this.s += l;
  }
  mirror(pos: V3, normal: V3, halfSize: number, kind: MirrorKind) {
    this.mirrors.push({ pos, normal, halfSize, sArrive: this.s, color: mul(this.color, GAIN), kind });
  }
}

interface Frame { halfW: number; yMin: number; yMax: number }

export function buildOptics(glyphs: Glyph[], L: Layout): OpticsScene {
  const B = new Builder();
  const frame: Frame = { halfW: L.glyphHalfW * 1.32, yMin: L.floorY + 0.12, yMax: L.glyphHalfH + 1.9 };
  const chains = splitChains(glyphs, L.lines);
  chains.forEach((chain, ci) => buildChain(B, chain, ci, L, frame));
  const pathLength = Math.max(0, ...B.segs.map((s) => s.sB));
  return { segments: B.segs, mirrors: B.mirrors, glass: B.glass, pathLength, sources: B.sources };
}

/** Two chains: words of a single line (split at the widest gap), or the two lines. Chain 1 runs right→left. */
function splitChains(glyphs: Glyph[], lines: number): Glyph[][] {
  let a: Glyph[], b: Glyph[];
  if (lines > 1) {
    a = glyphs.filter((g) => g.line === 0); b = glyphs.filter((g) => g.line !== 0);
  } else {
    const sorted = [...glyphs].sort((p, q) => p.minX - q.minX);
    let cut = 1, gap = -Infinity;
    for (let i = 1; i < sorted.length; i++) {
      const gp = sorted[i].minX - sorted[i - 1].maxX;
      if (gp > gap) { gap = gp; cut = i; }
    }
    a = sorted.slice(0, cut); b = sorted.slice(cut);
  }
  a.sort((p, q) => p.minX - q.minX);
  b.sort((p, q) => q.minX - p.minX);
  return [a, b].filter((c) => c.length);
}

function loopsOf(chain: Glyph[]): Loop[] {
  const out: Loop[] = [];
  for (const g of chain) for (const l of g.loops) if (l.pts.length >= 3) out.push(l);
  return out;
}
const P3 = (p: [number, number]): V3 => [p[0], p[1], 0];

function inFrame(p: V3, f: Frame): boolean {
  return Math.abs(p[0]) < f.halfW && p[1] > f.yMin && p[1] < f.yMax && Math.abs(p[2]) < 1.8;
}

function buildChain(B: Builder, chain: Glyph[], ci: number, L: Layout, frame: Frame) {
  B.s = 0;
  B.color = LASER[ci % LASER.length];
  const side = ci === 0 ? -1 : 1;                          // chain 0 enters from the left
  const loops = loopsOf(chain);
  if (!loops.length) return;
  const lineTop = Math.max(...chain.map((g) => g.maxY));

  // --- source + prism ---
  // prism stands on the floor beside the word (at letter height it would refract a letter into view)
  const lineBottom = Math.min(...chain.map((g) => g.minY));
  const prismC: V3 = [side * (L.glyphHalfW + 0.78), Math.max(L.floorY, lineBottom) + 0.36, 0.55];
  const src: V3 = add(prismC, [side * 3.2, 1.5, 1.1]);
  B.sources.push(src);
  const toPrism = norm(sub(prismC, src));
  const first = P3(loops[0].pts[0]);
  let best: { g: Glass; tr: NonNullable<ReturnType<typeof traceGlass>>; score: number } | null = null;
  for (let deg = 0; deg < 120; deg += 1) {
    const a = (deg * Math.PI) / 180;
    const g: Glass = { center: prismC, ax: [Math.cos(a), 0, -Math.sin(a)], ay: [Math.sin(a), 0, Math.cos(a)], az: [0, 1, 0], half: [0.42, 0, 0.36], kind: GlassKind.Prism };
    const tr = traceGlass(src, toPrism, g, IOR);
    if (!tr || tr.inner.length) continue;
    const score = dot(tr.dOut, norm(sub(first, tr.pOut)));
    if (!best || score > best.score) best = { g, tr, score };
  }
  let ray: { o: V3; d: V3 };
  if (best) {
    const { g, tr } = best;
    B.glass.push(g);
    B.seg(src, tr.pIn);
    B.seg(tr.pIn, tr.pOut, SegKind.InGlass);
    const sExit = B.s;
    for (const k of DISPERSION) {                          // dispersion fan: exact per-wavelength Snell
      const t2 = traceGlass(src, toPrism, g, k.ior);
      if (!t2) continue;
      B.segs.push({ a: t2.pOut, b: add(t2.pOut, mul(t2.dOut, 2.2)), sA: sExit, sB: sExit + 2.2, color: mul(k.color, GAIN * 0.55), kind: SegKind.Dispersion });
    }
    ray = { o: tr.pOut, d: tr.dOut };
  } else {
    ray = { o: src, d: norm(sub(first, src)) };
  }

  // --- glyph loops ---
  for (let i = 0; i < loops.length; i++) {
    const loop = loops[i];
    const pts = loop.pts.map(P3);
    const n = pts.length;
    const P0 = pts[0];
    const d1 = norm(sub(pts[1], P0));
    const dLast = norm(sub(P0, pts[n - 1]));
    const next: V3 = i + 1 < loops.length ? P3(loops[i + 1].pts[0]) : endTarget(chain, side, L);

    // choose router distance along the incoming ray
    let pick: { M: V3; f: V3; n0: V3; e: V3; score: number } | null = null;
    for (let l = 0.35; l <= 4.0; l += 0.05) {
      const M = add(ray.o, mul(ray.d, l));
      if (!inFrame(M, frame) && i > 0) continue;
      if (len(sub(P0, M)) < 0.3) continue;
      const f = norm(sub(P0, M));
      if (dot(f, ray.d) > 0.995) continue;                 // no deflection needed: degenerate mirror
      const n0 = bisector(f, d1);
      const e = reflect(dLast, n0);
      let score = dot(e, norm(sub(next, P0)));
      score -= Math.abs(Math.abs(e[2]) - 0.45) * 0.8;      // prefer exits leaving the text plane
      score -= Math.abs(M[2]) < 0.2 ? 0.6 : 0;              // routers off the ink plane
      score -= l * 0.04;
      score -= Math.max(0, M[1] - (lineTop + 0.45)) * 1.6;   // keep routers close to the letter tops (no tall spikes)
      score -= Math.max(0, Math.abs(e[1]) - 0.5) * 0.8;       // exits that shoot steeply up/down make long connectors
      score += exitHasRoom(P0, e, frame) ? 0 : -2;
      if (!pick || score > pick.score) pick = { M, f, n0, e, score };
    }
    if (!pick) {
      const M = add(ray.o, mul(ray.d, 1));
      const f = norm(sub(P0, M)), n0 = bisector(f, d1);
      pick = { M, f, n0, e: reflect(dLast, n0), score: 0 };
    }
    B.seg(ray.o, pick.M);
    B.mirror(pick.M, bisector(ray.d, pick.f), ROUTER, MirrorKind.Router);
    B.seg(pick.M, P0);
    B.mirror(P0, pick.n0, SPLITTER, MirrorKind.Splitter);
    for (let k = 0; k < n; k++) {
      const a = pts[k], b = pts[(k + 1) % n];
      B.seg(a, b);
      if (k + 1 < n) {                                      // outline mirror at every vertex except the entry
        const dIn = norm(sub(b, a)), dOut = norm(sub(pts[(k + 2) % n], b));
        B.mirror(b, bisector(dIn, dOut), OUTLINE_MIRROR, MirrorKind.Outline);
      }
    }
    ray = { o: P0, d: pick.e };
  }

  // --- exit: router → glass box → off frame ---
  const boxC = endTarget(chain, side, L);
  const tilt = side * 0.45;
  const box: Glass = {
    center: boxC,
    ax: norm([Math.cos(tilt), 0, -Math.sin(tilt)]), ay: norm([Math.sin(tilt) * 0.26, 0.966, Math.cos(tilt) * 0.26]),
    az: [0, 0, 0], half: [0.3, 0.3, 0.3], kind: GlassKind.Box,
  };
  box.az = norm([box.ax[1] * box.ay[2] - box.ax[2] * box.ay[1], box.ax[2] * box.ay[0] - box.ax[0] * box.ay[2], box.ax[0] * box.ay[1] - box.ax[1] * box.ay[0]]);
  box.ay = norm([box.az[1] * box.ax[2] - box.az[2] * box.ax[1], box.az[2] * box.ax[0] - box.az[0] * box.ax[2], box.az[0] * box.ax[1] - box.az[1] * box.ax[0]]);
  let M = add(ray.o, mul(ray.d, 0.6));
  for (let l = 0.4; l <= 3.0; l += 0.05) { const c = add(ray.o, mul(ray.d, l)); if (inFrame(c, frame) && len(sub(c, boxC)) > 0.8) { M = c; break; } }
  // aim slightly off-centre so the lateral shift through the tilted box is visible
  const aim = norm(sub(add(boxC, [side * 0.06, 0, 0.05]), M));
  B.seg(ray.o, M);
  B.mirror(M, bisector(ray.d, aim), ROUTER, MirrorKind.Router);
  const tr = traceGlass(M, aim, box, IOR);
  if (tr) {
    B.glass.push(box);
    B.seg(M, tr.pIn);
    let p = tr.pIn;
    for (const q of tr.inner) { B.seg(p, q, SegKind.InGlass); p = q; }
    B.seg(p, tr.pOut, SegKind.InGlass);
    B.seg(tr.pOut, add(tr.pOut, mul(tr.dOut, 30)));
  } else {
    B.seg(M, add(M, mul(aim, 30)));
  }
}

/** Where the chain's exit glass box sits: above the chain's last glyph, toward the centre line. */
function endTarget(chain: Glyph[], side: number, L: Layout): V3 {
  const last = chain[chain.length - 1];
  // lower line of a two-line layout: exit sideways past its last glyph (above would sit on the top line)
  if (L.lines > 1 && chain[0].line > 0) return [side > 0 ? last.minX - 0.85 : last.maxX + 0.85, (last.minY + last.maxY) / 2 - 0.15, 0.45];
  const x = (last.minX + last.maxX) / 2 - side * 0.1;
  return [x * 0.75, last.maxY + 1.05, 0.45];
}

/** True if some router position along the exit ray lies inside the frame (so the next hop is drawable). */
function exitHasRoom(P0: V3, e: V3, f: Frame): boolean {
  for (let l = 0.4; l <= 3.0; l += 0.3) if (inFrame(add(P0, mul(e, l)), f)) return true;
  return false;
}
