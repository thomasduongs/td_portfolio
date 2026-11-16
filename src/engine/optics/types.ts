// Optics scene: the laser paths, mirrors and glass objects that draw the name.
// Built on the CPU once per text layout (src/engine/optics/route.ts), uploaded once as storage buffers.
// GPU layouts are mirrored in shaders/frame.wgsl (Segment 48 B, Mirror 48 B, Glass 64 B).

export type V3 = [number, number, number];

/** A closed outline of one glyph contour on the plane z = 0 (world units, cap height = 1). */
export interface Loop {
  pts: [number, number][]; // >= 3 vertices, NOT repeated at the end. pts[0] = entry vertex (topmost, then leftmost).
  hole: boolean;            // inner contour (counter of O, A, D...)
}
/** One glyph: its outer loop(s) first, then holes. Glyphs ordered reading order (line by line, left to right). */
export interface Glyph { loops: Loop[]; line: number; minX: number; maxX: number; minY: number; maxY: number }

export const enum SegKind { Beam = 0, InGlass = 1, Dispersion = 2 }
export const enum MirrorKind { Outline = 0, Router = 1, Splitter = 2 }
export const enum GlassKind { Prism = 0, Box = 1 }

export interface Segment { a: V3; b: V3; sA: number; sB: number; color: V3; kind: SegKind }
export interface Mirror { pos: V3; normal: V3; halfSize: number; sArrive: number; color: V3; kind: MirrorKind }
export interface Glass { center: V3; ax: V3; ay: V3; az: V3; half: V3; kind: GlassKind }

export interface OpticsScene {
  segments: Segment[];
  mirrors: Mirror[];
  glass: Glass[];
  pathLength: number;       // longest path arc length (reveal animation ends here)
  sources: V3[];            // laser emitter positions (drawn as small bright housings)
}

export const SEGMENT_FLOATS = 12, MIRROR_FLOATS = 12, GLASS_FLOATS = 16;

/** Pack into Float32Arrays matching the WGSL structs. */
export function packOptics(s: OpticsScene): { segments: Float32Array; mirrors: Float32Array; glass: Float32Array } {
  const seg = new Float32Array(Math.max(1, s.segments.length) * SEGMENT_FLOATS);
  s.segments.forEach((g, i) => seg.set([...g.a, g.sA, ...g.b, g.sB, ...g.color, g.kind], i * SEGMENT_FLOATS));
  const mir = new Float32Array(Math.max(1, s.mirrors.length) * MIRROR_FLOATS);
  s.mirrors.forEach((m, i) => mir.set([...m.pos, m.halfSize, ...m.normal, m.sArrive, ...m.color, m.kind], i * MIRROR_FLOATS));
  const gl = new Float32Array(Math.max(1, s.glass.length) * GLASS_FLOATS);
  s.glass.forEach((g, i) => gl.set([...g.center, g.kind, ...g.ax, g.half[0], ...g.ay, g.half[1], ...g.az, g.half[2]], i * GLASS_FLOATS));
  return { segments: seg, mirrors: mir, glass: gl };
}
