// Laser pass: router/outline mirrors (alpha), anti-aliased screen-space beam quads (+ floor reflection), glints (additive).
// Output resolution, no depth. Everything is analytic: beams are distance-to-segment in pixels.
@group(1) @binding(0) var<storage, read> segments: array<Segment>;
@group(1) @binding(1) var<storage, read> mirrors: array<Mirror>;
@group(1) @binding(2) var<uniform> P: vec4f; // x = glowScale

const NEAR_W: f32 = 0.05;
const OFFSCREEN: vec4f = vec4f(2.0, 2.0, 2.0, 1.0);

fn pxToClip(p: vec2f) -> vec4f {
  return vec4f(p.x * F.resolution.z * 2.0 - 1.0, 1.0 - p.y * F.resolution.w * 2.0, 0.0, 1.0);
}
fn worldToPx(c: vec4f) -> vec2f {
  let n = c.xy / c.w;
  return vec2f((n.x * 0.5 + 0.5) * F.resolution.x, (0.5 - n.y * 0.5) * F.resolution.y);
}
fn clipW(p: vec3f) -> f32 { return (F.viewProj * vec4f(p, 1.0)).w; }

// Two triangles; corner (-1/+1, -1/+1).
fn corner(vi: u32) -> vec2f {
  var c = array<vec2f, 6>(vec2f(-1, -1), vec2f(1, -1), vec2f(1, 1), vec2f(-1, -1), vec2f(1, 1), vec2f(-1, 1));
  return c[vi];
}

fn pulseBoost(s: f32) -> f32 {
  if (F.misc.w < 0.0) { return 1.0; }
  let x = (s - F.misc.w) / 0.25;
  return 1.0 + 5.0 * exp(-x * x);
}

// ---------------------------------------------------------------- beams
struct BeamV {
  @builtin(position) pos: vec4f,
  @location(0) @interpolate(flat) pa: vec2f,
  @location(1) @interpolate(flat) pb: vec2f,
  @location(2) @interpolate(flat) sAB: vec2f,
  @location(3) @interpolate(flat) color: vec4f,
  @location(4) @interpolate(flat) wab: vec2f,
  @location(5) @interpolate(flat) wa3: vec3f,
  @location(6) @interpolate(flat) wb3: vec3f,
  @location(7) @interpolate(flat) refl: f32,
};

@vertex fn vs_beam(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> BeamV {
  var o: BeamV;
  o.pos = OFFSCREEN;
  let n = max(u32(F.counts.x), 1u);
  let refl = ii >= n;
  let seg = segments[ii % n];
  var a = seg.a.xyz;
  var b = seg.b.xyz;
  var sa = seg.a.w;
  var sb = seg.b.w;
  if (refl) {
    let fy = F.beamStyle.w;
    a.y = 2.0 * fy - a.y;
    b.y = 2.0 * fy - b.y;
  }
  // Clip against the near plane in world space.
  var wa = clipW(a);
  var wb = clipW(b);
  if (wa < NEAR_W && wb < NEAR_W) { return o; }
  if (wa < NEAR_W) {
    let t = (NEAR_W - wa) / (wb - wa);
    a = mix(a, b, t); sa = mix(sa, sb, t); wa = NEAR_W;
  } else if (wb < NEAR_W) {
    let t = (NEAR_W - wb) / (wa - wb);
    b = mix(b, a, t); sb = mix(sb, sa, t); wb = NEAR_W;
  }
  let pa = worldToPx(F.viewProj * vec4f(a, 1.0));
  let pb = worldToPx(F.viewProj * vec4f(b, 1.0));
  var d = pb - pa;
  let len = length(d);
  d = select(vec2f(1.0, 0.0), d / len, len > 1e-4);
  let nrm = vec2f(-d.y, d.x);
  let hw = F.beamStyle.y * P.x + 2.0;
  let c = corner(vi);
  let base = select(pa, pb, c.x > 0.0);
  let p = base + d * c.x * hw + nrm * c.y * hw;
  o.pos = pxToClip(p);
  o.pa = pa; o.pb = pb;
  o.sAB = vec2f(sa, sb);
  o.color = seg.color;
  o.wab = vec2f(wa, wb);
  o.wa3 = a; o.wb3 = b;
  o.refl = select(0.0, 1.0, refl);
  return o;
}

@fragment fn fs_beam(v: BeamV) -> @location(0) vec4f {
  let P2 = v.pos.xy;
  let d = v.pb - v.pa;
  let l2 = dot(d, d);
  let ts = select(0.0, clamp(dot(P2 - v.pa, d) / l2, 0.0, 1.0), l2 > 1e-6);
  let dist = length(P2 - (v.pa + d * ts));
  // Perspective-correct world parameter from the screen parameter.
  let tw = ts * v.wab.x / (v.wab.y * (1.0 - ts) + ts * v.wab.x);
  let s = mix(v.sAB.x, v.sAB.y, tw);
  let reveal = F.misc.z;
  if (s > reveal) { discard; }

  let pw = mix(v.wa3, v.wb3, tw);
  var gain = 1.0;
  if (v.refl > 0.5) {
    let fy = F.beamStyle.w;
    if (pw.y > fy) { discard; }
    gain = F.counts.w * exp(-(fy - pw.y) * 3.0);
  }

  let cw = F.beamStyle.x;
  let gw = F.beamStyle.y * P.x;
  let x = dist / cw;
  let core = exp(-x * x);
  let glow = F.beamStyle.z * exp(-dist / gw);
  var I = core * 2.2 + glow;

  let hd = (reveal - s) / 0.08;
  I += 6.0 * exp(-hd * hd);
  if (v.sAB.x < 1e-4) { I += 3.0 * exp(-(s / 0.05) * (s / 0.05)); }
  I *= pulseBoost(s);
  if (v.refl < 0.5) {
    I *= 1.0 + 0.6 * F.pointer.w * exp(-distance(F.pointer.xy, pw.xy) / 0.25);
  }
  let kind = v.color.w;
  if (kind > 0.5 && kind < 1.5) { I *= 0.55; }
  if (kind >= 1.5) { let f = 1.0 - tw; I *= f * f; }
  return vec4f(v.color.rgb * (I * gain), 0.0);
}

// ---------------------------------------------------------------- mirrors
struct MirV {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
  @location(1) @interpolate(flat) color: vec3f,
  @location(2) @interpolate(flat) fade: f32,
  @location(3) @interpolate(flat) spec: f32,
};

@vertex fn vs_mirror(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> MirV {
  var o: MirV;
  o.pos = OFFSCREEN;
  let m = mirrors[ii];
  let fade = clamp((F.misc.z - (m.normal.w - 0.2)) / 0.2, 0.0, 1.0);
  if (fade <= 0.0) { return o; }
  let N = normalize(m.normal.xyz);
  let up = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(N.y) > 0.99);
  let T = normalize(cross(up, N));
  let B = cross(N, T);
  let c = corner(vi);
  let p = m.pos.xyz + (T * c.x + B * c.y) * m.pos.w;
  o.pos = F.viewProj * vec4f(p, 1.0);
  o.uv = c;
  o.color = m.color.rgb;
  o.fade = fade;
  let V = normalize(F.camPos.xyz - m.pos.xyz);
  o.spec = pow(max(dot(N, V), 0.0), 24.0);
  return o;
}

@fragment fn fs_mirror(v: MirV) -> @location(0) vec4f {
  let e = max(abs(v.uv.x), abs(v.uv.y));
  let rim = smoothstep(0.84, 0.92, e);
  var col = vec3f(0.02) + v.color * (rim * 0.9 + v.spec * 0.7);
  col = min(col, vec3f(4.0));
  return vec4f(col * v.fade, 0.9 * v.fade);
}

// ---------------------------------------------------------------- glints
struct GlintV {
  @builtin(position) pos: vec4f,
  @location(0) q: vec2f,       // pixel offset from centre
  @location(1) @interpolate(flat) color: vec3f,
  @location(2) @interpolate(flat) size: f32,
};

@vertex fn vs_glint(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> GlintV {
  var o: GlintV;
  o.pos = OFFSCREEN;
  let m = mirrors[ii];
  let r = clamp((F.misc.z - m.normal.w) / 0.1, 0.0, 1.0);
  if (r <= 0.0) { return o; }
  let c = F.viewProj * vec4f(m.pos.xyz, 1.0);
  if (c.w < NEAR_W) { return o; }
  // Outline mirrors (kind 0) get a tiny steady point; only routers/splitters get a star (keeps the outline crisp).
  let outline = m.color.w < 0.5;
  let size = select(mix(8.0, 13.0, clamp(m.pos.w / 0.12, 0.0, 1.0)), 2.5, outline) * F.pointer.z;
  let ext = size * 2.5;
  let q = corner(vi) * ext;
  o.pos = vec4f(c.xy / c.w + vec2f(q.x * F.resolution.z * 2.0, -q.y * F.resolution.w * 2.0), 0.0, 1.0);
  o.q = q;
  let tw = select(0.85 + 0.15 * sin(F.time.x * 2.0 + f32(ii)), 1.0, outline);
  o.color = m.color.rgb * select(1.2, 0.45, outline) * tw * r * pulseBoost(m.normal.w);
  o.size = size;
  return o;
}

@fragment fn fs_glint(v: GlintV) -> @location(0) vec4f {
  let a = abs(v.q);
  let thin = 1.2;
  let armX = exp(-(a.y / thin) * (a.y / thin)) * exp(-a.x / (v.size * 0.6));
  let armY = exp(-(a.x / thin) * (a.x / thin)) * exp(-a.y / (v.size * 0.6));
  let k = v.size * 0.18;
  let core = exp(-dot(v.q, v.q) / (k * k));
  let arms = select(0.0, 0.8, v.size > 5.0);
  let I = (armX + armY) * arms + core * 1.2;
  return vec4f(v.color * I, 0.0);
}
