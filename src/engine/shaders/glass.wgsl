// Glass pass: analytic prism / box, screen-space refraction of hdrCopy, optional dispersion, crisp bright edges.
@group(1) @binding(0) var<storage, read> glass: array<Glass>;
@group(1) @binding(1) var hdrCopy: texture_2d<f32>;
@group(1) @binding(2) var samp: sampler;
@group(1) @binding(3) var<uniform> P: vec4f; // x = dispersion (1/0)

const NEAR_W: f32 = 0.05;

fn halfExt(g: Glass) -> vec3f {
  let prism = g.center.w < 0.5;
  return vec3f(g.ax.w, select(g.ay.w, g.ax.w, prism), g.az.w);
}

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) @interpolate(flat) idx: u32,
};

@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  var o: VOut;
  o.idx = ii;
  let g = glass[ii];
  let e = halfExt(g);
  var mn = vec2f(1e9);
  var mx = vec2f(-1e9);
  var behind = false;
  for (var k = 0u; k < 8u; k++) {
    let s = vec3f(select(-1.0, 1.0, (k & 1u) != 0u), select(-1.0, 1.0, (k & 2u) != 0u), select(-1.0, 1.0, (k & 4u) != 0u));
    let p = g.center.xyz + g.ax.xyz * (s.x * e.x) + g.ay.xyz * (s.y * e.y) + g.az.xyz * (s.z * e.z);
    let c = F.viewProj * vec4f(p, 1.0);
    if (c.w < NEAR_W) { behind = true; }
    let n = c.xy / max(c.w, NEAR_W);
    mn = min(mn, n);
    mx = max(mx, n);
  }
  if (behind) { mn = vec2f(-1.0); mx = vec2f(1.0); }
  let pad = F.resolution.zw * 4.0;
  mn -= pad; mx += pad;
  var cs = array<vec2f, 6>(vec2f(0, 0), vec2f(1, 0), vec2f(1, 1), vec2f(0, 0), vec2f(1, 1), vec2f(0, 1));
  let c = cs[vi];
  o.pos = vec4f(mix(mn.x, mx.x, c.x), mix(mn.y, mx.y, c.y), 0.0, 1.0);
  return o;
}

// Convex solid as the intersection of half-spaces n.p <= h (local space).
fn plane(prism: bool, k: u32, h: vec3f) -> vec4f {
  if (prism) {
    if (k < 3u) {
      let th = -1.5707963 + f32(k) * 2.0943951;
      return vec4f(cos(th), sin(th), 0.0, h.x * 0.5);
    }
    return vec4f(0.0, 0.0, select(1.0, -1.0, k == 4u), h.z);
  }
  let s = select(1.0, -1.0, (k & 1u) == 1u);
  if (k < 2u) { return vec4f(s, 0.0, 0.0, h.x); }
  if (k < 4u) { return vec4f(0.0, s, 0.0, h.y); }
  return vec4f(0.0, 0.0, s, h.z);
}

struct Hit { ok: bool, t0: f32, t1: f32, n0: vec3f, n1: vec3f };

fn intersect(o: vec3f, d: vec3f, prism: bool, h: vec3f) -> Hit {
  var r: Hit;
  var t0 = -1e9;
  var t1 = 1e9;
  var n0 = vec3f(0.0);
  var n1 = vec3f(0.0);
  let cnt = select(6u, 5u, prism);
  for (var k = 0u; k < cnt; k++) {
    let pl = plane(prism, k, h);
    let dn = dot(pl.xyz, d);
    let num = pl.w - dot(pl.xyz, o);
    if (abs(dn) < 1e-7) {
      if (num < 0.0) { t1 = -1e9; }
      continue;
    }
    let t = num / dn;
    if (dn < 0.0) {
      if (t > t0) { t0 = t; n0 = pl.xyz; }
    } else {
      if (t < t1) { t1 = t; n1 = pl.xyz; }
    }
  }
  r.ok = t0 > 0.0 && t0 < t1;
  r.t0 = t0; r.t1 = t1; r.n0 = n0; r.n1 = n1;
  return r;
}

// Screen-space refracted background colour for one IOR.
fn refractSample(I: vec3f, N0: vec3f, N1: vec3f, exitP: vec3f, ior: f32, fallbackUV: vec2f) -> vec3f {
  var T = refract(I, N0, 1.0 / ior);
  if (dot(T, T) < 1e-6) { T = reflect(I, N0); }
  var out = refract(T, -N1, ior);
  if (dot(out, out) < 1e-6) { out = reflect(T, -N1); }
  let pt = exitP + normalize(out) * 1.5;
  let c = F.viewProj * vec4f(pt, 1.0);
  var uv = fallbackUV;
  if (c.w > NEAR_W) {
    uv = vec2f(c.x / c.w * 0.5 + 0.5, 0.5 - c.y / c.w * 0.5);
  }
  uv = clamp(uv, vec2f(0.001), vec2f(0.999));
  return textureSampleLevel(hdrCopy, samp, uv, 0.0).rgb;
}

@fragment fn fs(v: VOut) -> @location(0) vec4f {
  let g = glass[v.idx];
  let prism = g.center.w < 0.5;
  let h = halfExt(g);
  let uv0 = v.pos.xy * F.resolution.zw;
  let dir = rayDir(uv0);
  let rel = F.camPos.xyz - g.center.xyz;
  let ol = vec3f(dot(rel, g.ax.xyz), dot(rel, g.ay.xyz), dot(rel, g.az.xyz));
  let dl = vec3f(dot(dir, g.ax.xyz), dot(dir, g.ay.xyz), dot(dir, g.az.xyz));
  let hit = intersect(ol, dl, prism, h);
  if (!hit.ok) { discard; }

  let N0 = normalize(g.ax.xyz * hit.n0.x + g.ay.xyz * hit.n0.y + g.az.xyz * hit.n0.z);
  let N1 = normalize(g.ax.xyz * hit.n1.x + g.ay.xyz * hit.n1.y + g.az.xyz * hit.n1.z);
  let exitP = F.camPos.xyz + dir * hit.t1;

  var refr: vec3f;
  if (P.x > 0.5) {
    refr = vec3f(
      refractSample(dir, N0, N1, exitP, 1.47, uv0).r,
      refractSample(dir, N0, N1, exitP, 1.50, uv0).g,
      refractSample(dir, N0, N1, exitP, 1.53, uv0).b);
  } else {
    refr = refractSample(dir, N0, N1, exitP, 1.5, uv0);
  }

  let cosT = max(dot(-dir, N0), 0.0);
  let f = 0.04 + 0.96 * pow(1.0 - cosT, 5.0);
  var col = refr * vec3f(0.92, 0.96, 1.0) * 0.85 * (1.0 - f) + vec3f(0.01) * f;

  // Distance from the entry point to the nearest other face = edge proximity.
  let pl = ol + dl * hit.t0;
  var m1 = 1e9;
  var m2 = 1e9;
  let cnt = select(6u, 5u, prism);
  for (var k = 0u; k < cnt; k++) {
    let q = plane(prism, k, h);
    let dd = abs(q.w - dot(q.xyz, pl));
    if (dd < m1) { m2 = m1; m1 = dd; } else if (dd < m2) { m2 = dd; }
  }
  let edge = 1.0 - smoothstep(0.0, 0.012, m2);
  col += vec3f(0.6, 0.7, 0.8) * edge * 0.6;
  return vec4f(col, 1.0);
}
