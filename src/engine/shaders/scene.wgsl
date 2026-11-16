// Scene pass: one fullscreen triangle, per-pixel hybrid raymarch at INTERNAL resolution.
//  - Glass text: 2D SDF texture extruded + beveled analytically, marched only inside its AABB.
//  - Chrome marbles: analytic ray/sphere (exact, no marching).
//  - Floor: analytic plane, black polished coat with raymarched planar reflection (roughness via
//    per-frame normal jitter that TAA integrates into a glossy blur — free blur, no extra taps).
//  - Glass: Fresnel split into reflection + interior-marched refraction, Beer–Lambert absorption,
//    TIR fallback, chromatic dispersion (full at Q>=3, emitter-only delta below).
//  - Floor shadows through glass are tinted + caustic-brightened at penumbra edges.
// Prepended: frame.wgsl (F, ign, ignT, rayDir, PI).

@group(1) @binding(0) var sdfTex: texture_2d<f32>;
@group(1) @binding(1) var samp: sampler;

const IOR: f32 = 1.5;
const IOR_RGB: vec3f = vec3f(1.475, 1.50, 1.53);
const ABSORB: vec3f = vec3f(0.34, 0.17, 0.10);      // per world unit: faint cyan crystal
const GLASS_SHADOW: vec3f = vec3f(0.30, 0.38, 0.44);
const FLOOR_ALBEDO: f32 = 0.012;
const CHROME_F0: vec3f = vec3f(0.92, 0.91, 0.93);
const POINTER_COLOR: vec3f = vec3f(0.55, 0.75, 1.0);
const NO_HIT: f32 = 1e9;

struct FOut { @location(0) color: vec4f, @builtin(frag_depth) depth: f32 };

@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}

// ---------------- geometry ----------------
fn sdf2(p: vec2f) -> f32 {
  let hw = F.textParams.xy;
  let uv = p / hw * 0.5 + 0.5; // SDF texture row 0 = bottom (world Y up), per B2 contract
  let d = textureSampleLevel(sdfTex, samp, clamp(uv, vec2f(0.0), vec2f(1.0)), 0.0).r;
  let o = length(max(abs(p) - hw, vec2f(0.0)));
  return select(d, max(d, 0.0) + o, o > 0.0);
}

fn sdText(p: vec3f) -> f32 {
  let r = F.textParams.w;
  let w = vec2f(sdf2(p.xy) + r, abs(p.z) - F.textParams.z + r);
  return min(max(w.x, w.y), 0.0) + length(max(w, vec2f(0.0))) - r;
}

fn textNormal(p: vec3f) -> vec3f {
  let e = vec2f(1.0, -1.0) * 0.0015;
  return normalize(e.xyy * sdText(p + e.xyy) + e.yyx * sdText(p + e.yyx) +
                   e.yxy * sdText(p + e.yxy) + e.xxx * sdText(p + e.xxx));
}

fn boxHit(ro: vec3f, rd: vec3f, bmax: vec3f) -> vec2f {
  let r = select(rd, vec3f(1e-6), abs(rd) < vec3f(1e-6));
  let inv = 1.0 / r;
  let t0 = (-bmax - ro) * inv;
  let t1 = (bmax - ro) * inv;
  let tn = min(t0, t1);
  let tf = max(t0, t1);
  return vec2f(max(max(tn.x, tn.y), tn.z), min(min(tf.x, tf.y), tf.z));
}

fn textBox() -> vec3f { return vec3f(F.textParams.xy, F.textParams.z); }

fn marchText(ro: vec3f, rd: vec3f, tmax: f32, steps: u32) -> f32 {
  let bb = boxHit(ro, rd, textBox());
  if (bb.y < max(bb.x, 0.0)) { return NO_HIT; }
  var t = max(bb.x, 0.0);
  let te = min(bb.y, tmax);
  for (var i = 0u; i < steps; i++) {
    let d = sdText(ro + rd * t);
    if (d < 0.0004 + 0.0006 * t) { return t; }
    t += d;
    if (t > te) { break; }
  }
  return NO_HIT;
}

fn sphereHit(ro: vec3f, rd: vec3f, s: vec4f) -> f32 {
  let oc = ro - s.xyz;
  let b = dot(oc, rd);
  let h = b * b - (dot(oc, oc) - s.w * s.w);
  if (h < 0.0) { return NO_HIT; }
  let t = -b - sqrt(h);
  return select(NO_HIT, t, t > 0.0);
}

// returns (t, index)
fn marblesHit(ro: vec3f, rd: vec3f) -> vec2f {
  var best = vec2f(NO_HIT, -1.0);
  let n = u32(F.misc.w);
  for (var i = 0u; i < min(n, MAX_MARBLES); i++) {
    let t = sphereHit(ro, rd, F.marbles[i]);
    if (t < best.x) { best = vec2f(t, f32(i)); }
  }
  return best;
}

fn floorHit(ro: vec3f, rd: vec3f) -> f32 {
  if (rd.y > -1e-4) { return NO_HIT; }
  let t = (F.spotColor.w - ro.y) / rd.y;
  return select(NO_HIT, t, t > 0.0);
}

// ---------------- lighting ----------------
fn rectMask(p: vec2f, h: vec2f, soft: f32) -> f32 {
  let a = 1.0 - smoothstep(h - vec2f(soft), h, abs(p));
  return a.x * a.y;
}

// Studio environment: emitters only (black stage). `direct` = seen by a primary ray (only the key light shows).
fn env(ro: vec3f, rd: vec3f, direct: bool) -> vec3f {
  var c = vec3f(0.0);
  let toL = F.spotPos.xyz - ro;
  let cosA = max(dot(rd, normalize(toL)), 0.0);
  let core = select(0.0, 0.55, sphereHit(ro, rd, vec4f(F.spotPos.xyz, 0.2)) < NO_HIT);
  c += F.spotColor.rgb * (core + 0.0035 * pow(cosA, 600.0) + 0.0004 * pow(cosA, 40.0));
  if (direct) { return c; }
  // overhead softbox
  if (rd.y > 0.0) {
    let q = ro + rd * ((4.5 - ro.y) / rd.y);
    c += vec3f(1.0, 0.96, 0.9) * 2.4 * rectMask(q.xz - vec2f(0.0, 0.5), vec2f(5.0, 1.0), 0.6);
  }
  // back rim bars (cool)
  if (rd.z < 0.0) {
    let q = ro + rd * ((-3.5 - ro.z) / rd.z);
    let fy = F.spotColor.w;
    let bars = rectMask(vec2f(abs(q.x) - 3.8, q.y - fy - 1.6), vec2f(0.12, 1.6), 0.1);
    c += vec3f(0.55, 0.72, 1.0) * 4.0 * bars;
  }
  // front horizontal strip behind the camera: draws a sheen line across glass faces
  if (rd.z > 0.0) {
    let q = ro + rd * ((14.0 - ro.z) / rd.z);
    c += vec3f(1.0, 0.98, 0.95) * 1.1 * rectMask(q.xy - vec2f(0.0, 2.2), vec2f(9.0, 0.35), 0.3);
  }
  c += vec3f(0.006, 0.007, 0.009) * (0.6 + 0.4 * rd.y);
  return c;
}

fn spotRadiance(p: vec3f) -> vec4f { // rgb radiance, w unused; returns light dir via spotL()
  let L = F.spotPos.xyz - p;
  let d2 = dot(L, L);
  let l = L * inverseSqrt(d2);
  let cone = smoothstep(F.spotPos.w, F.spotDir.w, dot(-l, F.spotDir.xyz));
  return vec4f(F.spotColor.rgb * cone / max(d2, 1.0), sqrt(d2));
}

fn pointerRadiance(p: vec3f) -> vec3f {
  let L = F.pointer.xyz - p;
  return POINTER_COLOR * (F.pointer.w * 0.5) / max(dot(L, L), 0.08);
}

fn ggx(n: vec3f, v: vec3f, l: vec3f, rough: f32) -> f32 {
  let h = normalize(v + l);
  let nl = max(dot(n, l), 0.0);
  let nv = max(dot(n, v), 1e-4);
  let nh = max(dot(n, h), 0.0);
  let a = rough * rough;
  let a2 = a * a;
  let dd = nh * nh * (a2 - 1.0) + 1.0;
  let k = a * 0.5;
  return a2 / (PI * dd * dd) * nl / (4.0 * (nl * (1.0 - k) + k) * (nv * (1.0 - k) + k));
}

fn fresnel(cosT: f32, f0: f32) -> f32 { return f0 + (1.0 - f0) * pow(1.0 - clamp(cosT, 0.0, 1.0), 5.0); }

// SDF soft shadow toward the key light, clipped to the text AABB (exits immediately if the segment misses it).
fn softShadow(ro: vec3f, rd: vec3f, tmax: f32, k: f32) -> f32 {
  let bb = boxHit(ro, rd, textBox() + vec3f(0.05));
  if (bb.y < max(bb.x, 0.0) || bb.x > tmax) { return 1.0; }
  var t = max(bb.x, 0.01);
  let te = min(bb.y, tmax);
  var res = 1.0;
  let n = u32(F.quality.w);
  for (var i = 0u; i < n; i++) {
    let h = sdText(ro + rd * t);
    res = min(res, k * h / t);
    t += clamp(h, 0.02, 0.35);
    if (res < 0.002 || t > te) { break; }
  }
  return clamp(res, 0.0, 1.0);
}

fn floorDirect(p: vec3f, v: vec3f) -> vec3f {
  let n = vec3f(0.0, 1.0, 0.0);
  var c = vec3f(0.0);
  let s = spotRadiance(p);
  if (max(s.r, max(s.g, s.b)) > 1e-4) {
    let l = (F.spotPos.xyz - p) / s.w;
    let sh = softShadow(p + n * 0.002, l, s.w, 10.0);
    // glass casts a tinted, partial shadow; caustic focusing concentrates light at the penumbra edge.
    let trans = mix(GLASS_SHADOW, vec3f(1.0), sh) + vec3f(1.0, 0.94, 0.84) * (sh * (1.0 - sh) * 3.2);
    c += s.rgb * trans * (FLOOR_ALBEDO / PI * max(l.y, 0.0) + 0.05 * ggx(n, v, l, 0.3));
  }
  let pl = normalize(F.pointer.xyz - p);
  c += pointerRadiance(p) * (FLOOR_ALBEDO / PI * max(pl.y, 0.0) + 0.05 * ggx(n, v, pl, 0.3));
  return c;
}

fn floorFade(p: vec3f) -> f32 { return exp(-max(length(p.xz) - 5.0, 0.0) * 0.35); }

fn chromeColor(p: vec3f, n: vec3f, rd: vec3f, refl: vec3f) -> vec3f {
  let v = -rd;
  let f = CHROME_F0 + (vec3f(1.0) - CHROME_F0) * pow(1.0 - max(dot(n, v), 0.0), 5.0);
  let s = spotRadiance(p);
  let l = normalize(F.spotPos.xyz - p);
  let pl = normalize(F.pointer.xyz - p);
  return f * (refl + s.rgb * ggx(n, v, l, 0.12) + pointerRadiance(p) * ggx(n, v, pl, 0.15));
}

// Cheap glass for secondary rays: Fresnel blend of env reflection and straight-through env with tint.
fn glassLite(p: vec3f, n: vec3f, rd: vec3f) -> vec3f {
  let f = fresnel(dot(n, -rd), 0.04);
  let through = env(p, refract(rd, n, 1.0 / IOR) * 0.5 + rd * 0.5, false) * exp(-ABSORB * 0.4);
  return env(p, reflect(rd, n), false) * f + through * (1.0 - f);
}

// One bounce of everything except the full glass solver. textSteps = 0 skips the text (no self-reflection).
fn traceSecondary(ro: vec3f, rd: vec3f, textSteps: u32) -> vec3f {
  var t = NO_HIT;
  var kind = 0u;
  let m = marblesHit(ro, rd);
  if (m.x < t) { t = m.x; kind = 2u; }
  let tf = floorHit(ro, rd);
  if (tf < t) { t = tf; kind = 1u; }
  if (textSteps > 0u) {
    let tt = marchText(ro, rd, t, textSteps);
    if (tt < t) { t = tt; kind = 3u; }
  }
  let p = ro + rd * t;
  switch kind {
    case 1u: { return floorDirect(p, -rd) * floorFade(p); }
    case 2u: {
      let n = normalize(p - F.marbles[u32(m.y)].xyz);
      return chromeColor(p, n, rd, env(p, reflect(rd, n), false));
    }
    case 3u: { return glassLite(p, textNormal(p), rd); }
    default: { return env(ro, rd, false); }
  }
}

// Full glass: reflection + interior march + dispersive exit + absorption + direct speculars + edge light runner.
fn shadeGlass(p: vec3f, n: vec3f, rd: vec3f) -> vec3f {
  let v = -rd;
  let f = fresnel(dot(n, v), 0.04);
  let refl = traceSecondary(p + n * 0.003, reflect(rd, n), 0u);

  let r1 = refract(rd, n, 1.0 / IOR);
  let ro = p - n * 0.004;
  var t = 0.0;
  let steps = u32(F.quality.y);
  for (var i = 0u; i < steps; i++) {
    let d = -sdText(ro + r1 * t);
    if (d < 0.0005) { break; }
    t += max(d, 0.004);
  }
  let xp = ro + r1 * t;
  let nx = textNormal(xp);                // outward normal at exit
  let xo = xp + nx * 0.004;
  var outG = refract(r1, -nx, IOR);
  let tirG = dot(outG, outG) < 0.5;
  outG = select(outG, reflect(r1, -nx), tirG);

  var trans: vec3f;
  if (F.time.w >= 3.0) {
    var c = vec3f(0.0);
    let iors = IOR_RGB;
    for (var k = 0u; k < 3u; k++) {
      var o = refract(r1, -nx, iors[k]);
      o = select(o, reflect(r1, -nx), dot(o, o) < 0.5);
      c[k] = traceSecondary(xo, o, 0u)[k];
    }
    trans = c;
  } else {
    // full trace for green, chromatic shift applied to emitters only (where dispersion is visible)
    let base = traceSecondary(xo, outG, 0u);
    var oR = refract(r1, -nx, IOR_RGB.x); oR = select(oR, outG, dot(oR, oR) < 0.5);
    var oB = refract(r1, -nx, IOR_RGB.z); oB = select(oB, outG, dot(oB, oB) < 0.5);
    let eG = env(xo, outG, false);
    trans = max(base + vec3f(env(xo, oR, false).r, eG.g, env(xo, oB, false).b) - eG, vec3f(0.0));
  }
  trans *= exp(-ABSORB * (t + 0.05));

  var c = refl * f + trans * (1.0 - f);

  let s = spotRadiance(p);
  let l = normalize(F.spotPos.xyz - p);
  let pl = normalize(F.pointer.xyz - p);
  c += s.rgb * ggx(n, v, l, 0.05) * f * 4.0 + pointerRadiance(p) * ggx(n, v, pl, 0.07) * 0.6;

  // Light runner: a soft band of light that travels along the glyph outlines (bevels + side walls).
  let d2 = sdf2(p.xy);
  let edge = 1.0 - smoothstep(0.0, 0.07, -d2);
  let span = F.textParams.x + 2.0;
  let sweep = (fract(F.time.x * 0.06) * 2.0 - 1.0) * span;
  let band = exp(-pow((p.x - sweep) / 0.45, 2.0));
  c += vec3f(1.0, 0.92, 0.8) * edge * band * 1.8 * (0.35 + 0.65 * max(n.z, 0.0));
  return c;
}

@fragment fn fs(@builtin(position) fc: vec4f) -> FOut {
  let uv = fc.xy * F.resolution.zw;
  let ro = F.camPos.xyz;
  let rd = rayDir(uv);

  var t = NO_HIT;
  var kind = 0u;
  let m = marblesHit(ro, rd);
  if (m.x < t) { t = m.x; kind = 2u; }
  let tf = floorHit(ro, rd);
  if (tf < t) { t = tf; kind = 1u; }
  let tt = marchText(ro, rd, t, u32(F.quality.x));
  if (tt < t) { t = tt; kind = 3u; }

  var col: vec3f;
  let p = ro + rd * t;
  switch kind {
    case 1u: {
      let v = -rd;
      var c = floorDirect(p, v);
      let rs = u32(F.quality.z);
      // glossy: jitter the normal per pixel per frame; TAA integrates the blur.
      let j = vec2f(ignT(fc.xy), ign(fc.yx + vec2f(17.0, 31.0))) - 0.5;
      let nr = normalize(vec3f(j.x * 0.07, 1.0, j.y * 0.07));
      let rr = reflect(rd, nr);
      var refl: vec3f;
      if (rs > 0u) { refl = traceSecondary(p + vec3f(0.0, 0.002, 0.0), rr, rs); }
      else { refl = env(p, rr, false) * 0.5; }
      c += refl * fresnel(v.y, 0.05);
      col = c * floorFade(p);
    }
    case 2u: {
      let n = normalize(p - F.marbles[u32(m.y)].xyz);
      let r = reflect(rd, n);
      col = chromeColor(p, n, rd, traceSecondary(p + n * 0.002, r, u32(F.quality.z)));
    }
    case 3u: { col = shadeGlass(p, textNormal(p), rd); }
    default: { col = env(ro, rd, true); }
  }

  var o: FOut;
  col = select(col, vec3f(0.0), col != col);   // NaN guard (WGSL: per-component)
  o.color = vec4f(min(col, vec3f(256.0)), 1.0);
  if (kind == 0u) {
    o.depth = 1.0;
  } else {
    let c = F.viewProj * vec4f(p, 1.0);
    o.depth = clamp(c.z / c.w, 0.0, 1.0);
  }
  return o;
}
