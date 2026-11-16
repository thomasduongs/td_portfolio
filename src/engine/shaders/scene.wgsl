// Scene pass v2 — the name is made of light. Fullscreen triangle, per-pixel at INTERNAL resolution.
// Prepended: frame.wgsl (F, hg, ign, ignT, rayDir) + beams.wgsl (beams(), beamI(), letterMask(), EXCITE_COLOR).
//
//  1. Coarse haze march over the whole view ray: each beam scatters (HG phase) in drifting 3D-noise haze,
//     Beer–Lambert transmittance; pair excitation where beams cross in mid-air.
//  2. Fine march inside the letter slab |z| < beamParams.z: two-step excitation (pair + triple product of
//     the three gobo beams) — where all three silhouettes coincide, the haze glows: the letters.
//  3. Surfaces: floor lit by the three projections (stretched coloured letters) with a glossy reflection of
//     the glowing slab; chrome marbles (Rapier) that also occlude the beams (beams.wgsl); lamp emitters.
// Blue-noise (IGN) jittered sample offsets per frame; TAA resolves the noise.

@group(1) @binding(0) var sdfTex: texture_2d<f32>;
@group(1) @binding(1) var samp: sampler;
@group(1) @binding(2) var noiseTex: texture_3d<f32>;
@group(1) @binding(3) var noiseSamp: sampler; // repeat

const NO_HIT: f32 = 1e9;
const FLOOR_ALBEDO: f32 = 0.08;
const FLOOR_IRR: f32 = 1.0;              // beam colour -> floor irradiance scale
const CHROME_F0: vec3f = vec3f(0.92, 0.91, 0.93);
const POINTER_COLOR: vec3f = vec3f(0.55, 0.75, 1.0);

struct FOut { @location(0) color: vec4f, @builtin(frag_depth) depth: f32 };

@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}

// ---------------- geometry ----------------
fn boxHit(ro: vec3f, rd: vec3f, bmax: vec3f) -> vec2f {
  let r = select(rd, vec3f(1e-6), abs(rd) < vec3f(1e-6));
  let inv = 1.0 / r;
  let t0 = (-bmax - ro) * inv;
  let t1 = (bmax - ro) * inv;
  let tn = min(t0, t1);
  let tf = max(t0, t1);
  return vec2f(max(max(tn.x, tn.y), tn.z), min(min(tf.x, tf.y), tf.z));
}

fn slabBox() -> vec3f { return vec3f(F.textParams.xy, F.beamParams.z); }

fn sphereHit(ro: vec3f, rd: vec3f, s: vec4f) -> f32 {
  let oc = ro - s.xyz;
  let b = dot(oc, rd);
  let h = b * b - (dot(oc, oc) - s.w * s.w);
  if (h < 0.0) { return NO_HIT; }
  let t = -b - sqrt(h);
  return select(NO_HIT, t, t > 0.0);
}

fn marblesHit(ro: vec3f, rd: vec3f) -> vec2f {
  var best = vec2f(NO_HIT, -1.0);
  let n = min(u32(F.misc.w), MAX_MARBLES);
  for (var i = 0u; i < n; i++) {
    let t = sphereHit(ro, rd, F.marbles[i]);
    if (t < best.x) { best = vec2f(t, f32(i)); }
  }
  return best;
}

fn floorHit(ro: vec3f, rd: vec3f) -> f32 {
  if (rd.y > -1e-4) { return NO_HIT; }
  let t = (F.textParams.z - ro.y) / rd.y;
  return select(NO_HIT, t, t > 0.0);
}

fn floorFade(p: vec3f) -> f32 { return exp(-max(length(p.xz) - 6.0, 0.0) * 0.25); }

// ---------------- haze ----------------
fn density(p: vec3f) -> f32 {
  let t = F.time.x;
  let n = textureSampleLevel(noiseTex, noiseSamp, p * 0.085 + vec3f(t * 0.011, t * 0.004, -t * 0.007), 0.0).r;
  let m = textureSampleLevel(noiseTex, noiseSamp, p * 0.29 + vec3f(-t * 0.006, t * 0.017, 0.0), 0.0).g;
  let h = exp(-max(p.y - F.textParams.z, 0.0) * 0.1);
  return F.misc.z * h * (0.3 + 1.4 * n * (0.55 + 0.9 * m));
}

// ---------------- lighting ----------------
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

fn pointerRadiance(p: vec3f) -> vec3f {
  let L = F.pointer.xyz - p;
  return POINTER_COLOR * (F.pointer.w * 0.35) / max(dot(L, L), 0.08);
}

// The three lamp lenses: bright discs when looked into along the beam, dim housings otherwise.
fn emitters(ro: vec3f, rd: vec3f) -> vec3f {
  var c = vec3f(0.0);
  for (var j = 0u; j < NUM_BEAMS; j++) {
    let b = F.beams[j];
    let toward = smoothstep(-0.1, 0.7, dot(-rd, b.dir.xyz));
    let oc = ro - b.pos.xyz;
    let bb = dot(oc, rd);
    let d2 = dot(oc, oc) - bb * bb;           // squared distance from lamp centre to the ray
    if (bb < 0.0) {
      let r = F.beamParams.w;
      let core = 1.0 - smoothstep(r * r * 0.6, r * r, d2);
      let glow = exp(-d2 / (r * r) * 0.35) * 0.08;
      c += b.color.rgb * (core * 4.0 + glow) * toward;
    }
  }
  return c;
}

// Excitation glow along a segment through the letter slab (= the letters). Also used for reflections.
fn slabGlow(ro: vec3f, rd: vec3f, tMax: f32, steps: u32, jitter: f32) -> f32 {
  let bb = boxHit(ro, rd, slabBox());
  let t0 = max(bb.x, 0.0);
  let t1 = min(bb.y, tMax);
  if (t1 <= t0 || steps == 0u) { return 0.0; }
  let dt = (t1 - t0) / f32(steps);
  var e = 0.0;
  for (var i = 0u; i < steps; i++) {
    let p = ro + rd * (t0 + (f32(i) + jitter) * dt);
    e += beams(p, -rd).excite;
  }
  return e * dt;
}

fn floorShade(p: vec3f, rd: vec3f, fc: vec2f) -> vec3f {
  let n = vec3f(0.0, 1.0, 0.0);
  let v = -rd;
  var c = vec3f(0.0);
  // Projections: each beam paints its stretched letter silhouette onto the floor.
  for (var j = 0u; j < NUM_BEAMS; j++) {
    let i = beamI(j, p);
    if (i > 0.0) {
      let l = normalize(F.beams[j].pos.xyz - p);
      c += F.beams[j].color.rgb * FLOOR_IRR * i * (FLOOR_ALBEDO / PI * max(l.y, 0.0) + 0.04 * ggx(n, v, l, 0.35));
    }
  }
  let pl = normalize(F.pointer.xyz - p);
  c += pointerRadiance(p) * (FLOOR_ALBEDO / PI * max(pl.y, 0.0) + 0.04 * ggx(n, v, pl, 0.3));
  // Glossy reflection of the glowing letters + lamps (normal jitter -> TAA integrates the blur).
  let j2 = vec2f(ignT(fc), ign(fc.yx + vec2f(17.0, 31.0))) - 0.5;
  let nr = normalize(vec3f(j2.x * 0.05, 1.0, j2.y * 0.05));
  let rr = reflect(rd, nr);
  let fr = fresnel(v.y, 0.04);
  let glow = slabGlow(p + vec3f(0.0, 1e-3, 0.0), rr, 60.0, u32(F.quality.z), ignT(fc + vec2f(5.0, 9.0)));
  c += (EXCITE_COLOR * glow * 0.35 + emitters(p, rr)) * fr;
  return c * floorFade(p);
}

fn marbleShade(p: vec3f, n: vec3f, rd: vec3f, fc: vec2f) -> vec3f {
  let v = -rd;
  let f = CHROME_F0 + (vec3f(1.0) - CHROME_F0) * pow(1.0 - max(dot(n, v), 0.0), 5.0);
  let r = reflect(rd, n);
  var c = emitters(p, r) + EXCITE_COLOR * slabGlow(p + n * 1e-3, r, 60.0, u32(F.quality.z), ignT(fc));
  if (r.y < 0.0) { c += vec3f(0.004); }
  for (var j = 0u; j < NUM_BEAMS; j++) {
    let i = beamI(j, p + n * 0.01);
    if (i > 0.0) {
      let l = normalize(F.beams[j].pos.xyz - p);
      c += F.beams[j].color.rgb * i * ggx(n, v, l, 0.12) * 0.5;
    }
  }
  let pl = normalize(F.pointer.xyz - p);
  c += pointerRadiance(p) * ggx(n, v, pl, 0.15);
  return f * c;
}

@fragment fn fs(@builtin(position) fc: vec4f) -> FOut {
  let uv = fc.xy * F.resolution.zw;
  let ro = F.camPos.xyz;
  let rd = rayDir(uv);
  let far = F.quality.w;

  // ---- surfaces ----
  var tS = NO_HIT;
  var kind = 0u;
  let m = marblesHit(ro, rd);
  if (m.x < tS) { tS = m.x; kind = 2u; }
  let tf = floorHit(ro, rd);
  if (tf < tS) { tS = tf; kind = 1u; }
  let tEnd = min(tS, far);

  // ---- letter slab (fine) ----
  let jit = ignT(fc.xy);
  let sb = boxHit(ro, rd, slabBox());
  let s0 = max(sb.x, 0.0);
  let s1 = min(sb.y, tEnd);
  let glow = slabGlow(ro, rd, tEnd, u32(F.quality.y), jit);

  // ---- haze (coarse) ----
  let n = u32(F.quality.x);
  let dt = tEnd / f32(n);
  var T = 1.0;
  var Ls = vec3f(0.0);
  var Tslab = 1.0;
  for (var i = 0u; i < n; i++) {
    let t = (f32(i) + jit) * dt;
    let p = ro + rd * t;
    let sig = density(p);
    let b = beams(p, -rd);
    var add = b.scatter * sig;
    // mid-air crossings outside the slab glow faintly (pair term); inside the slab the fine march owns it
    if (t < s0 || t > s1) { add += EXCITE_COLOR * b.excite * 0.25; }
    Ls += T * add * dt;
    T *= exp(-sig * dt);
    if (t < s0) { Tslab = T; }
  }

  // ---- surface shading ----
  var surf = vec3f(0.0);
  let p = ro + rd * tS;
  switch kind {
    case 1u: { surf = floorShade(p, rd, fc.xy); }
    case 2u: { surf = marbleShade(p, normalize(p - F.marbles[u32(m.y)].xyz), rd, fc.xy); }
    default: { surf = emitters(ro, rd); }
  }

  var col = surf * T + Ls + EXCITE_COLOR * glow * Tslab;
  col = select(col, vec3f(0.0), col != col);

  var o: FOut;
  o.color = vec4f(min(col, vec3f(256.0)), 1.0);
  // Depth: surfaces at their hit; glowing letters at the text plane (stable TAA reprojection); else far.
  var tDepth = -1.0;
  if (kind != 0u && tS < far) { tDepth = tS; }
  else if (glow > 0.02 && abs(rd.z) > 1e-4) { tDepth = -ro.z / rd.z; }
  if (tDepth > 0.0) {
    let c = F.viewProj * vec4f(ro + rd * tDepth, 1.0);
    o.depth = clamp(c.z / c.w, 0.0, 1.0);
  } else {
    o.depth = 1.0;
  }
  return o;
}
