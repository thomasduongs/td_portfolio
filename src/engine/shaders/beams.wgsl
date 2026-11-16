// Beam library (v2). Prepended AFTER frame.wgsl by consumers that evaluate light:
//   FRAME_WGSL + BEAMS_WGSL + passSource
// The consumer MUST declare these two bindings (any group/binding numbers) with exactly these names:
//   var sdfTex: texture_2d<f32>;   // text SDF, .r = signed distance (world units, negative inside), row 0 = world bottom
//   var samp: sampler;             // linear, clamp-to-edge
//
// Model: each of the 3 lamps is a gobo projector whose pattern is the text silhouette seen through the
// plane z = 0. A beam alone is a long extruded streak of letter-shaped light. The haze is a two-step
// excitation medium: it scatters each beam linearly AND glows where beams overlap (pair term) and
// intensely where all three overlap (triple term). All three silhouettes coincide only on the plane
// z = 0, so the triple term = the letters, formed purely by intersecting light (a visual hull).

fn sdf2(p: vec2f) -> f32 {
  let hw = F.textParams.xy;
  let uv = p / hw * 0.5 + 0.5;
  let d = textureSampleLevel(sdfTex, samp, clamp(uv, vec2f(0.0), vec2f(1.0)), 0.0).r;
  let o = length(max(abs(p) - hw, vec2f(0.0)));
  return select(d, max(d, 0.0) + o, o > 0.0);
}

// 1 inside letters, 0 outside, soft edge of width textParams.w.
fn letterMask(q: vec2f) -> f32 { return smoothstep(F.textParams.w, -F.textParams.w, sdf2(q)); }

// Unit vector from lamp j to p, and squared distance.
struct BeamGeo { l: vec3f, d2: f32 };
fn beamGeo(j: u32, p: vec3f) -> BeamGeo {
  let v = p - F.beams[j].pos.xyz;
  let d2 = max(dot(v, v), 1e-4);
  return BeamGeo(v * inverseSqrt(d2), d2);
}

// Gobo transmission for lamp j at point p: project p (shifted by the lamp's misalignment) from the lamp
// onto the plane z = 0 and look up the letter mask there.
fn gobo(j: u32, p: vec3f) -> f32 {
  let b = F.beams[j];
  let pp = p - b.align.xyz;
  let dz = pp.z - b.pos.z;
  if (abs(dz) < 1e-3) { return 0.0; }
  let s = -b.pos.z / dz;
  if (s <= 0.0) { return 0.0; }
  let q = b.pos.xy + (pp.xy - b.pos.xy) * s;
  return mix(b.color.w, 1.0, letterMask(q));
}

// Marble occlusion of lamp j at p: angular test against each sphere that lies between lamp and p.
fn marbleOcclusion(j: u32, p: vec3f, g: BeamGeo) -> f32 {
  let n = min(u32(F.misc.w), MAX_MARBLES);
  var vis = 1.0;
  let lp = F.beams[j].pos.xyz;
  for (var i = 0u; i < n; i++) {
    let m = F.marbles[i];
    let v = m.xyz - lp;
    let t = dot(v, g.l);                       // distance along the lamp->p ray to the sphere's closest point
    if (t <= 0.0 || t * t >= g.d2) { continue; } // sphere behind lamp or beyond p
    let h2 = dot(v, v) - t * t;                // squared perpendicular distance
    // soft edge: penumbra grows with distance from the sphere to p
    let pen = 0.02 + 0.04 * (sqrt(g.d2) - t);
    vis *= smoothstep(m.w - pen, m.w + pen, sqrt(h2));
  }
  return vis;
}

// Normalized beam intensity (~1 at the text plane inside the letters, cone + gobo + inverse-square + marbles).
fn beamI(j: u32, p: vec3f) -> f32 {
  let b = F.beams[j];
  let g = beamGeo(j, p);
  let cone = smoothstep(b.pos.w, b.dir.w, dot(g.l, b.dir.xyz));
  if (cone <= 0.0) { return 0.0; }
  var i = cone * gobo(j, p) * (b.align.w / g.d2);
  if (i > 1e-4 && F.misc.w > 0.5) { i *= marbleOcclusion(j, p, g); }
  return i;
}

struct BeamSample {
  scatter: vec3f,  // linear in-scattered radiance per unit sigma (already includes phase toward the viewer)
  excite: f32,     // two-step excitation emission per unit length (pair + triple terms)
  i: vec3f,        // raw normalized intensities of beams 0..2 (for lighting surfaces / particles)
};

// toViewer = unit vector from p toward the camera.
fn beams(p: vec3f, toViewer: vec3f) -> BeamSample {
  var s: BeamSample;
  var iv = vec3f(0.0);
  for (var j = 0u; j < NUM_BEAMS; j++) {
    let ij = beamI(j, p);
    iv[j] = ij;
    if (ij > 0.0) {
      let l = normalize(p - F.beams[j].pos.xyz);
      s.scatter += F.beams[j].color.rgb * ij * hg(dot(l, toViewer), 0.35);
    }
  }
  s.i = iv;
  let pair = iv.x * iv.y + iv.y * iv.z + iv.z * iv.x;
  let triple = iv.x * iv.y * iv.z;
  s.excite = F.beamParams.x * pair + F.beamParams.y * triple;
  return s;
}

// Colour of the excitation glow (warm white: the three beams combined).
const EXCITE_COLOR: vec3f = vec3f(1.0, 0.96, 0.9);
