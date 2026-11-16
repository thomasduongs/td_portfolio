// Shared per-frame uniforms (v2: light-beam text). Prepended to every WGSL module by the TS side:
//   device.createShaderModule({ code: FRAME_WGSL + passSource })
// Bind as @group(0) @binding(0) in EVERY pipeline. Byte layout mirrored exactly in frame.ts (FRAME_UBO_SIZE = 800).
const MAX_MARBLES: u32 = 16u;
const NUM_BEAMS: u32 = 3u;

// One gobo projector. The beam carries the text silhouette as seen through the plane z = 0.
struct Beam {
  pos:   vec4f, // xyz = lamp world position, w = cos(outer cone half-angle)
  dir:   vec4f, // xyz = normalized beam axis, w = cos(inner cone half-angle)
  color: vec4f, // rgb = scatter radiance scale, w = gobo leak (light passing outside the letters, 0..1)
  align: vec4f, // xyz = misalignment offset (world) applied to the gobo lookup, w = d0^2 (lamp->origin dist^2, normalizes intensity to ~1 at the text)
};

struct Frame {
  viewProj:     mat4x4f, //   0 jittered view-projection
  invViewProj:  mat4x4f, //  64 inverse of jittered viewProj (ray generation)
  prevViewProj: mat4x4f, // 128 previous frame UNjittered viewProj (TAA reprojection)
  camPos:       vec4f,   // 192 xyz camera world pos, w = tan(fovY/2)
  resolution:   vec4f,   // 208 internal W, H, 1/W, 1/H
  outRes:       vec4f,   // 224 output W, H, 1/W, 1/H
  jitter:       vec4f,   // 240 xy = current jitter in NDC, zw = previous jitter in NDC
  time:         vec4f,   // 256 x = seconds, y = dt, z = frame index, w = quality level
  pointer:      vec4f,   // 272 xyz = pointer light world pos, w = intensity
  textParams:   vec4f,   // 288 x = SDF texture half width, y = half height (world), z = floor Y, w = letter mask softness (world)
  quality:      vec4f,   // 304 x = coarse haze steps, y = fine (letter slab) steps, z = floor-reflection steps, w = far distance
  misc:         vec4f,   // 320 x = exposure, y = bloom strength, z = haze density (sigma), w = marble count
  beamParams:   vec4f,   // 336 x = pair excitation gain k2, y = triple excitation gain k3, z = letter slab half thickness, w = lamp emitter radius
  beams: array<Beam, 3>, // 352 (64 B each)
  marbles: array<vec4f, 16>, // 544 xyz = center, w = radius (first misc.w valid)
}; // total 800 bytes

@group(0) @binding(0) var<uniform> F: Frame;

// ---- shared helpers ----
const PI: f32 = 3.14159265359;

// Interleaved gradient noise (Jimenez 2014): blue-noise-like dither without a texture fetch.
fn ign(p: vec2f) -> f32 { return fract(52.9829189 * fract(dot(p, vec2f(0.06711056, 0.00583715)))); }
// Temporally varying IGN (golden-ratio offset per frame).
fn ignT(p: vec2f) -> f32 { return fract(ign(p) + F.time.z * 0.61803398875); }

// World-space ray direction for a pixel uv in [0,1] (top-left origin) using the jittered inverse VP.
fn rayDir(uv: vec2f) -> vec3f {
  let ndc = vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
  let w = F.invViewProj * vec4f(ndc, 1.0, 1.0);
  return normalize(w.xyz / w.w - F.camPos.xyz);
}

fn hg(cosT: f32, g: f32) -> f32 {
  return (1.0 - g * g) / (4.0 * PI * pow(max(1e-3, 1.0 + g * g - 2.0 * g * cosT), 1.5));
}
