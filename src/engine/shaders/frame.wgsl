// Shared per-frame uniforms. Prepended (string concat) to every WGSL module by the TS side:
//   device.createShaderModule({ code: FRAME_WGSL + passSource })
// Bind as @group(0) @binding(0) in EVERY pipeline (one bind group 0 for all passes -> no rebinding cost).
// Byte layout mirrored exactly in frame.ts (FRAME_UBO_SIZE = 656).
const MAX_MARBLES: u32 = 16u;

struct Frame {
  viewProj:     mat4x4f, //   0 jittered view-projection (scene pass)
  invViewProj:  mat4x4f, //  64 inverse of jittered viewProj (ray generation)
  prevViewProj: mat4x4f, // 128 previous frame UNjittered viewProj (TAA + froxel reprojection)
  camPos:       vec4f,   // 192 xyz camera world pos, w = tan(fovY/2)
  resolution:   vec4f,   // 208 internal W, H, 1/W, 1/H
  outRes:       vec4f,   // 224 output W, H, 1/W, 1/H
  jitter:       vec4f,   // 240 xy = current jitter in NDC, zw = previous jitter in NDC
  time:         vec4f,   // 256 x = seconds, y = dt, z = frame index, w = quality level
  pointer:      vec4f,   // 272 xyz = pointer light world pos, w = intensity (0 when idle)
  spotPos:      vec4f,   // 288 xyz = key spot light pos, w = cos(outer cone angle)
  spotDir:      vec4f,   // 304 xyz = normalized spot direction, w = cos(inner cone angle)
  spotColor:    vec4f,   // 320 rgb = radiance * intensity, w = floor plane world Y (glyphs stand on it)
  textParams:   vec4f,   // 336 x = half width, y = half height (world), z = half depth (extrusion), w = bevel radius
  quality:      vec4f,   // 352 primarySteps, interiorSteps, reflSteps, shadowSteps (as f32)
  froxel:       vec4f,   // 368 gridX, gridY, gridZ, far distance (world units along ray)
  misc:         vec4f,   // 384 x = exposure, y = bloom strength, z = fog density, w = marble count
  marbles: array<vec4f, 16>, // 400 xyz = center, w = radius (first misc.w are valid)
}; // total 656 bytes

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
