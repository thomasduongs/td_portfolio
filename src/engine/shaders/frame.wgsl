// Shared per-frame uniforms (v3: laser optics). Prepended to every WGSL module:
//   device.createShaderModule({ code: FRAME_WGSL + passSource })
// Bind as @group(0) @binding(0) in EVERY pipeline. Layout mirrored in frame.ts (FRAME_UBO_SIZE = 304).

struct Frame {
  viewProj:    mat4x4f, //   0 view-projection (no jitter; lines are analytically anti-aliased)
  invViewProj: mat4x4f, //  64
  view:        mat4x4f, // 128 world -> view
  camPos:      vec4f,   // 192 xyz camera, w = tan(fovY/2)
  resolution:  vec4f,   // 208 output W, H, 1/W, 1/H (device px)
  time:        vec4f,   // 224 x = seconds, y = dt, z = frame index, w = quality level
  pointer:     vec4f,   // 240 xy = pointer on plane z=0 (world), z = glint size scale (on-screen text size), w = activity 0..1
  misc:        vec4f,   // 256 x = exposure, y = bloom strength, z = reveal distance (path units lit so far), w = click pulse front (path units, < 0 = none)
  beamStyle:   vec4f,   // 272 x = core width px, y = glow width px, z = glow gain, w = floor Y (world)
  counts:      vec4f,   // 288 x = #segments, y = #mirrors, z = #glass objects, w = floor reflection gain
}; // 304 bytes

@group(0) @binding(0) var<uniform> F: Frame;

const PI: f32 = 3.14159265359;

fn ign(p: vec2f) -> f32 { return fract(52.9829189 * fract(dot(p, vec2f(0.06711056, 0.00583715)))); }
fn ignT(p: vec2f) -> f32 { return fract(ign(p) + F.time.z * 0.61803398875); }

fn rayDir(uv: vec2f) -> vec3f {
  let ndc = vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
  let w = F.invViewProj * vec4f(ndc, 1.0, 1.0);
  return normalize(w.xyz / w.w - F.camPos.xyz);
}

// ---- optics scene (storage buffers, built once per text layout on the CPU; see src/engine/optics/types.ts) ----
struct Segment {
  a:     vec4f, // xyz start, w = path arc length at start
  b:     vec4f, // xyz end,   w = path arc length at end
  color: vec4f, // rgb radiance, w = kind: 0 beam, 1 inside glass, 2 dispersion side-ray (fades along its length)
};
struct Mirror {
  pos:    vec4f, // xyz centre, w = half size (world)
  normal: vec4f, // xyz unit normal, w = path arc length where the beam arrives
  color:  vec4f, // rgb of the arriving beam, w = kind: 0 outline mirror, 1 router mirror, 2 beam splitter
};
struct Glass {
  center: vec4f, // xyz, w = kind: 0 triangular prism (equilateral cross-section in local XY, extruded along Z), 1 box
  ax:     vec4f, // local X axis (unit), w = half extent X (prism: circumradius of the triangle)
  ay:     vec4f, // local Y axis (unit), w = half extent Y (prism: unused)
  az:     vec4f, // local Z axis (unit), w = half extent Z
};
