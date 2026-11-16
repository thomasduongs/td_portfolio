@group(1) @binding(0) var<storage, read_write> positions: array<vec4f>;
@group(1) @binding(1) var<storage, read_write> velocities: array<vec4f>;
@group(1) @binding(2) var<storage, read_write> radiance: array<vec4f>;
@group(1) @binding(3) var sdfTex: texture_2d<f32>;
@group(1) @binding(4) var samp: sampler;
@group(1) @binding(5) var<uniform> params: vec4u;

fn hash(seed: u32) -> f32 {
  let state = seed * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return f32((word >> 22u) ^ word) / 4294967296.0;
}
@compute @workgroup_size(256)
fn initialize(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= arrayLength(&positions)) { return; }
  positions[id.x] = vec4f(0.0, 0.0, 0.0, -1.0);
  velocities[id.x] = vec4f(0.0); radiance[id.x] = vec4f(0.0);
}

@compute @workgroup_size(256)
fn update(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x; if (i >= params.x) { return; }
  var p = positions[i]; var v = velocities[i];
  let dt = min(F.time.y, 0.05);
  if (p.w < 0.0 || p.w > v.w || abs(p.x) > F.textParams.x + 1.0 || p.y < F.textParams.z || p.y > F.textParams.z + 3.5 || abs(p.z) > 3.0) {
    let seed = i * 7u + u32(F.time.z) * 131u;
    p = vec4f((hash(seed) * 2.0 - 1.0) * (F.textParams.x + 1.0), F.textParams.z + hash(seed + 1u) * 3.5, hash(seed + 2u) * 6.0 - 3.0, 0.0);
    v = vec4f((hash(seed + 3u) - 0.5) * 0.04, 0.0, (hash(seed + 4u) - 0.5) * 0.04, 8.0 + hash(seed + 5u) * 20.0);
  }
  let q = p.xyz * 0.35 + F.time.x * 0.05;
  // Curl of a cyclic sinusoidal vector potential: divergence-free flow.
  let curl = vec3f(cos(q.y) - cos(q.z), cos(q.z) - cos(q.x), cos(q.x) - cos(q.y));
  var velocity = v.xyz + (curl * 0.08 - v.xyz * 0.6 + vec3f(0.0, -0.005, 0.0)) * dt;
  let offset = p.xyz - F.pointer.xyz; let distanceToPointer = length(offset);
  if (distanceToPointer < 0.8) { velocity += offset / max(distanceToPointer, 0.01) * (1.0 - distanceToPointer / 0.8) * F.pointer.w * dt * 0.3; }
  let point = p.xyz + velocity * dt;
  positions[i] = vec4f(point, p.w + dt); velocities[i] = vec4f(velocity, v.w);
  let view = normalize(F.camPos.xyz - point);
  let b = beams(point, view);
  let pointerDistance2 = dot(point - F.pointer.xyz, point - F.pointer.xyz);
  let pointerLight = vec3f(0.55, 0.75, 1.0) * F.pointer.w * 0.3 / max(0.1, pointerDistance2);
  let rgb = b.scatter * 0.35 + EXCITE_COLOR * b.excite * 0.08 + pointerLight;
  // Soft volume borders + age fade: the spawn box and respawns never read as hard edges or pops.
  let bx = 1.0 - smoothstep(F.textParams.x - 0.2, F.textParams.x + 1.0, abs(point.x));
  let by = 1.0 - smoothstep(2.0, 3.5, point.y - F.textParams.z);
  let bz = 1.0 - smoothstep(1.8, 3.0, abs(point.z));
  let age = p.w + dt;
  let life = smoothstep(0.0, 1.5, age) * (1.0 - smoothstep(v.w - 1.5, v.w, age));
  radiance[i] = vec4f(min(vec3f(30.0), rgb) * bx * by * bz * life, 0.0);
}

