@group(1) @binding(0) var<storage, read_write> positions: array<vec4f>;
@group(1) @binding(1) var<storage, read_write> velocities: array<vec4f>;
@group(1) @binding(2) var<storage, read_write> radiance: array<f32>;
@group(1) @binding(3) var sdf: texture_2d<f32>;
@group(1) @binding(4) var linearSampler: sampler;
@group(1) @binding(5) var<uniform> params: vec4u;

fn hash(seed: u32) -> f32 {
  let state = seed * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return f32((word >> 22u) ^ word) / 4294967296.0;
}
fn text2D(p: vec2f) -> f32 {
  let uv = p / (2.0 * F.textParams.xy) + 0.5;
  let outside = length(max(abs(p) - F.textParams.xy, vec2f(0.0)));
  return textureSampleLevel(sdf, linearSampler, clamp(uv, vec2f(0.0), vec2f(1.0)), 0.0).r + outside;
}
fn textSDF(p: vec3f) -> f32 {
  let d = text2D(p.xy);
  let q = vec2f(d, abs(p.z) - F.textParams.z);
  return min(max(q.x, q.y), 0.0) + length(max(q, vec2f(0.0)));
}
fn visibility(p: vec3f, l: vec3f) -> f32 {
  let delta = l - p; let distanceToLight = length(delta); let rd = delta / max(distanceToLight, 0.001);
  let inv = 1.0 / select(vec3f(0.00001), rd, abs(rd) > vec3f(0.00001));
  let a = (-F.textParams.xyz - p) * inv; let b = (F.textParams.xyz - p) * inv;
  let lo = min(a, b); let hi = max(a, b);
  let start = max(0.02, max(lo.x, max(lo.y, lo.z)));
  let end = min(distanceToLight, min(hi.x, min(hi.y, hi.z)));
  if (start >= end) { return 1.0; }
  var t = start; var shadow = 1.0;
  for (var i = 0u; i < u32(F.quality.w); i++) {
    if (t > end) { break; }
    let d = textSDF(p + rd * t);
    if (d < 0.002) { return 0.38; } // glass transmits part of the light
    shadow = min(shadow, 12.0 * d / t); t += max(d, 0.015);
  }
  return mix(0.38, 1.0, clamp(shadow, 0.0, 1.0));
}
fn hg(cosTheta: f32, g: f32) -> f32 {
  return (1.0 - g*g) / (4.0 * PI * pow(max(0.001, 1.0 + g*g - 2.0*g*cosTheta), 1.5));
}

@compute @workgroup_size(256)
fn initialize(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= arrayLength(&positions)) { return; }
  positions[id.x] = vec4f(0.0, 0.0, 0.0, -1.0);
  velocities[id.x] = vec4f(0.0); radiance[id.x] = 0.0;
}

@compute @workgroup_size(256)
fn update(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x; if (i >= params.x) { return; }
  var p = positions[i]; var v = velocities[i];
  let dt = min(F.time.y, 0.05);
  if (p.w < 0.0 || p.w > v.w || abs(p.x) > F.textParams.x + 1.0 || p.y < F.spotColor.w || p.y > F.spotColor.w + 3.5 || abs(p.z) > 3.0) {
    let seed = i * 7u + u32(F.time.z) * 131u;
    p = vec4f((hash(seed) * 2.0 - 1.0) * (F.textParams.x + 1.0), F.spotColor.w + hash(seed + 1u) * 3.5, hash(seed + 2u) * 6.0 - 3.0, 0.0);
    v = vec4f((hash(seed + 3u) - 0.5) * 0.04, 0.0, (hash(seed + 4u) - 0.5) * 0.04, 8.0 + hash(seed + 5u) * 20.0);
  }
  let q = p.xyz * 0.35 + F.time.x * 0.05;
  // Curl of a cyclic sinusoidal vector potential: divergence-free flow.
  let curl = vec3f(cos(q.y) - cos(q.z), cos(q.z) - cos(q.x), cos(q.x) - cos(q.y));
  var velocity = v.xyz + (curl * 0.08 - v.xyz * 0.6 + vec3f(0.0, -0.005, 0.0)) * dt;
  let offset = p.xyz - F.pointer.xyz; let distanceToPointer = length(offset);
  if (distanceToPointer < 0.8) { velocity += offset / max(distanceToPointer, 0.01) * (1.0 - distanceToPointer / 0.8) * F.pointer.w * dt * 0.3; }
  var point = p.xyz + velocity * dt;
  if (abs(point.z) < F.textParams.z + 0.05) {
    let uv = point.xy / (2.0 * F.textParams.xy) + 0.5;
    if (all(uv >= vec2f(0.0)) && all(uv <= vec2f(1.0))) {
      let d = textureSampleLevel(sdf, linearSampler, uv, 0.0).r;
      if (d < 0.02) {
        let e = 0.008;
        let dx = text2D(point.xy + vec2f(e, 0.0)) - text2D(point.xy - vec2f(e, 0.0));
        let dy = text2D(point.xy + vec2f(0.0, e)) - text2D(point.xy - vec2f(0.0, e));
        let gradient = vec2f(dx, dy); let normal = gradient / max(length(gradient), 0.00001);
        point = vec3f(point.xy + normal * (0.02 - d), point.z);
        velocity -= vec3f(normal, 0.0) * min(0.0, dot(velocity.xy, normal));
      }
    }
  }
  positions[i] = vec4f(point, p.w + dt); velocities[i] = vec4f(velocity, v.w);
  let toLight = F.spotPos.xyz - point; let dist2 = max(1.0, dot(toLight, toLight)); let l = normalize(toLight);
  let view = normalize(F.camPos.xyz - point);
  let cone = smoothstep(F.spotPos.w, F.spotDir.w, dot(-l, F.spotDir.xyz));
  let incident = dot(F.spotColor.rgb, vec3f(0.2126, 0.7152, 0.0722)) * cone / dist2 * hg(dot(-l, view), 0.5) * visibility(point, F.spotPos.xyz);
  let pointerLight = F.pointer.w * 0.3 / max(0.1, dot(point - F.pointer.xyz, point - F.pointer.xyz));
  // Soft volume borders + age fade: the spawn box and respawns never read as hard edges or pops.
  let bx = 1.0 - smoothstep(F.textParams.x - 0.2, F.textParams.x + 1.0, abs(point.x));
  let by = 1.0 - smoothstep(2.0, 3.5, point.y - F.spotColor.w);
  let bz = 1.0 - smoothstep(1.8, 3.0, abs(point.z));
  let age = p.w + dt;
  let life = smoothstep(0.0, 1.5, age) * (1.0 - smoothstep(v.w - 1.5, v.w, age));
  radiance[i] = min(30.0, incident + pointerLight) * bx * by * bz * life;
}

