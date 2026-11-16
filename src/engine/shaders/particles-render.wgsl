@group(1) @binding(0) var<storage, read> positions: array<vec4f>;
@group(1) @binding(2) var<storage, read> radiance: array<vec4f>;
struct VertexOut { @builtin(position) position: vec4f, @location(0) local: vec2f, @location(1) light: vec3f }
@vertex
fn vs(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32) -> VertexOut {
  let corners = array<vec2f, 6>(vec2f(-1.0,-1.0), vec2f(1.0,-1.0), vec2f(-1.0,1.0), vec2f(-1.0,1.0), vec2f(1.0,-1.0), vec2f(1.0,1.0));
  let corner = corners[vertex];
  var clip = F.viewProj * vec4f(positions[instance].xyz, 1.0);
  let radiusPx = max(1.2, 0.012 * F.resolution.y / (2.0 * F.camPos.w * max(0.01, clip.w)));
  clip = vec4f(clip.xy + corner * radiusPx * 2.0 * F.resolution.zw * clip.w, clip.zw);
  var out: VertexOut; out.position = clip; out.local = corner; out.light = radiance[instance].rgb; return out;
}
@fragment
fn fs(in: VertexOut) -> @location(0) vec4f {
  let radius2 = dot(in.local, in.local); if (radius2 > 1.0) { discard; }
  return vec4f(exp(-radius2 * 4.0) * in.light * 0.22, 0.0);
}
