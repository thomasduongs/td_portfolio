struct Params { w: u32, h: u32, step: u32, pxPerUnit: f32 }
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> mask: array<u32>;
// xy = nearest inside seed, zw = nearest outside seed.
@group(0) @binding(2) var<storage, read> source: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> destination: array<vec4f>;
@group(0) @binding(4) var output: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn seed(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= P.w || id.y >= P.h) { return; }
  let i = id.y * P.w + id.x;
  let p = vec2f(id.xy);
  destination[i] = select(vec4f(vec2f(-1.0), p), vec4f(p, vec2f(-1.0)), mask[i] > 0u);
}

fn distanceSquared(p: vec2f, s: vec2f) -> f32 {
  if (s.x < 0.0) { return 1e20; }
  return dot(p - s, p - s);
}

@compute @workgroup_size(8, 8)
fn jump(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= P.w || id.y >= P.h) { return; }
  let i = id.y * P.w + id.x;
  let p = vec2f(id.xy);
  var best = source[i];
  var inside = distanceSquared(p, best.xy);
  var outside = distanceSquared(p, best.zw);
  for (var y = -1; y <= 1; y++) {
    for (var x = -1; x <= 1; x++) {
      let q = vec2i(id.xy) + vec2i(x, y) * i32(P.step);
      if (any(q < vec2i(0)) || q.x >= i32(P.w) || q.y >= i32(P.h)) { continue; }
      let candidate = source[u32(q.y) * P.w + u32(q.x)];
      let di = distanceSquared(p, candidate.xy);
      let dout = distanceSquared(p, candidate.zw);
      if (di < inside) { inside = di; best = vec4f(candidate.xy, best.zw); }
      if (dout < outside) { outside = dout; best = vec4f(best.xy, candidate.zw); }
    }
  }
  destination[i] = best;
}

@compute @workgroup_size(8, 8)
fn finish(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= P.w || id.y >= P.h) { return; }
  let seeds = source[id.y * P.w + id.x];
  let p = vec2f(id.xy);
  let far = length(vec2f(f32(P.w), f32(P.h))) * 2.0;
  let di = select(far, distance(p, seeds.xy), seeds.x >= 0.0);
  let dout = select(far, distance(p, seeds.zw), seeds.z >= 0.0);
  let sdf = (max(0.0, di - 0.5) - max(0.0, dout - 0.5)) / P.pxPerUnit;
  textureStore(output, vec2i(id.xy), vec4f(sdf, 0.0, 0.0, 1.0));
}
