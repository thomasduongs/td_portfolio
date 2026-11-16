// TAA upsample: jittered internal-res scene -> output-res history blend. YCoCg neighbourhood clip.
@group(1) @binding(0) var sceneTex: texture_2d<f32>;
@group(1) @binding(1) var depthTex: texture_depth_2d;
@group(1) @binding(2) var histTex: texture_2d<f32>;
@group(1) @binding(3) var samp: sampler;

@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}

fn toYCoCg(c: vec3f) -> vec3f {
  return vec3f(0.25 * c.r + 0.5 * c.g + 0.25 * c.b, 0.5 * c.r - 0.5 * c.b, -0.25 * c.r + 0.5 * c.g - 0.25 * c.b);
}
fn fromYCoCg(c: vec3f) -> vec3f {
  let t = c.x - c.z;
  return vec3f(t + c.y, c.x + c.z, t - c.y);
}
fn luma(c: vec3f) -> f32 { return dot(c, vec3f(0.2126, 0.7152, 0.0722)); }

// The scene is rendered with ndc' = ndc + jitter, so the texel holding the UNjittered sample
// for output uv sits at uv' = uv + jitter * (0.5, -0.5).
fn sceneUV(uv: vec2f) -> vec2f { return uv + F.jitter.xy * vec2f(0.5, -0.5); }

fn loadScene(uv: vec2f) -> vec3f {
  return min(textureSampleLevel(sceneTex, samp, uv, 0.0).rgb, vec3f(64.0));
}

@fragment fn fs_upsample(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let uv = pos.xy * F.outRes.zw;
  return vec4f(loadScene(sceneUV(uv)), 1.0);
}

// Clip history toward the AABB center (not clamp), box in YCoCg.
fn clipAABB(bmin: vec3f, bmax: vec3f, h: vec3f) -> vec3f {
  let center = 0.5 * (bmax + bmin);
  let ext = 0.5 * (bmax - bmin) + vec3f(1e-4);
  let v = h - center;
  let a = abs(v / ext);
  let m = max(a.x, max(a.y, a.z));
  if (m > 1.0) { return center + v / m; }
  return h;
}

@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let uv = pos.xy * F.outRes.zw;
  let suv = sceneUV(uv);
  let cur = loadScene(suv);

  var nmin = vec3f(1e9);
  var nmax = vec3f(-1e9);
  for (var y = -1; y <= 1; y++) {
    for (var x = -1; x <= 1; x++) {
      let c = toYCoCg(loadScene(suv + vec2f(f32(x), f32(y)) * F.resolution.zw));
      nmin = min(nmin, c);
      nmax = max(nmax, c);
    }
  }

  // Reprojection from depth (unjittered world position of this output pixel).
  let dims = vec2i(textureDimensions(depthTex));
  let ip = clamp(vec2i(suv * F.resolution.xy), vec2i(0), dims - vec2i(1));
  let z = textureLoad(depthTex, ip, 0);
  let ndcU = vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
  var world: vec3f;
  if (z >= 1.0) {
    world = F.camPos.xyz + rayDir(suv) * 100.0;
  } else {
    let w = F.invViewProj * vec4f(ndcU + F.jitter.xy, z, 1.0);
    world = w.xyz / w.w;
  }
  let pc = F.prevViewProj * vec4f(world, 1.0);
  let pndc = pc.xy / pc.w;
  let puv = vec2f(pndc.x * 0.5 + 0.5, 0.5 - pndc.y * 0.5);

  let reject = pc.w <= 0.0 || any(puv < vec2f(0.0)) || any(puv > vec2f(1.0)) || F.time.z < 0.5;
  if (reject) { return vec4f(cur, 1.0); }

  var hist = min(textureSampleLevel(histTex, samp, puv, 0.0).rgb, vec3f(64.0));
  hist = fromYCoCg(clipAABB(nmin, nmax, toYCoCg(hist)));

  let alpha = 0.1;
  let wc = alpha / (1.0 + luma(cur));
  let wh = (1.0 - alpha) / (1.0 + luma(hist));
  return vec4f((cur * wc + hist * wh) / (wc + wh), 1.0);
}
