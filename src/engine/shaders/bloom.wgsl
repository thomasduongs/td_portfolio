// Bloom chain: 13-tap downsample (Karis-averaged on the first), 3x3 tent upsample (additive blend in pipeline state).
@group(1) @binding(0) var srcTex: texture_2d<f32>;
@group(1) @binding(1) var samp: sampler;
@group(1) @binding(2) var<uniform> U: vec4f; // xy = destination size in texels

@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}

fn tap(uv: vec2f, o: vec2f, px: vec2f) -> vec3f {
  return min(textureSampleLevel(srcTex, samp, uv + o * px, 0.0).rgb, vec3f(64.0));
}
fn karis(c: vec3f) -> f32 {
  return 1.0 / (1.0 + dot(c, vec3f(0.2126, 0.7152, 0.0722)));
}

fn down(uv: vec2f, firstPass: bool) -> vec3f {
  let px = 1.0 / vec2f(textureDimensions(srcTex));
  let a = tap(uv, vec2f(-2, 2), px);  let b = tap(uv, vec2f(0, 2), px);  let c = tap(uv, vec2f(2, 2), px);
  let d = tap(uv, vec2f(-2, 0), px);  let e = tap(uv, vec2f(0, 0), px);  let f = tap(uv, vec2f(2, 0), px);
  let g = tap(uv, vec2f(-2, -2), px); let h = tap(uv, vec2f(0, -2), px); let i = tap(uv, vec2f(2, -2), px);
  let j = tap(uv, vec2f(-1, 1), px);  let k = tap(uv, vec2f(1, 1), px);
  let l = tap(uv, vec2f(-1, -1), px); let m = tap(uv, vec2f(1, -1), px);
  if (firstPass) {
    let g0 = (a + b + d + e) * 0.25;
    let g1 = (b + c + e + f) * 0.25;
    let g2 = (d + e + g + h) * 0.25;
    let g3 = (e + f + h + i) * 0.25;
    let g4 = (j + k + l + m) * 0.25;
    let w0 = karis(g0) * 0.125;
    let w1 = karis(g1) * 0.125;
    let w2 = karis(g2) * 0.125;
    let w3 = karis(g3) * 0.125;
    let w4 = karis(g4) * 0.5;
    return (g0 * w0 + g1 * w1 + g2 * w2 + g3 * w3 + g4 * w4) / (w0 + w1 + w2 + w3 + w4);
  }
  return e * 0.125 + (a + c + g + i) * 0.03125 + (b + d + f + h) * 0.0625 + (j + k + l + m) * 0.125;
}

@fragment fn fs_down_first(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  return vec4f(down(pos.xy / U.xy, true), 1.0);
}

@fragment fn fs_down(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  return vec4f(down(pos.xy / U.xy, false), 1.0);
}

@fragment fn fs_up(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let uv = pos.xy / U.xy;
  let px = 1.0 / vec2f(textureDimensions(srcTex));
  var s = tap(uv, vec2f(-1, 1), px) + tap(uv, vec2f(1, 1), px) + tap(uv, vec2f(-1, -1), px) + tap(uv, vec2f(1, -1), px);
  s += 2.0 * (tap(uv, vec2f(0, 1), px) + tap(uv, vec2f(0, -1), px) + tap(uv, vec2f(-1, 0), px) + tap(uv, vec2f(1, 0), px));
  s += 4.0 * tap(uv, vec2f(0, 0), px);
  return vec4f(s * (1.0 / 16.0) * 0.7, 1.0);
}
