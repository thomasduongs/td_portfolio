// Composite: froxel fog at pixel depth, bloom mix, exposure, AgX tonemap, vignette, grain+dither, OETF.
@group(1) @binding(0) var hdrTex: texture_2d<f32>;
@group(1) @binding(1) var bloomTex: texture_2d<f32>;
@group(1) @binding(2) var depthTex: texture_depth_2d;
@group(1) @binding(3) var fogTex: texture_3d<f32>;
@group(1) @binding(4) var samp: sampler;

const FOG_NEAR: f32 = 0.3;

@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}

fn agxContrast(x: vec3f) -> vec3f {
  let x2 = x * x;
  let x4 = x2 * x2;
  return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232;
}

// AgX (Troy Sobotka) polynomial approximation, linear sRGB in -> display-linear sRGB out.
fn agx(c0: vec3f) -> vec3f {
  let toRec2020 = mat3x3f(vec3f(0.6274, 0.0691, 0.0164), vec3f(0.3293, 0.9195, 0.0880), vec3f(0.0433, 0.0113, 0.8956));
  let toSrgb = mat3x3f(vec3f(1.6605, -0.1246, -0.0182), vec3f(-0.5876, 1.1329, -0.1006), vec3f(-0.0728, -0.0083, 1.1187));
  let inset = mat3x3f(vec3f(0.856627153315983, 0.137318972929847, 0.11189821299995), vec3f(0.0951212405381588, 0.761241990602591, 0.0767994186031903), vec3f(0.0482516061458583, 0.101439036467562, 0.811302368396859));
  let outset = mat3x3f(vec3f(1.1271005818144368, -0.1413297634984383, -0.14132976349843826), vec3f(-0.11060664309660323, 1.157823702216272, -0.11060664309660294), vec3f(-0.016493938717834573, -0.016493938717834257, 1.2519364065950405));
  let minEv = -12.47393;
  let maxEv = 4.026069;
  var c = inset * (toRec2020 * c0);
  c = log2(max(c, vec3f(1e-10)));
  c = clamp((c - minEv) / (maxEv - minEv), vec3f(0.0), vec3f(1.0));
  c = agxContrast(c);
  c = outset * c;
  c = pow(max(c, vec3f(0.0)), vec3f(2.2));
  return max(toSrgb * c, vec3f(0.0));
}

fn oetf(c: vec3f) -> vec3f {
  let lo = c * 12.92;
  let hi = 1.055 * pow(max(c, vec3f(0.0)), vec3f(1.0 / 2.4)) - 0.055;
  return select(hi, lo, c <= vec3f(0.0031308));
}

fn shade(pos: vec2f, toSrgb: bool) -> vec4f {
  let uv = pos * F.outRes.zw;
  var color = textureSampleLevel(hdrTex, samp, uv, 0.0).rgb;

  // Distance along the view ray from the internal-res depth.
  let dims = vec2i(textureDimensions(depthTex));
  let ip = clamp(vec2i(uv * F.resolution.xy), vec2i(0), dims - vec2i(1));
  let z = textureLoad(depthTex, ip, 0);
  let far = F.froxel.w;
  var d = far;
  if (z < 1.0) {
    let ndc = vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let w = F.invViewProj * vec4f(ndc, z, 1.0);
    d = min(length(w.xyz / w.w - F.camPos.xyz), far);
  }
  let sz = clamp(log(max(d, FOG_NEAR) / FOG_NEAR) / log(far / FOG_NEAR), 0.0, 1.0);
  let fog = textureSampleLevel(fogTex, samp, vec3f(uv, sz), 0.0);
  color = color * fog.a + fog.rgb;

  let bloom = textureSampleLevel(bloomTex, samp, uv, 0.0).rgb;
  color = mix(color, bloom, F.misc.y);
  color = color * F.misc.x;

  var t = agx(max(color, vec3f(0.0)));
  // Slight punch: gentle saturation boost and contrast around mid grey.
  let l = dot(t, vec3f(0.2126, 0.7152, 0.0722));
  t = mix(vec3f(l), t, 1.1);
  t = pow(max(t, vec3f(0.0)), vec3f(1.06));

  var o = select(t, oetf(t), toSrgb);
  // Vignette 0.25 in display space.
  let q = uv * 2.0 - 1.0;
  o *= 1.0 - 0.25 * smoothstep(0.35, 1.5, dot(q, q));
  // Grain + 8-bit dither.
  o += (ignT(pos) - 0.5) * 0.015;
  o += (ign(pos + vec2f(17.0, 31.0)) - 0.5) / 255.0;
  return vec4f(clamp(o, vec3f(0.0), vec3f(1.0)), 1.0);
}

// Non-sRGB swapchain: apply the sRGB OETF manually.
@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f { return shade(pos.xy, true); }
// '*-srgb' swapchain: hardware encodes.
@fragment fn fs_srgb(@builtin(position) pos: vec4f) -> @location(0) vec4f { return shade(pos.xy, false); }
