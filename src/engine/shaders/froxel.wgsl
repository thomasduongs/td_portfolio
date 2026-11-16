@group(1) @binding(0) var sdf: texture_2d<f32>;
@group(1) @binding(1) var linearSampler: sampler;
@group(1) @binding(2) var previousScatter: texture_3d<f32>;
@group(1) @binding(3) var scatterOut: texture_storage_3d<rgba16float, write>;
@group(1) @binding(4) var currentScatter: texture_3d<f32>;
@group(1) @binding(5) var integratedOut: texture_storage_3d<rgba16float, write>;

const FOG_NEAR: f32 = 0.3;
fn sliceDistance(z: f32) -> f32 { return FOG_NEAR * pow(F.froxel.w / FOG_NEAR, z / F.froxel.z); }
fn hash3(p: vec3f) -> f32 { return fract(sin(dot(p, vec3f(127.1, 311.7, 74.7))) * 43758.5453); }
fn noise3(p: vec3f) -> f32 {
  let i = floor(p); let f = fract(p); let u = f*f*(3.0-2.0*f);
  return mix(mix(mix(hash3(i), hash3(i+vec3f(1,0,0)), u.x), mix(hash3(i+vec3f(0,1,0)), hash3(i+vec3f(1,1,0)), u.x), u.y),
    mix(mix(hash3(i+vec3f(0,0,1)), hash3(i+vec3f(1,0,1)), u.x), mix(hash3(i+vec3f(0,1,1)), hash3(i+vec3f(1,1,1)), u.x), u.y), u.z);
}
fn textSDF(p: vec3f) -> f32 {
  let uv = p.xy / (2.0 * F.textParams.xy) + 0.5;
  let outside = length(max(abs(p.xy) - F.textParams.xy, vec2f(0.0)));
  let d = textureSampleLevel(sdf, linearSampler, clamp(uv, vec2f(0.0), vec2f(1.0)), 0.0).r + outside;
  let q = vec2f(d, abs(p.z) - F.textParams.z);
  return min(max(q.x, q.y), 0.0) + length(max(q, vec2f(0.0)));
}
fn visibility(p: vec3f, light: vec3f) -> f32 {
  let delta = light - p; let len = length(delta); let rd = delta / max(len, 0.001);
  let inv = 1.0 / select(vec3f(0.00001), rd, abs(rd) > vec3f(0.00001));
  let a = (-F.textParams.xyz - p) * inv; let b = (F.textParams.xyz - p) * inv;
  let lo = min(a,b); let hi = max(a,b);
  var t = max(0.02, max(lo.x, max(lo.y, lo.z)));
  let end = min(len, min(hi.x, min(hi.y, hi.z)));
  if (t >= end) { return 1.0; }
  var shadow = 1.0;
  for (var i = 0u; i < u32(F.quality.w); i++) {
    if (t > end) { break; }
    let d = textSDF(p + rd*t);
    if (d < 0.002) { return 0.38; } // glass transmits part of the light
    shadow = min(shadow, 10.0*d/t); t += max(d,0.02);
  }
  return mix(0.38, 1.0, clamp(shadow, 0.0, 1.0));
}
fn hg(cosTheta: f32, g: f32) -> f32 {
  return (1.0-g*g) / (4.0*PI*pow(max(0.001,1.0+g*g-2.0*g*cosTheta),1.5));
}

fn injectVoxel(id: vec3u, useHistory: bool) {
  if (any(id >= vec3u(F.froxel.xyz))) { return; }
  let uv = (vec2f(id.xy)+0.5)/F.froxel.xy;
  let direction = rayDir(uv);
  let distance = sliceDistance(f32(id.z)+ignT(vec2f(id.xy)));
  let p = F.camPos.xyz + direction*distance;
  let sigma = F.misc.z * (0.6+0.4*noise3(p*0.5+F.time.x*0.03)) * exp(-max(p.y-F.spotColor.w,0.0)*0.25);
  let delta = F.spotPos.xyz-p; let l = delta / max(length(delta),0.001);
  let cone = smoothstep(F.spotPos.w,F.spotDir.w,dot(-l,F.spotDir.xyz));
  let spot = F.spotColor.rgb * cone / max(1.0,dot(delta,delta)) * hg(dot(l,direction),0.55) * visibility(p,F.spotPos.xyz);
  let pointerDelta = F.pointer.xyz-p;
  let pointerDir = pointerDelta / max(length(pointerDelta),0.001);
  let pointer = vec3f(0.55,0.75,1.0) * F.pointer.w / max(1.0,dot(pointerDelta,pointerDelta)) * hg(dot(pointerDir,direction),0.3);
  var result = vec4f(sigma*(spot+pointer),sigma);
  if (useHistory) {
    let previousClip = F.prevViewProj * vec4f(p,1.0);
    if (previousClip.w > 0.0) {
      let ndc = previousClip.xy/previousClip.w;
      let previousUV = ndc*vec2f(0.5,-0.5)+0.5;
      // Clip W is camera-axis depth; convert to radial distance for exponential slices.
      let slope = ndc * vec2f(F.outRes.x/F.outRes.y,1.0) * F.camPos.w;
      let previousDistance = previousClip.w * sqrt(1.0+dot(slope,slope));
      let previousZ = log(max(previousDistance,0.0001)/FOG_NEAR)/log(F.froxel.w/FOG_NEAR);
      let coord = vec3f(previousUV,previousZ);
      if (all(coord >= vec3f(0.0)) && all(coord <= vec3f(1.0))) {
        let history = textureSampleLevel(previousScatter,linearSampler,coord,0.0);
        // Reject large density changes (e.g. camera crossing the fog boundary).
        if (abs(history.a-sigma) <= max(0.02,sigma*0.5)) { result = mix(result,history,0.9); }
      }
    }
  }
  textureStore(scatterOut,vec3i(id),result);
}

@compute @workgroup_size(4,4,4)
fn inject(@builtin(global_invocation_id) id: vec3u) { injectVoxel(id,true); }
@compute @workgroup_size(4,4,4)
fn injectFresh(@builtin(global_invocation_id) id: vec3u) { injectVoxel(id,false); }

@compute @workgroup_size(8,8)
fn integrate(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= vec2u(F.froxel.xy))) { return; }
  var transmittance = 1.0; var accumulated = vec3f(0.0);
  for (var z = 0u; z < u32(F.froxel.z); z++) {
    let voxel = textureLoad(currentScatter,vec3i(vec2i(id.xy),i32(z)),0);
    let thickness = sliceDistance(f32(z+1u))-sliceDistance(f32(z));
    let sliceT = exp(-voxel.a*thickness);
    let energy = voxel.rgb * (1.0-sliceT)/max(voxel.a,0.00001);
    accumulated += transmittance*energy; transmittance *= sliceT;
    textureStore(integratedOut,vec3i(vec2i(id.xy),i32(z)),vec4f(accumulated,transmittance));
  }
}
