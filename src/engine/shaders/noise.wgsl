// 64^3 tileable noise bake. Standalone: does not use frame.wgsl (check-wgsl.sh prepends it, so no
// shared names are redeclared here). Output lives in @group(1) to avoid the frame UBO at @group(0).
//   r = 4-octave gradient fBm (base 4 cells), g = 2-octave gradient fBm (base 8 cells),
//   b = worley-ish 1 - F1 (8 cells), a = 1.
// Every lattice wraps modulo its own cell count, so the texture repeats with period 1 (= 64 texels).

@group(1) @binding(0) var noiseOut: texture_storage_3d<rgba8unorm, write>;

const NOISE_SIZE: u32 = 64u;

fn nz_pcg(v: u32) -> u32 {
  let state = v * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

// Hash of a lattice cell, wrapped modulo `per` so the lattice is periodic.
fn nz_hash(c: vec3<i32>, per: i32) -> u32 {
  let p = vec3<i32>(per);
  let m = ((c % p) + p) % p;
  let u = vec3<u32>(m);
  return nz_pcg(u.x + nz_pcg(u.y + nz_pcg(u.z + 1013u)));
}

// Unit-ish gradient in [-1,1]^3 from a hash.
fn nz_grad(c: vec3<i32>, per: i32) -> vec3f {
  let h = nz_hash(c, per);
  let g = vec3f(f32(h & 1023u), f32((h >> 10u) & 1023u), f32((h >> 20u) & 1023u));
  return g * (2.0 / 1023.0) - vec3f(1.0);
}

// Perlin-style gradient noise on a lattice of `per` cells per unit (periodic).
fn nz_gradient(p: vec3f, per: i32) -> f32 {
  let i = vec3<i32>(floor(p));
  let f = p - floor(p);
  let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  var r = 0.0;
  for (var k = 0; k < 8; k++) {
    let o = vec3<i32>(k & 1, (k >> 1) & 1, (k >> 2) & 1);
    let g = nz_grad(i + o, per);
    let d = f - vec3f(o);
    let wx = mix(1.0 - u.x, u.x, f32(o.x));
    let wy = mix(1.0 - u.y, u.y, f32(o.y));
    let wz = mix(1.0 - u.z, u.z, f32(o.z));
    r += wx * wy * wz * dot(g, d);
  }
  return r;
}

// Octave sum, normalised by total amplitude. Octave o uses base<<o cells (still periodic).
fn nz_fbm(uvw: vec3f, base: i32, octaves: i32) -> f32 {
  var sum = 0.0;
  var norm = 0.0;
  var amp = 1.0;
  for (var o = 0; o < octaves; o++) {
    let per = base << u32(o);
    sum += amp * nz_gradient(uvw * f32(per), per);
    norm += amp;
    amp *= 0.5;
  }
  return sum / norm;
}

// Gradient noise is roughly in [-0.4, 0.4] after normalisation; spread it over [0,1].
fn nz_remap(v: f32) -> f32 {
  return clamp(0.5 + 0.9 * v, 0.0, 1.0);
}

// Worley-ish: 1 - distance to the nearest feature point (periodic over `per` cells).
fn nz_worley(uvw: vec3f, per: i32) -> f32 {
  let p = uvw * f32(per);
  let i = vec3<i32>(floor(p));
  let f = p - floor(p);
  var d2 = 1e9;
  for (var z = -1; z <= 1; z++) {
    for (var y = -1; y <= 1; y++) {
      for (var x = -1; x <= 1; x++) {
        let c = i + vec3<i32>(x, y, z);
        let h = nz_hash(c, per);
        let jit = vec3f(f32(h & 1023u), f32((h >> 10u) & 1023u), f32((h >> 20u) & 1023u)) / 1023.0;
        let q = vec3f(f32(x), f32(y), f32(z)) + jit - f;
        d2 = min(d2, dot(q, q));
      }
    }
  }
  return 1.0 - clamp(sqrt(d2), 0.0, 1.0);
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= NOISE_SIZE || id.y >= NOISE_SIZE || id.z >= NOISE_SIZE) { return; }
  let uvw = (vec3f(id) + vec3f(0.5)) / f32(NOISE_SIZE);
  let r = nz_remap(nz_fbm(uvw, 4, 4));
  let g = nz_remap(nz_fbm(uvw, 8, 2));
  let b = nz_worley(uvw, 8);
  textureStore(noiseOut, vec3<i32>(id), vec4f(r, g, b, 1.0));
}
