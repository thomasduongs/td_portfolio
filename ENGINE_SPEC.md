# v3 — Laser optics (CURRENT; supersedes v2 and v1 below)

The name is **outlined by crisp laser beams**. No haze, no fog, no particles, no marbles, no TAA.
White lasers enter from the page edges, refract through **glass prisms** (Snell's law; a faint
dispersion fan), bounce off **router mirrors** into each letter, and circulate around each glyph outline
via a tiny **mirror at every outline vertex** (normal = bisector, so the reflection lands exactly on the
next edge). When a loop closes, reflection off its entry splitter determines the exit ray; a router
mirror placed on that ray sends the beam to the next glyph. Each word's chain ends through a **glass box**
(parallel faces → lateral shift) and leaves the frame. All routing is computed exactly on the CPU.

Contracts (lead-owned): `shaders/frame.wgsl` (v3, 304 B + `Segment`/`Mirror`/`Glass` structs),
`frame.ts`, `types.ts` (QualitySettings v3, `Size {outW,outH}`, `FrameTargets {hdr,hdrCopy}`,
`EngineCtx.optics`), `optics/types.ts` (scene types + `packOptics`), `optics/route.ts` (lead), `host.ts` (lead).

Pass order (one command buffer): **LaserPass** (clear `hdr`; mirrors, beams, floor reflection, glints)
→ copy `hdr`→`hdrCopy` (host) → **GlassPass** (load `hdr`; glass objects refract `hdrCopy`) → **Bloom**
(reads `hdr`) → **Composite** (reads `hdr` + bloom). Everything at output resolution.

## v3 tasks

### contours.ts (src/engine/optics/contours.ts) — `export function extractGlyphs(r: TextRaster): Glyph[]`
- Input: `TextRaster` (mask rows bottom→top; pixel (c,row) at x=(c+.5)/pxPerUnit−halfW, y=(row+.5)/pxPerUnit−halfH).
- Marching squares on the mask (threshold 127, use the full-res mask; linear interpolation is not needed for a binary mask — use cell edge midpoints) → closed polylines. Remove degenerate/tiny loops (perimeter < 0.3).
- Douglas–Peucker simplify, tolerance 0.012 world; then merge vertices closer than 0.045; cap at 28 vertices per loop (raise tolerance until it fits). Keep corners sharp (DP does).
- Offset each vertex outward by 0.025 along the averaged edge normal (outer loops grow, holes shrink) so beams sit just outside the ink.
- Orientation: outer loops CCW, holes CW (signed area). Classify holes = loops contained in another loop (point-in-polygon of first vertex).
- Group into glyphs: each outer loop + the holes inside it. Order glyphs by line (top line first; line = round of loop centre y relative to line spacing 1.45 — simpler: sort by maxY descending in bands of 0.7, then minX ascending). Within a glyph: outer first, then holes ordered by minX.
- Rotate each loop so `pts[0]` is the topmost vertex (max y; ties → min x).
- `line` = 0-based line index. Fill min/max.
- Node-testable: export also `simplifyDP(pts, tol)`; add a small self-check script `scripts/test-contours.mjs` is NOT required.

### laser pass (src/engine/passes/lasers.ts + shaders/lasers.wgsl) — `class LaserPass implements Pass` name `'lasers'`
- Group 1: segments (read-only storage `array<Segment>`), mirrors (`array<Mirror>`), optional nothing else.
- **Beams**: instanced quad per segment, `draw(6, nSeg * (reflections ? 2 : 1))`. Instances ≥ nSeg are the floor reflection: mirror world y about `F.beamStyle.w` (y' = 2·floorY − y), gain `F.counts.w`, and fade with distance below the floor (`exp(-(floorY − y')·1.2)`), and discard fragments whose world y' > floorY (only below the floor line). VS: project a and b to clip; if either is behind the camera clip it; expand in **screen space** perpendicular to the segment by `halfWidth = glowWidthPx·glowScale + 2` px (pass endpoints in px to FS). Round caps: extend along the segment by halfWidth too.
- FS: d = pixel distance to the 2D segment (px). s = arc length at the closest point (lerp sA→sB). Intensity
  `core = exp(-(d/coreWidthPx)^2 * 1.0)`, `glow = glowGain * exp(-d/(glowWidthPx·glowScale))`, `I = core*2.2 + glow`.
  Reveal: `if s > reveal: discard`; head flare: `+ 6·exp(-((reveal−s)/0.08)^2)` while reveal < pathLength(+1).
  Pulse: if `misc.w ≥ 0`: `I *= 1 + 5·exp(-((s − misc.w)/0.25)^2)`.
  Shimmer near pointer: `I *= 1 + 0.6·pointer.w·exp(-dist2D_world(pointer, closest point)/0.25)` (use world-space closest point).
  Kind 1 (inside glass): ×0.55. Kind 2 (dispersion): × (1 − t_along)², colour as given.
  Out: `vec4(color·I, 0)`, additive blend (one, one). No depth.
- **Mirrors**: instanced oriented quad per mirror (drawn BEFORE beams, alpha blend), plane spanned by two tangents of `normal`, size `halfSize`. Shade: very dark glass-metal body `vec3(0.02)` alpha 0.9, plus a thin bright rim (edge 8%) and a specular term toward the camera from the arriving beam colour. Mirrors only appear once revealed (`reveal ≥ sArrive − 0.2`, fade in).
- **Glints**: instanced screen-aligned star sprite at each mirror centre (additive): 4-point star (two thin gaussians along x/y in px + small core), size 10–18 px scaled by `halfSize`, colour = mirror colour × 1.5, only when revealed; twinkle `0.8 + 0.2·sin(t·3 + i)`; pulse boost like beams using `sArrive`.
- Sources: host passes nothing; emitters are drawn as a segment's start flare: at segments with sA == 0 add a bright point (handled by head flare logic is fine: add `+ 3·exp(-(s/0.05)^2)`).
- `configure`: bind group from `ctx.optics` buffers; `encode`: render pass on `targets.hdr` (loadOp clear black), 3 draws.

### glass pass (src/engine/passes/glass.ts + shaders/glass.wgsl) — `class GlassPass` name `'glass'`
- Group 1: glass storage `array<Glass>`, `hdrCopy` texture, linear sampler.
- One instanced draw: per glass object a screen-space bounding quad (project the 8 corners of its OBB, take min/max in NDC → 6 vertices). FS: build view ray (`rayDir`), intersect the analytic object in local space (box: slab test; prism: intersection of 3 side planes of an equilateral triangle with circumradius `ax.w` in local XY + the two Z caps) → entry t, normal; exit point (second intersection) & normal. Discard on miss.
- Shading: Fresnel (Schlick, F0 0.04). Refraction: approximate screen-space — refracted direction through entry, then exit (Snell, IOR 1.5; TIR → reflect); project `exitPoint + outDir·1.5` to screen uv and sample `hdrCopy` there; dispersion when `q.glassDispersion`: IOR (1.47,1.50,1.53) → 3 samples, one per channel. Transmission tint vec3(0.92,0.96,1.0). Reflection: faint env `vec3(0.01)` + the brightest nearby beam is not needed.
  Edges: bright rim where the two nearest faces meet: `edge = 1 − smoothstep(0, 0.012, minDistToEdge)` → `+ vec3(0.6,0.7,0.8)·edge·0.6` (crisp glass edges catch light). Body: very slight darkening 0.85.
  Out: `vec4(col, 1)` with normal (replace) blending — the object fully re-renders its pixels.
- Load `targets.hdr` (loadOp load).

### cleanup (low tier)
- **quality.ts** presets v3 (fields: bloomMips, reflections, glowScale, glassDispersion): L0 {3,false,0.6,false} L1 {4,true,0.8,false} L2 {5,true,1.0,true} L3 {6,true,1.0,true} L4 {6,true,1.2,true}. Start: desktop 3, laptop 2, mobile 1; max mobile 3. Controller logic unchanged.
- **profiler.ts** `snapshot`: `renderScale: 1`. **bench.ts**: compile-clean.
- **bloom.ts/composite.ts**: read `targets.hdr` instead of `taa[n&1]` (single bind group, no parity). Composite unchanged otherwise.
- **Delete**: passes/particles.ts, passes/scene.ts, passes/taa.ts, shaders/{particles,particles-render,scene,taa,beams,noise}.wgsl, noise.ts, physics/physics.ts; remove `@dimforge/rapier3d-compat` from package.json (`npm uninstall`). `sdfbake.ts`: keep `rasterizeText` + CPU path; `bakeSDFGPU` and jfa.wgsl may be deleted.
- **main.ts / index.html**: hint text → `move · click to send a pulse of light`.

### fallback/webgl2.ts (mid)
Same optics scene (call `extractGlyphs` + `buildOptics` from route.ts), WebGL2 instanced drawing of beams + glints with additive blending into an RGBA16F (or RGBA8 if no EXT_color_buffer_float) target, simple glass (edge rim + uniform tint, no refraction), simple bloom (one 1/4-res blur) optional, AgX tonemap. Same camera/reveal/pulse logic as host.ts.

# v2 — Light-beam text (SUPERSEDES the glass-letter design below where they conflict)

The name is **not geometry**. Three coloured gobo projectors (R, G, B) shine through haze; each beam
carries the text silhouette as seen through the plane z = 0 (`beams.wgsl::gobo`). Alone, a beam is a
long letter-shaped streak. The haze is a two-step excitation medium: linear scatter per beam + glow
∝ `k2·(pairs) + k3·(R·G·B)`. The three silhouettes coincide only on z = 0 → the triple term is the
letters, formed purely by intersecting light. Lamps orbit (streaks swing, letters hold); pointer
agitation and the intro misalign the gobos (`align.xyz`) so the letters split into RGB ghosts and re-form.
Marbles (Rapier) fall **through** the light (no letter colliders) and cut shadow shafts in the beams.

Contracts (lead-owned, done): `shaders/frame.wgsl` (v2 layout, 800 B, `Beam` struct, `hg()` now shared),
`shaders/beams.wgsl` (beam library), `frame.ts` (`O`, `BEAMS_WGSL`, `BEAM_STRIDE`), `types.ts`
(`QualitySettings` v2, `EngineCtx.noiseView`, no froxel target), `host.ts` (rig + uniforms).
The froxel pass is deleted: beams are marched per pixel in the scene pass (letters need per-pixel detail).

## v2 tasks
- **noise.ts + shaders/noise.wgsl** — `export async function bakeNoise(device: GPUDevice): Promise<GPUTexture>`:
  64³ `rgba8unorm` 3D texture (usage STORAGE_BINDING | TEXTURE_BINDING), one compute pass (workgroup 4×4×4),
  **tileable** (period 64) gradient-noise fBm: r = 4-octave fBm, g = 2-octave, b = worley-ish (1 − F1),
  a = 1. Values remapped to [0,1]. Standalone WGSL (no frame.wgsl needed; must still pass check-wgsl.sh,
  which prepends frame.wgsl — so don't redeclare `F`, `PI`, `ign`, `hg`, `rayDir`). Await onSubmittedWorkDone.
- **quality.ts** presets (fields per `QualitySettings` v2):

  | L | scale | hazeSteps | letterSteps | reflSteps | particles | bloomMips |
  |---|---|---|---|---|---|---|
  | 0 | 0.50 | 16 | 16 | 0 | 8192 | 4 |
  | 1 | 0.60 | 20 | 20 | 8 | 16384 | 5 |
  | 2 | 0.70 | 24 | 28 | 10 | 32768 | 5 |
  | 3 | 0.80 | 32 | 36 | 12 | 65536 | 6 |
  | 4 | 1.00 | 40 | 48 | 16 | 131072 | 6 |
  taa true everywhere. Controller logic unchanged. Fix bench.ts if it referenced removed fields.
- **composite** — remove the froxel fog apply and its binding (fog is now inside the scene colour). Keep bloom, exposure, AgX, vignette, grain.
- **physics.ts** — no letter colliders (text is light): `rebuildText` keeps only ground + walls (walls use glyphHalfW). Marbles spawn above the pointer x, fall through the beams.
- **particles** — compile `FRAME_WGSL + BEAMS_WGSL + src`; declare `sdfTex`/`samp` (rename yours), delete local `hg`/`sdf2`-like duplicates. Radiance becomes **rgb** (`array<vec4f>`): `b = beams(point, view)`; `rgb = b.scatter * 0.35 + EXCITE_COLOR * b.excite * 0.08 + pointer light`. Motes inside the letters glow white. Particle SDF collision removed (letters are not solid). Spawn box: x ±(halfW+1), y [floorY, floorY+3.5], z ±3 (keep soft edges + age fade). Render shader uses rgb.
- **scene (lead)** — per-pixel beam march: coarse haze march over the whole ray (scatter, 3D-noise density, Beer–Lambert), fine march inside the letter slab |z| < beamParams.z (excitation = letters), floor lit by the projections (stretched coloured letters) + glossy reflection of the slab, chrome marbles, lamp emitters. Writes the text-plane depth where letters glow (TAA reprojection).
- **fallback/webgl2.ts** — port the v2 scene (same rig maths as host.ts `writeBeams`/`RIG`, fixed steps 16/20/0, no marbles, no 3D noise: use cheap 3D value noise in-shader or none).

# Prism engine — implementation spec

Portfolio landing page: the words **THOMAS DUONG** as extruded, beveled **glass** standing on a black
polished floor, lit by a slowly orbiting spotlight behind the letters (god rays through the glyph gaps
toward the camera), a pointer-following point light the glass refracts, GPU dust motes catching the
light, chrome marbles (Rapier physics) dropped on click that bounce off the letters and reflect them.

Contracts that are already written and MUST NOT change shape without the lead:
`src/engine/types.ts`, `src/engine/frame.ts`, `src/engine/shaders/frame.wgsl`, `src/engine/math.ts`,
`src/engine/host.ts` (frame loop; read it to see exactly how your module is called).

## 0. Decisions and why (performance-justified)

| Requested feature | Decision | Why |
|---|---|---|
| Geometry | **SDF raymarching** of a 2D text SDF extruded analytically. No meshes. | Glass text needs refraction with interior marching and exact bevels; a 2D SDF texture (one fetch per step, inside an AABB) is cheaper than a mesh + SSR refraction and gives exact normals at any resolution. |
| Forward+ clustered lighting | **Not built.** 3 analytic lights evaluated directly. | Clustering pays off at ≫16 lights; with 3 it is pure overhead (extra compute pass + indirection). |
| GPU culling / HZB / LOD / indirect mesh draws / meshopt / KTX2 | **Not built** (no meshes, no textures to stream). Indirect **dispatch + draw** used for particles so quality changes never re-record. | Nothing to cull: the scene is one fullscreen raymarch bounded by an AABB test. |
| Render bundles | Not used. | Each pass is 1 draw; bundles save nothing. |
| Reflections | Floor: **raymarched planar reflection** (mirror ray, reduced steps). Glass/chrome: analytic environment + one bounce into the SDF. | SSR has holes at screen edges; mirror-marching the SDF is exact and cheap within the AABB. |
| Refraction | Interior SDF march (negated SDF), refract at exit, TIR → reflect. Dispersion: 3 IORs at Q≥3, else 1 march + per-channel exit-dir offset. | Physically correct look per ms. |
| Volumetrics | **Froxel grid** (inject+temporal → integrate), SDF soft-shadow visibility, Beer–Lambert, HG phase, IGN blue-noise jitter, reprojection w/ rejection; applied in composite at pixel depth (trilinear = depth-aware upsample). | Standard AAA approach, cost independent of resolution. |
| Particles | GPU-resident SoA storage buffers, compute update (curl noise, SDF collision, pointer repulsion), lit in compute, instanced quads. | Zero CPU traffic. |
| Physics | **Rapier** (Rust→WASM via `@dimforge/rapier3d-compat`) in the render worker, fixed 120 Hz + interpolation, CCD on marbles. Letter colliders generated from the text mask (run-merged cuboids). | Rapier is the requested Rust/WASM engine; the npm build avoids requiring a wasm32 toolchain. Same thread as render = zero-copy hand-off (marble array written directly into the frame UBO). |
| Cloth / fluids / XPBD | **Not built.** | Not needed for this scene; would cost GPU time with no perceived gain. |
| Temporal | Jittered scene at internal res (renderScale 0.5–1.0) → TAA upsample to output res, YCoCg neighbourhood clamp. | Biggest single perf lever: raymarch cost ∝ internal pixels. |
| Rust/wgpu vs TS submission benchmark | Not built; TS submission is ~6 passes/frame, CPU encode cost is ≪0.2 ms, so the comparison can't move the needle. | Measured CPU ms shown in HUD. |

## 1. Files and owners (see .duet/plan.json)

```
src/main.ts                 main thread: feature detect, worker, input forwarding, resize, HUD, CSS fallback
src/style.css
src/ui/hud.ts               ?bench HUD overlay
src/engine/worker.ts        3 lines: createHost(postMessage)
src/engine/host.ts          [lead, done] frame loop
src/engine/device.ts        initDevice
src/engine/resources.ts     ResourcePool
src/engine/profiler.ts      GpuProfiler, FrameStats
src/engine/quality.ts       QualityController + presets
src/engine/bench.ts         BenchRunner
src/engine/sdfbake.ts       rasterizeText, bakeSDFGPU, bakeSDFCPU  (+ shaders/jfa.wgsl)
src/engine/passes/*.ts      one class per pass (+ shaders/<name>.wgsl)
src/physics/physics.ts      Physics (Rapier)
src/fallback/webgl2.ts      WebGL2Engine
```

All WGSL modules are compiled as `FRAME_WGSL + source` (import with `?raw`), bind `@group(0)` = frame UBO
via `ctx.frameBG`, pass resources at `@group(1)`. Use **explicit** pipeline layouts
`[ctx.frameBGL, ownBGL]` and `device.create{Render,Compute}PipelineAsync` inside `init()`.
Bind groups are created in `configure()` only. `encode()` must not create GPU objects (except
views passed in by the host). Fullscreen passes draw a single triangle: `draw(3)` with vertex index trick.

Coordinate conventions: world Y up, camera looks toward −Z at the text at origin, text in plane
z∈[−halfDepth, halfDepth]. Clip z in [0,1], depth clear 1.0, compare `less`. Screen uv top-left origin.

## 2. Module specs

### device.ts
`export async function initDevice(canvas: OffscreenCanvas|HTMLCanvasElement, tier: DeviceTier): Promise<{device: GPUDevice; context: GPUCanvasContext; caps: Caps} | null>`
- `navigator.gpu` missing → null. `requestAdapter({powerPreference: tier==='mobile' ? 'low-power' : 'high-performance'})`, null → null.
- Request optional features if present: `timestamp-query`, `shader-f16`, `rg11b10ufloat-renderable`, `float32-filterable`, `subgroups`.
- requiredLimits: none beyond defaults (portable).
- `context = canvas.getContext('webgpu')`, configure `{device, format: navigator.gpu.getPreferredCanvasFormat(), alphaMode: 'opaque'}`.
- caps: hdrFormat `'rgba16float'`, bloomFormat `rg11b10ufloat` iff feature, depthFormat `'depth32float'`, tier passes through (may downgrade `desktop`→`laptop` if `adapter.info?.type`/description suggests integrated — optional).
- `device.lost.then(info => console.error(...))`; `device.addEventListener('uncapturederror', e => console.error(e.error.message))`.

### resources.ts
`export class ResourcePool implements ResourcePoolLike` — Map<key,{obj, descHash}>. `texture(key, desc)`: hash = JSON of size/format/usage/dimension/mipLevelCount; same → return cached; else destroy old, create new with `label: key`. Same for `buffer`.

### profiler.ts
`export class GpuProfiler { constructor(device, caps, passNames: string[]); ts(name): TS; resolve(enc): void; afterSubmit(): void; gpuMs: number /* -1 if unsupported */; passMs: Record<string, number> }`
- QuerySet type 'timestamp', count = 2×passes. `ts(name)` returns `{querySet, beginningOfPassWriteIndex: 2i, endOfPassWriteIndex: 2i+1}` or undefined.
- `resolve(enc)`: `resolveQuerySet` → resolve buffer, then `copyBufferToBuffer` into a free readback buffer from a **ring of 3** (MAP_READ|COPY_DST). If none free, skip this frame's resolve (never stall).
- `afterSubmit()`: `mapAsync(READ)` the buffer just written; on resolve read BigInt64Array, per-pass ms = (end−begin)/1e6, EMA (α=0.1) into passMs; gpuMs = sum; unmap; return to ring.
- Timestamps may be 0/garbage on some drivers: ignore negative or > 1e9 ns deltas.

`export class FrameStats { refreshHz: number; lastInterval: number; interval(ms); cpu(ms); inputLatency(ms); snapshot(prof, q, backend): Stats }`
- Ring of last 240 intervals. refreshHz estimated from median interval over the ring snapped to the nearest of [60, 75, 90, 120, 144, 165, 240]. p50/p95/p99 via sorting a copy (only in snapshot, 4 Hz). dropped = count of intervals > 1.5×(1000/refreshHz). fps = 1000/mean.

### quality.ts
Presets (index = level):

| L | scale | primary | interior | refl | shadow | disp | froxel | particles | bloomMips | taa |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.50 | 48 | 12 | 0 | 6 | no | 80×45×32 | 8192 | 4 | yes |
| 1 | 0.60 | 64 | 16 | 24 | 8 | no | 96×54×48 | 16384 | 5 | yes |
| 2 | 0.75 | 80 | 20 | 32 | 10 | no | 128×72×64 | 32768 | 5 | yes |
| 3 | 0.85 | 96 | 24 | 40 | 12 | yes | 160×90×64 | 65536 | 6 | yes |
| 4 | 1.00 | 128 | 32 | 48 | 16 | yes | 160×90×96 | 131072 | 6 | yes |

`export class QualityController { constructor(tier); settings: QualitySettings; update(intervalMs, gpuMs, refreshHz): boolean; force(level): void; static MAX_PARTICLES = 131072 }`
- Start level: desktop 3, laptop 2, mobile 1. Max: mobile 2, others 4.
- Budget = 1000/refreshHz. Cost metric = gpuMs if ≥0 else intervalMs. Keep a 1-second window (by frame count ≈ refreshHz). Every window end: p90 of metric. If p90 > 0.92×budget for 2 consecutive windows → level−1. If p90 < 0.65×budget for 4 consecutive windows and level<max → level+1. After any change, ignore 1 window (settling). When gpuMs<0 (no timestamps), interval is vsync-quantized: only step up if p90 < 1.05×budget AND the level has been stable for 6 windows; step down when p90 > 1.3×budget.
- Return true when settings changed.

### bench.ts
`export class BenchRunner { constructor(mode: string, qc: QualityController, report: (r: unknown) => void); onFrame(now, stats: FrameStats, prof: GpuProfiler): boolean }`
- mode `'1'` or `''`: HUD-only — return `qc.update(stats.lastInterval, prof.gpuMs, stats.refreshHz)`.
- mode `'auto'`: for each level 0..4: `qc.force(level)` (return true that frame), warm 1.5 s, record 4 s of intervals + gpuMs + passMs; then report `{levels:[{level, p50,p95,p99, gpuMs, passes, dropped}], ua, refreshHz}` and console.table it, then resume adaptive.
- mode `'sustain'`: adaptive on; every 10 s push `{t, fps, p95, level, gpuMs}`; after 10 min report. (Thermal throttling shows as level drop/fps decay over time.)

### sdfbake.ts + shaders/jfa.wgsl
```ts
export interface TextRaster { lines: string[]; w: number; h: number; mask: Uint8Array; // 1 byte/px, 255 inside
  pxPerUnit: number; halfW: number; halfH: number;   // texture rect in world units, centered at origin
  glyphHalfW: number; glyphHalfH: number;            // tight glyph extents (world)
  floorY: number; }                                  // world Y of the lowest baseline
export function rasterizeText(lines: string[]): TextRaster
export async function bakeSDFGPU(device: GPUDevice, r: TextRaster): Promise<GPUTexture> // rgba16float, TEXTURE_BINDING
export function bakeSDFCPU(r: TextRaster): Float32Array // w*h, world units, for the WebGL2 fallback
```
- Raster: `OffscreenCanvas` (exists in worker + main). Font: `800 ${fontPx}px "Helvetica Neue", Helvetica, Arial, sans-serif`, letter-spacing ≈0.06 em (use `ctx.letterSpacing` if supported, else manual per-char advance). pxPerUnit = cap height in px (measure with `actualBoundingBoxAscent` of "H"): **cap height = 1.0 world unit**. Target texture width 2048 (1 line) / 1536 (2 lines); choose fontPx so text fits with padding 0.3 units each side. 2-line layout: line spacing 1.45 units; block centered vertically on origin. 1-line: cap centered on y=0 → baseline at y=−0.5 → floorY = −0.5. Canvas y down → world y up (flip). Mask = alpha>127.
- GPU JFA: storage buffers of `vec2<f32>` seed coords (−1 = none). Two runs (seeds = inside pixels → distance for outside; seeds = outside pixels → distance for inside), or one run with packed seeds per pass; step sizes w/2… 1 plus a final step 1 (JFA+1). Workgroup 8×8. Final pass writes `rgba16float` storage texture: `r = (dOut − dIn)/pxPerUnit` (signed, negative inside, world units) using pixel-center distances (subtract 0.5 px at the boundary so the zero crossing is on the edge). Then `device.queue.onSubmittedWorkDone()` before resolving.
- CPU: Felzenszwalb 1-D EDT squared distance, twice (inside/outside), same output convention.

### passes/scene.ts + shaders/scene.wgsl  — **lead owns**
Fullscreen raymarch → `sceneColor` (clear) + `sceneDepth` (frag_depth). Group 1: sdf texture + sampler.

### passes/particles.ts + shaders/particles.wgsl
- Buffers (allocated once at `QualityController.MAX_PARTICLES`): `pos: array<vec4f>` (xyz, age), `vel: array<vec4f>` (xyz, lifespan), `light: array<f32>` (incident radiance scalar, computed in compute). SoA → the render VS reads only pos+light (20 B/particle).
- Indirect buffer `[dispatchX,1,1]` and draw-indirect `[6, count, 0, 0]` written with `queue.writeBuffer` in `configure()` only.
- Init: compute pass at init writes random positions in box x∈[−halfW−1, halfW+1], y∈[floorY, floorY+3.5], z∈[−3, 3] (hash of index, PCG).
- Update (workgroup 256): vel += curlNoise(pos*0.35 + t*0.05)*0.08·dt − vel·0.6·dt + tiny gravity −0.005; pointer repulsion within 0.8 units (strength ∝ F.pointer.w); SDF collision: if |p.z|<halfDepth+0.05 and 2D SDF (sample at p.xy) < 0.02 → push out along SDF gradient. Respawn when age>lifespan or out of box. Light = spot cone attenuation × HG(g=0.5) toward camera × SDF soft shadow toward spot (F.quality.w steps, bounded to text AABB, early-out if segment misses AABB) + pointer light (inverse-square, clamp).
- Render: render pass loading `sceneColor` + `sceneDepth` (depth test `less`, depthWriteEnabled false), additive blend (one, one). VS: billboard quad (6 verts/instance), size 0.012 world units scaled by distance clamp to ≥1.2 px; FS: gaussian falloff × light × vec3(1.0,0.95,0.9) × 0.6 . Pass name `'particles'`.
- **Order**: `encode()` records ONLY the compute update (gets the `ts`). The draw is in `encodeDraw(enc: GPUCommandEncoder)`, which ScenePass calls right after its own render pass via `scene.setOverlay(particles)` (already wired in host.ts). ScenePass API: `setOverlay(o: { encodeDraw(enc: GPUCommandEncoder): void })`.

### passes/volumetric.ts + shaders/froxel.wgsl
- Textures: `scatter[2]` (ping-pong rgba16float 3D, own, pool keys `froxelA/B`), `ctx targets.froxelIntegrated`.
- Froxel (x,y,z) → uv=(x+.5)/X,(y+.5)/Y; slice distance exponential: `d(z) = near*(far/near)^(z/Z)`, near 0.3, far F.froxel.w; jitter z by `ignT(xy)` (blue-noise-ish) → `d((z+jitter)/Z)`. World pos = camPos + rayDir(uv)*d (rayDir uses jittered invVP — fine).
- Inject (wg 8×8×1 over x,y, loop z inside? No: dispatch X/4,Y/4,Z/4 with wg 4×4×4): density σ = F.misc.z × (0.6 + 0.4·noise3(p·0.5 + t·0.03)) × exp(−max(p.y−floorY,0)·0.25). In-scatter L = spot: cone smoothstep(cosOuter,cosInner) × inverse-square (min dist 1) × HG(g=0.55, viewDir·lightDir) × visibility(SDF soft shadow, F.quality.w steps, only if segment hits text AABB) × spotColor + pointer: point light inverse-square × HG(g=0.3) (no shadow) × vec3(0.55,0.75,1.0)×F.pointer.w. Store (σ·L, σ).
- Temporal: reproject world pos with F.prevViewProj → prev uv & prev distance → prev froxel coord; if inside [0,1]³ blend `mix(cur, prev, 0.9)` else cur. Read prev from scatter[(n+1)&1], write scatter[n&1] (needs trilinear sample of prev → bind prev as texture_3d<f32> + sampler, write cur as storage).
- Integrate (wg 8×8, loop z front→back): `T` starts 1, accum 0; per slice with thickness Δd: `Tslice = exp(−σΔd)`; energy-conserving `S = (L − L·Tslice)/max(σ,1e-5)`; accum += T·S; T *= Tslice; write (accum, T) to froxelIntegrated[x,y,z].
- Inject and integrate are **two compute passes** (storage write → sampled read needs a pass boundary). The `ts` goes on the inject pass only.

### passes/taa.ts + shaders/taa.wgsl
- Fragment fullscreen pass at output res writing `targets.taa[n&1]`. Inputs: sceneColor (internal res, linear sampler), sceneDepth (`texture_depth_2d`, textureLoad nearest), history `targets.taa[(n+1)&1]`. Two bind groups prebuilt (parity).
- Current sample: sceneColor at uv **minus jitter offset** (uv − F.jitter.xy·0.5·(1,−1)), Catmull-Rom/bicubic 5-tap optional, 3×3 neighbourhood min/max in YCoCg from internal-res texels around the uv.
- Reprojection: world pos from depth via invViewProj (depth 1 → use point at far distance along ray) → prev uv via F.prevViewProj. History sampled bilinear, clipped to neighbourhood AABB (clip toward center, not clamp). Blend α = 0.1 (0.2 if internal==output res… no, fixed 0.1); α=1 if prev uv off-screen or frame 0. Luma-weighted (1/(1+luma)) blend to stop firefly flicker.
- If `q.taa` is false just upsample current.

### passes/bloom.ts + shaders/bloom.wgsl
- Mip chain texture (pool key `bloom`), `caps.bloomFormat`, size = half of output, `bloomMips` levels. Downsample 13-tap (CoD:AW, Jimenez 2014) from `taa[n&1]` into mip0, then mip i→i+1. First downsample uses Karis average to kill fireflies. Upsample 3×3 tent with additive blend (`one, one`) mip i+1 → mip i. All fragment passes; per-mip views + bind groups prebuilt in configure (×2 parity for the source). TS: on the first downsample pass begin… — attach `ts` to the first render pass only.

### passes/composite.ts + shaders/composite.wgsl
`constructor(ctx, context: GPUCanvasContext)`, `setOutput(view)`.
- Inputs: taa[n&1], bloom mip0, sceneDepth (internal res; textureLoad at `uv*inRes`), froxelIntegrated (sample at uv, slice coordinate from linear distance: `z = log(d/near)/log(far/near)`), linear sampler.
- color = hdr; distance d from depth (reconstruct world pos with invViewProj; sky → far). Fog: `color = color*fog.a + fog.rgb`. Bloom: `mix(color, bloom, F.misc.y)`. Exposure F.misc.x. Tonemap: **AgX** (base, with slight punchy look). Vignette 0.25. Grain: (ignT(px)−0.5)·0.015 in display space + dither 1/255. Write to swapchain format (if format is `*-srgb` skip manual gamma; preferred canvas formats are non-srgb → apply sRGB OETF).

### physics/physics.ts
```ts
export class Physics {
  static async create(r: TextRaster): Promise<Physics>;    // await RAPIER.init()
  rebuildText(r: TextRaster): void;                        // remove/re-add letter colliders
  spawn(worldOnZ0: number[], eye: number[]): void;         // drop a marble
  step(dt: number, ray: { light: number[]; pusher: number[]; active: boolean }): void;
  writeMarbles(f32: Float32Array, offset: number, alpha: number): number; // write xyz,r ×count (interpolated), return count
}
```
- World gravity −9.81 (scaled ×0.6 for slower, floatier look → −6). Fixed ground: cuboid top at floorY, half extents (40, 0.5, 40). Invisible walls: back z=−1.6, front z=2.2, sides x=±(glyphHalfW+1.8), ceiling none.
- Letter colliders: downsample mask to cells of 0.06 world units; a cell is solid if >50% inside. Merge horizontal runs per row, then merge vertically identical runs → cuboids (half depth 0.16, minus a 0.02 margin). Restitution 0.35, friction 0.4.
- Marbles: max 16 (recycle oldest), radius 0.16, dynamic ball, `setCcdEnabled(true)`, restitution 0.55, friction 0.3, density 3, linear damping 0.05, angular damping 0.2. spawn at (x clamped to ±glyphHalfW, floorY + 2.6 + small random, z random ±0.1), small random velocity.
- Pointer pusher: kinematic position-based ball r=0.35 moved to `ray.pusher` each step via `setNextKinematicTranslation` when `active`, else parked at y=−50.
- Interpolation: keep prev+cur translation per marble (copied before each step), write `prev + (cur−prev)·alpha`.

### fallback/webgl2.ts
`export class WebGL2Engine { static create(canvas, init: InitMsg, post): Promise<WebGL2Engine|null>; handle(msg: MainToWorker): void }`
- `getContext('webgl2', {antialias:false, alpha:false, powerPreference:'high-performance'})`; null → return null.
- Text via `rasterizeText` + `bakeSDFCPU` → R16F texture (LINEAR filter; `EXT_color_buffer_float` not required for sampling).
- One fullscreen fragment shader (GLSL ES 3.00) porting scene.wgsl's essentials at fixed steps (primary 64, interior 16, floor reflection 24, shadow 8, no dispersion), analytic volumetric (16 jittered samples along the view ray with SDF shadow 4 steps), AgX + grain inline. Render to canvas at 0.6× resolution (canvas width = css·dpr·0.6, CSS stretches it). Own rAF loop, same camera math as host.ts (copy). Post `stats` at 4 Hz with backend 'webgl2' (gpuMs −1).

### main.ts / style.css / ui/hud.ts / worker.ts
- `worker.ts`: `import { createHost } from './host'; const h = createHost((m) => postMessage(m)); onmessage = (e) => h(e.data);`
- main.ts: tier detect (`matchMedia('(pointer:coarse)')` or `navigator.userAgentData?.mobile` → mobile; `navigator.hardwareConcurrency<=8 && !mobile` → laptop... keep simple: mobile / desktop, laptop if `deviceMemory<=8`). bench = URLSearchParams `bench`. If `'transferControlToOffscreen' in canvas`: worker `new Worker(new URL('./engine/worker.ts', import.meta.url), {type:'module'})`, post init with transfer. Else dynamic-import `./engine/host` and run on main thread with the canvas. Pointer: `pointermove`/`pointerdown`/`pointerup` on window, passive, normalized to canvas rect, `t = performance.timeOrigin + e.timeStamp`; use `getCoalescedEvents` not needed — send the latest only, throttled to one message per rAF on main thread? NO: send immediately (lowest latency; the worker reads the latest at frame start). Touch: `touch-action: none` on canvas. ResizeObserver on canvas → resize msg (css px, dpr). `visibilitychange` → visibility msg.
- On `ready`: add class `ready` to body (fades canvas in, fades hint after 4 s). On `error` 'nogpu' or init failure or no Worker: add class `nogpu` to body → CSS fallback.
- **Title** `<h1 id="title">` is visually hidden (sr-only) when `ready`; in `nogpu` mode it's shown, large, centered, with a CSS animated light sweep (background-clip:text gradient + text-shadow glow) on black — the lightweight visual fallback.
- style.css: html/body black, no scroll, canvas fixed inset 0 100vw/100vh, `.hint` bottom-center tiny uppercase letter-spaced #8a8a8a fading.
- hud.ts: `export function mountHud(): (s: Stats) => void` — fixed top-left monospace panel (fps, cpu, gpu, per-pass, p50/95/99, dropped, latency, level, scale, refresh, backend). On `bench-report` main logs + offers a JSON download link.
