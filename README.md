# Thomas Duong — portfolio

The name is made of light. Three coloured projector beams (red, green, blue) shine through drifting
haze. Each one carries the name's silhouette as a stencil, so on its own each beam is just long
letter-shaped rays. The haze glows where beams overlap, and **THOMAS DUONG** appears only where all
three cross. The lamps orbit slowly, so the rays swing around while the name holds. Move the pointer
fast and the beams knock out of alignment, splitting the name into fragments that re-form. Click to drop
chrome marbles (Rapier physics); they fall through the light and cut shadow shafts through the letters.

```bash
npm install
npm run dev        # http://localhost:5173
npm run build && npm run preview
```

URL flags: `?bench` (HUD: fps, per-pass GPU ms, p50/95/99, dropped frames, input latency, quality level),
`?bench=auto` (sweeps quality levels 0–4, logs a JSON report), `?bench=sustain` (10-min thermal run),
`?backend=webgl2` (force the fallback renderer), `?q=0..4` (pin a quality level).

## Architecture

- **Render worker** (`src/engine/worker.ts` → `host.ts`): WebGPU on an `OffscreenCanvas`, isolated from the DOM; main-thread host when OffscreenCanvas is unavailable.
- **Passes** (one command buffer per frame): particles (compute, lit by the beams) → scene (per-pixel beam march: coarse haze scatter + fine letter-slab excitation, projected floor, marbles) → TAA upsample → bloom → composite (AgX, grain).
- **Beams** (`shaders/beams.wgsl`): gobo projection through the plane z = 0, cone, inverse-square, marble occlusion, and a two-step excitation term `k2·pairs + k3·R·G·B` that forms the letters. The text is a 2D SDF baked on the GPU with jump flooding; there is no letter geometry.
- **Adaptive quality** (`quality.ts`): 5 presets driven by GPU timestamp queries and the measured refresh rate (internal resolution, step counts, froxel grid, particle count).
- **Physics**: Rapier (Rust→WASM), fixed 120 Hz with interpolation, CCD on the marbles. It's lazy-loaded so the ~2 MB WASM doesn't delay the first frame.
- **Fallbacks**: single-pass WebGL2 renderer, then a pure-CSS title.

Design decisions and the module contracts are in [`ENGINE_SPEC.md`](ENGINE_SPEC.md). Validate shaders with
`scripts/check-wgsl.sh` (needs `cargo install naga-cli`).
