# Thomas Duong — portfolio

A black, cinematic landing page: **THOMAS DUONG** as raymarched glass on a polished floor, lit by an
orbiting key light (volumetric god rays through the letters), a pointer light you steer, GPU dust
motes, and chrome marbles you drop with a click (Rapier physics).

```bash
npm install
npm run dev        # http://localhost:5173
npm run build && npm run preview
```

URL flags: `?bench` (HUD: fps, per-pass GPU ms, p50/95/99, dropped frames, input latency, quality level),
`?bench=auto` (sweeps quality levels 0–4, logs a JSON report), `?bench=sustain` (10-min thermal run),
`?backend=webgl2` (force the fallback renderer).

## Architecture

- **Render worker** (`src/engine/worker.ts` → `host.ts`): WebGPU on an `OffscreenCanvas`, isolated from the DOM; main-thread host when OffscreenCanvas is unavailable.
- **Passes** (one command buffer per frame): particles (compute) → froxel volumetrics (inject + temporal reprojection, integrate) → scene raymarch (+ particle draw) → TAA upsample → bloom → composite (fog apply, AgX, grain).
- **Text** is a 2D SDF baked on the GPU with jump flooding, extruded and beveled analytically, so no meshes are needed.
- **Adaptive quality** (`quality.ts`): 5 presets driven by GPU timestamp queries and the measured refresh rate (internal resolution, step counts, froxel grid, particle count).
- **Physics**: Rapier (Rust→WASM), fixed 120 Hz with interpolation, CCD on the marbles. It's lazy-loaded so the ~2 MB WASM doesn't delay the first frame.
- **Fallbacks**: single-pass WebGL2 renderer, then a pure-CSS title.

Design decisions and the module contracts are in [`ENGINE_SPEC.md`](ENGINE_SPEC.md). Validate shaders with
`scripts/check-wgsl.sh` (needs `cargo install naga-cli`).
