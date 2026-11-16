# Thomas Duong — portfolio

The name is drawn by crisp laser beams. Two white lasers enter from the page edges, bend through glass
prisms (with a thin rainbow from dispersion), and bounce off routing mirrors into the letters. A tiny
mirror at every corner of each letter outline reflects the beam exactly onto the next edge, so each
glyph is a closed loop of real reflections. Each word's beam finally passes through a tilted glass
block and leaves the frame. All of it is computed exactly (reflection and Snell's law) on the CPU.
On load the lasers trace the name letter by letter. Click to send a pulse of light through the
whole system, and move the pointer for parallax.

```bash
npm install
npm run dev        # http://localhost:5173
npm run build && npm run preview
```

URL flags: `?bench` (HUD: fps, per-pass GPU ms, p50/95/99, dropped frames, input latency, quality level),
`?bench=auto` (sweeps quality levels 0–4, logs a JSON report), `?bench=sustain` (10-min thermal run),
`?backend=webgl2` (force the fallback renderer), `?q=0..4` (pin a quality level).

## Architecture

- **Render worker** (`src/engine/worker.ts` → `host.ts`): WebGPU on an `OffscreenCanvas`, isolated from the DOM.
- **Optics** (`src/engine/optics/`): `contours.ts` traces glyph outlines from the rasterized text (marching squares + Douglas–Peucker); `route.ts` solves the installation (prism yaw and router positions are searched so every hop stays tidy and in frame).
- **Passes** (one command buffer, all at output resolution): lasers (anti-aliased beam quads, mirror chips, glints, floor reflection) → glass (analytic prism/box with screen-space refraction and dispersion) → bloom → composite (AgX).
- **Adaptive quality** (`quality.ts`): bloom depth, floor reflection, glow width, glass dispersion, driven by GPU timestamps.
- **Fallbacks**: WebGL2 renderer with the same optics scene, then a pure-CSS title.

Design decisions and the module contracts are in [`ENGINE_SPEC.md`](ENGINE_SPEC.md). Validate shaders with
`scripts/check-wgsl.sh` (needs `cargo install naga-cli`).
