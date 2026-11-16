// Rapier physics (WASM) for the marbles and the letter colliders. Runs on the render thread at a fixed 120 Hz step.
import * as RAPIER from '@dimforge/rapier3d-compat';
import type { TextRaster } from '../engine/sdfbake';

const GRAVITY_Y = -6;            // -9.81 scaled down for a floatier look
const MAX_MARBLES = 16;
const MARBLE_R = 0.16;
const POINTER_R = 0.35;
const CELL = 0.06;               // letter collider grid cell (world units)
const COLLIDER_HZ = 0.16;        // letter colliders' half depth (z)
const MARGIN = 0.01;             // shrink per half-extent so neighbouring cuboids don't catch
const PARK_Y = -50;
const WALL_THICK = 0.5;

interface Marble {
  body: RAPIER.RigidBody;
  prev: [number, number, number];
  cur: [number, number, number];
}

export class Physics {
  private world: RAPIER.World;
  private statics: RAPIER.RigidBody | null = null;
  private pusher: RAPIER.RigidBody;
  private marbles: Marble[] = [];
  private r: TextRaster;

  private constructor(r: TextRaster) {
    this.r = r;
    this.world = new RAPIER.World({ x: 0, y: GRAVITY_Y, z: 0 });
    this.pusher = this.world.createRigidBody(
      RAPIER.RigidBodyDesc.kinematicPositionBased().setTranslation(0, PARK_Y, 0),
    );
    this.world.createCollider(RAPIER.ColliderDesc.ball(POINTER_R).setRestitution(0.55).setFriction(0.3), this.pusher);
    this.rebuildStatics();
  }

  static async create(r: TextRaster): Promise<Physics> {
    await RAPIER.init();
    return new Physics(r);
  }

  /** Replace the letter colliders (and walls, which depend on the glyph width) after a re-bake. */
  rebuildText(r: TextRaster): void {
    this.r = r;
    this.rebuildStatics();
  }

  /** Drop a marble at the clicked point (x on the z=0 plane). `_eye` is reserved for camera-relative spawn bias. */
  spawn(worldOnZ0: number[], _eye: number[]): void {
    const r = this.r;
    const x = clamp(worldOnZ0[0], -r.glyphHalfW, r.glyphHalfW);
    const y = r.floorY + 2.6 + Math.random() * 0.2;
    const z = (Math.random() - 0.5) * 0.2;
    const vx = (Math.random() - 0.5) * 0.4, vz = (Math.random() - 0.5) * 0.2;

    if (this.marbles.length >= MAX_MARBLES) {
      const old = this.marbles.shift();
      if (old) this.world.removeRigidBody(old.body);
    }
    const body = this.world.createRigidBody(
      RAPIER.RigidBodyDesc.dynamic()
        .setTranslation(x, y, z)
        .setLinvel(vx, 0, vz)
        .setCcdEnabled(true)
        .setLinearDamping(0.05)
        .setAngularDamping(0.2),
    );
    this.world.createCollider(
      RAPIER.ColliderDesc.ball(MARBLE_R).setDensity(3).setRestitution(0.55).setFriction(0.3),
      body,
    );
    this.marbles.push({ body, prev: [x, y, z], cur: [x, y, z] });
  }

  /** Advance the simulation by one fixed step. Pointer pusher follows `ray.pusher` while active, else is parked. */
  step(dt: number, ray: { light: number[]; pusher: number[]; active: boolean }): void {
    for (const m of this.marbles) m.prev = m.cur;
    if (ray.active) this.pusher.setNextKinematicTranslation({ x: ray.pusher[0], y: ray.pusher[1], z: ray.pusher[2] });
    else this.pusher.setNextKinematicTranslation({ x: 0, y: PARK_Y, z: 0 });
    this.world.timestep = dt;
    this.world.step();
    for (const m of this.marbles) {
      const t = m.body.translation();
      m.cur = [t.x, t.y, t.z];
    }
  }

  /** Writes xyz + radius per marble into the frame UBO (vec4 array), interpolated between the last two steps. */
  writeMarbles(f32: Float32Array, offset: number, alpha: number): number {
    const n = this.marbles.length;
    for (let i = 0; i < n; i++) {
      const m = this.marbles[i];
      const o = offset + i * 4;
      f32[o] = m.prev[0] + (m.cur[0] - m.prev[0]) * alpha;
      f32[o + 1] = m.prev[1] + (m.cur[1] - m.prev[1]) * alpha;
      f32[o + 2] = m.prev[2] + (m.cur[2] - m.prev[2]) * alpha;
      f32[o + 3] = MARBLE_R;
    }
    return n;
  }

  /** Ground, invisible walls and letter colliders in one fixed body. */
  private rebuildStatics(): void {
    const r = this.r;
    if (this.statics) this.world.removeRigidBody(this.statics);
    const body = this.world.createRigidBody(RAPIER.RigidBodyDesc.fixed());
    this.statics = body;
    const add = (hx: number, hy: number, hz: number, x: number, y: number, z: number, restitution: number, friction: number) => {
      this.world.createCollider(
        RAPIER.ColliderDesc.cuboid(hx, hy, hz).setTranslation(x, y, z).setRestitution(restitution).setFriction(friction),
        body,
      );
    };

    // Ground: top surface at floorY.
    add(40, 0.5, 40, 0, r.floorY - 0.5, 0, 0.35, 0.4);

    // Invisible walls: back, front, sides. Tall enough that nothing escapes upward.
    const wallH = 10;
    const sideX = r.glyphHalfW + 1.8;
    add(40, wallH, WALL_THICK, 0, r.floorY + wallH, -1.6 - WALL_THICK, 0.35, 0.4);
    add(40, wallH, WALL_THICK, 0, r.floorY + wallH, 2.2 + WALL_THICK, 0.35, 0.4);
    add(WALL_THICK, wallH, 40, -sideX - WALL_THICK, r.floorY + wallH, 0, 0.35, 0.4);
    add(WALL_THICK, wallH, 40, sideX + WALL_THICK, r.floorY + wallH, 0, 0.35, 0.4);

    // Letters: downsample the mask to CELL-sized cells (solid if >50% inside), then merge runs.
    for (const c of letterCuboids(r)) {
      add(c.hx, c.hy, COLLIDER_HZ - MARGIN, c.cx, c.cy, 0, 0.35, 0.4);
    }
  }
}

interface Cuboid { cx: number; cy: number; hx: number; hy: number }

/**
 * Mask rows are bottom-to-top world Y (row 0 = bottom). Pixel (c, row) lies at
 * x = (c + 0.5) / pxPerUnit - halfW, y = (row + 0.5) / pxPerUnit - halfH.
 */
function letterCuboids(r: TextRaster): Cuboid[] {
  const cols = Math.max(1, Math.ceil((2 * r.halfW) / CELL));
  const rows = Math.max(1, Math.ceil((2 * r.halfH) / CELL));
  const inside = new Uint32Array(cols * rows);
  const total = new Uint32Array(cols * rows);
  const cellPx = r.pxPerUnit * CELL;

  for (let row = 0; row < r.h; row++) {
    const cj = Math.min(rows - 1, Math.floor((row + 0.5) / cellPx));
    for (let c = 0; c < r.w; c++) {
      const ci = Math.min(cols - 1, Math.floor((c + 0.5) / cellPx));
      const k = cj * cols + ci;
      total[k]++;
      if (r.mask[row * r.w + c] > 127) inside[k]++;
    }
  }

  // Per-row runs of solid cells, merged vertically when identical.
  const out: Cuboid[] = [];
  let open = new Map<string, { i0: number; i1: number; j0: number }>();
  for (let j = 0; j < rows; j++) {
    const runs: { i0: number; i1: number }[] = [];
    let start = -1;
    for (let i = 0; i <= cols; i++) {
      const solid = i < cols && total[j * cols + i] > 0 && inside[j * cols + i] * 2 > total[j * cols + i];
      if (solid && start < 0) start = i;
      if (!solid && start >= 0) { runs.push({ i0: start, i1: i - 1 }); start = -1; }
    }
    const next = new Map<string, { i0: number; i1: number; j0: number }>();
    for (const run of runs) {
      const key = run.i0 + ':' + run.i1;
      next.set(key, open.get(key) ?? { i0: run.i0, i1: run.i1, j0: j });
      open.delete(key);
    }
    for (const run of open.values()) out.push(toCuboid(r, run.i0, run.i1, run.j0, j - 1));
    open = next;
  }
  for (const run of open.values()) out.push(toCuboid(r, run.i0, run.i1, run.j0, rows - 1));
  return out;
}

function toCuboid(r: TextRaster, i0: number, i1: number, j0: number, j1: number): Cuboid {
  const x0 = -r.halfW + i0 * CELL, x1 = -r.halfW + (i1 + 1) * CELL;
  const y0 = -r.halfH + j0 * CELL, y1 = -r.halfH + (j1 + 1) * CELL;
  return { cx: (x0 + x1) / 2, cy: (y0 + y1) / 2, hx: Math.max(0.005, (x1 - x0) / 2 - MARGIN), hy: Math.max(0.005, (y1 - y0) / 2 - MARGIN) };
}

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}
