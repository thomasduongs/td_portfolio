// Shared engine contracts. Every pass module implements `Pass`.
// Keep this file dependency-free: it is imported by worker, passes, physics and main thread.

/** Formats picked once at device init. */
export interface Caps {
  timestamps: boolean;          // 'timestamp-query' feature enabled
  f16: boolean;                 // 'shader-f16' enabled (WGSL may `enable f16;` only when true)
  rg11b10Renderable: boolean;   // 'rg11b10ufloat-renderable' enabled -> bloom chain uses rg11b10ufloat
  float32Filterable: boolean;
  subgroups: boolean;
  presentFormat: GPUTextureFormat;
  hdrFormat: GPUTextureFormat;  // always 'rgba16float' (scene + TAA history)
  bloomFormat: GPUTextureFormat;// 'rg11b10ufloat' if renderable else 'rgba16float'
  depthFormat: GPUTextureFormat;// 'depth32float'
  tier: DeviceTier;
  maxWorkgroupInvocations: number;
}

export type DeviceTier = 'desktop' | 'laptop' | 'mobile';

/** Quality level 0 (lowest) .. 4 (highest). Produced by QualityController, consumed by passes. */
export interface QualitySettings {
  level: number;
  bloomMips: number;        // 3..6
  reflections: boolean;     // floor mirror-image of the beams
  glowScale: number;        // multiplies beamStyle glow width (cheap halo vs wide halo)
  glassDispersion: boolean; // glass pass samples R/G/B separately (3 taps) vs 1 tap
}

/** Output size in device pixels. Everything renders at output resolution (lines are analytically AA'd). */
export interface Size { outW: number; outH: number }

/** Shared per-frame textures (owned by host via ResourcePool). */
export interface FrameTargets {
  hdr: GPUTexture;      // hdrFormat, output size. LaserPass clears+draws; GlassPass loads+draws; Bloom+Composite read.
  hdrCopy: GPUTexture;  // copy of hdr after beams, before glass (GlassPass samples it for screen-space refraction)
}

/** GPU buffers of the optics scene (storage, read-only). Replaced when the text layout changes; host then calls configure(). */
export interface OpticsBuffers {
  segments: GPUBuffer; mirrors: GPUBuffer; glass: GPUBuffer;
  numSegments: number; numMirrors: number; numGlass: number;
}

/** Context every pass receives in its constructor. */
export interface EngineCtx {
  device: GPUDevice;
  caps: Caps;
  frameUBO: GPUBuffer;
  frameBGL: GPUBindGroupLayout; // @group(0) shared by all pipelines
  frameBG: GPUBindGroup;
  linearClamp: GPUSampler;
  pool: ResourcePoolLike;
  optics: OpticsBuffers;        // mutable: host swaps buffers on re-layout, then calls configure()
}

export interface ResourcePoolLike {
  texture(key: string, desc: GPUTextureDescriptor): GPUTexture; // returns cached texture if desc matches, else recreates
  buffer(key: string, desc: GPUBufferDescriptor): GPUBuffer;
}

/** Timestamp writes for one pass; undefined when timestamps unsupported or profiling disabled. */
export type TS = GPURenderPassTimestampWrites | GPUComputePassTimestampWrites | undefined;

export interface Pass {
  readonly name: string;
  /** Async pipeline compilation (create*PipelineAsync). All passes' init() run concurrently via Promise.all. */
  init(): Promise<void>;
  /** Called on init and whenever Size, QualitySettings, or targets change. Recreate bind groups here, never per frame. */
  configure(size: Size, q: QualitySettings, targets: FrameTargets): void;
  /** Record GPU work. Must not allocate GPU objects. `ts` gives begin/end timestamp indices for this pass. */
  encode(enc: GPUCommandEncoder, ts: TS, frameIndex: number): void;
}

// ---------- worker <-> main thread messages ----------
export type MainToWorker =
  | { type: 'init'; canvas: OffscreenCanvas | HTMLCanvasElement; width: number; height: number; dpr: number; tier: DeviceTier; bench: string | null; reducedMotion: boolean; forceWebGL2?: boolean; forceLevel?: number }
  | { type: 'resize'; width: number; height: number; dpr: number }
  | { type: 'pointer'; x: number; y: number; down: boolean; click: boolean; t: number } // click = send a light pulse // x,y in [0,1] top-left origin; t = performance.timeOrigin + event.timeStamp
  | { type: 'visibility'; hidden: boolean };

export interface Stats {
  fps: number;
  cpuMs: number;              // worker CPU time per frame (encode + physics)
  gpuMs: number;              // total GPU time (timestamps) or -1
  passes: Record<string, number>; // per-pass GPU ms
  p50: number; p95: number; p99: number; // frame interval ms over last window
  dropped: number;            // frames with interval > 1.5x refresh in last window
  inputLatencyMs: number;     // pointer event -> GPU work done for the frame that consumed it
  level: number;
  renderScale: number;
  refreshHz: number;
  backend: 'webgpu' | 'webgl2';
}

export type WorkerToMain =
  | { type: 'ready'; backend: 'webgpu' | 'webgl2' }
  | { type: 'stats'; stats: Stats }
  | { type: 'bench-report'; report: unknown }
  | { type: 'error'; message: string };
