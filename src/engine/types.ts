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
  renderScale: number;      // internal scene resolution / output resolution
  primarySteps: number;     // scene raymarch steps
  interiorSteps: number;    // glass interior march steps
  reflSteps: number;        // floor reflection march steps
  shadowSteps: number;      // SDF soft-shadow steps (scene, froxels, particles)
  dispersion: boolean;      // 3-channel refraction (true) or 1 march + offsets (false)
  froxel: [number, number, number]; // grid dims x,y,z
  particles: number;        // active particle count (multiple of 256)
  bloomMips: number;        // 4..6
  taa: boolean;
}

/** Output and internal render sizes in device pixels. */
export interface Size {
  outW: number; outH: number;   // swapchain / TAA output
  inW: number; inH: number;     // internal scene resolution = round(out * renderScale)
}

/** Shared per-frame textures. Owned by the worker (via ResourcePool), read/written by passes. */
export interface FrameTargets {
  sceneColor: GPUTexture;   // hdrFormat, internal size. Written: ScenePass (clear), ParticlePass (load, additive)
  sceneDepth: GPUTexture;   // depth32float, internal size. Written: ScenePass (frag_depth), read: Particles(test), Composite(fog)
  taa: [GPUTexture, GPUTexture]; // hdrFormat, output size. Frame n: TAA writes taa[n&1], reads history taa[(n+1)&1]; Bloom+Composite read taa[n&1]
  froxelIntegrated: GPUTexture; // rgba16float 3D: rgb=in-scatter accumulated, a=transmittance. Written: VolumetricPass
}

/** Context every pass receives in its constructor. */
export interface EngineCtx {
  device: GPUDevice;
  caps: Caps;
  frameUBO: GPUBuffer;         // FRAME_UBO_SIZE bytes, layout in frame.ts / frame.wgsl
  frameBGL: GPUBindGroupLayout;// @group(0) layout shared by ALL pipelines (explicit pipeline layouts: [frameBGL, passBGL...])
  frameBG: GPUBindGroup;       // set once per pass encoder at group 0
  sdfView: GPUTextureView;     // rgba16float 2D text SDF, .r = signed distance in WORLD units (negative inside)
  linearClamp: GPUSampler;
  pool: ResourcePoolLike;
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
  | { type: 'init'; canvas: OffscreenCanvas | HTMLCanvasElement; width: number; height: number; dpr: number; tier: DeviceTier; bench: string | null; reducedMotion: boolean; forceWebGL2?: boolean }
  | { type: 'resize'; width: number; height: number; dpr: number }
  | { type: 'pointer'; x: number; y: number; down: boolean; click: boolean; t: number } // x,y in [0,1] top-left origin; t = performance.timeOrigin + event.timeStamp
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
