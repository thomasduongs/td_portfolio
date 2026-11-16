import type { MainToWorker, WorkerToMain } from '../engine/types';
import { rasterizeText, type TextRaster } from '../engine/sdfbake';
import { extractGlyphs } from '../engine/optics/contours';
import { buildOptics } from '../engine/optics/route';
import { packOptics, type OpticsScene } from '../engine/optics/types';
import { m4, perspective, lookAt, mul, invert } from '../engine/math';

type InitMsg = Extract<MainToWorker, { type: 'init' }>;
type Post = (message: WorkerToMain) => void;
type Program = { program: WebGLProgram; uniforms: Record<string, WebGLUniformLocation | null> };
const FOV = 32 * Math.PI / 180;
const MAX_PIXELS = 3_700_000;

/** The same CPU-routed installation as WebGPU, with instanced WebGL2 quads and simple tinted glass. */
export class WebGL2Engine {
  private raster!: TextRaster;
  private scene!: OpticsScene;
  private beam!: Program; private mirror!: Program; private glint!: Program; private glass!: Program; private tone!: Program;
  private segmentVAO!: WebGLVertexArrayObject; private mirrorVAO!: WebGLVertexArrayObject; private glassVAO!: WebGLVertexArrayObject;
  private buffers: WebGLBuffer[] = [];
  private hdr!: WebGLTexture; private framebuffer!: WebGLFramebuffer;
  private floatHDR = false;
  private width: number; private height: number; private dpr: number;
  private hidden = false; private lost = false;
  private time = 0; private last = 0; private reportAt = 0; private pulseStart = -1;
  private intervals: number[] = [];
  private pointer = { x: 0.5, y: 0.5, activity: 0, lastMove: -1e9, pendingT: 0 };
  private parallax = [0, 0]; private eye = [0, 0.5, 8];
  private projection = m4(); private view = m4(); private vp = m4(); private inverse = m4();

  static async create(canvas: OffscreenCanvas | HTMLCanvasElement, init: InitMsg, post: Post): Promise<WebGL2Engine | null> {
    const gl = canvas.getContext('webgl2', { antialias: false, alpha: false, powerPreference: 'high-performance' }) as WebGL2RenderingContext | null;
    if (!gl) return null;
    const engine = new WebGL2Engine(canvas, gl, init, post);
    try { engine.setup(); return engine; }
    catch (error) { console.error('WebGL2 initialization failed', error); return null; }
  }

  private constructor(private canvas: OffscreenCanvas | HTMLCanvasElement, private gl: WebGL2RenderingContext, private init: InitMsg, private post: Post) {
    this.width = init.width; this.height = init.height; this.dpr = init.dpr;
  }

  private compile(vertexSource: string, fragmentSource: string): Program {
    const gl = this.gl;
    const shader = (type: number, source: string): WebGLShader => {
      const object = gl.createShader(type); if (!object) throw new Error('Cannot allocate shader');
      gl.shaderSource(object, source); gl.compileShader(object);
      if (!gl.getShaderParameter(object, gl.COMPILE_STATUS)) { const error = gl.getShaderInfoLog(object); gl.deleteShader(object); throw new Error(error ?? 'GLSL compile failure'); }
      return object;
    };
    const vertex = shader(gl.VERTEX_SHADER, vertexSource), fragment = shader(gl.FRAGMENT_SHADER, fragmentSource);
    const program = gl.createProgram(); if (!program) throw new Error('Cannot allocate GL program');
    gl.attachShader(program, vertex); gl.attachShader(program, fragment); gl.linkProgram(program);
    gl.deleteShader(vertex); gl.deleteShader(fragment);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) { const error = gl.getProgramInfoLog(program); gl.deleteProgram(program); throw new Error(error ?? 'GLSL link failure'); }
    const uniforms: Program['uniforms'] = {};
    for (const name of ['uVP', 'uInvVP', 'uEye', 'uRes', 'uPointer', 'uStyle', 'uMisc', 'uReflect', 'uHDR']) uniforms[name] = gl.getUniformLocation(program, name);
    return { program, uniforms };
  }

  private setup(): void {
    const gl = this.gl;
    this.floatHDR = !!gl.getExtension('EXT_color_buffer_float');
    this.beam = this.compile(HEADER + BEAM_VERTEX, HEADER + BEAM_FRAGMENT);
    this.mirror = this.compile(HEADER + MIRROR_VERTEX, HEADER + MIRROR_FRAGMENT);
    this.glint = this.compile(HEADER + GLINT_VERTEX, HEADER + GLINT_FRAGMENT);
    this.glass = this.compile(HEADER + GLASS_VERTEX, HEADER + GLASS_FRAGMENT);
    this.tone = this.compile(HEADER + FULLSCREEN_VERTEX, HEADER + TONEMAP_FRAGMENT);
    const texture = gl.createTexture(), framebuffer = gl.createFramebuffer();
    if (!texture || !framebuffer) throw new Error('Cannot allocate HDR target');
    this.hdr = texture; this.framebuffer = framebuffer;
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.hdr);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.disable(gl.DEPTH_TEST); gl.disable(gl.CULL_FACE);
    this.resize();
    this.canvas.addEventListener('webglcontextlost', event => { event.preventDefault(); this.lost = true; this.post({ type: 'error', message: 'WebGL context lost' }); });
    this.schedule();
  }

  private instanceVAO(data: Float32Array, attributes: number): WebGLVertexArrayObject {
    const gl = this.gl, vao = gl.createVertexArray(), buffer = gl.createBuffer();
    if (!vao || !buffer) throw new Error('Cannot allocate optics buffers');
    this.buffers.push(buffer); gl.bindVertexArray(vao); gl.bindBuffer(gl.ARRAY_BUFFER, buffer); gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
    for (let i = 0; i < attributes; i++) { gl.enableVertexAttribArray(i); gl.vertexAttribPointer(i, 4, gl.FLOAT, false, attributes * 16, i * 16); gl.vertexAttribDivisor(i, 1); }
    return vao;
  }

  private resize(): void {
    const gl = this.gl;
    let outW = Math.max(1, Math.round(this.width * Math.min(this.dpr, 2))), outH = Math.max(1, Math.round(this.height * Math.min(this.dpr, 2)));
    if (outW * outH > MAX_PIXELS) { const k = Math.sqrt(MAX_PIXELS / (outW * outH)); outW = Math.round(outW * k); outH = Math.round(outH * k); }
    this.canvas.width = outW; this.canvas.height = outH;
    gl.bindTexture(gl.TEXTURE_2D, this.hdr);
    gl.texImage2D(gl.TEXTURE_2D, 0, this.floatHDR ? gl.RGBA16F : gl.RGBA8, outW, outH, 0, gl.RGBA, this.floatHDR ? gl.HALF_FLOAT : gl.UNSIGNED_BYTE, null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffer); gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.hdr, 0);
    if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) throw new Error('Incomplete optics framebuffer');
    gl.bindFramebuffer(gl.FRAMEBUFFER, null); gl.viewport(0, 0, outW, outH);
    const lines = this.width / this.height < 0.9 ? ['THOMAS', 'DUONG'] : ['THOMAS DUONG'];
    if (!this.raster || this.raster.lines.length !== lines.length) {
      this.raster = rasterizeText(lines);
      this.scene = buildOptics(extractGlyphs(this.raster), { glyphHalfW: this.raster.glyphHalfW, glyphHalfH: this.raster.glyphHalfH, floorY: this.raster.floorY, lines: lines.length });
      this.buffers.forEach(buffer => gl.deleteBuffer(buffer)); this.buffers = [];
      [this.segmentVAO, this.mirrorVAO, this.glassVAO].forEach(vao => { if (vao) gl.deleteVertexArray(vao); });
      const packed = packOptics(this.scene);
      this.segmentVAO = this.instanceVAO(packed.segments, 3); this.mirrorVAO = this.instanceVAO(packed.mirrors, 3); this.glassVAO = this.instanceVAO(packed.glass, 4);
    }
  }

  handle(message: MainToWorker): void {
    switch (message.type) {
      case 'resize': this.width = message.width; this.height = message.height; this.dpr = message.dpr; this.resize(); break;
      case 'visibility': this.hidden = message.hidden; if (!message.hidden) this.last = 0; break;
      case 'pointer':
        if (Math.hypot(message.x - this.pointer.x, message.y - this.pointer.y) > 1e-4) this.pointer.lastMove = this.time;
        this.pointer.x = message.x; this.pointer.y = message.y; this.pointer.pendingT = message.t;
        if (message.click) this.pulseStart = this.time;
        break;
    }
  }

  private schedule(): void {
    if (this.lost) return;
    if (typeof self.requestAnimationFrame === 'function') self.requestAnimationFrame(this.render);
    else setTimeout(() => this.render(performance.now()), 1000 / 60);
  }

  private updateCamera(): void {
    const r = this.raster, aspect = this.canvas.width / this.canvas.height, tanH = Math.tan(FOV / 2);
    const distance = Math.max(((r.glyphHalfW + 1.4) * 1.08) / (tanH * aspect), ((r.glyphHalfH + 1.6) * 1.15) / tanH);
    const k = this.init.reducedMotion ? 0.02 : 0.1;
    this.parallax[0] += ((this.pointer.x - 0.5) - this.parallax[0]) * k;
    this.parallax[1] += ((this.pointer.y - 0.5) - this.parallax[1]) * k;
    this.eye[0] = this.parallax[0] * distance * 0.12;
    this.eye[1] = 0.25 + r.glyphHalfH * 0.15 - this.parallax[1] * distance * 0.06;
    this.eye[2] = distance;
    perspective(this.projection, FOV, aspect, 0.1, distance + 60); lookAt(this.view, this.eye, [0, 0.15, 0], [0, 1, 0]);
    mul(this.vp, this.projection, this.view); invert(this.inverse, this.vp);
  }

  private render = (now: number): void => {
    this.schedule(); if (this.hidden || this.lost) return;
    const start = performance.now(), interval = this.last ? now - this.last : 1000 / 60;
    const dt = Math.min(interval / 1000, 0.1); this.last = now; this.time += dt;
    this.intervals.push(interval); if (this.intervals.length > 240) this.intervals.shift();
    this.updateCamera();
    const p = this.pointer, gl = this.gl, width = this.canvas.width, height = this.canvas.height;
    p.activity += ((this.time - p.lastMove < 0.8 ? 1 : 0) - p.activity) * Math.min(1, dt * 3);
    const nx = p.x * 2 - 1, ny = 1 - p.y * 2, m = this.inverse;
    const w = m[3]*nx + m[7]*ny + m[11] + m[15];
    const fx = (m[0]*nx + m[4]*ny + m[8] + m[12])/w, fy = (m[1]*nx + m[5]*ny + m[9] + m[13])/w, fz = (m[2]*nx + m[6]*ny + m[10] + m[14])/w;
    const scale = -this.eye[2] / (fz - this.eye[2]), pointerX = this.eye[0] + (fx-this.eye[0])*scale, pointerY = this.eye[1] + (fy-this.eye[1])*scale;
    const capPx = height / (2 * this.eye[2] * Math.tan(FOV / 2)), glintScale = Math.min(1.6, Math.max(0.45, capPx / 95));
    const reveal = this.init.reducedMotion ? 1e6 : Math.max(0, this.time - 0.4) * 9;
    let pulse = this.pulseStart >= 0 ? (this.time - this.pulseStart) * 14 : -1;
    if (pulse > this.scene.pathLength + 3) { pulse = -1; this.pulseStart = -1; }
    const use = (program: Program) => {
      gl.useProgram(program.program); const u = program.uniforms;
      gl.uniformMatrix4fv(u.uVP, false, this.vp); gl.uniformMatrix4fv(u.uInvVP, false, this.inverse);
      gl.uniform3fv(u.uEye, this.eye); gl.uniform2f(u.uRes, width, height);
      gl.uniform4f(u.uPointer, pointerX, pointerY, glintScale, p.activity);
      gl.uniform4f(u.uStyle, Math.max(1, 1.2 * height / 1000), 4.5 * height / 1000, 0.07, this.raster.floorY - 0.06);
      gl.uniform4f(u.uMisc, this.time, reveal, pulse, this.scene.pathLength);
    };
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffer); gl.viewport(0, 0, width, height); gl.clearColor(0,0,0,1); gl.clear(gl.COLOR_BUFFER_BIT);
    gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    use(this.mirror); gl.bindVertexArray(this.mirrorVAO); gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, this.scene.mirrors.length);
    gl.blendFunc(gl.ONE, gl.ONE); use(this.beam); gl.bindVertexArray(this.segmentVAO);
    gl.uniform1f(this.beam.uniforms.uReflect, 0); gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, this.scene.segments.length);
    gl.uniform1f(this.beam.uniforms.uReflect, 1); gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, this.scene.segments.length);
    use(this.glint); gl.bindVertexArray(this.mirrorVAO); gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, this.scene.mirrors.length);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA); use(this.glass); gl.bindVertexArray(this.glassVAO); gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, this.scene.glass.length);
    gl.disable(gl.BLEND); gl.bindFramebuffer(gl.FRAMEBUFFER, null); gl.bindVertexArray(null);
    use(this.tone); gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.hdr); gl.uniform1i(this.tone.uniforms.uHDR, 0); gl.drawArrays(gl.TRIANGLES, 0, 3);
    const latency = p.pendingT ? performance.timeOrigin + performance.now() - p.pendingT : 0; p.pendingT = 0;
    if (now - this.reportAt >= 250) {
      this.reportAt = now; const sorted = [...this.intervals].sort((a,b) => a-b), n = sorted.length;
      const percentile = (q: number) => sorted[Math.min(n-1, Math.floor(n*q))];
      const refreshHz = [60,75,90,120,144,165,240].reduce((best,hz) => Math.abs(1000/hz-percentile(0.5)) < Math.abs(1000/best-percentile(0.5)) ? hz : best,60);
      this.post({ type: 'stats', stats: { fps: 1000/(this.intervals.reduce((a,b)=>a+b,0)/n), cpuMs: performance.now()-start,
        gpuMs: -1, passes: {}, p50: percentile(0.5), p95: percentile(0.95), p99: percentile(0.99),
        dropped: this.intervals.filter(ms => ms > 1.5*1000/refreshHz).length, inputLatencyMs: latency,
        level: 1, renderScale: 1, refreshHz, backend: 'webgl2' } });
    }
  };
}

const HEADER = `#version 300 es
precision highp float;
precision highp int;
uniform mat4 uVP, uInvVP;
uniform vec3 uEye;
uniform vec2 uRes;
uniform vec4 uPointer, uStyle, uMisc;
const vec4 OFFSCREEN=vec4(2.0,2.0,2.0,1.0);
vec2 corner(int i) {
  const vec2 c[6]=vec2[6](vec2(-1,-1),vec2(1,-1),vec2(1,1),vec2(-1,-1),vec2(1,1),vec2(-1,1));
  return c[i];
}
vec2 worldToPx(vec4 c) { return (c.xy/c.w*0.5+0.5)*uRes; }
vec4 pxToClip(vec2 p) { return vec4(p/uRes*2.0-1.0,0.0,1.0); }
float pulseBoost(float s) { if(uMisc.z<0.0) return 1.0; float x=(s-uMisc.z)/0.25; return 1.0+5.0*exp(-x*x); }
`;

const BEAM_VERTEX = `
layout(location=0) in vec4 aStart;
layout(location=1) in vec4 aEnd;
layout(location=2) in vec4 aColor;
uniform float uReflect;
flat out vec2 vA,vB,vS,vW;
flat out vec3 vWA,vWB;
flat out vec4 vColor;
void main() {
  gl_Position=OFFSCREEN; vA=vec2(0);vB=vec2(0);vS=vec2(0);vW=vec2(1);vWA=vec3(0);vWB=vec3(0);vColor=vec4(0);
  vec3 a=aStart.xyz,b=aEnd.xyz; float sa=aStart.w,sb=aEnd.w;
  if(uReflect>0.5) { a.y=2.0*uStyle.w-a.y; b.y=2.0*uStyle.w-b.y; }
  float wa=(uVP*vec4(a,1)).w,wb=(uVP*vec4(b,1)).w;
  if(wa<0.05&&wb<0.05) return;
  if(wa<0.05) { float t=(0.05-wa)/(wb-wa);a=mix(a,b,t);sa=mix(sa,sb,t);wa=0.05; }
  else if(wb<0.05) { float t=(0.05-wb)/(wa-wb);b=mix(b,a,t);sb=mix(sb,sa,t);wb=0.05; }
  vec2 pa=worldToPx(uVP*vec4(a,1)),pb=worldToPx(uVP*vec4(b,1)),delta=pb-pa;
  vec2 d=length(delta)>0.0001?normalize(delta):vec2(1,0),n=vec2(-d.y,d.x),c=corner(gl_VertexID);
  float halfWidth=uStyle.y+2.0;
  vec2 base=c.x>0.0?pb:pa;
  gl_Position=pxToClip(base+d*c.x*halfWidth+n*c.y*halfWidth);
  vA=pa;vB=pb;vS=vec2(sa,sb);vW=vec2(wa,wb);vWA=a;vWB=b;vColor=aColor;
}`;
const BEAM_FRAGMENT = `
flat in vec2 vA,vB,vS,vW;
flat in vec3 vWA,vWB;
flat in vec4 vColor;
uniform float uReflect;
out vec4 fragColor;
void main() {
  vec2 d=vB-vA;float l2=dot(d,d),t=l2>1e-6?clamp(dot(gl_FragCoord.xy-vA,d)/l2,0.0,1.0):0.0;
  float dist=length(gl_FragCoord.xy-(vA+d*t));
  float tw=t*vW.x/(vW.y*(1.0-t)+t*vW.x),s=mix(vS.x,vS.y,tw);
  if(s>uMisc.y) discard;
  vec3 point=mix(vWA,vWB,tw);float gain=1.0;
  if(uReflect>0.5) { if(point.y>uStyle.w) discard;gain=0.12*exp(-(uStyle.w-point.y)*3.0); }
  float x=dist/uStyle.x,intensity=exp(-x*x)*2.2+uStyle.z*exp(-dist/max(uStyle.y,0.001));
  float hd=(uMisc.y-s)/0.08;intensity+=6.0*exp(-hd*hd);
  if(vS.x<0.0001) intensity+=3.0*exp(-pow(s/0.05,2.0));
  intensity*=pulseBoost(s);
  if(uReflect<0.5) intensity*=1.0+0.6*uPointer.w*exp(-distance(uPointer.xy,point.xy)/0.25);
  if(vColor.w>0.5&&vColor.w<1.5) intensity*=0.55;
  if(vColor.w>=1.5) intensity*=pow(1.0-tw,2.0);
  fragColor=vec4(vColor.rgb*intensity*gain,0.0);
}`;
const MIRROR_VERTEX = `
layout(location=0) in vec4 aPos;
layout(location=1) in vec4 aNormal;
layout(location=2) in vec4 aColor;
out vec2 vUV;
flat out vec3 vColor;
flat out float vFade,vSpec;
void main() {
  gl_Position=OFFSCREEN;vUV=vec2(0);vColor=vec3(0);vFade=0.0;vSpec=0.0;
  float fade=clamp((uMisc.y-(aNormal.w-0.2))/0.2,0.0,1.0);if(fade<=0.0) return;
  vec3 n=normalize(aNormal.xyz),up=abs(n.y)>0.99?vec3(1,0,0):vec3(0,1,0),t=normalize(cross(up,n)),b=cross(n,t);
  vec2 c=corner(gl_VertexID);vec3 p=aPos.xyz+(t*c.x+b*c.y)*aPos.w;
  gl_Position=uVP*vec4(p,1.0);vUV=c;vColor=aColor.rgb;vFade=fade;vSpec=pow(max(dot(n,normalize(uEye-aPos.xyz)),0.0),24.0);
}`;
const MIRROR_FRAGMENT = `
in vec2 vUV;
flat in vec3 vColor;
flat in float vFade,vSpec;
out vec4 fragColor;
void main() {
  float rim=smoothstep(0.84,0.92,max(abs(vUV.x),abs(vUV.y)));
  vec3 col=min(vec3(0.02)+vColor*(rim*0.9+vSpec*0.7),vec3(4.0));
  fragColor=vec4(col*vFade,0.9*vFade);
}`;
const GLINT_VERTEX = `
layout(location=0) in vec4 aPos;
layout(location=1) in vec4 aNormal;
layout(location=2) in vec4 aColor;
out vec2 vQ;
flat out vec3 vColor;
flat out float vSize,vArms;
void main() {
  gl_Position=OFFSCREEN;vQ=vec2(0);vColor=vec3(0);vSize=1.0;vArms=0.0;
  float revealed=clamp((uMisc.y-aNormal.w)/0.1,0.0,1.0);if(revealed<=0.0) return;
  vec4 c=uVP*vec4(aPos.xyz,1.0);if(c.w<0.05) return;
  bool outline=aColor.w<0.5;
  float size=(outline?2.5:mix(8.0,13.0,clamp(aPos.w/0.12,0.0,1.0)))*uPointer.z;
  vec2 q=corner(gl_VertexID)*size*2.5;
  gl_Position=vec4(c.xy/c.w+q/uRes*2.0,0.0,1.0);vQ=q;vSize=size;vArms=outline?0.0:0.8;
  float twinkle=outline?1.0:0.85+0.15*sin(uMisc.x*2.0+float(gl_InstanceID));
  vColor=aColor.rgb*(outline?0.45:1.2)*twinkle*revealed*pulseBoost(aNormal.w);
}`;
const GLINT_FRAGMENT = `
in vec2 vQ;
flat in vec3 vColor;
flat in float vSize,vArms;
out vec4 fragColor;
void main() {
  vec2 a=abs(vQ);float thin=1.2;
  float armX=exp(-pow(a.y/thin,2.0))*exp(-a.x/(vSize*0.6)),armY=exp(-pow(a.x/thin,2.0))*exp(-a.y/(vSize*0.6));
  float k=vSize*0.18,core=exp(-dot(vQ,vQ)/(k*k));
  fragColor=vec4(vColor*((armX+armY)*vArms+core*1.2),0.0);
}`;
const GLASS_VERTEX = `
layout(location=0) in vec4 aCenter;
layout(location=1) in vec4 aX;
layout(location=2) in vec4 aY;
layout(location=3) in vec4 aZ;
flat out vec4 vCenter,vX,vY,vZ;
void main() {
  vCenter=aCenter;vX=aX;vY=aY;vZ=aZ;
  vec3 halfSize=vec3(aX.w,aCenter.w<0.5?aX.w:aY.w,aZ.w);
  vec2 lo=vec2(1.0),hi=vec2(-1.0);bool behind=false;
  for(int i=0;i<8;i++) {
    vec3 sign3=vec3((i&1)==0?-1.0:1.0,(i&2)==0?-1.0:1.0,(i&4)==0?-1.0:1.0);
    vec3 p=aCenter.xyz+aX.xyz*halfSize.x*sign3.x+aY.xyz*halfSize.y*sign3.y+aZ.xyz*halfSize.z*sign3.z;
    vec4 clip=uVP*vec4(p,1.0);if(clip.w<0.05) behind=true;
    vec2 ndc=clip.xy/max(0.05,clip.w);lo=min(lo,ndc);hi=max(hi,ndc);
  }
  if(behind) {lo=vec2(-1.0);hi=vec2(1.0);}
  vec2 c=corner(gl_VertexID)*0.5+0.5;gl_Position=vec4(mix(lo,hi,c),0.0,1.0);
}`;
const GLASS_FRAGMENT = `
flat in vec4 vCenter,vX,vY,vZ;
out vec4 fragColor;
vec4 planeAt(int i) {
  if(vCenter.w<0.5) {
    if(i<3) { float angle=radians(i==0?270.0:(i==1?30.0:150.0));return vec4(cos(angle),sin(angle),0.0,vX.w*0.5); }
    return vec4(0,0,i==3?1.0:-1.0,vZ.w);
  }
  int axis=i/2;vec3 n=vec3(0);n[axis]=(i&1)==0?1.0:-1.0;
  return vec4(n,axis==0?vX.w:(axis==1?vY.w:vZ.w));
}
void main() {
  vec4 farPoint=uInvVP*vec4(gl_FragCoord.xy/uRes*2.0-1.0,1.0,1.0);
  vec3 rd=normalize(farPoint.xyz/farPoint.w-uEye),delta=uEye-vCenter.xyz;
  vec3 ro=vec3(dot(delta,vX.xyz),dot(delta,vY.xyz),dot(delta,vZ.xyz));
  vec3 localDir=vec3(dot(rd,vX.xyz),dot(rd,vY.xyz),dot(rd,vZ.xyz));
  float entry=-1e20,exitT=1e20;int count=vCenter.w<0.5?5:6;
  for(int i=0;i<6;i++) {
    if(i>=count) break;vec4 plane=planeAt(i);float den=dot(plane.xyz,localDir),num=plane.w-dot(plane.xyz,ro);
    if(abs(den)<1e-7) {if(num<0.0) discard;continue;}
    float t=num/den;if(den<0.0) entry=max(entry,t);else exitT=min(exitT,t);
  }
  if(entry>=exitT||exitT<0.0) discard;
  vec3 hit=ro+localDir*max(entry,0.0);float first=1e20,second=1e20;
  for(int i=0;i<6;i++) {if(i>=count) break;vec4 plane=planeAt(i);float d=abs(dot(plane.xyz,hit)-plane.w);if(d<first){second=first;first=d;}else second=min(second,d);}
  float edge=1.0-smoothstep(0.0,0.012,second);
  fragColor=vec4(vec3(0.02,0.035,0.05)+vec3(0.6,0.7,0.8)*edge*0.6,0.12+edge*0.65);
}`;
const FULLSCREEN_VERTEX = `
void main() {
  vec2 p=vec2(float((gl_VertexID<<1)&2),float(gl_VertexID&2));gl_Position=vec4(p*2.0-1.0,0.0,1.0);
}`;
const TONEMAP_FRAGMENT = `
uniform sampler2D uHDR;
out vec4 fragColor;
vec3 agx(vec3 c) {
  mat3 inset=mat3(0.8424790623,0.0423282423,0.0423756549,0.0784336,0.8784686365,0.0784336,0.0792237451,0.0791661275,0.8791429738);
  mat3 outset=mat3(1.1968790051,-0.0528968518,-0.0529716355,-0.0980208811,1.1519031299,-0.0980434501,-0.0990297441,-0.0989611768,1.1510736726);
  c=clamp((log2(max(inset*max(c,0.0),vec3(1e-6)))+12.47393)/16.5,0.0,1.0);
  vec3 c2=c*c,c4=c2*c2;
  c=15.5*c4*c2-40.14*c4*c+31.96*c4-6.868*c2*c+0.4298*c2+0.1191*c-0.00232;
  float luma=dot(c,vec3(0.2126,0.7152,0.0722));c=mix(vec3(luma),pow(max(c,0.0),vec3(1.05)),1.1);
  return pow(max(outset*c,0.0),vec3(2.2));
}
vec3 srgb(vec3 c) { return mix(c*12.92,1.055*pow(max(c,0.0),vec3(1.0/2.4))-0.055,step(vec3(0.0031308),c)); }
void main() {
  vec2 uv=gl_FragCoord.xy/uRes;vec3 color=agx(texture(uHDR,uv).rgb);
  color*=1.0-0.25*pow(length((uv-0.5)*1.414),2.0);color=srgb(color);
  float noise=fract(52.9829189*fract(dot(gl_FragCoord.xy,vec2(0.06711056,0.00583715)))+uMisc.x*0.618);
  color+=(noise-0.5)*(0.015+1.0/255.0);fragColor=vec4(clamp(color,0.0,1.0),1.0);
}`;
