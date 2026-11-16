import type { MainToWorker, WorkerToMain } from '../engine/types';
import { rasterizeText, bakeSDFCPU, type TextRaster } from '../engine/sdfbake';
import { m4, perspective, lookAt, mul, invert } from '../engine/math';

type InitMsg = Extract<MainToWorker, { type: 'init' }>;
type Post = (message: WorkerToMain) => void;
const FOV = 32 * Math.PI / 180;
// Mirrors host.ts RIG: keep the fallback's projectors and convergence in sync.
const RIG = [
  { pos: [-7.5, 4.0, -4.5], orbit: [0.9, 0.8, 1.0], freq: 0.19, color: [9.0, 1.1, 0.55], scatter: [-1.4, 0.5] },
  { pos: [7.5, 4.0, -4.5], orbit: [0.9, 0.8, 1.0], freq: 0.17, color: [0.8, 8.0, 1.6], scatter: [1.3, -0.6] },
  { pos: [0.0, 7.6, -5.5], orbit: [1.8, 0.6, 0.8], freq: 0.13, color: [1.2, 3.2, 11.0], scatter: [0.2, 1.2] },
] as const;
const VERTEX = `#version 300 es
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2),float(gl_VertexID & 2));
  gl_Position = vec4(p*2.0-1.0,0.0,1.0);
}`;

export class WebGL2Engine {
  private raster!: TextRaster;
  private texture!: WebGLTexture;
  private program!: WebGLProgram;
  private uniforms: Record<string, WebGLUniformLocation | null> = {};
  private width: number; private height: number; private dpr: number;
  private hidden = false; private lost = false;
  private time = 0; private last = 0; private reportAt = 0;
  private intervals: number[] = [];
  private pointer = { x: 0.5, y: 0.5, lastMove: -1e9, intensity: 0, pendingT: 0 };
  private agitation = 0;
  private beamPos = new Float32Array(12); private beamDir = new Float32Array(12);
  private beamColor = new Float32Array(12); private beamAlign = new Float32Array(12);
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

  private setup(): void {
    const gl = this.gl;
    const compile = (type: number, source: string) => {
      const shader = gl.createShader(type); if (!shader) throw new Error('Cannot allocate shader');
      gl.shaderSource(shader, source); gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) { const error = gl.getShaderInfoLog(shader); gl.deleteShader(shader); throw new Error(error ?? 'GLSL compile failure'); }
      return shader;
    };
    const vertex = compile(gl.VERTEX_SHADER, VERTEX), fragment = compile(gl.FRAGMENT_SHADER, FRAGMENT);
    const program = gl.createProgram(); if (!program) throw new Error('Cannot allocate GL program');
    gl.attachShader(program, vertex); gl.attachShader(program, fragment); gl.linkProgram(program);
    gl.deleteShader(vertex); gl.deleteShader(fragment);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program) ?? 'GLSL link failure');
    this.program = program; gl.useProgram(program); gl.bindVertexArray(gl.createVertexArray());
    for (const name of ['uSDF', 'uInvVP', 'uEye', 'uRes', 'uText', 'uTime', 'uFar', 'uBeamPos[0]', 'uBeamDir[0]', 'uBeamColor[0]', 'uBeamAlign[0]', 'uPointer']) this.uniforms[name] = gl.getUniformLocation(program, name);
    const texture = gl.createTexture(); if (!texture) throw new Error('Cannot allocate SDF texture');
    this.texture = texture; gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.uniform1i(this.uniforms.uSDF, 0); gl.disable(gl.DEPTH_TEST); gl.disable(gl.BLEND);
    this.resize();
    this.canvas.addEventListener('webglcontextlost', event => { event.preventDefault(); this.lost = true; this.post({ type: 'error', message: 'WebGL context lost' }); });
    this.schedule();
  }

  handle(message: MainToWorker): void {
    switch (message.type) {
      case 'resize': this.width = message.width; this.height = message.height; this.dpr = message.dpr; this.resize(); break;
      case 'visibility': this.hidden = message.hidden; if (!message.hidden) this.last = 0; break;
      case 'pointer':
        this.agitation = Math.min(1.6, this.agitation + Math.hypot(message.x - this.pointer.x, message.y - this.pointer.y) * 3.5);
        if (Math.abs(message.x - this.pointer.x) + Math.abs(message.y - this.pointer.y) > 1e-4 || message.click) this.pointer.lastMove = this.time;
        this.pointer.x = message.x; this.pointer.y = message.y; this.pointer.pendingT = message.t; break;
    }
  }

  private resize(): void {
    this.canvas.width = Math.max(1, Math.round(this.width * this.dpr * 0.6));
    this.canvas.height = Math.max(1, Math.round(this.height * this.dpr * 0.6));
    this.gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    const lines = this.width / this.height < 0.9 ? ['THOMAS', 'DUONG'] : ['THOMAS DUONG'];
    if (!this.raster || this.raster.lines.length !== lines.length) {
      this.raster = rasterizeText(lines);
      const sdf = bakeSDFCPU(this.raster), gl = this.gl;
      gl.bindTexture(gl.TEXTURE_2D, this.texture); gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.R16F, this.raster.w, this.raster.h, 0, gl.RED, gl.FLOAT, sdf);
    }
  }

  private schedule(): void {
    if (this.lost) return;
    if (typeof self.requestAnimationFrame === 'function') self.requestAnimationFrame(this.render);
    else setTimeout(() => this.render(performance.now()), 1000 / 60);
  }

  private writeBeams(dt: number): void {
    const t = this.time, still = this.init.reducedMotion;
    this.agitation *= Math.exp(-dt * 2.2);
    const intro = still ? 0 : Math.exp(-1.35 * t) * Math.cos(2.1 * t);
    const reach = Math.hypot(this.raster.glyphHalfW, this.raster.glyphHalfH) * 1.12;
    for (let j = 0; j < 3; j++) {
      const rig = RIG[j], o = j * 4, w = still ? 0 : 1;
      const px = rig.pos[0] + w * rig.orbit[0] * Math.sin(t * rig.freq + j * 2.1);
      const py = rig.pos[1] + w * rig.orbit[1] * Math.sin(t * rig.freq * 1.3 + j);
      const pz = rig.pos[2] + w * rig.orbit[2] * Math.cos(t * rig.freq * 0.8 + j * 1.7);
      const d0 = Math.hypot(px, py, pz);
      const outer = Math.atan(reach / d0) + 3 * Math.PI / 180, inner = outer - 4 * Math.PI / 180;
      this.beamPos.set([px, py, pz, Math.cos(outer)], o);
      this.beamDir.set([-px / d0, -py / d0, -pz / d0, Math.cos(inner)], o);
      this.beamColor.set([...rig.color, 0.06], o);
      const angle = t * 0.9 + j * 2.094;
      this.beamAlign.set([rig.scatter[0] * intro + Math.cos(angle) * this.agitation * 0.35,
        rig.scatter[1] * intro + Math.sin(angle) * this.agitation * 0.35, 0, d0 * d0], o);
    }
  }

  private render = (now: number): void => {
    this.schedule(); if (this.hidden || this.lost) return;
    const start = performance.now();
    const interval = this.last ? now - this.last : 1000 / 60;
    const dt = Math.min(interval / 1000, 0.1); this.last = now;
    this.intervals.push(interval); if (this.intervals.length > 240) this.intervals.shift();
    this.time += dt * (this.init.reducedMotion ? 0.25 : 1);
    const r = this.raster, aspect = this.canvas.width / this.canvas.height, tanH = Math.tan(FOV / 2);
    const distance = Math.max(r.glyphHalfW * 1.3 / (tanH * aspect), r.glyphHalfH * 2.4 / tanH);
    const k = this.init.reducedMotion ? 0.02 : 0.12;
    this.parallax[0] += ((this.pointer.x - 0.5) - this.parallax[0]) * k;
    this.parallax[1] += ((this.pointer.y - 0.5) - this.parallax[1]) * k;
    this.eye[0] = this.parallax[0] * distance * 0.08;
    this.eye[1] = 0.35 + r.glyphHalfH * 0.2 - this.parallax[1] * distance * 0.04;
    this.eye[2] = distance;
    perspective(this.projection, FOV, aspect, 0.1, distance + 30); lookAt(this.view, this.eye, [0, 0.05, 0], [0, 1, 0]);
    mul(this.vp, this.projection, this.view); invert(this.inverse, this.vp);
    const nx = this.pointer.x * 2 - 1, ny = 1 - this.pointer.y * 2, m = this.inverse;
    const w = m[3]*nx + m[7]*ny + m[11] + m[15];
    const far = [(m[0]*nx + m[4]*ny + m[8] + m[12])/w, (m[1]*nx + m[5]*ny + m[9] + m[13])/w, (m[2]*nx + m[6]*ny + m[10] + m[14])/w];
    const rayScale = (0.9 - this.eye[2]) / (far[2] - this.eye[2]);
    this.pointer.intensity += ((this.time - this.pointer.lastMove < 1.2 ? 1 : 0.15) - this.pointer.intensity) * Math.min(1, dt * 4);
    this.writeBeams(dt);
    const gl = this.gl, u = this.uniforms;
    gl.useProgram(this.program); gl.uniformMatrix4fv(u.uInvVP, false, this.inverse);
    gl.uniform3fv(u.uEye, this.eye); gl.uniform2f(u.uRes, this.canvas.width, this.canvas.height);
    gl.uniform4f(u.uText, r.halfW, r.halfH, r.floorY, 0.01); gl.uniform1f(u.uTime, this.time); gl.uniform1f(u.uFar, distance + 16);
    gl.uniform4fv(u['uBeamPos[0]'], this.beamPos); gl.uniform4fv(u['uBeamDir[0]'], this.beamDir);
    gl.uniform4fv(u['uBeamColor[0]'], this.beamColor); gl.uniform4fv(u['uBeamAlign[0]'], this.beamAlign);
    gl.uniform4f(u.uPointer, this.eye[0]+(far[0]-this.eye[0])*rayScale, this.eye[1]+(far[1]-this.eye[1])*rayScale, 0.9, this.pointer.intensity*6);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    const latency = this.pointer.pendingT ? performance.timeOrigin + performance.now() - this.pointer.pendingT : 0; this.pointer.pendingT = 0;
    if (now - this.reportAt >= 250) {
      this.reportAt = now;
      const sorted = [...this.intervals].sort((a,b) => a-b), n = sorted.length;
      const percentile = (p: number) => sorted[Math.min(n-1, Math.floor(n*p))];
      const refreshHz = [60,75,90,120,144,165,240].reduce((best,hz) => Math.abs(1000/hz-percentile(0.5)) < Math.abs(1000/best-percentile(0.5)) ? hz : best,60);
      this.post({ type: 'stats', stats: { fps: 1000/(this.intervals.reduce((a,b)=>a+b,0)/n), cpuMs: performance.now()-start,
        gpuMs: -1, passes: {}, p50: percentile(0.5), p95: percentile(0.95), p99: percentile(0.99),
        dropped: this.intervals.filter(ms => ms > 1.5*1000/refreshHz).length, inputLatencyMs: latency,
        level: 0, renderScale: 0.6, refreshHz, backend: 'webgl2' } });
    }
  };
}

// Fixed-cost v2 beam rendering: 16 haze samples, 20 letter-slab samples, no reflection march.
const FRAGMENT = `#version 300 es
precision highp float;
precision highp int;
uniform sampler2D uSDF;
uniform mat4 uInvVP;
uniform vec3 uEye;
uniform vec2 uRes;
uniform vec4 uText, uPointer;
uniform vec4 uBeamPos[3], uBeamDir[3], uBeamColor[3], uBeamAlign[3];
uniform float uTime, uFar;
out vec4 fragColor;
const float PI=3.14159265359, MISS=1e9;
const vec3 EXCITE_COLOR=vec3(1.0,0.96,0.9);
float ign(vec2 p) { return fract(52.9829189*fract(dot(p,vec2(0.06711056,0.00583715)))); }
float hg(float c,float g) { return (1.0-g*g)/(4.0*PI*pow(max(0.001,1.0+g*g-2.0*g*c),1.5)); }
float sdf2(vec2 p) {
  vec2 uv=p/uText.xy*0.5+0.5;
  float d=textureLod(uSDF,clamp(uv,0.0,1.0),0.0).r;
  float outside=length(max(abs(p)-uText.xy,0.0));
  return outside>0.0 ? max(d,0.0)+outside : d;
}
float letterMask(vec2 p) { return 1.0-smoothstep(-uText.w,uText.w,sdf2(p)); }
float gobo(int j,vec3 p) {
  vec3 pp=p-uBeamAlign[j].xyz;
  float dz=pp.z-uBeamPos[j].z;
  if(abs(dz)<0.001) return 0.0;
  float s=-uBeamPos[j].z/dz;
  if(s<=0.0) return 0.0;
  vec2 q=uBeamPos[j].xy+(pp.xy-uBeamPos[j].xy)*s;
  return mix(uBeamColor[j].w,1.0,letterMask(q));
}
float beamI(int j,vec3 p) {
  vec3 delta=p-uBeamPos[j].xyz; float d2=max(dot(delta,delta),0.0001);
  vec3 l=delta*inversesqrt(d2);
  float cone=smoothstep(uBeamPos[j].w,uBeamDir[j].w,dot(l,uBeamDir[j].xyz));
  if(cone<=0.0) return 0.0;
  return cone*gobo(j,p)*uBeamAlign[j].w/d2;
}
struct BeamSample { vec3 scatter; float excite; };
BeamSample beams(vec3 p,vec3 toViewer) {
  BeamSample b; b.scatter=vec3(0.0); vec3 intensity=vec3(0.0);
  for(int j=0;j<3;j++) {
    float i=beamI(j,p); intensity[j]=i;
    if(i>0.0) { vec3 l=normalize(p-uBeamPos[j].xyz); b.scatter+=uBeamColor[j].rgb*i*hg(dot(l,toViewer),0.35); }
  }
  float pair=intensity.x*intensity.y+intensity.y*intensity.z+intensity.z*intensity.x;
  b.excite=0.06*pair+14.0*intensity.x*intensity.y*intensity.z;
  return b;
}
vec2 boxHit(vec3 ro,vec3 rd,vec3 halfSize) {
  vec3 safe=vec3(abs(rd.x)<1e-6?1e-6:rd.x,abs(rd.y)<1e-6?1e-6:rd.y,abs(rd.z)<1e-6?1e-6:rd.z);
  vec3 a=(-halfSize-ro)/safe,b=(halfSize-ro)/safe;
  vec3 lo=min(a,b),hi=max(a,b);
  return vec2(max(lo.x,max(lo.y,lo.z)),min(hi.x,min(hi.y,hi.z)));
}
float floorHit(vec3 ro,vec3 rd) { if(rd.y> -0.0001) return MISS; float t=(uText.z-ro.y)/rd.y; return t>0.0?t:MISS; }
float floorFade(vec3 p) { return exp(-max(length(p.xz)-6.0,0.0)*0.25); }
float ggx(vec3 n,vec3 v,vec3 l,float rough) {
  vec3 h=normalize(v+l); float nl=max(dot(n,l),0.0),nv=max(dot(n,v),0.0001),nh=max(dot(n,h),0.0);
  float a=rough*rough,a2=a*a,dd=nh*nh*(a2-1.0)+1.0,k=a*0.5;
  return a2/(PI*dd*dd)*nl/(4.0*(nl*(1.0-k)+k)*(nv*(1.0-k)+k));
}
vec3 pointerRadiance(vec3 p) { vec3 d=uPointer.xyz-p; return vec3(0.55,0.75,1.0)*uPointer.w*0.35/max(dot(d,d),0.08); }
vec3 emitters(vec3 ro,vec3 rd) {
  vec3 c=vec3(0.0);
  for(int j=0;j<3;j++) {
    float toward=smoothstep(-0.1,0.7,dot(-rd,uBeamDir[j].xyz));
    vec3 oc=ro-uBeamPos[j].xyz; float b=dot(oc,rd),d2=dot(oc,oc)-b*b;
    if(b<0.0) {
      float r=0.14,core=1.0-smoothstep(r*r*0.6,r*r,d2),glow=exp(-d2/(r*r)*0.35)*0.08;
      c+=uBeamColor[j].rgb*(core*4.0+glow)*toward;
    }
  }
  return c;
}
vec3 floorShade(vec3 p,vec3 rd) {
  vec3 n=vec3(0.0,1.0,0.0),v=-rd,c=vec3(0.0);
  for(int j=0;j<3;j++) {
    float i=beamI(j,p);
    if(i>0.0) { vec3 l=normalize(uBeamPos[j].xyz-p); c+=uBeamColor[j].rgb*i*(0.08/PI*max(l.y,0.0)+0.04*ggx(n,v,l,0.35)); }
  }
  vec3 pl=normalize(uPointer.xyz-p);
  c+=pointerRadiance(p)*(0.08/PI*max(pl.y,0.0)+0.04*ggx(n,v,pl,0.3));
  // Level-zero fallback omits the expensive reflected letter-slab march.
  c+=emitters(p,reflect(rd,n))*(0.04+0.96*pow(1.0-clamp(v.y,0.0,1.0),5.0));
  return c*floorFade(p);
}
float hash3(vec3 p) { return fract(sin(dot(p,vec3(127.1,311.7,74.7)))*43758.5453); }
float noise3(vec3 p) {
  vec3 i=floor(p),f=fract(p),u=f*f*(3.0-2.0*f);
  return mix(mix(mix(hash3(i),hash3(i+vec3(1,0,0)),u.x),mix(hash3(i+vec3(0,1,0)),hash3(i+vec3(1,1,0)),u.x),u.y),
    mix(mix(hash3(i+vec3(0,0,1)),hash3(i+vec3(1,0,1)),u.x),mix(hash3(i+vec3(0,1,1)),hash3(i+vec3(1,1,1)),u.x),u.y),u.z);
}
float density(vec3 p) {
  float n=noise3(p*0.3+vec3(uTime*0.011,uTime*0.004,-uTime*0.007));
  return 0.05*exp(-max(p.y-uText.z,0.0)*0.1)*(0.3+1.4*n);
}
float slabGlow(vec3 ro,vec3 rd,float maxT,float jitter) {
  vec2 box=boxHit(ro,rd,vec3(uText.xy,0.15));
  float t0=max(box.x,0.0),t1=min(box.y,maxT);
  if(t1<=t0) return 0.0;
  float stepSize=(t1-t0)/20.0,e=0.0;
  for(int i=0;i<20;i++) { vec3 p=ro+rd*(t0+(float(i)+jitter)*stepSize); e+=beams(p,-rd).excite; }
  return e*stepSize;
}
vec3 agx(vec3 c) {
  mat3 inset=mat3(0.8424790623,0.0423282423,0.0423756549,0.0784336,0.8784686365,0.0784336,0.0792237451,0.0791661275,0.8791429738);
  mat3 outset=mat3(1.1968790051,-0.0528968518,-0.0529716355,-0.0980208811,1.1519031299,-0.0980434501,-0.0990297441,-0.0989611768,1.1510736726);
  c=clamp((log2(max(inset*max(c,0.0),vec3(1e-6)))+12.47393)/16.5,0.0,1.0);
  vec3 c2=c*c,c4=c2*c2;
  c=15.5*c4*c2-40.14*c4*c+31.96*c4-6.868*c2*c+0.4298*c2+0.1191*c-0.00232;
  float luma=dot(c,vec3(0.2126,0.7152,0.0722)); c=mix(vec3(luma),pow(max(c,0.0),vec3(1.05)),1.1);
  return pow(max(outset*c,0.0),vec3(2.2));
}
vec3 srgb(vec3 c) { return mix(c*12.92,1.055*pow(max(c,0.0),vec3(1.0/2.4))-0.055,step(vec3(0.0031308),c)); }
void main() {
  vec2 uv=gl_FragCoord.xy/uRes;
  vec4 farPoint=uInvVP*vec4(uv*2.0-1.0,1.0,1.0);
  vec3 ro=uEye,rd=normalize(farPoint.xyz/farPoint.w-ro);
  float floorT=floorHit(ro,rd),tEnd=min(floorT,uFar);
  float jitter=fract(ign(gl_FragCoord.xy)+uTime*0.61803398875);
  vec2 slab=boxHit(ro,rd,vec3(uText.xy,0.15));
  float s0=max(slab.x,0.0),s1=min(slab.y,tEnd);
  float glow=slabGlow(ro,rd,tEnd,jitter);
  float stepSize=tEnd/16.0,trans=1.0,slabTrans=1.0;
  vec3 scatter=vec3(0.0);
  for(int i=0;i<16;i++) {
    float t=(float(i)+jitter)*stepSize;
    vec3 p=ro+rd*t; float sigma=density(p); BeamSample b=beams(p,-rd);
    vec3 add=b.scatter*sigma;
    if(t<s0||t>s1) add+=EXCITE_COLOR*b.excite*0.25;
    scatter+=trans*add*stepSize; trans*=exp(-sigma*stepSize);
    if(t<s0) slabTrans=trans;
  }
  vec3 surface=floorT<MISS?floorShade(ro+rd*floorT,rd):emitters(ro,rd);
  vec3 color=surface*trans+scatter+EXCITE_COLOR*glow*slabTrans;
  if(any(isnan(color))||any(isinf(color))) color=vec3(0.0);
  color=agx(min(color,vec3(256.0))); color*=1.0-0.25*pow(length((uv-0.5)*1.414),2.0);
  color=srgb(color); color+=(fract(ign(gl_FragCoord.xy)+uTime*0.618)-0.5)*(0.015+1.0/255.0);
  fragColor=vec4(clamp(color,0.0,1.0),1.0);
}`;
