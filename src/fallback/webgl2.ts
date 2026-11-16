import type { MainToWorker, WorkerToMain } from '../engine/types';
import { rasterizeText, bakeSDFCPU, type TextRaster } from '../engine/sdfbake';
import { m4, perspective, lookAt, mul, invert } from '../engine/math';

type InitMsg = Extract<MainToWorker, { type: 'init' }>;
type Post = (message: WorkerToMain) => void;
const FOV = 32 * Math.PI / 180;
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
    for (const name of ['uSDF', 'uInvVP', 'uEye', 'uRes', 'uText', 'uFloor', 'uTime', 'uSpot', 'uSpotDir', 'uPointer']) this.uniforms[name] = gl.getUniformLocation(program, name);
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
    perspective(this.projection, FOV, aspect, 0.1, 40); lookAt(this.view, this.eye, [0, 0.05, 0], [0, 1, 0]);
    mul(this.vp, this.projection, this.view); invert(this.inverse, this.vp);
    const nx = this.pointer.x * 2 - 1, ny = 1 - this.pointer.y * 2, m = this.inverse;
    const w = m[3]*nx + m[7]*ny + m[11] + m[15];
    const far = [(m[0]*nx + m[4]*ny + m[8] + m[12])/w, (m[1]*nx + m[5]*ny + m[9] + m[13])/w, (m[2]*nx + m[6]*ny + m[10] + m[14])/w];
    const rayScale = (0.9 - this.eye[2]) / (far[2] - this.eye[2]);
    this.pointer.intensity += ((this.time - this.pointer.lastMove < 1.2 ? 1 : 0.15) - this.pointer.intensity) * Math.min(1, dt * 4);
    const sx = 3.2 * Math.sin(this.time * 0.11), spot = [sx, 3.6, -5.2], target = [-sx * 0.25, r.floorY, 2.5];
    const length = Math.hypot(target[0]-spot[0], target[1]-spot[1], target[2]-spot[2]);
    const gl = this.gl, u = this.uniforms;
    gl.useProgram(this.program); gl.uniformMatrix4fv(u.uInvVP, false, this.inverse);
    gl.uniform3fv(u.uEye, this.eye); gl.uniform2f(u.uRes, this.canvas.width, this.canvas.height);
    gl.uniform4f(u.uText, r.halfW, r.halfH, 0.16, 0.035); gl.uniform1f(u.uFloor, r.floorY); gl.uniform1f(u.uTime, this.time);
    gl.uniform3fv(u.uSpot, spot); gl.uniform3f(u.uSpotDir, (target[0]-spot[0])/length, (target[1]-spot[1])/length, (target[2]-spot[2])/length);
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

const FRAGMENT = `#version 300 es
precision highp float;
precision highp int;
uniform sampler2D uSDF;
uniform mat4 uInvVP;
uniform vec3 uEye;
uniform vec2 uRes;
uniform vec4 uText;
uniform float uFloor, uTime;
uniform vec3 uSpot, uSpotDir;
uniform vec4 uPointer;
out vec4 fragColor;
const float PI=3.14159265359, MISS=1e9;
const vec3 SPOT_COLOR=vec3(90.0,81.0,70.2), ABSORB=vec3(0.34,0.17,0.10);
float ign(vec2 p) { return fract(52.9829189*fract(dot(p,vec2(0.06711056,0.00583715)))); }
float sdf2(vec2 p) {
  vec2 uv=p/uText.xy*0.5+0.5;
  float d=textureLod(uSDF,clamp(uv,0.0,1.0),0.0).r;
  float outside=length(max(abs(p)-uText.xy,0.0));
  return outside>0.0 ? max(d,0.0)+outside : d;
}
float sdText(vec3 p) {
  vec2 q=vec2(sdf2(p.xy)+uText.w,abs(p.z)-uText.z+uText.w);
  return min(max(q.x,q.y),0.0)+length(max(q,0.0))-uText.w;
}
vec3 normalAt(vec3 p) {
  vec2 e=vec2(0.0015,-0.0015);
  return normalize(e.xyy*sdText(p+e.xyy)+e.yyx*sdText(p+e.yyx)+e.yxy*sdText(p+e.yxy)+e.xxx*sdText(p+e.xxx));
}
vec2 boxHit(vec3 ro,vec3 rd) {
  vec3 safe=vec3(abs(rd.x)<1e-6?1e-6:rd.x,abs(rd.y)<1e-6?1e-6:rd.y,abs(rd.z)<1e-6?1e-6:rd.z);
  vec3 a=(-uText.xyz-ro)/safe,b=(uText.xyz-ro)/safe;
  vec3 lo=min(a,b),hi=max(a,b);
  return vec2(max(lo.x,max(lo.y,lo.z)),min(hi.x,min(hi.y,hi.z)));
}
float marchText(vec3 ro,vec3 rd,float maxT,int steps) {
  vec2 box=boxHit(ro,rd); if(box.y<max(box.x,0.0)) return MISS;
  float t=max(0.0,box.x),end=min(maxT,box.y);
  if(t>end) return MISS;
  for(int i=0;i<64;i++) {
    if(i>=steps) break;
    float d=sdText(ro+rd*t); if(d<0.0004+0.0006*t) return t;
    t+=d; if(t>end) break;
  }
  return MISS;
}
float shadow(vec3 p,vec3 light,int steps) {
  vec3 delta=light-p; float len=length(delta); vec3 rd=delta/max(len,0.001);
  vec2 box=boxHit(p,rd); float t=max(0.01,box.x),end=min(box.y,len),result=1.0;
  if(t>=end) return 1.0;
  for(int i=0;i<8;i++) {
    if(i>=steps) break;
    float d=sdText(p+rd*t); result=min(result,10.0*d/t);
    t+=clamp(d,0.02,0.35); if(result<0.002||t>end) break;
  }
  return clamp(result,0.0,1.0);
}
float rectMask(vec2 p,vec2 h,float soft) { vec2 a=1.0-smoothstep(h-vec2(soft),h,abs(p)); return a.x*a.y; }
vec3 env(vec3 ro,vec3 rd,bool direct) {
  float cosA=max(dot(rd,normalize(uSpot-ro)),0.0);
  vec3 c=SPOT_COLOR*(0.0035*pow(cosA,600.0));
  if(direct) return c;
  if(rd.y>0.0001) { vec3 q=ro+rd*((4.5-ro.y)/rd.y); c+=vec3(1.0,0.96,0.9)*2.4*rectMask(q.xz-vec2(0.0,0.5),vec2(5.0,1.0),0.6); }
  if(rd.z< -0.0001) { vec3 q=ro+rd*((-3.5-ro.z)/rd.z); c+=vec3(0.55,0.72,1.0)*2.0*rectMask(vec2(abs(q.x)-3.8,q.y-uFloor-1.6),vec2(0.12,1.6),0.1); }
  if(rd.z>0.0001) { vec3 q=ro+rd*((14.0-ro.z)/rd.z); c+=vec3(1.0,0.98,0.95)*1.1*rectMask(q.xy-vec2(0.0,2.2),vec2(9.0,0.35),0.3); }
  if(rd.z< -0.0001) { vec3 q=ro+rd*((-9.0-ro.z)/rd.z); float g=rectMask(q.xy-vec2(0.0,1.0),vec2(13.0,4.5),4.0); c+=mix(vec3(0.35,0.45,0.7),vec3(1.0,0.85,0.7),smoothstep(-3.0,5.0,q.y))*0.45*g; }
  if(rd.z>0.0001) { vec3 q=ro+rd*((30.0-ro.z)/rd.z); c+=vec3(0.9,0.93,1.0)*0.5*rectMask(q.xy-vec2(0.0,3.0),vec2(18.0,7.0),6.0)*smoothstep(-4.0,10.0,q.y); }
  return c+vec3(0.006,0.007,0.009)*(0.6+0.4*rd.y);
}
vec3 spotRadiance(vec3 p) {
  vec3 delta=uSpot-p;
  return SPOT_COLOR*smoothstep(cos(radians(26.0)),cos(radians(15.0)),dot(-normalize(delta),uSpotDir))/max(dot(delta,delta),1.0);
}
vec3 pointerRadiance(vec3 p) { vec3 d=uPointer.xyz-p; return vec3(0.55,0.75,1.0)*uPointer.w*0.5/max(dot(d,d),0.08); }
float ggx(vec3 n,vec3 v,vec3 l,float rough) {
  vec3 h=normalize(v+l); float nl=max(dot(n,l),0.0),nv=max(dot(n,v),0.0001),nh=max(dot(n,h),0.0);
  float a=rough*rough,a2=a*a,dd=nh*nh*(a2-1.0)+1.0,k=a*0.5;
  return a2/(PI*dd*dd)*nl/(4.0*(nl*(1.0-k)+k)*(nv*(1.0-k)+k));
}
float fresnel(float c,float f0) { return f0+(1.0-f0)*pow(1.0-clamp(c,0.0,1.0),5.0); }
float floorHit(vec3 ro,vec3 rd) { if(rd.y> -0.0001) return MISS; float t=(uFloor-ro.y)/rd.y; return t>0.0?t:MISS; }
float floorFade(vec3 p) { return exp(-max(length(p.xz)-5.0,0.0)*0.35); }
vec3 floorDirect(vec3 p,vec3 v) {
  vec3 n=vec3(0,1,0),l=normalize(uSpot-p),pl=normalize(uPointer.xyz-p);
  float sh=shadow(p+n*0.002,uSpot,8);
  vec3 trans=mix(vec3(0.3,0.38,0.44),vec3(1),sh)+vec3(1,0.94,0.84)*sh*(1.0-sh)*3.2;
  return spotRadiance(p)*trans*(0.012/PI*max(l.y,0.0)+0.05*ggx(n,v,l,0.3))
    +pointerRadiance(p)*(0.012/PI*max(pl.y,0.0)+0.05*ggx(n,v,pl,0.3));
}
vec3 glassLite(vec3 p,vec3 n,vec3 rd) {
  float f=fresnel(dot(n,-rd),0.04);
  return env(p,reflect(rd,n),false)*f+env(p,normalize(refract(rd,n,1.0/1.5)*0.5+rd*0.5),false)*exp(-ABSORB*0.4)*(1.0-f);
}
vec3 secondary(vec3 ro,vec3 rd,int steps) {
  float tf=floorHit(ro,rd),tt=steps>0?marchText(ro,rd,tf,steps):MISS;
  if(tt<tf) { vec3 p=ro+rd*tt; return glassLite(p,normalAt(p),rd); }
  if(tf<MISS) { vec3 p=ro+rd*tf; return floorDirect(p,-rd)*floorFade(p); }
  return env(ro,rd,false);
}
vec3 shadeGlass(vec3 p,vec3 n,vec3 rd) {
  float f=fresnel(dot(n,-rd),0.04);
  vec3 reflected=secondary(p+n*0.003,reflect(rd,n),0);
  vec3 inside=refract(rd,n,1.0/1.5),ro=p-n*0.004; float t=0.0;
  for(int i=0;i<16;i++) { float d=-sdText(ro+inside*t); if(d<0.0005) break; t+=max(d,0.004); }
  vec3 exitPoint=ro+inside*t,exitNormal=normalAt(exitPoint),outDir=refract(inside,-exitNormal,1.5);
  if(dot(outDir,outDir)<0.5) outDir=reflect(inside,-exitNormal);
  vec3 transmitted=secondary(exitPoint+exitNormal*0.004,outDir,0)*exp(-ABSORB*(t+0.05));
  vec3 c=reflected*f+transmitted*(1.0-f);
  c+=spotRadiance(p)*ggx(n,-rd,normalize(uSpot-p),0.05)*f*4.0;
  c+=pointerRadiance(p)*ggx(n,-rd,normalize(uPointer.xyz-p),0.07)*0.6;
  float edge=1.0-smoothstep(0.0,0.07,-sdf2(p.xy));
  float sweep=(fract(uTime*0.06)*2.0-1.0)*(uText.x+2.0);
  c+=vec3(1.0,0.92,0.8)*edge*exp(-pow((p.x-sweep)/0.45,2.0))*1.8*(0.35+0.65*max(n.z,0.0));
  return c;
}
float hg(float c,float g) { return (1.0-g*g)/(4.0*PI*pow(max(0.001,1.0+g*g-2.0*g*c),1.5)); }
vec4 fog(vec3 ro,vec3 rd,float distance) {
  vec3 accum=vec3(0); float trans=1.0,stepSize=min(distance,uEye.z+14.0)/16.0;
  float jitter=fract(ign(gl_FragCoord.xy)+uTime*0.618);
  for(int i=0;i<16;i++) {
    vec3 p=ro+rd*((float(i)+jitter)*stepSize);
    float density=0.035*(0.6+0.4*sin(p.x*0.5+uTime*0.03)*sin(p.z*0.5)*0.5+0.2)*exp(-max(p.y-uFloor,0.0)*0.25);
    vec3 l=normalize(uSpot-p),pl=normalize(uPointer.xyz-p);
    vec3 light=spotRadiance(p)*hg(dot(l,rd),0.55)*mix(0.38,1.0,shadow(p,uSpot,4))+pointerRadiance(p)*hg(dot(pl,rd),0.3);
    float sliceT=exp(-density*stepSize); accum+=trans*light*(1.0-sliceT); trans*=sliceT;
  }
  return vec4(accum,trans);
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
  vec3 rd=normalize(farPoint.xyz/farPoint.w-uEye);
  float tf=floorHit(uEye,rd),tt=marchText(uEye,rd,tf,64),t=min(tf,tt);
  vec3 color;
  if(tt<tf) { vec3 p=uEye+rd*tt; color=shadeGlass(p,normalAt(p),rd); }
  else if(tf<MISS) {
    vec3 p=uEye+rd*tf; color=(floorDirect(p,-rd)+secondary(p+vec3(0,0.002,0),reflect(rd,vec3(0,1,0)),24)*fresnel(-rd.y,0.05))*floorFade(p);
  } else color=env(uEye,rd,true);
  vec4 volume=fog(uEye,rd,t); color=color*volume.a+volume.rgb;
  color=agx(color); color*=1.0-0.25*pow(length((uv-0.5)*1.414),2.0);
  color=srgb(color); color+=(fract(ign(gl_FragCoord.xy)+uTime*0.618)-0.5)*(0.015+1.0/255.0);
  fragColor=vec4(clamp(color,0.0,1.0),1.0);
}`;
