import * as THREE from 'three';

// Venus mean radius; the Magellan DEM is relative to this datum.
const R = 6051800;
const DEG = Math.PI / 180;

// IAU feature coordinates, east-positive longitude.
const PLACES = [
  ['Maxwell Montes (highest point)', 65.2, 3.3],
  ['Cleopatra crater', 65.8, 7.1],
  ['Lakshmi Planum', 68.6, 339.3],
  ['Fortuna Tessera', 69.9, 45.1],
  ['Maat Mons', 0.5, 194.6],
  ['Sapas Mons', 8.5, 188.3],
  ['Ozza Mons', 4.5, 201.0],
  ['Gula Mons', 21.9, 359.1],
  ['Sif Mons', 22.0, 352.4],
  ['Theia Mons (Beta Regio)', 22.7, 281.0],
  ['Rhea Mons (Beta Regio)', 32.4, 282.2],
  ['Alpha Regio', -25.5, 1.0],
  ['Pancake domes, Seoritsu Farra', -30.0, 11.8],
  ['Ovda Regio', -2.8, 85.6],
  ['Thetis Regio', -11.4, 129.9],
  ['Artemis Corona', -35.0, 135.0],
  ['Diana Chasma', -14.5, 155.0],
  ['Baltis Vallis', 37.3, 161.4],
  ['Mead crater', 12.5, 57.2],
  ['Idunn Mons', -46.0, 214.5],
].map(([name, lat, lon]) => ({ name, lat: lat * DEG, lon: (((lon + 180) % 360) - 180) * DEG }));

// ---------------------------------------------------------------- data loading

async function loadHeight() {
  const meta = await (await fetch('data/height.json')).json();
  let buf = new Uint8Array(await (await fetch('data/height.bin.gz')).arrayBuffer());
  // Some static servers transparently decode .gz; only decompress if it's still gzip.
  if (buf[0] === 0x1f && buf[1] === 0x8b) {
    const stream = new Blob([buf]).stream().pipeThrough(new DecompressionStream('gzip'));
    buf = new Uint8Array(await new Response(stream).arrayBuffer());
  }
  const { width: W, height: H } = meta;
  const delta = new Int16Array(buf.buffer, buf.byteOffset, W * H);
  const data = new Float32Array(W * H);
  for (let y = 0; y < H; y++) {
    let acc = 0;
    for (let x = 0; x < W; x++) data[y * W + x] = acc += delta[y * W + x];
  }
  return { W, H, data };
}

// Half-float texture with a CPU-built mip chain (R16F can't be mipmapped by the GPU everywhere).
function heightTexture({ W, H, data }) {
  const mips = [];
  let w = W, h = H, cur = data;
  for (;;) {
    const half = new Uint16Array(w * h);
    for (let i = 0; i < cur.length; i++) half[i] = THREE.DataUtils.toHalfFloat(cur[i]);
    mips.push({ data: half, width: w, height: h });
    if (w === 1 && h === 1) break;
    const nw = Math.max(1, w >> 1), nh = Math.max(1, h >> 1), next = new Float32Array(nw * nh);
    for (let y = 0; y < nh; y++) for (let x = 0; x < nw; x++) {
      const x0 = Math.min(2 * x, w - 1), x1 = Math.min(2 * x + 1, w - 1);
      const y0 = Math.min(2 * y, h - 1), y1 = Math.min(2 * y + 1, h - 1);
      next[y * nw + x] = 0.25 * (cur[y0 * w + x0] + cur[y0 * w + x1] + cur[y1 * w + x0] + cur[y1 * w + x1]);
    }
    w = nw; h = nh; cur = next;
  }
  const tex = new THREE.DataTexture(mips[0].data, W, H, THREE.RedFormat, THREE.HalfFloatType);
  tex.mipmaps = mips;
  tex.generateMipmaps = false;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.needsUpdate = true;
  return tex;
}

function loadRadar(url, renderer) {
  return new THREE.TextureLoader().loadAsync(url).then((tex) => {
    tex.format = THREE.RedFormat;
    tex.flipY = false;
    tex.colorSpace = THREE.NoColorSpace;
    tex.wrapS = THREE.RepeatWrapping;
    tex.wrapT = THREE.ClampToEdgeWrapping;
    tex.minFilter = THREE.LinearMipmapLinearFilter;
    tex.anisotropy = renderer.capabilities.getMaxAnisotropy();
    return tex;
  });
}

// Bilinear terrain lookup on the CPU (for collision and the HUD), metres above datum.
function makeSampler({ W, H, data }) {
  return (lat, lon) => {
    const x = ((lon / (2 * Math.PI) + 0.5) * W - 0.5 + W) % W;
    const y = Math.min(Math.max((0.5 - lat / Math.PI) * H - 0.5, 0), H - 1);
    const x0 = Math.floor(x), y0 = Math.floor(y), fx = x - x0, fy = y - y0;
    const x1 = (x0 + 1) % W, y1 = Math.min(y0 + 1, H - 1);
    const a = data[y0 * W + x0], b = data[y0 * W + x1], c = data[y1 * W + x0], d = data[y1 * W + x1];
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy;
  };
}

// ---------------------------------------------------------------- terrain shader
//
// The terrain is a polar grid centred under the camera and rebuilt in the shader every
// frame. Each vertex is an (east, north) offset on the tangent plane; the shader wraps it
// onto the sphere, samples the DEM, and outputs a camera-relative position computed
// without large-number cancellation, so float32 stays precise from 10 m to 5000 km up.

// Phones get a lighter terrain mesh.
const COARSE = matchMedia('(pointer: coarse)').matches;
const RINGS = COARSE ? 200 : 320, SEGMENTS = COARSE ? 320 : 512;

function gridGeometry() {
  const pos = new Float32Array((RINGS + 1) * (SEGMENTS + 1) * 3);
  let k = 0;
  for (let i = 0; i <= RINGS; i++) for (let j = 0; j <= SEGMENTS; j++) {
    pos[k++] = i / RINGS; pos[k++] = (j / SEGMENTS) * 2 * Math.PI; pos[k++] = 0;
  }
  const idx = [];
  for (let i = 0; i < RINGS; i++) for (let j = 0; j < SEGMENTS; j++) {
    const a = i * (SEGMENTS + 1) + j, b = a + SEGMENTS + 1;
    idx.push(a, b, a + 1, a + 1, b, b + 1);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setIndex(idx);
  return g;
}

const common = /* glsl */`
  #include <common>
  uniform sampler2D uHeight, uRadar;
  uniform vec3 uC, uE, uN;          // centre point and its east/north axes (ECEF, unit)
  uniform float uR, uExag, uCamAlt;
  vec2 dirToUv(vec3 d) {
    return vec2(atan(d.y, d.x) / (2.0 * PI) + 0.5, 0.5 - asin(clamp(d.z, -1.0, 1.0)) / PI);
  }
  vec3 offsetToDir(vec2 en) {
    float s = length(en);
    vec2 t = s > 0.0 ? en / s : vec2(0.0);
    float a = s / uR;
    return uC * cos(a) + (uE * t.x + uN * t.y) * sin(a);
  }
`;

const terrainVert = /* glsl */`
  ${common}
  #include <logdepthbuf_pars_vertex>
  uniform float uSMax, uL, uTexelM;
  varying vec2 vEN;
  varying vec3 vRel;
  varying float vH;
  void main() {
    float t = position.x, az = position.y;
    float eL = exp(t * uL), denom = exp(uL) - 1.0;
    float s = uSMax * (eL - 1.0) / denom;
    float spacing = uSMax * uL * eL / denom / ${RINGS}.0;
    vec2 dir2 = vec2(sin(az), cos(az));
    vec2 en = dir2 * s;
    vec3 d = offsetToDir(en);
    float lod = max(0.0, log2(spacing / uTexelM));
    float h = textureLod(uHeight, dirToUv(d), lod).r * uExag;
    float a = s / uR;
    float sa = sin(a), ha = sin(0.5 * a);
    // Local frame: x east, y up, z south. Vertical term avoids (R+h)cos(a) - R cancellation.
    vec3 local = vec3((uR + h) * sa * dir2.x, h * cos(a) - 2.0 * uR * ha * ha - uCamAlt, -(uR + h) * sa * dir2.y);
    vEN = en; vRel = local; vH = h;
    gl_Position = projectionMatrix * viewMatrix * vec4(local, 1.0);
    #include <logdepthbuf_vertex>
  }
`;

const terrainFrag = /* glsl */`
  ${common}
  #include <logdepthbuf_pars_fragment>
  uniform float uDetail, uFogRho, uFogH;
  uniform vec2 uHeightSize;
  uniform vec3 uFogColor;
  varying vec2 vEN;
  varying vec3 vRel;
  varying float vH;

  // Screen-space UV derivatives that ignore the 0/1 longitude wrap.
  vec4 uvGrad(vec2 uv) {
    vec2 dx = dFdx(uv), dy = dFdy(uv);
    float wx = dFdx(fract(uv.x + 0.5)), wy = dFdy(fract(uv.x + 0.5));
    if (abs(wx) < abs(dx.x)) dx.x = wx;
    if (abs(wy) < abs(dy.x)) dy.x = wy;
    return vec4(dx, dy);
  }

  void main() {
    #include <logdepthbuf_fragment>
    vec3 d = offsetToDir(vEN);
    vec2 uv = dirToUv(d);
    vec4 g = uvGrad(uv);
    float lat = asin(clamp(d.z, -1.0, 1.0));

    // Terrain normal from central differences of the DEM, at a step matched to screen size.
    vec2 texel = 1.0 / uHeightSize;
    vec2 stp = max(texel, vec2(max(abs(g.x), abs(g.z)), max(abs(g.y), abs(g.w))));
    float lod = max(0.0, log2(max(stp.x / texel.x, stp.y / texel.y)));
    float hE = textureLod(uHeight, uv + vec2(stp.x, 0.0), lod).r, hW = textureLod(uHeight, uv - vec2(stp.x, 0.0), lod).r;
    float hN = textureLod(uHeight, uv - vec2(0.0, stp.y), lod).r, hS = textureLod(uHeight, uv + vec2(0.0, stp.y), lod).r;
    float mx = 2.0 * stp.x * 2.0 * PI * uR * max(cos(lat), 0.02), my = 2.0 * stp.y * PI * uR;
    vec3 n = normalize(vec3(-(hE - hW) / mx * uExag, -(hN - hS) / my * uExag, 1.0));

    float radar = textureGrad(uRadar, uv, g.xy, g.zw).r;
    float dist = length(vRel);
    if (uDetail > 0.0) {
      // Synthetic sub-pixel detail so low passes don't look like a magnified blur: two octaves of
      // the radar mosaic itself, tiled and faded in only where the real data is magnified.
      float d1 = textureGrad(uRadar, uv * 24.0, g.xy * 24.0, g.zw * 24.0).r;
      float d2 = textureGrad(uRadar, uv * 384.0, g.xy * 384.0, g.zw * 384.0).r;
      float w1 = uDetail * smoothstep(250000.0, 30000.0, dist);
      float w2 = uDetail * smoothstep(30000.0, 2000.0, dist);
      radar *= mix(1.0, 0.55 + 0.9 * d1, w1 * 0.45) * mix(1.0, 0.55 + 0.9 * d2, w2 * 0.45);
    }

    // Diffuse light under a thick overcast: mostly from above, slightly from the north-west.
    vec3 L = normalize(vec3(-0.35, 0.35, 0.85));
    float shade = 0.25 + 0.9 * max(dot(n, L), 0.0);
    vec3 albedo = mix(vec3(0.07, 0.025, 0.01), vec3(1.05, 0.80, 0.50), pow(clamp(radar, 0.0, 1.0), 1.4) * 1.25);
    vec3 col = albedo * shade;

    // Exponential-atmosphere fog integrated along the ray between camera and point.
    float zc = uCamAlt, zp = vH, dz = zp - zc;
    float od = abs(dz) > 1.0
      ? uFogRho * uFogH * dist * (exp(-zc / uFogH) - exp(-zp / uFogH)) / dz
      : uFogRho * dist * exp(-zc / uFogH);
    col = mix(uFogColor, col, exp(-od));
    gl_FragColor = vec4(col, 1.0);
  }
`;

const skyVert = /* glsl */`
  varying vec3 vRay;
  uniform mat4 uInvViewProj;
  void main() {
    vec4 p = uInvViewProj * vec4(position.xy, 1.0, 1.0);
    vRay = p.xyz / p.w;
    gl_Position = vec4(position.xy, 0.0, 1.0);
  }
`;

const skyFrag = /* glsl */`
  varying vec3 vRay;
  uniform float uCamAlt, uR, uFogH;
  uniform vec3 uFogColor;
  void main() {
    vec3 r = normalize(vRay);
    float e = asin(clamp(r.y, -1.0, 1.0));
    float dip = acos(clamp(uR / (uR + max(uCamAlt, 0.0)), -1.0, 1.0));
    float air = exp(-max(uCamAlt, 0.0) / (uFogH * 4.0));
    vec3 surfaceSky = mix(uFogColor, vec3(0.42, 0.22, 0.09), smoothstep(0.0, 1.2, e));
    float limb = exp(-max(e + dip, 0.0) / 0.04);
    vec3 space = uFogColor * limb * 0.9;
    gl_FragColor = vec4(mix(space, surfaceSky, clamp(air * 1.5, 0.0, 1.0)), 1.0);
  }
`;

// ---------------------------------------------------------------- flight model

class Plane {
  constructor() {
    this.pos = new THREE.Vector3();  // ECEF metres (JS doubles)
    this.fwd = new THREE.Vector3();
    this.up = new THREE.Vector3();
    this.throttle = 3.7;             // log10(m/s)
    this.rates = { pitch: 0, roll: 0, yaw: 0 };
  }
  get speed() { return 10 ** this.throttle; }
  frame() {
    const p = this.pos, r = p.length();
    const lat = Math.asin(p.z / r), lon = Math.atan2(p.y, p.x);
    const E = new THREE.Vector3(-Math.sin(lon), Math.cos(lon), 0);
    const N = new THREE.Vector3(-Math.sin(lat) * Math.cos(lon), -Math.sin(lat) * Math.sin(lon), Math.cos(lat));
    const U = p.clone().divideScalar(r);
    return { lat, lon, E, N, U, alt: r - R };
  }
  place(lat, lon, alt, heading) {
    this.pos.set(Math.cos(lat) * Math.cos(lon), Math.cos(lat) * Math.sin(lon), Math.sin(lat)).multiplyScalar(R + alt);
    const { E, N, U } = this.frame();
    this.fwd.copy(N).multiplyScalar(Math.cos(heading)).addScaledVector(E, Math.sin(heading));
    this.up.copy(U);
  }
  rotate(axis, angle) {
    const q = new THREE.Quaternion().setFromAxisAngle(axis, angle);
    this.fwd.applyQuaternion(q); this.up.applyQuaternion(q);
  }
}

// ---------------------------------------------------------------- main

async function main() {
  const renderer = new THREE.WebGLRenderer({ antialias: true, logarithmicDepthBuffer: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, COARSE ? 1.5 : 2));
  renderer.setSize(innerWidth, innerHeight);
  document.body.prepend(renderer.domElement);

  const [hm, radar8k] = await Promise.all([loadHeight(), loadRadar('data/radar_8k.jpg', renderer)]);
  const heightAt = makeSampler(hm);
  document.getElementById('mapimg').src = 'data/radar_8k.jpg';

  const fogColor = new THREE.Color(0.80, 0.52, 0.27);
  const uniforms = {
    uHeight: { value: heightTexture(hm) }, uRadar: { value: radar8k },
    uHeightSize: { value: new THREE.Vector2(hm.W, hm.H) },
    uC: { value: new THREE.Vector3() }, uE: { value: new THREE.Vector3() }, uN: { value: new THREE.Vector3() },
    uR: { value: R }, uExag: { value: 3 }, uCamAlt: { value: 0 },
    uSMax: { value: 1e6 }, uL: { value: 8 }, uTexelM: { value: (2 * Math.PI * R) / hm.W },
    uDetail: { value: 1 }, uFogRho: { value: 3e-6 }, uFogH: { value: 15000 }, uFogColor: { value: fogColor },
    uInvViewProj: { value: new THREE.Matrix4() },
  };

  const scene = new THREE.Scene();
  const terrain = new THREE.Mesh(gridGeometry(), new THREE.ShaderMaterial({
    uniforms, vertexShader: terrainVert, fragmentShader: terrainFrag, side: THREE.DoubleSide,
  }));
  terrain.frustumCulled = false;
  scene.add(terrain);

  const sky = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.ShaderMaterial({
    uniforms, vertexShader: skyVert, fragmentShader: skyFrag, depthTest: false, depthWrite: false,
  }));
  sky.frustumCulled = false;
  sky.renderOrder = -1;
  scene.add(sky);

  const camera = new THREE.PerspectiveCamera(65, innerWidth / innerHeight, 0.5, 2e7);
  addEventListener('resize', () => {
    renderer.setSize(innerWidth, innerHeight);
    camera.aspect = innerWidth / innerHeight;
    camera.updateProjectionMatrix();
  });

  // Upgrade to the 16k radar mosaic in the background when the GPU can hold it.
  if (renderer.capabilities.maxTextureSize >= 16384) {
    loadRadar('data/radar_16k.jpg', renderer).then((t) => { uniforms.uRadar.value = t; radar8k.dispose(); }).catch(() => {});
  }

  const plane = new Plane();
  const teleport = (p, heading = 0) => {
    // Arrive 600 km south of the feature, ~100 km up and nosed down. The radar mosaic is
    // ~2 km/px, so it reads best from tens to hundreds of km up (low passes get blurry).
    const lat = p.lat - 600000 / R;
    const top = Math.max(heightAt(lat, p.lon), heightAt(p.lat, p.lon), 0);
    plane.place(lat, p.lon, (top + 100000) * uniforms.uExag.value, heading);
    plane.rotate(new THREE.Vector3().crossVectors(plane.fwd, plane.up).normalize(), -14 * DEG);
  };
  // Shareable position in the URL hash: #lat,lon,alt(km, real),heading,pitch (degrees).
  const fromHash = () => {
    const v = location.hash.slice(1).split(',').map(Number);
    if (v.length < 3 || v.some(isNaN)) return false;
    const [lat, lon, alt, hdg = 0, pitch = 0] = v;
    plane.place(lat * DEG, lon * DEG, alt * 1000 * uniforms.uExag.value, hdg * DEG);
    const right = new THREE.Vector3().crossVectors(plane.fwd, plane.up).normalize();
    plane.rotate(right, pitch * DEG);
    return true;
  };
  if (!fromHash()) teleport(PLACES[0]);

  // UI
  const $ = (id) => document.getElementById(id);
  const sel = $('goto');
  PLACES.forEach((p, i) => sel.add(new Option(p.name, i)));
  sel.onchange = () => { if (sel.value !== '') teleport(PLACES[+sel.value]); sel.value = ''; sel.blur(); };
  $('map').onclick = (ev) => {
    const r = $('mapimg').getBoundingClientRect();
    const lon = ((ev.clientX - r.left) / r.width - 0.5) * 2 * Math.PI;
    const lat = (0.5 - (ev.clientY - r.top) / r.height) * Math.PI;
    plane.place(lat, lon, (Math.max(heightAt(lat, lon), 0) + 100000) * uniforms.uExag.value, 0);
  };

  const setExag = (v) => {
    // Rescale our altitude too, so the real (unexaggerated) altitude stays put.
    const e0 = uniforms.uExag.value, e1 = THREE.MathUtils.clamp(v, 1, 20);
    plane.pos.setLength(R + ((plane.pos.length() - R) * e1) / e0);
    uniforms.uExag.value = e1;
    $('exagbtn').textContent = `×${e1}`;
  };
  const EXAG_STEPS = [1, 2, 3, 5, 8];
  $('exagbtn').onclick = () => setExag(EXAG_STEPS.find((s) => s > uniforms.uExag.value) ?? 1);

  const keys = new Set();
  let invert = false, paused = false;
  addEventListener('keydown', (e) => {
    keys.add(e.code);
    if (e.code === 'BracketRight') setExag(uniforms.uExag.value + 1);
    if (e.code === 'BracketLeft') setExag(uniforms.uExag.value - 1);
    if (e.code === 'KeyT') uniforms.uDetail.value = 1 - uniforms.uDetail.value;
    if (e.code === 'KeyI') { invert = !invert; $('invbtn').classList.toggle('on', invert); }
    if (e.code === 'Escape') showAbout(about.hidden);
    if (e.code === 'KeyP') paused = !paused;
    if (e.code === 'KeyH') $('help').style.display = $('help').style.display === 'none' ? '' : 'none';
    if (e.code.startsWith('Arrow') || e.code === 'Space') e.preventDefault();
  });
  addEventListener('keyup', (e) => keys.delete(e.code));
  addEventListener('blur', () => keys.clear());
  addEventListener('wheel', (e) => {
    plane.throttle = THREE.MathUtils.clamp(plane.throttle - Math.sign(e.deltaY) * 0.1, 1, 5);
  }, { passive: true });

  // Pointer flying: drag to use a virtual stick (mouse anywhere; touch on the left half),
  // and on touch, drag the right half up/down for throttle.
  const stick = { x: 0, y: 0, id: null, ox: 0, oy: 0, r: 1 };
  let throttleDrag = null;
  const canvas = renderer.domElement;
  const ring = $('stick'), knob = $('knob');
  canvas.addEventListener('pointerdown', (e) => {
    if (e.pointerType === 'mouse' && e.button !== 0) return;
    const touch = e.pointerType !== 'mouse';
    if (touch && e.clientX > innerWidth / 2) {
      if (throttleDrag === null) throttleDrag = { id: e.pointerId, y: e.clientY, t: plane.throttle };
    } else if (stick.id === null) {
      Object.assign(stick, { id: e.pointerId, ox: e.clientX, oy: e.clientY, r: touch ? 60 : 110 });
      ring.style.cssText = `display:block;left:${e.clientX}px;top:${e.clientY}px;width:${2 * stick.r}px;height:${2 * stick.r}px`;
      knob.style.transform = 'translate(-50%, -50%)';
    } else return;
    canvas.setPointerCapture(e.pointerId);
    e.preventDefault();
  });
  canvas.addEventListener('pointermove', (e) => {
    if (e.pointerId === stick.id) {
      let dx = (e.clientX - stick.ox) / stick.r, dy = (e.clientY - stick.oy) / stick.r;
      const len = Math.hypot(dx, dy);
      if (len > 1) { dx /= len; dy /= len; }
      stick.x = dx; stick.y = dy;
      knob.style.transform = `translate(calc(-50% + ${dx * stick.r}px), calc(-50% + ${dy * stick.r}px))`;
    } else if (throttleDrag && e.pointerId === throttleDrag.id) {
      plane.throttle = THREE.MathUtils.clamp(throttleDrag.t + (throttleDrag.y - e.clientY) / 150, 1, 5);
    }
  });
  const release = (e) => {
    if (e.pointerId === stick.id) { Object.assign(stick, { x: 0, y: 0, id: null }); ring.style.display = 'none'; }
    if (throttleDrag && e.pointerId === throttleDrag.id) throttleDrag = null;
  };
  canvas.addEventListener('pointerup', release);
  canvas.addEventListener('pointercancel', release);

  // About panel: shown on first load; flight is paused while it's open.
  const about = $('about');
  const showAbout = (open) => { about.hidden = !open; };
  $('aboutbtn').onclick = () => showAbout(true);
  $('fly').onclick = () => showAbout(false);
  $('invbtn').onclick = () => { invert = !invert; $('invbtn').classList.toggle('on', invert); };
  if (COARSE) document.body.classList.add('touch');

  $('loading').remove();

  let last = performance.now(), hudT = 0, warnT = 0, hashT = 1;
  const tmp = new THREE.Vector3(), right = new THREE.Vector3();
  const basis = new THREE.Matrix4();

  function step(dt) {
    const k = (c) => (keys.has(c) ? 1 : 0);
    // Squared stick response gives fine control near centre. Drag down = pull back = nose up.
    const sq = (v) => v * Math.abs(v);
    const clamp1 = (v) => Math.max(-1, Math.min(1, v));
    const pitchIn = clamp1(k('KeyS') + k('ArrowDown') - k('KeyW') - k('ArrowUp') + sq(stick.y)) * (invert ? -1 : 1);
    const rollIn = clamp1(k('KeyD') + k('ArrowRight') - k('KeyA') - k('ArrowLeft') + sq(stick.x));
    const yawIn = k('KeyE') - k('KeyQ');
    if (keys.has('ShiftLeft') || keys.has('ShiftRight')) plane.throttle = Math.min(plane.throttle + dt * 0.6, 5);
    if (keys.has('ControlLeft') || keys.has('ControlRight')) plane.throttle = Math.max(plane.throttle - dt * 0.6, 1);

    // Smoothed control rates (rad/s).
    const r = plane.rates, a = 1 - Math.exp(-dt * 6);
    r.pitch += (pitchIn * 1.1 - r.pitch) * a;
    r.roll += (rollIn * 2.0 - r.roll) * a;
    r.yaw += (yawIn * 0.5 - r.yaw) * a;

    const f = plane.frame();
    right.crossVectors(plane.fwd, plane.up).normalize();
    // A gentle coordinated turn: banking also yaws the nose toward the low wing.
    const bank = -right.dot(f.U);
    plane.rotate(right, r.pitch * dt);
    plane.rotate(plane.fwd, r.roll * dt);
    plane.rotate(plane.up, -(r.yaw + bank * 0.6) * dt);

    plane.pos.addScaledVector(plane.fwd, plane.speed * dt);

    // Re-orthonormalise.
    plane.fwd.normalize();
    plane.up.addScaledVector(plane.fwd, -plane.up.dot(plane.fwd)).normalize();

    // Terrain collision: clamp to the ground and level the nose out.
    const g = plane.frame();
    const ground = heightAt(g.lat, g.lon) * uniforms.uExag.value;
    const minAlt = ground + 3;
    if (g.alt < minAlt) {
      plane.pos.copy(g.U).multiplyScalar(R + minAlt);
      const down = plane.fwd.dot(g.U);
      if (down < 0) {
        plane.fwd.addScaledVector(g.U, -down).normalize();
        plane.up.addScaledVector(plane.fwd, -plane.up.dot(plane.fwd)).normalize();
      }
      warnT = 1.2;
    }
    if (g.alt > 1.5e7) plane.pos.copy(g.U).multiplyScalar(R + 1.5e7);
  }

  function render() {
    const f = plane.frame();
    uniforms.uC.value.copy(f.U); uniforms.uE.value.copy(f.E); uniforms.uN.value.copy(f.N);
    uniforms.uCamAlt.value = f.alt;
    // The atmosphere scales with the terrain so haze sits at the same real altitudes.
    uniforms.uFogH.value = 15000 * uniforms.uExag.value;

    const ground = heightAt(f.lat, f.lon) * uniforms.uExag.value;
    const agl = Math.max(f.alt - ground, 1);
    // Grid extent: past the horizon far enough to see 11 km peaks (times exaggeration) beyond it.
    const peak = 11500 * uniforms.uExag.value;
    const horizon = Math.sqrt(2 * R * Math.max(f.alt, 1)) + Math.sqrt(2 * R * peak);
    const sMax = Math.min(horizon, 1.45 * R);
    const first = THREE.MathUtils.clamp(agl * 0.12, 4, sMax / RINGS);
    // Solve for the growth rate L so the first ring sits at `first`.
    let lo = 1e-4, hi = 40;
    for (let i = 0; i < 50; i++) {
      const L = 0.5 * (lo + hi);
      const s1 = (sMax * (Math.exp(L / RINGS) - 1)) / (Math.exp(L) - 1);
      if (s1 > first) lo = L; else hi = L;
    }
    uniforms.uSMax.value = sMax;
    uniforms.uL.value = 0.5 * (lo + hi);

    // Camera at the origin of the local frame (x east, y up, z south).
    const toLocal = (v, out) => out.set(v.dot(f.E), v.dot(f.U), -v.dot(f.N));
    const fw = toLocal(plane.fwd, new THREE.Vector3());
    const up = toLocal(plane.up, new THREE.Vector3());
    const rt = new THREE.Vector3().crossVectors(fw, up);
    basis.makeBasis(rt, up, fw.clone().negate());
    camera.quaternion.setFromRotationMatrix(basis);
    camera.position.set(0, 0, 0);
    camera.near = THREE.MathUtils.clamp(agl * 0.05, 0.3, 1000);
    camera.far = sMax * 1.5 + f.alt * 2;
    camera.updateProjectionMatrix();
    camera.updateMatrixWorld();
    uniforms.uInvViewProj.value.multiplyMatrices(camera.matrixWorld, camera.projectionMatrixInverse);

    renderer.render(scene, camera);
    return { f, ground, agl };
  }

  const fmtAlt = (m) => (Math.abs(m) >= 10000 ? `${(m / 1000).toFixed(1)} km` : `${Math.round(m)} m`);
  const fmtDeg = (r, p, n) => `${Math.abs(r / DEG).toFixed(2)}°${r >= 0 ? p : n}`;

  function hud({ f, ground, agl }) {
    const exag = uniforms.uExag.value;
    const hdg = (Math.atan2(plane.fwd.dot(f.E), plane.fwd.dot(f.N)) / DEG + 360) % 360;
    $('pos').textContent = `${fmtDeg(f.lat, 'N', 'S')} ${((f.lon / DEG + 360) % 360).toFixed(2)}°E`;
    $('hdg').textContent = `${hdg.toFixed(0).padStart(3, '0')}°`;
    // Altitudes are shown in real (unexaggerated) metres.
    $('alt').textContent = fmtAlt(f.alt / exag);
    $('agl').textContent = fmtAlt(agl / exag);
    $('elev').textContent = fmtAlt(ground / exag);
    const v = plane.speed;
    $('spd').textContent = v >= 1000 ? `${(v / 1000).toFixed(1)} km/s` : `${Math.round(v)} m/s`;
    $('exag').textContent = `×${exag}`;
    $('thrfill').style.height = `${((plane.throttle - 1) / 4) * 100}%`;
    $('thrlabel').textContent = $('spd').textContent;

    let best = null, bestD = Infinity;
    for (const p of PLACES) {
      const c = Math.sin(f.lat) * Math.sin(p.lat) + Math.cos(f.lat) * Math.cos(p.lat) * Math.cos(f.lon - p.lon);
      const d = Math.acos(Math.min(1, Math.max(-1, c))) * R;
      if (d < bestD) { bestD = d; best = p; }
    }
    $('place').textContent = bestD < 800000 ? `${best.name} · ${Math.round(bestD / 1000)} km` : '';

    const mk = $('marker');
    mk.setAttribute('transform', `translate(${(f.lon / DEG + 180).toFixed(1)} ${(90 - f.lat / DEG).toFixed(1)}) rotate(${hdg.toFixed(0)})`);
  }

  function loop(now) {
    const dt = Math.min((now - last) / 1000, 0.1);
    last = now;
    if (!paused && about.hidden) step(dt);
    const state = render();
    warnT -= dt;
    $('warn').style.display = warnT > 0 ? 'block' : 'none';
    if ((hudT -= dt) <= 0) { hud(state); hudT = 0.1; }
    if ((hashT -= dt) <= 0) {
      const { f } = state, pitch = Math.asin(plane.fwd.dot(f.U)) / DEG;
      const hdg = (Math.atan2(plane.fwd.dot(f.E), plane.fwd.dot(f.N)) / DEG + 360) % 360;
      const v = [f.lat / DEG, ((f.lon / DEG + 360) % 360), f.alt / uniforms.uExag.value / 1000, hdg, pitch];
      history.replaceState(null, '', '#' + v.map((x) => x.toFixed(3)).join(','));
      hashT = 1;
    }
    requestAnimationFrame(loop);
  }
  requestAnimationFrame(loop);
}

main().catch((e) => {
  console.error(e);
  document.getElementById('loading').textContent = `Failed to load: ${e.message}. Serve this folder over HTTP (see README).`;
});
