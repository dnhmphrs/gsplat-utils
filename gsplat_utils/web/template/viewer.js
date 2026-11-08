// viewer.js — WebGPU Gaussian splat viewer with named poses and animated transitions.
import { loadModel } from './splat-loader.js';
import {
  Transition, DEFAULTS, EFFECT_IDS, v3, lookRotation, cameraAxes, quatFromAxisAngle, quatMultiply,
  quatNormalize, quatRotate, quatFromMatrix, matrixFromQuat, normalizePose, horizontalBasis, clamp,
} from './transitions.js';
import { UI } from './ui.js';
import { SHADER } from './shaders.js';
import { Camera } from './camera.js';
export { Camera };


// ---------------------------------------------------------------------------
// renderer
// ---------------------------------------------------------------------------
export class Renderer {
  static async create(canvas) {
    if (!navigator.gpu) throw new Error('WebGPU is not available in this browser.');
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('No WebGPU adapter found (try Chrome/Edge 113+, Safari 18+ or recent Firefox).');
    const wanted = {};
    for (const k of ['maxStorageBufferBindingSize', 'maxBufferSize']) if (adapter.limits[k]) wanted[k] = adapter.limits[k];
    const device = await adapter.requestDevice({ requiredLimits: wanted });
    return new Renderer(canvas, adapter, device);
  }
  constructor(canvas, adapter, device) {
    this.canvas = canvas; this.adapter = adapter; this.device = device;
    this.context = canvas.getContext('webgpu');
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({ device, format: this.format, alphaMode: 'opaque' });
    this.count = 0;
    this.background = [0.06, 0.06, 0.07, 1];
    this.uniformData = new ArrayBuffer(128);
    this.uf = new Float32Array(this.uniformData);
    this.uu = new Uint32Array(this.uniformData);
    this.uniformBuffer = device.createBuffer({ size: 128, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const module = device.createShaderModule({ code: SHADER });
    this.bgl = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      ],
    });
    this.pipeline = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.bgl] }),
      vertex: { module, entryPoint: 'vs_main' },
      fragment: {
        module, entryPoint: 'fs_main',
        targets: [{
          format: this.format,
          blend: {
            color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
          },
        }],
      },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
    });
    this.bindGroup = null;
    this.splatBuffer = null; this.colorBuffer = null; this.orderBuffer = null;
    this.worker = null; this.sortInFlight = false; this.sortPending = null; this.lastSortView = null;
    this.onSorted = null;
    this.spareIndexBuffer = null;
  }

  setModel(model) {
    const device = this.device;
    const limit = device.limits.maxStorageBufferBindingSize || 134217728;
    let n = model.count;
    const maxN = Math.floor(limit / 48);
    if (n > maxN) { console.warn(`model has ${n} splats, device limit allows ${maxN}; truncating`); n = maxN; }
    this.count = n;
    const data = new Float32Array(n * 12);
    for (let i = 0; i < n; i++) {
      const o = i * 12;
      data[o] = model.positions[i * 3]; data[o + 1] = model.positions[i * 3 + 1]; data[o + 2] = model.positions[i * 3 + 2];
      data[o + 4] = model.scales[i * 3]; data[o + 5] = model.scales[i * 3 + 1]; data[o + 6] = model.scales[i * 3 + 2];
      data[o + 8] = model.quats[i * 4]; data[o + 9] = model.quats[i * 4 + 1]; data[o + 10] = model.quats[i * 4 + 2]; data[o + 11] = model.quats[i * 4 + 3];
    }
    const colors = new Uint32Array(model.colors.buffer, model.colors.byteOffset, n);
    for (const b of [this.splatBuffer, this.colorBuffer, this.orderBuffer]) if (b) b.destroy();
    this.splatBuffer = device.createBuffer({ size: Math.max(16, data.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.colorBuffer = device.createBuffer({ size: Math.max(16, n * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.orderBuffer = device.createBuffer({ size: Math.max(16, n * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(this.splatBuffer, 0, data);
    device.queue.writeBuffer(this.colorBuffer, 0, colors);
    const identity = new Uint32Array(n);
    for (let i = 0; i < n; i++) identity[i] = i;
    device.queue.writeBuffer(this.orderBuffer, 0, identity);
    this.bindGroup = device.createBindGroup({
      layout: this.bgl,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: this.splatBuffer } },
        { binding: 2, resource: { buffer: this.colorBuffer } },
        { binding: 3, resource: { buffer: this.orderBuffer } },
      ],
    });
    // sorting worker owns a copy of the positions
    if (this.worker) this.worker.terminate();
    this.worker = new Worker(new URL('./sort-worker.js', import.meta.url));
    const positions = new Float32Array(model.positions.subarray(0, n * 3));
    this.workerReady = false;
    this.worker.onmessage = (e) => {
      const m = e.data;
      if (m.type === 'ready') { this.workerReady = true; if (this.sortPending) { const v = this.sortPending; this.sortPending = null; this.requestSort(v, true); } return; }
      if (m.type === 'sorted') {
        this.device.queue.writeBuffer(this.orderBuffer, 0, m.indices);
        this.spareIndexBuffer = m.indices.buffer;
        this.sortInFlight = false;
        this.lastSortView = m.view;
        if (this.onSorted) this.onSorted();
        if (this.sortPending) { const v = this.sortPending; this.sortPending = null; this.requestSort(v, true); }
      }
    };
    this.worker.postMessage({ type: 'init', positions }, [positions.buffer]);
    this.lastSortView = null;
  }

  /** Ask for a re-sort if the view moved enough (or force). */
  requestSort(view, force = false) {
    if (!this.worker || this.count === 0) return;
    if (!force && this.lastSortView) {
      const a = this.lastSortView, b = view;
      const dot = a[2] * b[2] + a[6] * b[6] + a[10] * b[10];
      const dp = Math.hypot(a[14] - b[14], a[13] - b[13], a[12] - b[12]);
      if (dot > 0.999 && dp < 0.01) return;
    }
    if (this.sortInFlight || !this.workerReady) { this.sortPending = view; return; }
    this.sortInFlight = true;
    const msg = { type: 'sort', view: new Float32Array(view) };
    const transfer = [msg.view.buffer];
    if (this.spareIndexBuffer) { msg.buffer = this.spareIndexBuffer; transfer.push(this.spareIndexBuffer); this.spareIndexBuffer = null; }
    this.worker.postMessage(msg, transfer);
  }

  resize(dprCap = 2) {
    const dpr = Math.min(window.devicePixelRatio || 1, dprCap);
    const w = Math.max(1, Math.floor(this.canvas.clientWidth * dpr));
    const h = Math.max(1, Math.floor(this.canvas.clientHeight * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) { this.canvas.width = w; this.canvas.height = h; return true; }
    return false;
  }

  render(camera, fx = { effectId: 0, tEffect: 0, strength: 1 }, sizeScale = 1) {
    const view = camera.viewMatrix();
    const W = this.canvas.width, H = this.canvas.height;
    const f = camera.focalPx(H);
    this.uf.set(view, 0);
    this.uf[16] = f; this.uf[17] = f;
    this.uf[18] = W; this.uf[19] = H;
    this.uf[20] = camera.position[0]; this.uf[21] = camera.position[1]; this.uf[22] = camera.position[2];
    this.uf[23] = performance.now() / 1000;
    this.uu[24] = fx.effectId | 0;
    this.uf[25] = fx.tEffect || 0;
    this.uf[26] = fx.strength ?? 1;
    this.uf[27] = sizeScale;
    this.device.queue.writeBuffer(this.uniformBuffer, 0, this.uniformData);
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: this.context.getCurrentTexture().createView(),
        clearValue: { r: this.background[0], g: this.background[1], b: this.background[2], a: 1 },
        loadOp: 'clear', storeOp: 'store',
      }],
    });
    if (this.count > 0 && this.bindGroup) {
      pass.setPipeline(this.pipeline);
      pass.setBindGroup(0, this.bindGroup);
      pass.draw(6, this.count, 0, 0);
    }
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    return view;
  }
}

// ---------------------------------------------------------------------------
// viewer (model + camera + controls + transitions + UI)
// ---------------------------------------------------------------------------
export class Viewer {
  constructor(canvas, config) {
    this.canvas = canvas;
    this.config = config;
    this.up = v3.normalize(config.up || [0, 0, 1]);
    this.camera = new Camera(this.up);
    this.poses = (config.poses || []).map((p) => ({ ...p }));
    this.transitionOptions = { ...DEFAULTS, ...((config.transitions || {}).default || {}) };
    this.route = config.route || null;
    this.transition = null;
    this.fx = { effectId: 0, tEffect: 0, strength: 1 };
    this.sizeScale = 1;
    this.needsRender = true;
    this.currentPoseId = null;
    this.orbitTarget = null;
    this.keys = new Set();
    this.pointers = new Map();
    this.frameTimes = [];
    this.lastFrame = performance.now();
    this.model = null;
    this.ui = new UI(this, config);
  }

  async init() {
    try {
      this.renderer = await Renderer.create(this.canvas);
    } catch (e) {
      this.ui.showMessage(`${e.message}<br><small>This page renders Gaussian splats with WebGPU. Use Chrome/Edge 113+, Safari 18+ or a recent Firefox.</small>`);
      throw e;
    }
    this.renderer.device.lost.then((info) => this.ui.showMessage(`GPU device lost: ${info.message}`));
    this.renderer.onSorted = () => { this.needsRender = true; };
    this.ui.setProgress(0, 0, 'loading model');
    this.model = await loadModel(this.config.model, {
      onProgress: (r, t) => this.ui.setProgress(r, t, 'loading model'),
      pointSize: this.config.fallback_point_size || 0.02,
    });
    this.ui.setProgress(1, 1, 'uploading to GPU');
    await new Promise((r) => setTimeout(r, 0));
    this.renderer.setModel(this.model);
    this.bounds = this.model.bounds;
    this.ui.hideLoading();
    this.ui.setModelInfo(this.model);
    this.installControls();
    window.addEventListener('resize', () => { this.needsRender = true; });
    // initial view: deep link, first pose, or an overview of the scene
    const hashId = decodeURIComponent((location.hash || '').slice(1));
    const start = this.poses.find((p) => p.id === hashId) || this.poses[0];
    if (start) this.jumpTo(start.id);
    else this.overview();
    this.loop();
  }

  // --- poses & transitions -------------------------------------------------
  poseById(id) { return this.poses.find((p) => p.id === id); }
  focusDistance() { return clamp(this.transitionOptions.focus_distance || 2.5, 0.2, 100); }
  setOrbitTargetFromCamera() {
    this.orbitTarget = v3.add(this.camera.position, v3.scale(this.camera.axes().forward, this.focusDistance()));
  }
  overview() {
    const b = this.bounds;
    const { e1 } = horizontalBasis(this.up);
    this.camera.position = v3.add(v3.add(b.center, v3.scale(e1, -b.radius * 1.4)), v3.scale(this.up, b.radius * 0.5));
    this.camera.lookAt(b.center);
    this.camera.fov = 60;
    this.orbitTarget = b.center.slice();
    this.currentPoseId = null;
    this.needsRender = true;
  }
  jumpTo(id) {
    const p = this.poseById(id);
    if (!p) return;
    this.transition = null;
    this.fx = { effectId: 0, tEffect: 0, strength: 1 };
    this.camera.setPose(p);
    this.setOrbitTargetFromCamera();
    this.currentPoseId = id;
    this.ui.setActivePose(id);
    this.setHash(id);
    this.needsRender = true;
  }
  goTo(id, overrides = {}) {
    const p = this.poseById(id);
    if (!p) return null;
    const opts = { ...this.transitionOptions, ...(p.transition || {}), ...overrides };
    if (p.via) opts.via = p.via;
    const ctx = { up: this.up, route: this.route, pivot: null };
    this.transition = new Transition(this.camera.pose(), p, opts, ctx);
    this.transitionTargetId = id;
    this.currentPoseId = id;
    this.ui.setActivePose(id);
    this.setHash(id);
    this.needsRender = true;
    return this.transition;
  }
  setHash(id) {
    try { history.replaceState(null, '', `#${encodeURIComponent(id)}`); } catch (_) { /* file:// etc. */ }
  }
  cancelTransition() {
    if (this.transition) { this.transition = null; this.fx = { effectId: 0, tEffect: 0, strength: 1 }; this.setOrbitTargetFromCamera(); }
  }
  nextPose(dir = 1) {
    if (!this.poses.length) return;
    const i = this.poses.findIndex((p) => p.id === this.currentPoseId);
    const j = ((i < 0 ? 0 : i + dir) + this.poses.length) % this.poses.length;
    this.goTo(this.poses[j].id);
  }

  /** Current viewpoint as a poses.json entry (COLMAP frame) plus the model-frame pose. */
  currentPoseJSON() {
    const pose = this.camera.pose();
    const c2m = this.config.colmap_to_model;
    let colmap = { position: pose.position, quaternion: pose.quaternion };
    if (c2m && c2m.rotation) {
      const R = c2m.rotation, t = c2m.translation, s = c2m.scale || 1;
      const Rt = [[R[0][0], R[1][0], R[2][0]], [R[0][1], R[1][1], R[2][1]], [R[0][2], R[1][2], R[2][2]]];
      const d = v3.sub(pose.position, t);
      const pos = [Rt[0][0] * d[0] + Rt[0][1] * d[1] + Rt[0][2] * d[2], Rt[1][0] * d[0] + Rt[1][1] * d[1] + Rt[1][2] * d[2], Rt[2][0] * d[0] + Rt[2][1] * d[1] + Rt[2][2] * d[2]].map((v) => v / s);
      const Rq = matrixFromQuat(pose.quaternion);
      const m = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
      for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) m[i][j] = Rt[i][0] * Rq[0][j] + Rt[i][1] * Rq[1][j] + Rt[i][2] * Rq[2][j];
      colmap = { position: pos, quaternion: quatFromMatrix(m) };
    }
    const n = this.poses.length + 1;
    const round = (a) => a.map((v) => Math.round(v * 1e5) / 1e5);
    return {
      id: `pose-${n}`, label: `Pose ${n}`,
      position: round(colmap.position), quaternion: round(colmap.quaternion), fov_y_deg: Math.round(pose.fov * 100) / 100,
      frame: 'colmap',
      model_frame_pose: { frame: this.config.frame, position: round(pose.position), quaternion: round(pose.quaternion) },
    };
  }

  // --- controls -------------------------------------------------------------
  installControls() {
    const c = this.canvas;
    c.style.touchAction = 'none';
    c.addEventListener('contextmenu', (e) => e.preventDefault());
    c.addEventListener('pointerdown', (e) => {
      c.setPointerCapture(e.pointerId);
      this.pointers.set(e.pointerId, { x: e.clientX, y: e.clientY, button: e.button, shift: e.shiftKey });
      if (this.transition) this.cancelTransition();
      if (!this.orbitTarget) this.setOrbitTargetFromCamera();
    });
    c.addEventListener('pointermove', (e) => {
      const p = this.pointers.get(e.pointerId);
      if (!p) return;
      const dx = e.clientX - p.x, dy = e.clientY - p.y;
      const prev = { ...p };
      p.x = e.clientX; p.y = e.clientY;
      if (this.pointers.size === 2) {
        const [a, b] = [...this.pointers.values()];
        const curDist = Math.hypot(a.x - b.x, a.y - b.y);
        const prevA = a === p ? prev : a, prevB = b === p ? prev : b;
        const prevDist = Math.hypot(prevA.x - prevB.x, prevA.y - prevB.y);
        if (prevDist > 0) this.dolly(Math.log(prevDist / curDist) * 1.5);
        this.pan(dx * 0.5, dy * 0.5);
      } else if (p.button === 2 || p.button === 1 || p.shift) {
        this.pan(dx, dy);
      } else {
        this.orbit(-dx * 0.005, -dy * 0.005);
      }
    });
    const up = (e) => { this.pointers.delete(e.pointerId); };
    c.addEventListener('pointerup', up); c.addEventListener('pointercancel', up); c.addEventListener('lostpointercapture', up);
    c.addEventListener('wheel', (e) => { e.preventDefault(); if (this.transition) this.cancelTransition(); this.dolly(e.deltaY * 0.0012); }, { passive: false });
    window.addEventListener('keydown', (e) => {
      if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA')) return;
      const k = e.key.toLowerCase();
      if ('wasdqe'.includes(k) && k.length === 1) { this.keys.add(k); if (this.transition) this.cancelTransition(); e.preventDefault(); }
    });
    window.addEventListener('keyup', (e) => this.keys.delete(e.key.toLowerCase()));
    window.addEventListener('blur', () => this.keys.clear());
  }
  orbit(dyaw, dpitch) {
    if (!this.orbitTarget) this.setOrbitTargetFromCamera();
    let off = v3.sub(this.camera.position, this.orbitTarget);
    const dist = v3.length(off) || 1e-3;
    const qy = quatFromAxisAngle(this.up, dyaw);
    off = quatRotate(qy, off);
    const right = v3.normalize(v3.cross(off, this.up));
    if (v3.length(right) > 0) {
      // limit elevation to avoid flipping over the pole
      const el = Math.asin(clamp(v3.dot(v3.normalize(off), this.up), -1, 1));
      const newEl = clamp(el + dpitch, -1.48, 1.48);
      const qp = quatFromAxisAngle(right, -(newEl - el));
      off = quatRotate(qp, off);
    }
    this.camera.position = v3.add(this.orbitTarget, v3.scale(v3.normalize(off), dist));
    this.camera.lookAt(this.orbitTarget);
    this.currentPoseId = null; this.ui.setActivePose(null);
    this.needsRender = true;
  }
  pan(dx, dy) {
    if (!this.orbitTarget) this.setOrbitTargetFromCamera();
    const { right, down } = this.camera.axes();
    const dist = v3.length(v3.sub(this.camera.position, this.orbitTarget)) || 1;
    const k = (dist * 2 * Math.tan((this.camera.fov * Math.PI) / 360)) / this.canvas.clientHeight;
    const move = v3.add(v3.scale(right, -dx * k), v3.scale(down, -dy * k));
    this.camera.position = v3.add(this.camera.position, move);
    this.orbitTarget = v3.add(this.orbitTarget, move);
    this.currentPoseId = null; this.ui.setActivePose(null);
    this.needsRender = true;
  }
  dolly(amount) {
    if (!this.orbitTarget) this.setOrbitTargetFromCamera();
    const off = v3.sub(this.camera.position, this.orbitTarget);
    const dist = v3.length(off);
    const nd = clamp(dist * Math.exp(amount), 0.05, 1e4);
    this.camera.position = v3.add(this.orbitTarget, v3.scale(v3.normalize(off), nd));
    this.currentPoseId = null; this.ui.setActivePose(null);
    this.needsRender = true;
  }
  fly(dt) {
    if (!this.keys.size) return;
    const speed = 2.0 * dt * (this.keys.has('shift') ? 3 : 1);
    const { right, forward } = this.camera.axes();
    const fwd = v3.normalize(v3.sub(forward, v3.scale(this.up, v3.dot(forward, this.up))));
    let move = [0, 0, 0];
    if (this.keys.has('w')) move = v3.add(move, fwd);
    if (this.keys.has('s')) move = v3.sub(move, fwd);
    if (this.keys.has('d')) move = v3.add(move, right);
    if (this.keys.has('a')) move = v3.sub(move, right);
    if (this.keys.has('e')) move = v3.add(move, this.up);
    if (this.keys.has('q')) move = v3.sub(move, this.up);
    if (v3.length(move) === 0) return;
    move = v3.scale(v3.normalize(move), speed);
    this.camera.position = v3.add(this.camera.position, move);
    if (this.orbitTarget) this.orbitTarget = v3.add(this.orbitTarget, move);
    this.currentPoseId = null; this.ui.setActivePose(null);
    this.needsRender = true;
  }

  // --- main loop --------------------------------------------------------------
  loop() {
    const tick = (now) => {
      const dt = Math.min(0.1, (now - this.lastFrame) / 1000);
      this.lastFrame = now;
      this.fly(dt);
      if (this.transition) {
        const st = this.transition.update(now);
        this.camera.setPose(st.pose);
        this.fx = { effectId: st.effectId, tEffect: st.tEffect, strength: st.strength };
        this.needsRender = true;
        if (st.done) {
          this.transition = null;
          this.fx = { effectId: 0, tEffect: 0, strength: 1 };
          this.setOrbitTargetFromCamera();
          this.renderer.requestSort(this.camera.viewMatrix(), true);
        }
      }
      if (this.renderer.resize()) this.needsRender = true;
      if (this.needsRender) {
        this.needsRender = false;
        const view = this.renderer.render(this.camera, this.fx, this.sizeScale);
        this.renderer.requestSort(view);
        this.frameTimes.push(now);
        while (this.frameTimes.length && this.frameTimes[0] < now - 1000) this.frameTimes.shift();
        this.ui.updateStatus({ fps: this.frameTimes.length, count: this.renderer.count, pose: this.currentPoseId, transition: this.transition });
      }
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }
}

// ---------------------------------------------------------------------------
async function main() {
  const canvas = document.getElementById('gl');
  let config;
  try {
    const res = await fetch('viewer-config.json');
    config = await res.json();
  } catch (e) {
    document.getElementById('message').hidden = false;
    document.getElementById('message').innerHTML = `Could not load viewer-config.json (${e.message}). Serve this folder over HTTP, e.g. <code>python -m http.server</code>.`;
    return;
  }
  document.title = config.title || 'gsplat';
  const viewer = new Viewer(canvas, config);
  window.viewer = viewer;
  try { await viewer.init(); } catch (e) { console.warn('viewer not started:', e.message); }
}
main();
