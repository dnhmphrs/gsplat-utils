// Run with:  node --test tests/web/
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import * as T from '../../gsplat_utils/web/template/transitions.js';
import { Camera } from '../../gsplat_utils/web/template/camera.js';
import { parseSplat, parsePly, computeBounds, detectFormat } from '../../gsplat_utils/web/template/splat-loader.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const TEMPLATE = path.resolve(here, '../../gsplat_utils/web/template');
const UP = [0, 0, 1];
const close = (a, b, eps = 1e-9, msg = '') => {
  assert.equal(a.length, b.length, msg);
  for (let i = 0; i < a.length; i++) assert.ok(Math.abs(a[i] - b[i]) < eps, `${msg} index ${i}: ${a[i]} vs ${b[i]}`);
};
const quatClose = (a, b, eps = 1e-9) => {
  const d = Math.abs(T.quatDot(T.quatNormalize(a), T.quatNormalize(b)));
  assert.ok(1 - d < eps, `quaternions differ: ${a} vs ${b} (|dot|=${d})`);
};

const A = { position: [0, 0, 1.5], quaternion: T.lookRotation([1, 0, 0], UP), fov: 60 };
const B = { position: [6, 4, 1.5], quaternion: T.lookRotation([0, 1, 0], UP), fov: 48 };
const ROUTE = [[0, 0, 1.5], [3, 0, 1.5], [3, 4, 1.5], [8, 4, 1.5]];
const VIA = [{ position: [3, 1, 1.5] }];

test('endpoints are reproduced exactly at t=0 and t=1 for every mode', () => {
  for (const mode of T.MODES) {
    for (const easing of ['linear', 'easeInOutCubic', 'easeOutBack', [0.65, 0, 0.35, 1]]) {
      const ctx = { up: UP, route: ROUTE };
      const opts = { mode, easing, via: mode === 'spline' ? VIA : undefined, fov_pulse: 7 };
      const p0 = T.interpolatePose(A, B, 0, opts, ctx);
      const p1 = T.interpolatePose(A, B, 1, opts, ctx);
      assert.deepEqual(p0.position, A.position, mode);
      assert.deepEqual(p0.quaternion, A.quaternion, mode);
      assert.equal(p0.fov, A.fov, mode);
      assert.deepEqual(p1.position, B.position, mode);
      assert.deepEqual(p1.quaternion, B.quaternion, mode);
      assert.equal(p1.fov, B.fov, mode);
      // and the open interval approaches the endpoints continuously
      const pe = T.interpolatePose(A, B, 1e-6, { ...opts, easing: 'linear' }, ctx);
      assert.ok(T.v3.dist(pe.position, A.position) < 1e-3, `${mode} continuous at start`);
      const pl = T.interpolatePose(A, B, 1 - 1e-6, { ...opts, easing: 'linear' }, ctx);
      assert.ok(T.v3.dist(pl.position, B.position) < 1e-3, `${mode} continuous at end`);
      quatClose(pl.quaternion, B.quaternion, 1e-6);
    }
  }
});

test('mid-transition poses are finite and modes actually differ', () => {
  const ctx = { up: UP, route: ROUTE };
  const mids = {};
  for (const mode of T.MODES) {
    const p = T.interpolatePose(A, B, 0.5, { mode, easing: 'linear' }, ctx);
    for (const v of [...p.position, ...p.quaternion, p.fov]) assert.ok(Number.isFinite(v), mode);
    mids[mode] = p.position;
  }
  assert.ok(T.v3.dist(mids.linear, mids.arc) > 0.1);
  const q = 0.25; // at t=0.5 the straight line and this particular route coincide, so compare at a quarter
  assert.ok(T.v3.dist(T.interpolatePose(A, B, q, { mode: 'linear', easing: 'linear' }, ctx).position,
    T.interpolatePose(A, B, q, { mode: 'route', easing: 'linear' }, ctx).position) > 0.1);
  assert.ok(T.v3.dist(mids.linear, mids.dolly) > 0.1);
  // route stays on the walk path when route_blend = 1 (corners rounded by < 0.18 * route_smooth)
  for (let i = 1; i < 40; i++) {
    const pr = T.interpolatePose(A, B, i / 40, { mode: 'route', easing: 'linear', route_blend: 1 }, ctx);
    assert.ok(T.nearestOnPolyline(ROUTE, pr.position).dist < 0.11, `route position follows the polyline at ${i / 40}`);
  }
  const sm = T.smoothPolyline([[0, 0, 0], [2, 0, 0], [2, 2, 0]], 0.1, 0.6);
  assert.deepEqual(sm[0], [0, 0, 0]); assert.deepEqual(sm[sm.length - 1], [2, 2, 0]);
  assert.ok(T.nearestOnPolyline([[0, 0, 0], [2, 0, 0], [2, 2, 0]], sm[Math.floor(sm.length / 2)]).dist < 0.11);
});

test('slerp takes the shortest path and handles antipodal quaternions', () => {
  const q0 = [0, 0, 0, 1];
  const q350 = T.quatFromAxisAngle([0, 0, 1], (350 * Math.PI) / 180);
  const mid = T.slerp(q0, q350, 0.5);
  const ang = (T.quatAngle(q0, mid) * 180) / Math.PI;
  assert.ok(Math.abs(ang - 5) < 1e-6, `expected 5deg, got ${ang}`);
  const neg = q0.map((v) => -v);
  quatClose(T.slerp(q0, neg, 0.5), q0);
  const q90 = T.quatFromAxisAngle([0, 1, 0], Math.PI / 2);
  const h = T.slerp(q0, q90, 0.5);
  assert.ok(Math.abs((T.quatAngle(q0, h) * 180) / Math.PI - 45) < 1e-6);
});

test('easings hit 0 and 1 at the ends and are monotonic (except easeOutBack, which overshoots)', () => {
  const specs = [...Object.keys(T.easings), [0.42, 0, 0.58, 1], [0.65, 0, 0.35, 1], 'cubicBezier(0.2,0.8,0.8,0.2)'];
  for (const spec of specs) {
    const f = T.resolveEasing(spec);
    assert.ok(Math.abs(f(0)) < 1e-9, `${spec} f(0)`);
    assert.ok(Math.abs(f(1) - 1) < 1e-9, `${spec} f(1)`);
    let prev = -Infinity, maxv = -Infinity;
    for (let i = 0; i <= 400; i++) {
      const v = f(i / 400);
      maxv = Math.max(maxv, v);
      if (spec !== 'easeOutBack') assert.ok(v >= prev - 1e-9, `${spec} not monotonic at ${i / 400}`);
      prev = v;
    }
    if (spec === 'easeOutBack') assert.ok(maxv > 1.05, 'easeOutBack overshoots');
  }
  assert.equal(T.resolveEasing('nope'), T.easings.linear);
  assert.ok(Math.abs(T.resolveEasing('0.42,0,0.58,1')(0.5) - 0.5) < 1e-6);
});

test('catmull-rom passes through its control points', () => {
  const pts = [[0, 0, 0], [1, 2, 0], [3, 2.5, 1], [4, 0, 1], [7, -1, 2]];
  const cum = T.polylineLengths(pts), total = cum[cum.length - 1];
  for (let i = 0; i < pts.length; i++) close(T.sampleSpline(pts, cum[i] / total), pts[i], 1e-9, `point ${i}`);
  close(T.catmullRom(pts[0], pts[1], pts[2], pts[3], 0), pts[1]);
  close(T.catmullRom(pts[0], pts[1], pts[2], pts[3], 1), pts[2]);
  // smooth: the midpoint of a segment is near the chord but not on a kink
  const mid = T.catmullRom(pts[0], pts[1], pts[2], pts[3], 0.5);
  assert.ok(T.v3.dist(mid, T.v3.lerp(pts[1], pts[2], 0.5)) < T.v3.dist(pts[1], pts[2]));
});

test('route slicing picks the right part of the polyline in both directions', () => {
  const route = Array.from({ length: 11 }, (_, i) => [i, 0, 0]);
  const a = [2.3, 0.5, 0], b = [7.6, -0.4, 0];
  const na = T.nearestOnPolyline(route, a);
  assert.equal(na.index, 2); assert.ok(Math.abs(na.t - 0.3) < 1e-9); assert.ok(Math.abs(na.length - 2.3) < 1e-9);
  close(na.point, [2.3, 0, 0]); assert.ok(Math.abs(na.dist - 0.5) < 1e-9);
  const fwd = T.sliceRoute(route, a, b);
  assert.equal(fwd.reversed, false);
  close(fwd.points[0], [2.3, 0, 0]);
  close(fwd.points[fwd.points.length - 1], [7.6, 0, 0]);
  assert.deepEqual(fwd.points.slice(1, -1).map((p) => p[0]), [3, 4, 5, 6, 7]);
  assert.ok(Math.abs(fwd.length - 5.3) < 1e-9);
  const back = T.sliceRoute(route, b, a);
  assert.equal(back.reversed, true);
  assert.deepEqual(back.points.slice(1, -1).map((p) => p[0]), [7, 6, 5, 4, 3]);
  // poses far from the route fall back to a spline
  const far = T.buildRoutePath(T.normalizePose({ position: [2, 50, 0], quaternion: [0, 0, 0, 1] }), T.normalizePose(B), { ...T.DEFAULTS }, { up: UP, route });
  assert.equal(far, null);
});

test('up_lock keeps the horizon level; plain slerp does not', () => {
  const a = { position: [0, 0, 0], quaternion: T.lookRotation([1, 0, 0.2], UP), fov: 60 };
  const b = { position: [1, 1, 0], quaternion: T.lookRotation([0, 1, -0.4], UP), fov: 60 };
  let maxRollLocked = 0, maxRollFree = 0;
  for (let i = 1; i < 50; i++) {
    const t = i / 50;
    const locked = T.interpolatePose(a, b, t, { mode: 'linear', easing: 'linear', up_lock: true });
    const free = T.interpolatePose(a, b, t, { mode: 'linear', easing: 'linear', up_lock: false });
    maxRollLocked = Math.max(maxRollLocked, Math.abs(T.rollOf(locked.quaternion, UP)));
    maxRollFree = Math.max(maxRollFree, Math.abs(T.rollOf(free.quaternion, UP)));
  }
  assert.ok(maxRollLocked < 1e-9, `locked roll ${maxRollLocked}`);
  assert.ok(maxRollFree > 1e-3, `slerp should introduce some roll here, got ${maxRollFree}`);
  // endpoints with roll: roll blends linearly and endpoints stay exact
  const rolled = T.withRoll(a.quaternion, 0.3, UP);
  assert.ok(Math.abs(T.rollOf(rolled, UP) - 0.3) < 1e-9);
  const m = T.interpRotation(rolled, b.quaternion, 0.5, UP, true);
  assert.ok(Math.abs(T.rollOf(m, UP) - 0.15) < 1e-6);
});

test('lookRotation produces OpenCV camera axes (right = forward x up, y down)', () => {
  const q = T.lookRotation([1, 0, 0], UP);
  const ax = T.cameraAxes(q);
  close(ax.forward, [1, 0, 0]); close(ax.right, [0, -1, 0]); close(ax.down, [0, 0, -1]);
  const { yaw, pitch } = T.yawPitchOf(T.lookRotation([0, 1, 1], UP), UP);
  assert.ok(Math.abs(pitch - Math.PI / 4) < 1e-9);
  assert.ok(Math.abs(yaw - Math.PI / 2) < 1e-9);
  // quatFromMatrix / matrixFromQuat round trip
  const m = T.matrixFromQuat(q);
  quatClose(T.quatFromMatrix(m), q);
  quatClose(T.quatFromMatrix(m.flat()), q);
});

test('Transition timing, duration_per_metre and effect envelope', () => {
  const tr = new T.Transition(A, B, { mode: 'linear', duration_ms: 1000, duration_per_metre: 500, effect: 'scatter', effect_strength: 0.8 }, { up: UP });
  const dist = Math.hypot(6, 4);
  assert.ok(Math.abs(tr.duration - dist * 500) < 1e-9);
  const s0 = tr.update(1000);
  assert.equal(s0.t, 0); assert.equal(s0.done, false); assert.equal(s0.effectId, T.EFFECT_IDS.scatter); assert.equal(s0.tEffect, 0);
  const sm = tr.update(1000 + tr.duration / 2);
  assert.ok(Math.abs(sm.tEffect - 1) < 1e-9); assert.equal(sm.strength, 0.8);
  const s1 = tr.update(1000 + tr.duration + 5);
  assert.equal(s1.t, 1); assert.equal(s1.done, true); assert.deepEqual(s1.pose.position, B.position);
  const none = new T.Transition(A, B, { effect: 'none', duration_ms: 10 }, { up: UP });
  assert.equal(none.update(5).tEffect, 0);
  assert.equal(none.update(5).effectId, 0);
});

test('Camera view matrix: forward -> +z, right -> +x, image-down -> +y', () => {
  const cam = new Camera(UP);
  cam.setPose({ position: [2, 3, 1.5], quaternion: T.lookRotation([0, 1, 0], UP), fov_y_deg: 90 });
  const V = cam.viewMatrix();
  const apply = (p) => [
    V[0] * p[0] + V[4] * p[1] + V[8] * p[2] + V[12],
    V[1] * p[0] + V[5] * p[1] + V[9] * p[2] + V[13],
    V[2] * p[0] + V[6] * p[1] + V[10] * p[2] + V[14],
  ];
  close(apply([2, 3, 1.5]), [0, 0, 0]);
  close(apply([2, 5, 1.5]), [0, 0, 2]);          // 2m ahead
  close(apply([3, 3, 1.5]), [1, 0, 0]);          // world +x is to the right when looking along +y (right = f x up = +x)
  close(apply([2, 3, 2.5]), [0, -1, 0]);         // world up is image-up (negative y)
  assert.ok(Math.abs(cam.focalPx(1000) - 500) < 1e-9); // 90deg vertical fov: f = H/2
  // the sort worker reads camera z from view[2], view[6], view[10], view[14]
  const z = V[2] * 2 + V[6] * 5 + V[10] * 1.5 + V[14];
  assert.ok(Math.abs(z - 2) < 1e-9);
});

test('splat loader parses .splat, gaussian .ply and point .ply', () => {
  // .splat: 2 splats
  const buf = new ArrayBuffer(64);
  const f = new Float32Array(buf), u = new Uint8Array(buf);
  f.set([1, 2, 3, 0.1, 0.2, 0.3], 0); u.set([255, 128, 0, 200, 128 + 128 * 0.0, 128 + 128 * 0.0, 128 + 128 * 0.0, 255], 24); // quat w=0? use normalised (0,0,0,1): w byte 128 -> 0
  f.set([-1, -2, -3, 0.5, 0.5, 0.5], 8); u.set([10, 20, 30, 40, 255, 128, 128, 128], 56); // w=1 -> identity
  const m = parseSplat(buf);
  assert.equal(m.count, 2);
  close(Array.from(m.positions.subarray(3, 6)), [-1, -2, -3], 1e-6);
  close(Array.from(m.scales.subarray(0, 3)), [0.1, 0.2, 0.3], 1e-6);
  assert.deepEqual(Array.from(m.colors.subarray(4, 8)), [10, 20, 30, 40]);
  close(Array.from(m.quats.subarray(4, 8)), [0, 0, 0, 1], 1e-6); // stored wxyz -> xyzw
  assert.equal(detectFormat(buf, 'model.splat'), 'splat');
  // gaussian ply, binary little endian, all float32
  const props = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3'];
  let header = `ply\nformat binary_little_endian 1.0\nelement vertex 1\n${props.map((p) => `property float ${p}`).join('\n')}\nend_header\n`;
  const pad = (4 - ((header.length + 'comment \n'.length) % 4)) % 4;   // 4-byte align the body for the fast path
  header = header.replace('end_header', `comment ${'x'.repeat(pad)}\nend_header`);
  assert.equal(header.length % 4, 0);
  const hb = new TextEncoder().encode(header);
  const vals = new Float32Array([1, 2, 3, 0, 0, 0, 0, 1.7724538509 /* -> 1.0 */, -1.7724538509 /* -> 0 */, 0 /* alpha .5 */, Math.log(0.5), Math.log(0.25), Math.log(2), 1, 0, 0, 0]);
  const ply = new Uint8Array(hb.length + vals.byteLength); ply.set(hb, 0); ply.set(new Uint8Array(vals.buffer), hb.length);
  assert.equal(detectFormat(ply.buffer, 'x.bin'), 'ply');
  const g = parsePly(ply.buffer);
  assert.equal(g.kind, 'gaussian-ply'); assert.equal(g.count, 1);
  assert.deepEqual(Array.from(g.colors), [128, 255, 0, 128]);
  close(Array.from(g.scales), [0.5, 0.25, 2], 1e-6);
  close(Array.from(g.quats), [0, 0, 0, 1], 1e-6);
  // point ply with uchar colours (DataView path)
  const ph = new TextEncoder().encode('ply\nformat binary_little_endian 1.0\nelement vertex 2\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n');
  const body = new Uint8Array(2 * 15); const dv = new DataView(body.buffer);
  dv.setFloat32(0, 5, true); dv.setFloat32(4, 6, true); dv.setFloat32(8, 7, true); body.set([1, 2, 3], 12);
  dv.setFloat32(15, -5, true); dv.setFloat32(19, 0, true); dv.setFloat32(23, 0, true); body.set([200, 100, 50], 27);
  const pp = new Uint8Array(ph.length + body.length); pp.set(ph, 0); pp.set(body, ph.length);
  const pm = parsePly(pp.buffer, { pointSize: 0.03 });
  assert.equal(pm.kind, 'point-ply'); assert.equal(pm.count, 2);
  close(Array.from(pm.positions), [5, 6, 7, -5, 0, 0], 1e-6);
  assert.deepEqual(Array.from(pm.colors), [1, 2, 3, 255, 200, 100, 50, 255]);
  close(Array.from(pm.scales.subarray(0, 3)), [0.03, 0.03, 0.03], 1e-9);
  const b = computeBounds(pm.positions);
  assert.ok(b.radius > 0 && Number.isFinite(b.center[0]));
});

test('sort worker orders splats back to front for the given view', async () => {
  const src = readFileSync(path.join(TEMPLATE, 'sort-worker.js'), 'utf8');
  const posted = [];
  const sandbox = { self: {}, Float32Array, Uint32Array, Int32Array, Infinity, Math };
  sandbox.self.postMessage = (m) => posted.push(m);
  vm.createContext(sandbox);
  vm.runInContext(src, sandbox);
  const n = 1000;
  const positions = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) { positions[i * 3] = Math.sin(i); positions[i * 3 + 1] = Math.cos(i * 1.3); positions[i * 3 + 2] = (i * 7919) % n; }
  sandbox.self.onmessage({ data: { type: 'init', positions } });
  assert.equal(posted[0].type, 'ready');
  const cam = new Camera(UP);
  cam.setPose({ position: [0, 0, -5], quaternion: T.lookRotation([0, 0, 1], UP), fov: 60 }); // looking along +z
  sandbox.self.onmessage({ data: { type: 'sort', view: cam.viewMatrix() } });
  const out = posted[1];
  assert.equal(out.type, 'sorted'); assert.equal(out.indices.length, n);
  const seen = new Set(out.indices); assert.equal(seen.size, n, 'permutation');
  let prevZ = Infinity;
  for (const i of out.indices) { const z = positions[i * 3 + 2]; assert.ok(z <= prevZ + 1e-6, 'descending depth'); prevZ = z; }
});

test('WGSL shader has balanced braces and the expected entry points', () => {
  const src = readFileSync(path.join(TEMPLATE, 'shaders.js'), 'utf8');
  const code = src.slice(src.indexOf('`') + 1, src.lastIndexOf('`'));
  let depth = 0;
  for (const ch of code) { if (ch === '{') depth++; if (ch === '}') depth--; assert.ok(depth >= 0); }
  assert.equal(depth, 0);
  assert.match(code, /@vertex\s+fn vs_main/); assert.match(code, /@fragment\s+fn fs_main/);
  for (const name of ['effect', 'tEffect', 'strength', 'sizeScale']) assert.ok(code.includes(`u.${name}`), name);
});
