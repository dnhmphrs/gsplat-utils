// transitions.js — camera pose interpolation for the gsplat-utils viewer.
// Dependency-free ES module; every helper is exported so it can be unit tested in Node.
//
// Conventions
//   pose       {position:[x,y,z], quaternion:[x,y,z,w], fov}   (fov = vertical, degrees)
//   quaternion rotates camera-frame vectors into world space. Camera axes follow
//              OpenCV/COLMAP: +x right, +y down, +z forward.
//   up         configurable world up vector (default [0,0,1]).

// ----------------------------------------------------------------------------
// vectors
// ----------------------------------------------------------------------------
export const v3 = {
  add: (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]],
  sub: (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]],
  scale: (a, s) => [a[0] * s, a[1] * s, a[2] * s],
  dot: (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2],
  cross: (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]],
  length: (a) => Math.hypot(a[0], a[1], a[2]),
  dist: (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]),
  normalize(a) {
    const l = Math.hypot(a[0], a[1], a[2]);
    return l > 1e-12 ? [a[0] / l, a[1] / l, a[2] / l] : [0, 0, 0];
  },
  lerp: (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t],
};
export const lerp = (a, b, t) => a + (b - a) * t;
export const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
export const tent = (t) => 1 - Math.abs(2 * clamp(t, 0, 1) - 1);

// ----------------------------------------------------------------------------
// quaternions [x, y, z, w]
// ----------------------------------------------------------------------------
export function quatNormalize(q) {
  const l = Math.hypot(q[0], q[1], q[2], q[3]) || 1;
  return [q[0] / l, q[1] / l, q[2] / l, q[3] / l];
}
export const quatDot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
export const quatConjugate = (q) => [-q[0], -q[1], -q[2], q[3]];
export function quatMultiply(a, b) {
  const [ax, ay, az, aw] = a, [bx, by, bz, bw] = b;
  return [
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
    aw * bw - ax * bx - ay * by - az * bz,
  ];
}
export function quatRotate(q, v) {
  const [x, y, z, w] = q;
  // v' = v + 2w(q×v) + 2 q×(q×v)
  const tx = 2 * (y * v[2] - z * v[1]);
  const ty = 2 * (z * v[0] - x * v[2]);
  const tz = 2 * (x * v[1] - y * v[0]);
  return [
    v[0] + w * tx + (y * tz - z * ty),
    v[1] + w * ty + (z * tx - x * tz),
    v[2] + w * tz + (x * ty - y * tx),
  ];
}
export function quatFromAxisAngle(axis, angle) {
  const a = v3.normalize(axis), s = Math.sin(angle / 2);
  return [a[0] * s, a[1] * s, a[2] * s, Math.cos(angle / 2)];
}
/** Angle (radians) between two rotations. */
export function quatAngle(a, b) {
  return 2 * Math.acos(clamp(Math.abs(quatDot(quatNormalize(a), quatNormalize(b))), 0, 1));
}
/** 3x3 rotation matrix (array of rows) -> quaternion [x,y,z,w]. Accepts 3x3 or 4x4 nested rows or a flat row-major 9/16 array. */
export function quatFromMatrix(m) {
  let r;
  if (Array.isArray(m[0]) || ArrayBuffer.isView(m[0])) {
    r = [[m[0][0], m[0][1], m[0][2]], [m[1][0], m[1][1], m[1][2]], [m[2][0], m[2][1], m[2][2]]];
  } else {
    const n = m.length === 16 ? 4 : 3;
    r = [[m[0], m[1], m[2]], [m[n], m[n + 1], m[n + 2]], [m[2 * n], m[2 * n + 1], m[2 * n + 2]]];
  }
  const [[m00, m01, m02], [m10, m11, m12], [m20, m21, m22]] = r;
  const tr = m00 + m11 + m22;
  let x, y, z, w;
  if (tr > 0) {
    const s = Math.sqrt(tr + 1) * 2;
    w = 0.25 * s; x = (m21 - m12) / s; y = (m02 - m20) / s; z = (m10 - m01) / s;
  } else if (m00 > m11 && m00 > m22) {
    const s = Math.sqrt(1 + m00 - m11 - m22) * 2;
    w = (m21 - m12) / s; x = 0.25 * s; y = (m01 + m10) / s; z = (m02 + m20) / s;
  } else if (m11 > m22) {
    const s = Math.sqrt(1 + m11 - m00 - m22) * 2;
    w = (m02 - m20) / s; x = (m01 + m10) / s; y = 0.25 * s; z = (m12 + m21) / s;
  } else {
    const s = Math.sqrt(1 + m22 - m00 - m11) * 2;
    w = (m10 - m01) / s; x = (m02 + m20) / s; y = (m12 + m21) / s; z = 0.25 * s;
  }
  const q = quatNormalize([x, y, z, w]);
  return q[3] < 0 ? [-q[0], -q[1], -q[2], -q[3]] : q;
}
/** quaternion -> 3x3 rotation matrix as array of rows. Columns are the camera axes (right, down, forward) in world space. */
export function matrixFromQuat(q) {
  const [x, y, z, w] = quatNormalize(q);
  return [
    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
  ];
}
export function slerp(a, b, t) {
  a = quatNormalize(a); b = quatNormalize(b);
  let d = quatDot(a, b);
  if (d < 0) { b = [-b[0], -b[1], -b[2], -b[3]]; d = -d; } // shortest path
  if (d > 0.9995) return quatNormalize([lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t), lerp(a[3], b[3], t)]);
  const th = Math.acos(d), s = Math.sin(th);
  const wa = Math.sin((1 - t) * th) / s, wb = Math.sin(t * th) / s;
  return [a[0] * wa + b[0] * wb, a[1] * wa + b[1] * wb, a[2] * wa + b[2] * wb, a[3] * wa + b[3] * wb];
}

// ----------------------------------------------------------------------------
// camera orientation helpers
// ----------------------------------------------------------------------------
export function cameraAxes(q) {
  return { right: quatRotate(q, [1, 0, 0]), down: quatRotate(q, [0, 1, 0]), forward: quatRotate(q, [0, 0, 1]) };
}
/** A deterministic horizontal basis (e1, e2) perpendicular to `up`, right-handed with e1 × e2 = up. */
export function horizontalBasis(up) {
  const u = v3.normalize(up);
  const ref = Math.abs(u[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
  const e2 = v3.normalize(v3.cross(u, ref));   // perpendicular to ref and up
  const e1 = v3.normalize(v3.cross(e2, u));    // e1 × e2 = up
  return { e1, e2, up: u };
}
/**
 * Rotation looking along `forward` with zero roll relative to `up`
 * (OpenCV camera: right = forward × up, down = forward × right).
 * If forward is parallel to up, `fallbackRight` (or a basis vector) breaks the tie.
 */
export function lookRotation(forward, up = [0, 0, 1], fallbackRight = null) {
  const f = v3.normalize(forward);
  let right = v3.cross(f, up);
  if (v3.length(right) < 1e-6) {
    right = fallbackRight ? v3.sub(fallbackRight, v3.scale(f, v3.dot(fallbackRight, f))) : null;
    if (!right || v3.length(right) < 1e-6) {
      const { e1, e2 } = horizontalBasis(up);
      right = Math.abs(v3.dot(e2, f)) < 0.9 ? v3.cross(f, e2) : v3.cross(f, e1);
      right = v3.scale(right, -1);
    }
  }
  right = v3.normalize(right);
  const down = v3.cross(f, right);
  // matrix columns = right, down, forward
  return quatFromMatrix([
    [right[0], down[0], f[0]],
    [right[1], down[1], f[1]],
    [right[2], down[2], f[2]],
  ]);
}
/** yaw/pitch (radians) of the camera forward axis relative to `up`: yaw around up from e1, pitch positive looking up. */
export function yawPitchOf(q, up = [0, 0, 1]) {
  const { e1, e2, up: u } = horizontalBasis(up);
  const f = cameraAxes(q).forward;
  const pitch = Math.asin(clamp(v3.dot(f, u), -1, 1));
  const yaw = Math.atan2(v3.dot(f, e2), v3.dot(f, e1));
  return { yaw, pitch };
}
export function forwardFromYawPitch(yaw, pitch, up = [0, 0, 1]) {
  const { e1, e2, up: u } = horizontalBasis(up);
  const c = Math.cos(pitch);
  return v3.add(v3.add(v3.scale(e1, c * Math.cos(yaw)), v3.scale(e2, c * Math.sin(yaw))), v3.scale(u, Math.sin(pitch)));
}
/** Roll (radians) of the camera about its forward axis relative to the level orientation for `up`. 0 = horizon level. */
export function rollOf(q, up = [0, 0, 1]) {
  const { right, forward } = cameraAxes(q);
  let levelRight = v3.cross(forward, up);
  if (v3.length(levelRight) < 1e-6) return 0;
  levelRight = v3.normalize(levelRight);
  const levelDown = v3.cross(forward, levelRight);
  return Math.atan2(v3.dot(right, levelDown), v3.dot(right, levelRight));
}
/** Level rotation with the same forward as q, then re-apply `roll` about forward. */
export function withRoll(q, roll, up = [0, 0, 1]) {
  const { forward, right } = cameraAxes(q);
  const level = lookRotation(forward, up, right);
  if (Math.abs(roll) < 1e-12) return level;
  return quatMultiply(level, quatFromAxisAngle([0, 0, 1], roll));
}
const wrapAngle = (a) => Math.atan2(Math.sin(a), Math.cos(a));
/**
 * Interpolate rotations. upLock=true: yaw and pitch move independently (horizon stays
 * level; any roll present at the endpoints is blended linearly so the endpoints are
 * reproduced exactly). upLock=false: plain shortest-path slerp.
 */
export function interpRotation(qa, qb, s, up = [0, 0, 1], upLock = true) {
  if (s <= 0) return qa.slice();
  if (s >= 1) return qb.slice();
  if (!upLock) return slerp(qa, qb, s);
  const a = yawPitchOf(qa, up), b = yawPitchOf(qb, up);
  const ra = rollOf(qa, up), rb = rollOf(qb, up);
  // near the poles yaw is ill-defined: fall back to slerp there
  if (Math.abs(a.pitch) > 1.45 || Math.abs(b.pitch) > 1.45) return slerp(qa, qb, s);
  const yaw = a.yaw + wrapAngle(b.yaw - a.yaw) * s;
  const pitch = lerp(a.pitch, b.pitch, s);
  const roll = ra + wrapAngle(rb - ra) * s;
  const f = forwardFromYawPitch(yaw, pitch, up);
  const q = lookRotation(f, up);
  return Math.abs(roll) < 1e-9 ? q : quatMultiply(q, quatFromAxisAngle([0, 0, 1], roll));
}

// ----------------------------------------------------------------------------
// easing
// ----------------------------------------------------------------------------
export const easings = {
  linear: (t) => t,
  smoothstep: (t) => t * t * (3 - 2 * t),
  easeInOutCubic: (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2),
  easeInOutQuint: (t) => (t < 0.5 ? 16 * t * t * t * t * t : 1 - Math.pow(-2 * t + 2, 5) / 2),
  easeInOutExpo: (t) => (t <= 0 ? 0 : t >= 1 ? 1 : t < 0.5 ? Math.pow(2, 20 * t - 10) / 2 : (2 - Math.pow(2, -20 * t + 10)) / 2),
  easeOutBack: (t) => { const c1 = 1.70158, c3 = c1 + 1; return 1 + c3 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2); },
  easeInOutSine: (t) => -(Math.cos(Math.PI * t) - 1) / 2,
};
export const EASINGS = Object.keys(easings).concat(['cubicBezier']);
/** CSS-style cubic-bezier(x1,y1,x2,y2) easing. */
export function cubicBezier(x1, y1, x2, y2) {
  const A = (a1, a2) => 1 - 3 * a2 + 3 * a1, B = (a1, a2) => 3 * a2 - 6 * a1, C = (a1) => 3 * a1;
  const bez = (t, a1, a2) => ((A(a1, a2) * t + B(a1, a2)) * t + C(a1)) * t;
  const slope = (t, a1, a2) => 3 * A(a1, a2) * t * t + 2 * B(a1, a2) * t + C(a1);
  const solveX = (x) => {
    let t = x;
    for (let i = 0; i < 8; i++) {
      const s = slope(t, x1, x2);
      if (Math.abs(s) < 1e-6) break;
      t -= (bez(t, x1, x2) - x) / s;
    }
    if (t < 0 || t > 1 || Math.abs(bez(t, x1, x2) - x) > 1e-4) { // bisection fallback
      let lo = 0, hi = 1;
      for (let i = 0; i < 40; i++) { t = (lo + hi) / 2; if (bez(t, x1, x2) < x) lo = t; else hi = t; }
    }
    return t;
  };
  return (x) => (x <= 0 ? 0 : x >= 1 ? 1 : bez(solveX(x), y1, y2));
}
/** 'easeInOutCubic' | [x1,y1,x2,y2] | function -> function */
export function resolveEasing(spec) {
  if (typeof spec === 'function') return spec;
  if (Array.isArray(spec) && spec.length === 4) return cubicBezier(...spec.map(Number));
  if (typeof spec === 'string') {
    const m = spec.match(/^cubicBezier\s*\(([^)]*)\)$/i) || spec.match(/^cubic-bezier\s*\(([^)]*)\)$/i);
    if (m) return cubicBezier(...m[1].split(',').map(Number));
    if (easings[spec]) return easings[spec];
    const nums = spec.split(',').map((x) => Number(x.trim()));
    if (nums.length === 4 && nums.every(Number.isFinite)) return cubicBezier(...nums); // "0.65,0,0.35,1"
  }
  return easings.linear;
}

// ----------------------------------------------------------------------------
// curves / polylines
// ----------------------------------------------------------------------------
/** Centripetal Catmull-Rom segment between p1 and p2 (t in [0,1]); alpha=0.5 centripetal, 0 uniform. */
export function catmullRom(p0, p1, p2, p3, t, alpha = 0.5) {
  if (t <= 0) return p1.slice();
  if (t >= 1) return p2.slice();
  const tj = (ti, pi, pj) => Math.pow(Math.max(v3.dist(pi, pj), 1e-9), alpha) + ti;
  const t0 = 0, t1 = tj(t0, p0, p1), t2 = tj(t1, p1, p2), t3 = tj(t2, p2, p3);
  const tt = lerp(t1, t2, t);
  const L = (pa, pb, ta, tb) => (tb - ta < 1e-12 ? pa : v3.add(v3.scale(pa, (tb - tt) / (tb - ta)), v3.scale(pb, (tt - ta) / (tb - ta))));
  const A1 = L(p0, p1, t0, t1), A2 = L(p1, p2, t1, t2), A3 = L(p2, p3, t2, t3);
  const B1 = L(A1, A2, t0, t2), B2 = L(A2, A3, t1, t3);
  return L(B1, B2, t1, t2);
}
export function polylineLengths(points) {
  const cum = [0];
  for (let i = 1; i < points.length; i++) cum.push(cum[i - 1] + v3.dist(points[i - 1], points[i]));
  return cum;
}
/** Locate arc-length L on a polyline -> {index, t}. */
export function locateLength(cum, L) {
  const total = cum[cum.length - 1];
  if (total <= 0) return { index: 0, t: 0 };
  L = clamp(L, 0, total);
  let i = 0;
  while (i < cum.length - 2 && cum[i + 1] < L) i++;
  const seg = cum[i + 1] - cum[i];
  return { index: i, t: seg > 1e-12 ? (L - cum[i]) / seg : 0 };
}
/** Smooth spline through `points`, parameterised by chord length s in [0,1]. Passes through every point. */
export function sampleSpline(points, s, alpha = 0.5) {
  const n = points.length;
  if (n === 0) return [0, 0, 0];
  if (n === 1) return points[0].slice();
  if (s <= 0) return points[0].slice();
  if (s >= 1) return points[n - 1].slice();
  const cum = polylineLengths(points);
  const { index: i, t } = locateLength(cum, s * cum[n - 1]);
  const p1 = points[i], p2 = points[Math.min(i + 1, n - 1)];
  const p0 = i > 0 ? points[i - 1] : v3.sub(v3.scale(p1, 2), p2);
  const p3 = i + 2 < n ? points[i + 2] : v3.sub(v3.scale(p2, 2), p1);
  return catmullRom(p0, p1, p2, p3, t, alpha);
}
export function pointAtLength(points, cum, L) {
  const { index: i, t } = locateLength(cum, L);
  return v3.lerp(points[i], points[Math.min(i + 1, points.length - 1)], t);
}
/** Nearest point on a polyline -> {index (segment start), t, point, dist, length (arc position)}. */
export function nearestOnPolyline(points, p) {
  let best = null;
  let acc = 0;
  for (let i = 0; i < points.length - 1; i++) {
    const a = points[i], b = points[i + 1];
    const ab = v3.sub(b, a), ap = v3.sub(p, a);
    const l2 = v3.dot(ab, ab);
    const t = l2 > 1e-12 ? clamp(v3.dot(ap, ab) / l2, 0, 1) : 0;
    const q = v3.add(a, v3.scale(ab, t));
    const d = v3.dist(p, q);
    if (!best || d < best.dist) best = { index: i, t, point: q, dist: d, length: acc + t * Math.sqrt(l2) };
    acc += Math.sqrt(l2);
  }
  if (!best) return points.length ? { index: 0, t: 0, point: points[0].slice(), dist: v3.dist(p, points[0]), length: 0 } : null;
  return best;
}
/** Piece of the route between the projections of a and b (ordered a -> b). */
export function sliceRoute(points, a, b) {
  if (!points || points.length < 2) return { points: [], reversed: false, length: 0, startDist: Infinity, endDist: Infinity };
  const na = nearestOnPolyline(points, a), nb = nearestOnPolyline(points, b);
  const keyA = na.index + na.t, keyB = nb.index + nb.t;
  const out = [na.point];
  if (keyA <= keyB) {
    for (let i = na.index + 1; i <= nb.index; i++) out.push(points[i].slice());
  } else {
    for (let i = na.index; i >= nb.index + 1; i--) out.push(points[i].slice());
  }
  out.push(nb.point);
  const cleaned = dedupePoints(out);
  const cum = polylineLengths(cleaned);
  return { points: cleaned, reversed: keyA > keyB, length: cum[cum.length - 1], startDist: na.dist, endDist: nb.dist };
}
export function dedupePoints(points, eps = 1e-6) {
  const out = [];
  for (const p of points) if (!out.length || v3.dist(out[out.length - 1], p) > eps) out.push(p);
  return out;
}
/** Resample a polyline every `step` metres (endpoints kept). */
export function resamplePolyline(points, step) {
  const pts = dedupePoints(points);
  if (pts.length < 2 || !(step > 0)) return pts;
  const cum = polylineLengths(pts), total = cum[cum.length - 1];
  const n = Math.max(1, Math.ceil(total / step));
  const out = [];
  for (let i = 0; i <= n; i++) out.push(pointAtLength(pts, cum, (i / n) * total));
  return out;
}
/**
 * Moving-average smoothing of a polyline with a window of `windowLen` metres.
 * Endpoints stay fixed (the window shrinks symmetrically towards the ends), so a
 * 90-degree corner is rounded by at most ~0.18 * windowLen.
 */
export function smoothPolyline(points, step = 0.1, windowLen = 0.6) {
  const pts = resamplePolyline(points, step);
  const n = pts.length;
  if (n < 3 || !(windowLen > 0)) return pts;
  const h = Math.max(0, Math.round(windowLen / (2 * step)));
  const out = [];
  for (let i = 0; i < n; i++) {
    const hi = Math.min(h, i, n - 1 - i);
    let acc = [0, 0, 0];
    for (let j = i - hi; j <= i + hi; j++) acc = v3.add(acc, pts[j]);
    out.push(v3.scale(acc, 1 / (2 * hi + 1)));
  }
  return out;
}

// ----------------------------------------------------------------------------
// pose interpolation
// ----------------------------------------------------------------------------
export const MODES = ['linear', 'spline', 'arc', 'dolly', 'route'];
export const EFFECTS = ['none', 'fade', 'scatter', 'pulse', 'desaturate', 'dissolve'];
export const EFFECT_IDS = { none: 0, fade: 1, scatter: 2, pulse: 3, desaturate: 4, dissolve: 5 };
export const DEFAULTS = {
  mode: 'spline',
  easing: 'easeInOutCubic',
  duration_ms: 1800,
  duration_per_metre: 0,      // if > 0, duration = max(duration_ms, distance * this)
  effect: 'none',
  effect_strength: 1.0,
  up_lock: true,
  fov_pulse: 0,               // degrees added at mid transition (dolly-zoom feel)
  focus_distance: 2.5,        // metres; where a pose "looks" (arc pivot)
  // arc
  arc_side: 0.35,             // sideways swing as a fraction of the travel distance
  arc_height: 0.12,           // upward swing as a fraction of the travel distance
  arc_look: 0.5,              // 0..1 how much the camera keeps looking at the pivot mid-arc
  // dolly
  dolly_overlap: 0.3,         // 0 = move then turn, 1 = together
  dolly_order: 'move-then-turn',
  // route
  route_blend: 1.0,           // 1 = follow the walk path, 0 = straight line
  look_ahead: 1.5,            // metres ahead on the path the camera looks at
  look_blend_in: 0.15,        // fraction of the move used to turn from the start pose into travel direction
  look_blend_out: 0.3,        // fraction used to settle into the target pose
  route_max_snap: 6.0,        // metres; poses farther from the route than this fall back to spline
  route_smooth: 0.6,          // metres; moving-average window that rounds corners of the walk path
};

export function normalizePose(p) {
  if (!p) throw new Error('pose required');
  const fov = p.fov ?? p.fov_y_deg ?? 60;
  return {
    position: [p.position[0], p.position[1], p.position[2]],
    quaternion: quatNormalize(p.quaternion ?? [0, 0, 0, 1]),
    fov: Number(fov),
    via: p.via || null,
    id: p.id,
  };
}
const exact = (p) => ({ position: p.position.slice(), quaternion: p.quaternion.slice(), fov: p.fov });

function lookAtPoint(p, dist) {
  return v3.add(p.position, v3.scale(cameraAxes(p.quaternion).forward, dist));
}

function posePath(a, b, opts, ctx) {
  // control points for spline mode: a, via..., b
  const pts = [a.position];
  const rots = [a.quaternion];
  const via = opts.via || a.via || [];
  for (const v of via) {
    const pos = Array.isArray(v) ? v : v.position;
    pts.push([pos[0], pos[1], pos[2]]);
    rots.push(v && v.quaternion ? quatNormalize(v.quaternion) : null);
  }
  pts.push(b.position);
  rots.push(b.quaternion);
  // fill missing rotations by arc-length fraction between neighbours that have one
  const cum = polylineLengths(pts), total = cum[cum.length - 1] || 1;
  for (let i = 0; i < rots.length; i++) if (!rots[i]) rots[i] = interpRotation(a.quaternion, b.quaternion, cum[i] / total, ctx.up, opts.up_lock);
  return { pts, rots, cum, total };
}

function modeSpline(a, b, s, opts, ctx) {
  const { pts, rots, cum, total } = posePath(a, b, opts, ctx);
  const position = sampleSpline(pts, s);
  const { index: i, t } = locateLength(cum, s * total);
  const quaternion = interpRotation(rots[i], rots[Math.min(i + 1, rots.length - 1)], t, ctx.up, opts.up_lock);
  return { position, quaternion };
}

function modeArc(a, b, s, opts, ctx) {
  const up = ctx.up;
  const d = v3.sub(b.position, a.position), dist = v3.length(d);
  let side = v3.cross(v3.normalize(d), up);
  if (v3.length(side) < 1e-6) side = horizontalBasis(up).e1;
  side = v3.normalize(side);
  const bump = Math.sin(Math.PI * s);
  const swing = v3.add(v3.scale(side, opts.arc_side * dist * bump), v3.scale(up, opts.arc_height * dist * bump));
  const position = v3.add(v3.lerp(a.position, b.position, s), swing);
  let quaternion = interpRotation(a.quaternion, b.quaternion, s, up, opts.up_lock);
  if (opts.arc_look > 0) {
    const pivot = ctx.pivot || v3.scale(v3.add(lookAtPoint(a, opts.focus_distance), lookAtPoint(b, opts.focus_distance)), 0.5);
    const toPivot = v3.sub(pivot, position);
    if (v3.length(toPivot) > 1e-6) {
      const qLook = lookRotation(toPivot, up, cameraAxes(quaternion).right);
      quaternion = slerp(quaternion, qLook, clamp(opts.arc_look, 0, 1) * bump);
    }
  }
  return { position, quaternion };
}

function modeDolly(a, b, s, opts, ctx) {
  const o = clamp(opts.dolly_overlap, 0, 1), span = 0.5 + 0.5 * o;
  let first = easings.smoothstep(clamp(s / span, 0, 1));
  let second = easings.smoothstep(clamp((s - (1 - span)) / span, 0, 1));
  if (opts.dolly_order === 'turn-then-move') [first, second] = [second, first];
  return {
    position: v3.lerp(a.position, b.position, first),
    quaternion: interpRotation(a.quaternion, b.quaternion, second, ctx.up, opts.up_lock),
    fovT: second,
  };
}

/** Build the route path a -> ... -> b once (cached per transition). */
export function buildRoutePath(a, b, opts, ctx) {
  const route = ctx.route;
  if (!route || route.length < 2) return null;
  const sl = sliceRoute(route, a.position, b.position);
  if (sl.points.length < 2 || sl.length < 1e-3) return null;
  if (sl.startDist > opts.route_max_snap || sl.endDist > opts.route_max_snap) return null;
  let pts = dedupePoints([a.position, ...sl.points, b.position]);
  if (pts.length < 2) return null;
  pts = smoothPolyline(pts, 0.1, opts.route_smooth);
  const cum = polylineLengths(pts);
  return { pts, cum, total: cum[cum.length - 1] };
}

function modeRoute(a, b, s, opts, ctx, cache) {
  const path = cache.route !== undefined ? cache.route : (cache.route = buildRoutePath(a, b, opts, ctx));
  if (!path) return modeSpline(a, b, s, opts, ctx);
  const up = ctx.up;
  const pRoute = pointAtLength(path.pts, path.cum, s * path.total);
  const pLine = v3.lerp(a.position, b.position, s);
  const position = v3.lerp(pLine, pRoute, clamp(opts.route_blend, 0, 1));
  // look along the path
  const aheadL = Math.min(path.total, s * path.total + opts.look_ahead);
  let dir = v3.sub(v3.lerp(pLine, pointAtLength(path.pts, path.cum, aheadL), clamp(opts.route_blend, 0, 1)), position);
  if (v3.length(dir) < 1e-6) dir = v3.sub(b.position, a.position);
  if (v3.length(dir) < 1e-6) dir = cameraAxes(b.quaternion).forward;
  // keep the travel view level in pitch unless the path really climbs (stairs)
  let qTravel = lookRotation(dir, up, cameraAxes(a.quaternion).right);
  const win = clamp(opts.look_blend_in, 0.01, 0.9), wout = clamp(opts.look_blend_out, 0.01, 0.9);
  const wA = 1 - easings.smoothstep(clamp(s / win, 0, 1));
  const wB = easings.smoothstep(clamp((s - (1 - wout)) / wout, 0, 1));
  let quaternion = qTravel;
  if (wA > 0) quaternion = slerp(quaternion, a.quaternion, wA);
  if (wB > 0) quaternion = slerp(quaternion, b.quaternion, wB);
  return { position, quaternion };
}

/**
 * Pose at progress t in [0,1] between poses a and b.
 * opts: see DEFAULTS. ctx: {up:[..], route:[[x,y,z]...]|null, pivot:[..]|null, cache:{}}
 */
export function interpolatePose(a, b, t, opts = {}, ctx = {}) {
  a = normalizePose(a); b = normalizePose(b);
  const o = { ...DEFAULTS, ...opts };
  const c = { up: v3.normalize(ctx.up || [0, 0, 1]), route: ctx.route || null, pivot: ctx.pivot || null, cache: ctx.cache || {} };
  if (!(t > 0)) return exact(a);
  if (t >= 1) return exact(b);
  const s = clamp(resolveEasing(o.easing)(t), -0.5, 1.5); // easeOutBack overshoots on purpose
  let r;
  switch (o.mode) {
    case 'linear': r = { position: v3.lerp(a.position, b.position, s), quaternion: interpRotation(a.quaternion, b.quaternion, s, c.up, o.up_lock) }; break;
    case 'arc': r = modeArc(a, b, clamp(s, 0, 1), o, c); break;
    case 'dolly': r = modeDolly(a, b, clamp(s, 0, 1), o, c); break;
    case 'route': r = modeRoute(a, b, clamp(s, 0, 1), o, c, c.cache); break;
    case 'spline':
    default: r = modeSpline(a, b, clamp(s, 0, 1), o, c); break;
  }
  const fovT = r.fovT ?? clamp(s, 0, 1);
  const fov = lerp(a.fov, b.fov, fovT) + (o.fov_pulse || 0) * Math.sin(Math.PI * clamp(t, 0, 1));
  return { position: r.position, quaternion: quatNormalize(r.quaternion), fov };
}

/** Travel distance used for duration_per_metre (route length when following a route). */
export function transitionDistance(a, b, opts = {}, ctx = {}) {
  a = normalizePose(a); b = normalizePose(b);
  const o = { ...DEFAULTS, ...opts };
  if (o.mode === 'route') {
    const path = buildRoutePath(a, b, o, { up: ctx.up || [0, 0, 1], route: ctx.route || null });
    if (path) return path.total;
  }
  if (o.mode === 'spline') {
    const { total } = posePath(a, b, o, { up: v3.normalize(ctx.up || [0, 0, 1]) });
    return total;
  }
  return v3.dist(a.position, b.position);
}

/** Drives one transition over wall-clock time. */
export class Transition {
  constructor(from, to, opts = {}, ctx = {}) {
    this.from = normalizePose(from);
    this.to = normalizePose(to);
    this.opts = { ...DEFAULTS, ...opts };
    this.ctx = { up: v3.normalize(ctx.up || [0, 0, 1]), route: ctx.route || null, pivot: ctx.pivot || null, cache: {} };
    const dist = transitionDistance(this.from, this.to, this.opts, this.ctx);
    this.distance = dist;
    let d = Number(this.opts.duration_ms) || 0;
    if (this.opts.duration_per_metre > 0) d = Math.max(d, dist * this.opts.duration_per_metre);
    this.duration = Math.max(1, d);
    this.start = null;
    this.done = false;
    this.t = 0;
  }
  /** @returns {{pose, t, done, tEffect, effect, strength}} */
  update(nowMs) {
    if (this.start === null) this.start = nowMs;
    let t = (nowMs - this.start) / this.duration;
    if (t >= 1) { t = 1; this.done = true; }
    this.t = t;
    const pose = interpolatePose(this.from, this.to, t, this.opts, this.ctx);
    const effect = this.opts.effect || 'none';
    return {
      pose, t, done: this.done,
      effect,
      effectId: EFFECT_IDS[effect] ?? 0,
      tEffect: effect === 'none' ? 0 : tent(t),
      strength: Number(this.opts.effect_strength ?? 1),
    };
  }
  cancel() { this.done = true; }
}
