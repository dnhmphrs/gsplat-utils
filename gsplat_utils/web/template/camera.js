// camera.js — pinhole camera with OpenCV/COLMAP axes (x right, y down, z forward).
import { v3, lookRotation, cameraAxes, normalizePose } from './transitions.js';

// ---------------------------------------------------------------------------
// camera
// ---------------------------------------------------------------------------
export class Camera {
  constructor(up = [0, 0, 1]) {
    this.up = v3.normalize(up);
    this.position = [0, 0, 0];
    this.quaternion = [0, 0, 0, 1];
    this.fov = 60;
    this.near = 0.05;
  }
  setPose(p) {
    const q = normalizePose(p);
    this.position = q.position; this.quaternion = q.quaternion; this.fov = q.fov;
  }
  pose() { return { position: this.position.slice(), quaternion: this.quaternion.slice(), fov: this.fov }; }
  axes() { return cameraAxes(this.quaternion); }
  /** world -> camera, column-major Float32Array(16) (camera axes: x right, y down, z forward). */
  viewMatrix() {
    const { right: r, down: d, forward: f } = this.axes();
    const p = this.position;
    return new Float32Array([
      r[0], d[0], f[0], 0,
      r[1], d[1], f[1], 0,
      r[2], d[2], f[2], 0,
      -v3.dot(r, p), -v3.dot(d, p), -v3.dot(f, p), 1,
    ]);
  }
  focalPx(height) { return (0.5 * height) / Math.tan((this.fov * Math.PI) / 360); }
  lookAt(target) { this.quaternion = lookRotation(v3.sub(target, this.position), this.up, this.axes().right); }
}
