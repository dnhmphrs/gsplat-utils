// shaders.js — WGSL for the splat renderer (kept separate so tools can validate it).
export const SHADER = /* wgsl */ `
struct Uniforms {
  view : mat4x4<f32>,
  focal : vec2<f32>,
  viewport : vec2<f32>,
  camPos : vec3<f32>,
  time : f32,
  effect : u32,
  tEffect : f32,
  strength : f32,
  sizeScale : f32,
};
struct Splat { pos : vec3<f32>, pad0 : f32, scale : vec3<f32>, pad1 : f32, quat : vec4<f32>, };
@group(0) @binding(0) var<uniform> u : Uniforms;
@group(0) @binding(1) var<storage, read> splats : array<Splat>;
@group(0) @binding(2) var<storage, read> colors : array<u32>;
@group(0) @binding(3) var<storage, read> order : array<u32>;

struct VSOut {
  @builtin(position) pos : vec4<f32>,
  @location(0) uv : vec2<f32>,
  @location(1) color : vec4<f32>,
};

fn pcg(v : u32) -> u32 {
  let state = v * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}
fn hash3(i : u32) -> vec3<f32> {
  let a = pcg(i + 1u); let b = pcg(a); let c = pcg(b);
  return vec3<f32>(f32(a), f32(b), f32(c)) / 4294967295.0;
}
// q = (x, y, z, w); returns the rotation matrix whose columns are the rotated basis vectors
fn quatToMat(q : vec4<f32>) -> mat3x3<f32> {
  let x = q.x; let y = q.y; let z = q.z; let w = q.w;
  return mat3x3<f32>(
    vec3<f32>(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + z * w), 2.0 * (x * z - y * w)),
    vec3<f32>(2.0 * (x * y - z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + x * w)),
    vec3<f32>(2.0 * (x * z + y * w), 2.0 * (y * z - x * w), 1.0 - 2.0 * (x * x + y * y)));
}

@vertex
fn vs_main(@builtin(vertex_index) vi : u32, @builtin(instance_index) ii : u32) -> VSOut {
  var out : VSOut;
  out.pos = vec4<f32>(0.0, 0.0, 2.0, 1.0); // off-screen by default (clipped)
  out.uv = vec2<f32>(0.0, 0.0);
  out.color = vec4<f32>(0.0, 0.0, 0.0, 0.0);
  let idx = order[ii];
  let s = splats[idx];
  var p = s.pos;
  var scale = s.scale * u.sizeScale;
  let packed = colors[idx];
  var color = vec4<f32>(f32(packed & 0xffu), f32((packed >> 8u) & 0xffu),
                        f32((packed >> 16u) & 0xffu), f32((packed >> 24u) & 0xffu)) / 255.0;
  if (u.effect != 0u) {
    let h = hash3(idx);
    let k = u.strength * u.tEffect;
    if (u.effect == 1u) {            // fade
      color.a = color.a * (1.0 - clamp(k, 0.0, 1.0));
    } else if (u.effect == 2u) {     // scatter
      let dir = normalize(h * 2.0 - vec3<f32>(1.0, 1.0, 1.0));
      let dist = length(p - u.camPos);
      p = p + dir * (k * 0.25 * dist);
    } else if (u.effect == 3u) {     // pulse
      scale = scale * (1.0 + 2.0 * k);
    } else if (u.effect == 4u) {     // desaturate
      let l = dot(color.rgb, vec3<f32>(0.299, 0.587, 0.114));
      color = vec4<f32>(mix(color.rgb, vec3<f32>(l, l, l), clamp(k, 0.0, 1.0)), color.a);
    } else if (u.effect == 5u) {     // dissolve
      if (h.x < k) { color.a = 0.0; }
    }
  }
  let cam = (u.view * vec4<f32>(p, 1.0)).xyz;
  let W = u.viewport.x;
  let H = u.viewport.y;
  if (cam.z < 0.05 || color.a < 0.002) { return out; }
  // image-plane centre in pixels relative to the principal point (y down)
  let center = vec2<f32>(u.focal.x * cam.x / cam.z, u.focal.y * cam.y / cam.z);
  if (abs(center.x) > W || abs(center.y) > H) { return out; }
  // 3D covariance -> camera space -> 2D (EWA splatting)
  let R = quatToMat(s.quat);
  let S = mat3x3<f32>(vec3<f32>(scale.x, 0.0, 0.0), vec3<f32>(0.0, scale.y, 0.0), vec3<f32>(0.0, 0.0, scale.z));
  let M = R * S;
  let cov3d = M * transpose(M);
  let Wm = mat3x3<f32>(u.view[0].xyz, u.view[1].xyz, u.view[2].xyz);
  let z2 = cam.z * cam.z;
  let J = mat3x3<f32>(
    vec3<f32>(u.focal.x / cam.z, 0.0, 0.0),
    vec3<f32>(0.0, u.focal.y / cam.z, 0.0),
    vec3<f32>(-u.focal.x * cam.x / z2, -u.focal.y * cam.y / z2, 0.0));
  let T = J * Wm;
  let cov2 = T * cov3d * transpose(T);
  let a = cov2[0][0] + 0.3;
  let b = cov2[0][1];
  let d = cov2[1][1] + 0.3;
  let mid = 0.5 * (a + d);
  let det = a * d - b * b;
  let disc = sqrt(max(mid * mid - det, 0.1));
  let l1 = mid + disc;
  let l2 = max(mid - disc, 0.1);
  var diag : vec2<f32>;
  if (abs(b) < 1e-6) {
    diag = select(vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0), a >= d);
  } else {
    diag = normalize(vec2<f32>(b, l1 - a));
  }
  let major = min(sqrt(2.0 * l1), 1024.0) * diag;
  let minor = min(sqrt(2.0 * l2), 1024.0) * vec2<f32>(diag.y, -diag.x);
  var corner = vec2<f32>(2.0, 2.0);
  switch (vi) {
    case 0u: { corner = vec2<f32>(-2.0, -2.0); }
    case 1u: { corner = vec2<f32>(2.0, -2.0); }
    case 2u: { corner = vec2<f32>(-2.0, 2.0); }
    case 3u: { corner = vec2<f32>(-2.0, 2.0); }
    case 4u: { corner = vec2<f32>(2.0, -2.0); }
    default: { corner = vec2<f32>(2.0, 2.0); }
  }
  let px = center + corner.x * major + corner.y * minor;
  // pixels (y down) -> clip space (y up); depth unused (sorted, no depth test)
  out.pos = vec4<f32>(px.x * 2.0 / W, -px.y * 2.0 / H, 0.5, 1.0);
  out.uv = corner;
  out.color = color;
  return out;
}

@fragment
fn fs_main(in : VSOut) -> @location(0) vec4<f32> {
  let A = -dot(in.uv, in.uv);
  if (A < -4.0) { discard; }
  let B = exp(A) * in.color.a;
  return vec4<f32>(in.color.rgb * B, B);
}
`;
