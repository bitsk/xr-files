// SPDX-License-Identifier: GPL-3.0-or-later
//! 3D spatial mathematics, OpenXR asymmetric FOV matrices (`wgpu` [0, 1] depth),
//! spherical/cylindrical arc ergonomics, and user-facing (`Roll = 0°`) pose solver.

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EyeView {
    pub position: [f32; 3],
    /// Quaternion `[x, y, z, w]`.
    pub orientation: [f32; 4],
    /// OpenXR asymmetric FOV angles in radians: `[left, right, up, down]`.
    pub fov: [f32; 4],
}

impl Default for EyeView {
    fn default() -> Self {
        Self {
            position: [0.0, 1.60, 0.0],
            orientation: [0.0, 0.0, 0.0, 1.0],
            fov: [-0.82, 0.82, 0.78, -0.78],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pose3D {
    pub position: [f32; 3],
    pub orientation: [f32; 4],
}

impl Default for Pose3D {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0, 0.0],
            orientation: [0.0, 0.0, 0.0, 1.0],
        }
    }
}

/// Computes the stereo midpoint between left and right eyes (`ISSUE-XR-04`).
#[inline]
pub fn center_eye_position(eyes: &[EyeView; 2]) -> [f32; 3] {
    [
        0.5 * (eyes[0].position[0] + eyes[1].position[0]),
        0.5 * (eyes[0].position[1] + eyes[1].position[1]),
        0.5 * (eyes[0].position[2] + eyes[1].position[2]),
    ]
}

#[inline]
pub fn vec3_add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[inline]
pub fn vec3_sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
pub fn vec3_scale(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

#[inline]
pub fn vec3_dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
pub fn vec3_cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
pub fn vec3_len(v: [f32; 3]) -> f32 {
    vec3_dot(v, v).sqrt()
}

#[inline]
pub fn vec3_normalize(v: [f32; 3]) -> [f32; 3] {
    let l = vec3_len(v);
    if l > 1e-7 && l.is_finite() {
        [v[0] / l, v[1] / l, v[2] / l]
    } else {
        [0.0, 0.0, -1.0]
    }
}

#[inline]
pub fn quat_normalize(q: [f32; 4]) -> [f32; 4] {
    let l = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if l > 1e-7 && l.is_finite() {
        [q[0] / l, q[1] / l, q[2] / l, q[3] / l]
    } else {
        [0.0, 0.0, 0.0, 1.0]
    }
}

#[inline]
pub fn quat_mul(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [
        a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
        a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
        a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
        a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
    ]
}

#[inline]
pub fn quat_from_yaw_pitch(yaw_rad: f32, pitch_rad: f32) -> [f32; 4] {
    let (sy, cy) = (0.5 * yaw_rad).sin_cos();
    let (sp, cp) = (0.5 * pitch_rad).sin_cos();
    quat_normalize([cy * sp, sy * cp, -sy * sp, cy * cp])
}

#[inline]
pub fn quat_rotate_vec3(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
    let qv = [q[0], q[1], q[2]];
    let t = vec3_scale(vec3_cross(qv, v), 2.0);
    vec3_add(v, vec3_add(vec3_scale(t, q[3]), vec3_cross(qv, t)))
}

/// Forward ray direction (`-Z` in OpenXR local grip/aim space) transformed by `q`.
#[inline]
pub fn quat_forward(q: [f32; 4]) -> [f32; 3] {
    vec3_normalize(quat_rotate_vec3(q, [0.0, 0.0, -1.0]))
}

/// Solves a roll-free (`Roll = 0°`) orientation so a panel's `+Z` normal faces `eye_center`,
/// clamping pitch compensation to comfortable bounds (`Constraint 1.1 & 1.2`).
pub fn user_facing_orientation(panel_pos: [f32; 3], eye_center: [f32; 3]) -> [f32; 4] {
    let to_eye = vec3_sub(eye_center, panel_pos);
    let horiz = (to_eye[0] * to_eye[0] + to_eye[2] * to_eye[2]).sqrt();
    if horiz < 1e-4 {
        return [0.0, 0.0, 0.0, 1.0];
    }
    // Panel local +Z normal points toward viewer: sin(yaw) = dx/horiz, cos(yaw) = dz/horiz
    let yaw = to_eye[0].atan2(to_eye[2]);
    // Positive pitch tilts panel face upward when panel sits below eye level (`to_eye.y > 0`)
    let pitch = (-to_eye[1]).atan2(horiz).clamp(-26.0f32.to_radians(), 18.0f32.to_radians());
    quat_from_yaw_pitch(yaw, pitch)
}

/// Places a panel on a cylindrical/spherical horizon arc around `anchor_pos` + `anchor_yaw`.
/// Positive `arc_yaw_deg` places the panel to the user's left (`-X` when `anchor_yaw = 0`? Wait:
/// in right-handed `+Y` up, `-Z` forward: rotating `-Z` by `+yaw` around `+Y` gives
/// `[-sin(yaw), 0, -cos(yaw)]` -> left of center! And `-yaw` gives `+X` -> right of center!).
pub fn arc_panel_pose(
    anchor_pos: [f32; 3],
    anchor_yaw_rad: f32,
    arc_yaw_deg: f32,
    elevation_m: f32,
    radius_m: f32,
    eye_center: [f32; 3],
) -> Pose3D {
    let total_yaw = anchor_yaw_rad + arc_yaw_deg.to_radians();
    let (s, c) = total_yaw.sin_cos();
    let pos = [
        anchor_pos[0] - s * radius_m,
        anchor_pos[1] + elevation_m,
        anchor_pos[2] - c * radius_m,
    ];
    let orientation = user_facing_orientation(pos, eye_center);
    Pose3D {
        position: pos,
        orientation,
    }
}

/// Column-major 4x4 matrix compatible with WGSL `mat4x4<f32>`.
pub type Mat4 = [[f32; 4]; 4];

pub fn mat4_identity() -> Mat4 {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

pub fn mat4_mul(a: Mat4, b: Mat4) -> Mat4 {
    let mut out = [[0.0f32; 4]; 4];
    for col in 0..4 {
        for row in 0..4 {
            out[col][row] = a[0][row] * b[col][0]
                + a[1][row] * b[col][1]
                + a[2][row] * b[col][2]
                + a[3][row] * b[col][3];
        }
    }
    out
}

pub fn mat4_from_trs(translation: [f32; 3], rotation: [f32; 4], scale: [f32; 3]) -> Mat4 {
    let [x, y, z, w] = quat_normalize(rotation);
    let x2 = x + x;
    let y2 = y + y;
    let z2 = z + z;
    let xx = x * x2;
    let xy = x * y2;
    let xz = x * z2;
    let yy = y * y2;
    let yz = y * z2;
    let zz = z * z2;
    let wx = w * x2;
    let wy = w * y2;
    let wz = w * z2;

    [
        [
            (1.0 - (yy + zz)) * scale[0],
            (xy + wz) * scale[0],
            (xz - wy) * scale[0],
            0.0,
        ],
        [
            (xy - wz) * scale[1],
            (1.0 - (xx + zz)) * scale[1],
            (yz + wx) * scale[1],
            0.0,
        ],
        [
            (xz + wy) * scale[2],
            (yz - wx) * scale[2],
            (1.0 - (xx + yy)) * scale[2],
            0.0,
        ],
        [translation[0], translation[1], translation[2], 1.0],
    ]
}

/// Computes the camera view matrix (inverse of eye world pose).
pub fn mat4_view_from_eye(eye: &EyeView) -> Mat4 {
    let q = quat_normalize(eye.orientation);
    let inv_q = [-q[0], -q[1], -q[2], q[3]];
    let rot = mat4_from_trs([0.0, 0.0, 0.0], inv_q, [1.0, 1.0, 1.0]);
    let neg_p = [-eye.position[0], -eye.position[1], -eye.position[2]];
    let trans = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [neg_p[0], neg_p[1], neg_p[2], 1.0],
    ];
    mat4_mul(rot, trans)
}

/// Asymmetric OpenXR `Fovf` projection matrix for `wgpu` (`+X` right, `+Y` up, `Z` in `[0, 1]`).
pub fn mat4_projection_xr_fov(fov: [f32; 4], near: f32, far: f32) -> Mat4 {
    let tan_l = fov[0].tan();
    let tan_r = fov[1].tan();
    let tan_u = fov[2].tan();
    let tan_d = fov[3].tan();

    let w = (tan_r - tan_l).max(1e-5);
    let h = (tan_u - tan_d).max(1e-5);

    [
        [2.0 / w, 0.0, 0.0, 0.0],
        [0.0, 2.0 / h, 0.0, 0.0],
        [
            (tan_r + tan_l) / w,
            (tan_u + tan_d) / h,
            far / (near - far),
            -1.0,
        ],
        [0.0, 0.0, (far * near) / (near - far), 0.0],
    ]
}

pub fn mat4_view_proj(eye: &EyeView, near: f32, far: f32) -> Mat4 {
    mat4_mul(mat4_projection_xr_fov(eye.fov, near, far), mat4_view_from_eye(eye))
}

/// Transforms a 3D point by a column-major 4x4 matrix (returns `[x, y, z, w]`).
pub fn mat4_mul_vec4(m: Mat4, v: [f32; 4]) -> [f32; 4] {
    let mut out = [0.0f32; 4];
    for row in 0..4 {
        out[row] =
            m[0][row] * v[0] + m[1][row] * v[1] + m[2][row] * v[2] + m[3][row] * v[3];
    }
    out
}

/// Unprojects normalized window coordinates `ndc_x, ndc_y in [-1, 1]` into a world-space ray.
pub fn ray_from_window_ndc(eye: &EyeView, ndc_x: f32, ndc_y: f32) -> ([f32; 3], [f32; 3]) {
    let tan_l = eye.fov[0].tan();
    let tan_r = eye.fov[1].tan();
    let tan_u = eye.fov[2].tan();
    let tan_d = eye.fov[3].tan();

    let u = 0.5 * (ndc_x + 1.0);
    let v = 0.5 * (ndc_y + 1.0);
    let local_dir = vec3_normalize([
        tan_l + u * (tan_r - tan_l),
        tan_d + v * (tan_u - tan_d),
        -1.0,
    ]);
    let world_dir = quat_rotate_vec3(eye.orientation, local_dir);
    (eye.position, vec3_normalize(world_dir))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xr_projection_maps_forward_to_wgpu_zero_one_depth() {
        let eye = EyeView {
            position: [0.0, 1.6, 0.0],
            orientation: [0.0, 0.0, 0.0, 1.0],
            fov: [-0.8, 0.8, 0.75, -0.75],
        };
        let vp = mat4_view_proj(&eye, 0.05, 50.0);
        let clip = mat4_mul_vec4(vp, [0.0, 1.6, -1.75, 1.0]);
        assert!(clip[3] > 1.74);
        let ndc = [clip[0] / clip[3], clip[1] / clip[3], clip[2] / clip[3]];
        assert!(ndc[0].abs() < 1e-5);
        assert!(ndc[1].abs() < 1e-5);
        assert!(ndc[2] > 0.0 && ndc[2] < 1.0);

        // User-facing panel on left wing (+27°) aims its +Z normal back toward eye center
        let panel = arc_panel_pose([0.0, 1.6, 0.0], 0.0, 27.0, -0.08, 1.75, eye.position);
        let normal = quat_rotate_vec3(panel.orientation, [0.0, 0.0, 1.0]);
        let to_eye = vec3_normalize(vec3_sub(eye.position, panel.position));
        assert!(vec3_dot(normal, to_eye) > 0.99);
    }
}
