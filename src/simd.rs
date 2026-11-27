// SPDX-License-Identifier: GPL-3.0-or-later
//! Cross-architecture SIMD acceleration kernels for `xr-files`:
//! - `x86_64`: Runtime-dispatched AVX2 (256-bit) with baseline SSE2 (128-bit)
//! - `aarch64`: Hardware NEON (128-bit)
//! - Portable scalar fallback (for `riscv64` and parity verification)
//!
//! Accelerated hot paths:
//! 1. SoA 3D Ray-AABB batch intersection with AUDIT-01 direct-vs-slop priority
//! 2. Vectorized damped spring animation step for spatial Z-travel & glow
//! 3. SIMD ASCII case-folded substring & boundary-aware fuzzy filename search
//! 4. SIMD RGBA8 2x2 box downsampling (mipmap/thumbnail chain) & premultiplied alpha
//! 5. SIMD file byte inspection (newline count, binary check, Shannon entropy)

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// Returns the active hardware SIMD backend identifier for telemetry and diagnostics.
pub fn active_simd_backend() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            "x86_64-avx2"
        } else {
            "x86_64-sse2"
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        "aarch64-neon"
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        "scalar"
    }
}

// ============================================================================
// 1. Structure-of-Arrays (SoA) 3D Ray-AABB Batch Intersection + Hit Slop
// ============================================================================

/// Result of picking against a batch of spatial 3D bounding boxes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RayPickHit {
    pub index: usize,
    pub distance: f32,
    pub hit_point: [f32; 3],
    /// `true` when the ray directly pierces the physical AABB; `false` when hit via angular slop.
    pub direct: bool,
    /// Metric distance from the ray at `distance` to the true AABB surface (`0.0` for direct hits).
    pub border_distance: f32,
}

/// Padded Structure-of-Arrays (SoA) batch of world-space 3D AABBs for SIMD ray picking.
#[derive(Debug, Clone, Default)]
pub struct SoaAabbBatch {
    len: usize,
    pub min_x: Vec<f32>,
    pub min_y: Vec<f32>,
    pub min_z: Vec<f32>,
    pub max_x: Vec<f32>,
    pub max_y: Vec<f32>,
    pub max_z: Vec<f32>,
    /// Per-item angular hit-slop tangent (`tan(slop_angle)`), typically `tan(2.2°)` ≈ `0.0384`.
    pub slop_tan: Vec<f32>,
}

impl SoaAabbBatch {
    pub fn with_capacity(cap: usize) -> Self {
        let padded = (cap + 7) & !7;
        Self {
            len: 0,
            min_x: Vec::with_capacity(padded),
            min_y: Vec::with_capacity(padded),
            min_z: Vec::with_capacity(padded),
            max_x: Vec::with_capacity(padded),
            max_y: Vec::with_capacity(padded),
            max_z: Vec::with_capacity(padded),
            slop_tan: Vec::with_capacity(padded),
        }
    }

    pub fn clear(&mut self) {
        self.len = 0;
        self.min_x.clear();
        self.min_y.clear();
        self.min_z.clear();
        self.max_x.clear();
        self.max_y.clear();
        self.max_z.clear();
        self.slop_tan.clear();
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Appends a world-space AABB `[min, max]` with angular slop `slop_rad` (radians).
    pub fn push(&mut self, min: [f32; 3], max: [f32; 3], slop_rad: f32) -> usize {
        // Remove any previous tail padding if push is called after finalize_padding
        self.min_x.truncate(self.len);
        self.min_y.truncate(self.len);
        self.min_z.truncate(self.len);
        self.max_x.truncate(self.len);
        self.max_y.truncate(self.len);
        self.max_z.truncate(self.len);
        self.slop_tan.truncate(self.len);

        let idx = self.len;
        self.min_x.push(min[0].min(max[0]));
        self.min_y.push(min[1].min(max[1]));
        self.min_z.push(min[2].min(max[2]));
        self.max_x.push(min[0].max(max[0]));
        self.max_y.push(min[1].max(max[1]));
        self.max_z.push(min[2].max(max[2]));
        self.slop_tan.push(slop_rad.max(0.0).tan());
        self.len += 1;
        idx
    }

    /// Pads internal lanes to an 8-lane boundary with degenerate far-away AABBs.
    pub fn finalize_padding(&mut self) {
        let padded = (self.len + 7) & !7;
        while self.min_x.len() < padded {
            self.min_x.push(1.0e9);
            self.min_y.push(1.0e9);
            self.min_z.push(1.0e9);
            self.max_x.push(1.0e9);
            self.max_y.push(1.0e9);
            self.max_z.push(1.0e9);
            self.slop_tan.push(0.0);
        }
    }

    /// Picks the best hit along `origin + dir * t` (`t in [min_t, max_t]`).
    ///
    /// Enforces openKylin XR `AUDIT-01` priority:
    /// At near-equal depth (`|t_a - t_b| < 2.5e-3`), a direct geometry hit (`direct == true`)
    /// unconditionally wins over an expanded angular-slop hit; if both are slop hits at the
    /// same plane depth, the one with smaller metric `border_distance` wins.
    pub fn pick_ray(
        &self,
        origin: [f32; 3],
        dir: [f32; 3],
        min_t: f32,
        max_t: f32,
    ) -> Option<RayPickHit> {
        let Some(unit_dir) = normalize3(dir) else {
            return None;
        };
        if self.len == 0 {
            return None;
        }
        #[cfg(target_arch = "x86_64")]
        {
            if self.min_x.len() >= ((self.len + 7) & !7) {
                if std::is_x86_feature_detected!("avx2") {
                    // SAFETY: AVX2 availability verified above and SoA slices are padded to 8 lanes.
                    return unsafe { self.pick_ray_avx2(origin, unit_dir, min_t, max_t) };
                }
                // SAFETY: SSE2 is guaranteed on all x86_64 targets and SoA slices are padded to >=4 lanes.
                return unsafe { self.pick_ray_sse2(origin, unit_dir, min_t, max_t) };
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            if self.min_x.len() >= ((self.len + 7) & !7) {
                // SAFETY: NEON is standard on aarch64 and SoA slices are padded to >=4 lanes.
                return unsafe { self.pick_ray_neon(origin, unit_dir, min_t, max_t) };
            }
        }
        self.pick_ray_scalar(origin, unit_dir, min_t, max_t)
    }

    pub fn pick_ray_scalar(
        &self,
        origin: [f32; 3],
        unit_dir: [f32; 3],
        min_t: f32,
        max_t: f32,
    ) -> Option<RayPickHit> {
        let inv = safe_inv_dir(unit_dir);
        let mut best: Option<RayPickHit> = None;
        for i in 0..self.len {
            if let Some(candidate) =
                eval_candidate(self, i, origin, unit_dir, inv, min_t, max_t)
            {
                best = Some(match best {
                    None => candidate,
                    Some(cur) => choose_better_hit(cur, candidate),
                });
            }
        }
        best
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn pick_ray_avx2(
        &self,
        origin: [f32; 3],
        unit_dir: [f32; 3],
        min_t: f32,
        max_t: f32,
    ) -> Option<RayPickHit> {
        let inv = safe_inv_dir(unit_dir);
        let ox = _mm256_set1_ps(origin[0]);
        let oy = _mm256_set1_ps(origin[1]);
        let oz = _mm256_set1_ps(origin[2]);
        let ix = _mm256_set1_ps(inv[0]);
        let iy = _mm256_set1_ps(inv[1]);
        let iz = _mm256_set1_ps(inv[2]);
        let vmin_t = _mm256_set1_ps(min_t);
        let vmax_t = _mm256_set1_ps(max_t);

        let padded = (self.len + 7) & !7;
        let mut best: Option<RayPickHit> = None;

        let mut base = 0usize;
        while base < padded {
            let min_x = _mm256_loadu_ps(self.min_x.as_ptr().add(base));
            let min_y = _mm256_loadu_ps(self.min_y.as_ptr().add(base));
            let min_z = _mm256_loadu_ps(self.min_z.as_ptr().add(base));
            let max_x = _mm256_loadu_ps(self.max_x.as_ptr().add(base));
            let max_y = _mm256_loadu_ps(self.max_y.as_ptr().add(base));
            let max_z = _mm256_loadu_ps(self.max_z.as_ptr().add(base));
            let slop = _mm256_loadu_ps(self.slop_tan.as_ptr().add(base));

            // Expand AABBs conservatively by max_t * slop_tan for fast 8-wide broadphase culling;
            // exact direct-vs-slop test runs only on surviving lanes.
            let pad = _mm256_mul_ps(vmax_t, slop);
            let emin_x = _mm256_sub_ps(min_x, pad);
            let emin_y = _mm256_sub_ps(min_y, pad);
            let emin_z = _mm256_sub_ps(min_z, pad);
            let emax_x = _mm256_add_ps(max_x, pad);
            let emax_y = _mm256_add_ps(max_y, pad);
            let emax_z = _mm256_add_ps(max_z, pad);

            let tx1 = _mm256_mul_ps(_mm256_sub_ps(emin_x, ox), ix);
            let tx2 = _mm256_mul_ps(_mm256_sub_ps(emax_x, ox), ix);
            let tmin_x = _mm256_min_ps(tx1, tx2);
            let tmax_x = _mm256_max_ps(tx1, tx2);

            let ty1 = _mm256_mul_ps(_mm256_sub_ps(emin_y, oy), iy);
            let ty2 = _mm256_mul_ps(_mm256_sub_ps(emax_y, oy), iy);
            let tmin_y = _mm256_min_ps(ty1, ty2);
            let tmax_y = _mm256_max_ps(ty1, ty2);

            let tz1 = _mm256_mul_ps(_mm256_sub_ps(emin_z, oz), iz);
            let tz2 = _mm256_mul_ps(_mm256_sub_ps(emax_z, oz), iz);
            let tmin_z = _mm256_min_ps(tz1, tz2);
            let tmax_z = _mm256_max_ps(tz1, tz2);

            let t_enter = _mm256_max_ps(_mm256_max_ps(tmin_x, tmin_y), _mm256_max_ps(tmin_z, vmin_t));
            let t_exit = _mm256_min_ps(_mm256_min_ps(tmax_x, tmax_y), _mm256_min_ps(tmax_z, vmax_t));
            let cmp = _mm256_cmp_ps(t_enter, t_exit, _CMP_LE_OQ);
            let mut mask = _mm256_movemask_ps(cmp) as u32;

            while mask != 0 {
                let lane = mask.trailing_zeros() as usize;
                mask &= mask - 1;
                let idx = base + lane;
                if idx < self.len {
                    if let Some(candidate) =
                        eval_candidate(self, idx, origin, unit_dir, inv, min_t, max_t)
                    {
                        best = Some(match best {
                            None => candidate,
                            Some(cur) => choose_better_hit(cur, candidate),
                        });
                    }
                }
            }
            base += 8;
        }
        best
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse2")]
    unsafe fn pick_ray_sse2(
        &self,
        origin: [f32; 3],
        unit_dir: [f32; 3],
        min_t: f32,
        max_t: f32,
    ) -> Option<RayPickHit> {
        let inv = safe_inv_dir(unit_dir);
        let ox = _mm_set1_ps(origin[0]);
        let oy = _mm_set1_ps(origin[1]);
        let oz = _mm_set1_ps(origin[2]);
        let ix = _mm_set1_ps(inv[0]);
        let iy = _mm_set1_ps(inv[1]);
        let iz = _mm_set1_ps(inv[2]);
        let vmin_t = _mm_set1_ps(min_t);
        let vmax_t = _mm_set1_ps(max_t);

        let padded = (self.len + 3) & !3;
        let mut best: Option<RayPickHit> = None;

        let mut base = 0usize;
        while base < padded {
            let min_x = _mm_loadu_ps(self.min_x.as_ptr().add(base));
            let min_y = _mm_loadu_ps(self.min_y.as_ptr().add(base));
            let min_z = _mm_loadu_ps(self.min_z.as_ptr().add(base));
            let max_x = _mm_loadu_ps(self.max_x.as_ptr().add(base));
            let max_y = _mm_loadu_ps(self.max_y.as_ptr().add(base));
            let max_z = _mm_loadu_ps(self.max_z.as_ptr().add(base));
            let slop = _mm_loadu_ps(self.slop_tan.as_ptr().add(base));

            let pad = _mm_mul_ps(vmax_t, slop);
            let tx1 = _mm_mul_ps(_mm_sub_ps(_mm_sub_ps(min_x, pad), ox), ix);
            let tx2 = _mm_mul_ps(_mm_sub_ps(_mm_add_ps(max_x, pad), ox), ix);
            let ty1 = _mm_mul_ps(_mm_sub_ps(_mm_sub_ps(min_y, pad), oy), iy);
            let ty2 = _mm_mul_ps(_mm_sub_ps(_mm_add_ps(max_y, pad), oy), iy);
            let tz1 = _mm_mul_ps(_mm_sub_ps(_mm_sub_ps(min_z, pad), oz), iz);
            let tz2 = _mm_mul_ps(_mm_sub_ps(_mm_add_ps(max_z, pad), oz), iz);

            let t_enter = _mm_max_ps(
                _mm_max_ps(_mm_min_ps(tx1, tx2), _mm_min_ps(ty1, ty2)),
                _mm_max_ps(_mm_min_ps(tz1, tz2), vmin_t),
            );
            let t_exit = _mm_min_ps(
                _mm_min_ps(_mm_max_ps(tx1, tx2), _mm_max_ps(ty1, ty2)),
                _mm_min_ps(_mm_max_ps(tz1, tz2), vmax_t),
            );
            let mut mask = _mm_movemask_ps(_mm_cmple_ps(t_enter, t_exit)) as u32;
            while mask != 0 {
                let lane = mask.trailing_zeros() as usize;
                mask &= mask - 1;
                let idx = base + lane;
                if idx < self.len {
                    if let Some(candidate) =
                        eval_candidate(self, idx, origin, unit_dir, inv, min_t, max_t)
                    {
                        best = Some(match best {
                            None => candidate,
                            Some(cur) => choose_better_hit(cur, candidate),
                        });
                    }
                }
            }
            base += 4;
        }
        best
    }

    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn pick_ray_neon(
        &self,
        origin: [f32; 3],
        unit_dir: [f32; 3],
        min_t: f32,
        max_t: f32,
    ) -> Option<RayPickHit> {
        let inv = safe_inv_dir(unit_dir);
        let ox = vdupq_n_f32(origin[0]);
        let oy = vdupq_n_f32(origin[1]);
        let oz = vdupq_n_f32(origin[2]);
        let ix = vdupq_n_f32(inv[0]);
        let iy = vdupq_n_f32(inv[1]);
        let iz = vdupq_n_f32(inv[2]);
        let vmin_t = vdupq_n_f32(min_t);
        let vmax_t = vdupq_n_f32(max_t);

        let padded = (self.len + 3) & !3;
        let mut best: Option<RayPickHit> = None;

        let mut base = 0usize;
        while base < padded {
            let min_x = vld1q_f32(self.min_x.as_ptr().add(base));
            let min_y = vld1q_f32(self.min_y.as_ptr().add(base));
            let min_z = vld1q_f32(self.min_z.as_ptr().add(base));
            let max_x = vld1q_f32(self.max_x.as_ptr().add(base));
            let max_y = vld1q_f32(self.max_y.as_ptr().add(base));
            let max_z = vld1q_f32(self.max_z.as_ptr().add(base));
            let slop = vld1q_f32(self.slop_tan.as_ptr().add(base));

            let pad = vmulq_f32(vmax_t, slop);
            let tx1 = vmulq_f32(vsubq_f32(vsubq_f32(min_x, pad), ox), ix);
            let tx2 = vmulq_f32(vsubq_f32(vaddq_f32(max_x, pad), ox), ix);
            let ty1 = vmulq_f32(vsubq_f32(vsubq_f32(min_y, pad), oy), iy);
            let ty2 = vmulq_f32(vsubq_f32(vaddq_f32(max_y, pad), oy), iy);
            let tz1 = vmulq_f32(vsubq_f32(vsubq_f32(min_z, pad), oz), iz);
            let tz2 = vmulq_f32(vsubq_f32(vaddq_f32(max_z, pad), oz), iz);

            let t_enter = vmaxq_f32(
                vmaxq_f32(vminq_f32(tx1, tx2), vminq_f32(ty1, ty2)),
                vmaxq_f32(vminq_f32(tz1, tz2), vmin_t),
            );
            let t_exit = vminq_f32(
                vminq_f32(vmaxq_f32(tx1, tx2), vmaxq_f32(ty1, ty2)),
                vminq_f32(vmaxq_f32(tz1, tz2), vmax_t),
            );
            let cmp = vcleq_f32(t_enter, t_exit);
            let mut lanes = [0u32; 4];
            vst1q_u32(lanes.as_mut_ptr(), cmp);
            for (lane, &flag) in lanes.iter().enumerate() {
                if flag != 0 {
                    let idx = base + lane;
                    if idx < self.len {
                        if let Some(candidate) =
                            eval_candidate(self, idx, origin, unit_dir, inv, min_t, max_t)
                        {
                            best = Some(match best {
                                None => candidate,
                                Some(cur) => choose_better_hit(cur, candidate),
                            });
                        }
                    }
                }
            }
            base += 4;
        }
        best
    }
}

#[inline]
fn normalize3(v: [f32; 3]) -> Option<[f32; 3]> {
    let sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    if !sq.is_finite() || sq < 1e-12 {
        return None;
    }
    let inv = sq.sqrt().recip();
    Some([v[0] * inv, v[1] * inv, v[2] * inv])
}

#[inline]
fn safe_inv_dir(d: [f32; 3]) -> [f32; 3] {
    d.map(|c| {
        if c.abs() < 1e-8 {
            if c.is_sign_negative() {
                -1.0e8
            } else {
                1.0e8
            }
        } else {
            1.0 / c
        }
    })
}

#[inline]
fn eval_candidate(
    batch: &SoaAabbBatch,
    i: usize,
    origin: [f32; 3],
    unit_dir: [f32; 3],
    inv: [f32; 3],
    min_t: f32,
    max_t: f32,
) -> Option<RayPickHit> {
    let bmin = [batch.min_x[i], batch.min_y[i], batch.min_z[i]];
    let bmax = [batch.max_x[i], batch.max_y[i], batch.max_z[i]];

    // 1. Exact direct slab intersection
    let tx1 = (bmin[0] - origin[0]) * inv[0];
    let tx2 = (bmax[0] - origin[0]) * inv[0];
    let ty1 = (bmin[1] - origin[1]) * inv[1];
    let ty2 = (bmax[1] - origin[1]) * inv[1];
    let tz1 = (bmin[2] - origin[2]) * inv[2];
    let tz2 = (bmax[2] - origin[2]) * inv[2];

    let t_enter = tx1
        .min(tx2)
        .max(ty1.min(ty2))
        .max(tz1.min(tz2))
        .max(min_t);
    let t_exit = tx1
        .max(tx2)
        .min(ty1.max(ty2))
        .min(tz1.max(tz2))
        .min(max_t);
    if t_enter <= t_exit {
        let hit_point = [
            origin[0] + unit_dir[0] * t_enter,
            origin[1] + unit_dir[1] * t_enter,
            origin[2] + unit_dir[2] * t_enter,
        ];
        return Some(RayPickHit {
            index: i,
            distance: t_enter,
            hit_point,
            direct: true,
            border_distance: 0.0,
        });
    }

    // 2. Angular hit-slop check (2.2° minimum angular target rule)
    let slop_tan = batch.slop_tan[i];
    if slop_tan <= 0.0 {
        return None;
    }
    let center = [
        0.5 * (bmin[0] + bmax[0]),
        0.5 * (bmin[1] + bmax[1]),
        0.5 * (bmin[2] + bmax[2]),
    ];
    let to_center = [
        center[0] - origin[0],
        center[1] - origin[1],
        center[2] - origin[2],
    ];
    let t_proj = (to_center[0] * unit_dir[0]
        + to_center[1] * unit_dir[1]
        + to_center[2] * unit_dir[2])
        .clamp(min_t, max_t);
    let p = [
        origin[0] + unit_dir[0] * t_proj,
        origin[1] + unit_dir[1] * t_proj,
        origin[2] + unit_dir[2] * t_proj,
    ];
    let clamped = [
        p[0].clamp(bmin[0], bmax[0]),
        p[1].clamp(bmin[1], bmax[1]),
        p[2].clamp(bmin[2], bmax[2]),
    ];
    let dx = p[0] - clamped[0];
    let dy = p[1] - clamped[1];
    let dz = p[2] - clamped[2];
    let border_dist = (dx * dx + dy * dy + dz * dz).sqrt();
    let allowed = t_proj * slop_tan;
    if border_dist <= allowed {
        let to_clamped = [
            clamped[0] - origin[0],
            clamped[1] - origin[1],
            clamped[2] - origin[2],
        ];
        let hit_t = (to_clamped[0] * unit_dir[0]
            + to_clamped[1] * unit_dir[1]
            + to_clamped[2] * unit_dir[2])
            .clamp(min_t, max_t);
        Some(RayPickHit {
            index: i,
            distance: hit_t,
            hit_point: clamped,
            direct: false,
            border_distance: border_dist,
        })
    } else {
        None
    }
}

/// Enforces `AUDIT-01`: coplanar or near-coplanar direct geometry hits always beat slop hits.
#[inline]
fn choose_better_hit(a: RayPickHit, b: RayPickHit) -> RayPickHit {
    const COPLANAR_EPS: f32 = 2.5e-3;
    if (a.distance - b.distance).abs() <= COPLANAR_EPS {
        match (a.direct, b.direct) {
            (true, false) => return a,
            (false, true) => return b,
            (false, false) => {
                return if b.border_distance < a.border_distance {
                    b
                } else {
                    a
                };
            }
            (true, true) => {}
        }
    }
    if b.distance < a.distance {
        b
    } else {
        a
    }
}

// ============================================================================
// 2. Vectorized Animation Smoothing (`current += (target - current) * alpha`)
// ============================================================================

/// Smoothly interpolates `values` toward `targets` by `alpha` (`0.0..=1.0`) in-place using SIMD.
pub fn lerp_f32_slice_in_place(values: &mut [f32], targets: &[f32], alpha: f32) {
    let n = values.len().min(targets.len());
    let alpha = alpha.clamp(0.0, 1.0);
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 checked at runtime; slice bounds respected.
            unsafe { lerp_avx2(&mut values[..n], &targets[..n], alpha) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is standard on aarch64; slice bounds respected.
        unsafe { lerp_neon(&mut values[..n], &targets[..n], alpha) };
        return;
    }
    #[allow(unreachable_code)]
    for i in 0..n {
        values[i] += (targets[i] - values[i]) * alpha;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn lerp_avx2(values: &mut [f32], targets: &[f32], alpha: f32) {
    let n = values.len();
    let va = _mm256_set1_ps(alpha);
    let mut i = 0usize;
    while i + 8 <= n {
        let cur = _mm256_loadu_ps(values.as_ptr().add(i));
        let tgt = _mm256_loadu_ps(targets.as_ptr().add(i));
        let next = _mm256_add_ps(cur, _mm256_mul_ps(_mm256_sub_ps(tgt, cur), va));
        _mm256_storeu_ps(values.as_mut_ptr().add(i), next);
        i += 8;
    }
    while i < n {
        values[i] += (targets[i] - values[i]) * alpha;
        i += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn lerp_neon(values: &mut [f32], targets: &[f32], alpha: f32) {
    let n = values.len();
    let va = vdupq_n_f32(alpha);
    let mut i = 0usize;
    while i + 4 <= n {
        let cur = vld1q_f32(values.as_ptr().add(i));
        let tgt = vld1q_f32(targets.as_ptr().add(i));
        let next = vaddq_f32(cur, vmulq_f32(vsubq_f32(tgt, cur), va));
        vst1q_f32(values.as_mut_ptr().add(i), next);
        i += 4;
    }
    while i < n {
        values[i] += (targets[i] - values[i]) * alpha;
        i += 1;
    }
}

// ============================================================================
// 3. SIMD Case-Folded Substring & Fuzzy Filename Search
// ============================================================================

/// Fast SIMD case-insensitive substring test (`needle_lower` MUST already be ASCII-lowercased).
pub fn contains_case_insensitive_prelowered(haystack: &str, needle_lower: &str) -> bool {
    let hb = haystack.as_bytes();
    let nb = needle_lower.as_bytes();
    if nb.is_empty() {
        return true;
    }
    if nb.len() > hb.len() {
        return false;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && hb.len() >= 32 {
            // SAFETY: AVX2 checked at runtime.
            return unsafe { contains_ci_avx2(hb, nb) };
        }
        if hb.len() >= 16 {
            // SAFETY: SSE2 always present on x86_64.
            return unsafe { contains_ci_sse2(hb, nb) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if hb.len() >= 16 {
            // SAFETY: NEON always present on aarch64.
            return unsafe { contains_ci_neon(hb, nb) };
        }
    }
    contains_ci_scalar(hb, nb)
}

#[inline]
fn fold_ascii_byte(b: u8) -> u8 {
    if b.is_ascii_uppercase() {
        b + 32
    } else {
        b
    }
}

pub fn contains_ci_scalar(hb: &[u8], nb: &[u8]) -> bool {
    if nb.is_empty() {
        return true;
    }
    if nb.len() > hb.len() {
        return false;
    }
    let first = nb[0];
    let last_pos = hb.len() - nb.len();
    for i in 0..=last_pos {
        if fold_ascii_byte(hb[i]) == first
            && hb[i + 1..i + nb.len()]
                .iter()
                .zip(&nb[1..])
                .all(|(&h, &n)| fold_ascii_byte(h) == n)
        {
            return true;
        }
    }
    false
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn contains_ci_avx2(hb: &[u8], nb: &[u8]) -> bool {
    let first = _mm256_set1_epi8(nb[0] as i8);
    let upper_a_minus_1 = _mm256_set1_epi8((b'A' - 1) as i8);
    let upper_z_plus_1 = _mm256_set1_epi8((b'Z' + 1) as i8);
    let case_bit = _mm256_set1_epi8(0x20);
    let last_pos = hb.len() - nb.len();
    let mut i = 0usize;
    while i + 32 <= hb.len() && i <= last_pos {
        let chunk = _mm256_loadu_si256(hb.as_ptr().add(i) as *const __m256i);
        let ge_a = _mm256_cmpgt_epi8(chunk, upper_a_minus_1);
        let le_z = _mm256_cmpgt_epi8(upper_z_plus_1, chunk);
        let is_upper = _mm256_and_si256(ge_a, le_z);
        let folded = _mm256_or_si256(chunk, _mm256_and_si256(is_upper, case_bit));
        let eq = _mm256_cmpeq_epi8(folded, first);
        let mut mask = _mm256_movemask_epi8(eq) as u32;
        while mask != 0 {
            let lane = mask.trailing_zeros() as usize;
            mask &= mask - 1;
            let pos = i + lane;
            if pos <= last_pos
                && hb[pos + 1..pos + nb.len()]
                    .iter()
                    .zip(&nb[1..])
                    .all(|(&h, &n)| fold_ascii_byte(h) == n)
            {
                return true;
            }
        }
        i += 32;
    }
    if i <= last_pos {
        contains_ci_scalar(&hb[i..], nb)
    } else {
        false
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn contains_ci_sse2(hb: &[u8], nb: &[u8]) -> bool {
    let first = _mm_set1_epi8(nb[0] as i8);
    let upper_a_minus_1 = _mm_set1_epi8((b'A' - 1) as i8);
    let upper_z_plus_1 = _mm_set1_epi8((b'Z' + 1) as i8);
    let case_bit = _mm_set1_epi8(0x20);
    let last_pos = hb.len() - nb.len();
    let mut i = 0usize;
    while i + 16 <= hb.len() && i <= last_pos {
        let chunk = _mm_loadu_si128(hb.as_ptr().add(i) as *const __m128i);
        let ge_a = _mm_cmpgt_epi8(chunk, upper_a_minus_1);
        let le_z = _mm_cmpgt_epi8(upper_z_plus_1, chunk);
        let is_upper = _mm_and_si128(ge_a, le_z);
        let folded = _mm_or_si128(chunk, _mm_and_si128(is_upper, case_bit));
        let eq = _mm_cmpeq_epi8(folded, first);
        let mut mask = _mm_movemask_epi8(eq) as u32;
        while mask != 0 {
            let lane = mask.trailing_zeros() as usize;
            mask &= mask - 1;
            let pos = i + lane;
            if pos <= last_pos
                && hb[pos + 1..pos + nb.len()]
                    .iter()
                    .zip(&nb[1..])
                    .all(|(&h, &n)| fold_ascii_byte(h) == n)
            {
                return true;
            }
        }
        i += 16;
    }
    if i <= last_pos {
        contains_ci_scalar(&hb[i..], nb)
    } else {
        false
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn contains_ci_neon(hb: &[u8], nb: &[u8]) -> bool {
    let first = vdupq_n_u8(nb[0]);
    let upper_a = vdupq_n_u8(b'A');
    let upper_z = vdupq_n_u8(b'Z');
    let case_bit = vdupq_n_u8(0x20);
    let last_pos = hb.len() - nb.len();
    let mut i = 0usize;
    while i + 16 <= hb.len() && i <= last_pos {
        let chunk = vld1q_u8(hb.as_ptr().add(i));
        let is_upper = vandq_u8(vcgeq_u8(chunk, upper_a), vcleq_u8(chunk, upper_z));
        let folded = vorrq_u8(chunk, vandq_u8(is_upper, case_bit));
        let eq = vceqq_u8(folded, first);
        if vmaxvq_u8(eq) != 0 {
            let mut lanes = [0u8; 16];
            vst1q_u8(lanes.as_mut_ptr(), eq);
            for (lane, &flag) in lanes.iter().enumerate() {
                if flag != 0 {
                    let pos = i + lane;
                    if pos <= last_pos
                        && hb[pos + 1..pos + nb.len()]
                            .iter()
                            .zip(&nb[1..])
                            .all(|(&h, &n)| fold_ascii_byte(h) == n)
                    {
                        return true;
                    }
                }
            }
        }
        i += 16;
    }
    if i <= last_pos {
        contains_ci_scalar(&hb[i..], nb)
    } else {
        false
    }
}

/// Scores `candidate` against a user search `query` (`None` when unmatched).
/// Combines SIMD exact/substring acceleration with subsequence boundary scoring.
pub fn score_filename_match(candidate: &str, query: &str) -> Option<u32> {
    let q = query.trim();
    if q.is_empty() {
        return Some(1);
    }
    let q_lower = q.to_ascii_lowercase();

    // Fast path 1: SIMD exact / prefix / substring match
    if contains_case_insensitive_prelowered(candidate, &q_lower) {
        let cand_lower = candidate.to_ascii_lowercase();
        if cand_lower == q_lower {
            return Some(10_000);
        }
        if cand_lower.starts_with(&q_lower) {
            return Some(8_000u32.saturating_sub(candidate.len() as u32));
        }
        let pos = cand_lower.find(&q_lower).unwrap_or(0);
        let boundary = pos == 0
            || matches!(
                cand_lower.as_bytes().get(pos.wrapping_sub(1)),
                Some(b'_' | b'-' | b'.' | b' ' | b'/')
            );
        let base = if boundary { 6_000u32 } else { 4_000u32 };
        return Some(base.saturating_sub((pos * 8 + candidate.len()) as u32));
    }

    // Fast path 2: Fuzzy subsequence match (ASCII + UTF-8 chars such as Pinyin/CJK)
    let mut score = 1_000u32;
    let mut c_chars = candidate.chars().peekable();
    let mut prev_sep = true;
    let mut streak = 0u32;
    for qc in q_lower.chars() {
        let mut matched = false;
        while let Some(cc) = c_chars.next() {
            let cc_fold = cc.to_ascii_lowercase();
            let is_sep = matches!(cc, '_' | '-' | '.' | ' ' | '/');
            if cc_fold == qc {
                matched = true;
                streak += 1;
                score += streak * 18 + if prev_sep { 45 } else { 0 };
                prev_sep = is_sep;
                break;
            }
            streak = 0;
            prev_sep = is_sep;
        }
        if !matched {
            return None;
        }
    }
    Some(score.saturating_sub(candidate.len().min(200) as u32))
}

/// Batch filters and ranks `(index, score)` for a slice of names, highest score first.
pub fn filter_and_rank_names<S: AsRef<str>>(names: &[S], query: &str) -> Vec<(usize, u32)> {
    let q = query.trim();
    if q.is_empty() {
        return (0..names.len()).map(|i| (i, 1)).collect();
    }
    let mut hits: Vec<(usize, u32)> = names
        .iter()
        .enumerate()
        .filter_map(|(i, name)| score_filename_match(name.as_ref(), q).map(|s| (i, s)))
        .collect();
    hits.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    hits
}

// ============================================================================
// 4. SIMD RGBA8 Thumbnail / Mipmap 2x2 Box Downsampler & Premultiplied Alpha
// ============================================================================

/// Converts straight-alpha RGBA8 pixels to premultiplied-alpha RGBA8 in-place using SIMD.
pub fn premultiply_rgba_in_place(rgba: &mut [u8]) {
    let len = rgba.len() & !3;
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && len >= 32 {
            // SAFETY: AVX2 checked at runtime; slice length aligned to 4.
            unsafe { premultiply_avx2(&mut rgba[..len]) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if len >= 16 {
            // SAFETY: NEON standard on aarch64.
            unsafe { premultiply_neon(&mut rgba[..len]) };
            return;
        }
    }
    premultiply_scalar(&mut rgba[..len]);
}

pub fn premultiply_scalar(rgba: &mut [u8]) {
    for px in rgba.chunks_exact_mut(4) {
        let a = px[3] as u16;
        px[0] = ((px[0] as u16 * a + 127) / 255) as u8;
        px[1] = ((px[1] as u16 * a + 127) / 255) as u8;
        px[2] = ((px[2] as u16 * a + 127) / 255) as u8;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn premultiply_avx2(rgba: &mut [u8]) {
    let mut i = 0usize;
    let zero = _mm256_setzero_si256();
    let bias = _mm256_set1_epi16(128);
    let alpha_shuf = _mm256_setr_epi8(
        6, 7, 6, 7, 6, 7, -1, -1, 14, 15, 14, 15, 14, 15, -1, -1,
        6, 7, 6, 7, 6, 7, -1, -1, 14, 15, 14, 15, 14, 15, -1, -1,
    );
    let alpha_keep = _mm256_setr_epi16(
        0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255,
    );
    while i + 32 <= rgba.len() {
        let raw = _mm256_loadu_si256(rgba.as_ptr().add(i) as *const __m256i);
        let lo = _mm256_unpacklo_epi8(raw, zero);
        let hi = _mm256_unpackhi_epi8(raw, zero);

        let a_lo = _mm256_or_si256(_mm256_shuffle_epi8(lo, alpha_shuf), alpha_keep);
        let a_hi = _mm256_or_si256(_mm256_shuffle_epi8(hi, alpha_shuf), alpha_keep);

        // Exact division by 255 for x in 0..=65025: ((x + 128) + ((x + 128) >> 8)) >> 8
        let prod_lo = _mm256_add_epi16(_mm256_mullo_epi16(lo, a_lo), bias);
        let res_lo = _mm256_srli_epi16(_mm256_add_epi16(prod_lo, _mm256_srli_epi16(prod_lo, 8)), 8);

        let prod_hi = _mm256_add_epi16(_mm256_mullo_epi16(hi, a_hi), bias);
        let res_hi = _mm256_srli_epi16(_mm256_add_epi16(prod_hi, _mm256_srli_epi16(prod_hi, 8)), 8);

        let packed = _mm256_packus_epi16(res_lo, res_hi);
        _mm256_storeu_si256(rgba.as_mut_ptr().add(i) as *mut __m256i, packed);
        i += 32;
    }
    if i < rgba.len() {
        premultiply_scalar(&mut rgba[i..]);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn premultiply_neon(rgba: &mut [u8]) {
    let mut i = 0usize;
    let half = vdupq_n_u16(128);
    // vld4_u8/vst4_u8 process eight interleaved RGBA pixels (32 bytes).
    while i + 32 <= rgba.len() {
        let px = vld4_u8(rgba.as_ptr().add(i));
        let a16 = vmovl_u8(px.3);
        let scale = |chan: uint8x8_t| -> uint8x8_t {
            let c16 = vmovl_u8(chan);
            let prod = vaddq_u16(vmulq_u16(c16, a16), half);
            let div = vshrq_n_u16(vaddq_u16(prod, vshrq_n_u16(prod, 8)), 8);
            vmovn_u16(div)
        };
        let out = uint8x8x4_t(scale(px.0), scale(px.1), scale(px.2), px.3);
        vst4_u8(rgba.as_mut_ptr().add(i), out);
        i += 32;
    }
    if i < rgba.len() {
        premultiply_scalar(&mut rgba[i..]);
    }
}

/// Downsamples an RGBA8 image by 2x using a 2x2 box filter (`max(1, width/2) x max(1, height/2)`).
pub fn downsample_rgba_2x(src: &[u8], width: u32, height: u32) -> (Vec<u8>, u32, u32) {
    let w = width as usize;
    let h = height as usize;
    assert!(src.len() >= w * h * 4, "RGBA source buffer too small");
    let dst_w = (width / 2).max(1);
    let dst_h = (height / 2).max(1);
    if width < 2 || height < 2 {
        return downsample_rgba_2x_scalar(src, width, height);
    }
    let mut dst = vec![0u8; (dst_w as usize) * (dst_h as usize) * 4];
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && dst_w >= 8 {
            // SAFETY: AVX2 checked at runtime; row bounds validated.
            unsafe { downsample_2x_avx2(src, w, dst_w as usize, dst_h as usize, &mut dst) };
            return (dst, dst_w, dst_h);
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if dst_w >= 4 {
            // SAFETY: NEON standard on aarch64; row bounds validated.
            unsafe { downsample_2x_neon(src, w, dst_w as usize, dst_h as usize, &mut dst) };
            return (dst, dst_w, dst_h);
        }
    }
    downsample_rgba_2x_scalar(src, width, height)
}

pub fn downsample_rgba_2x_scalar(src: &[u8], width: u32, height: u32) -> (Vec<u8>, u32, u32) {
    let w = width as usize;
    let h = height as usize;
    let dst_w = (w / 2).max(1);
    let dst_h = (h / 2).max(1);
    let mut dst = vec![0u8; dst_w * dst_h * 4];
    for dy in 0..dst_h {
        let sy0 = (dy * 2).min(h - 1);
        let sy1 = (dy * 2 + 1).min(h - 1);
        for dx in 0..dst_w {
            let sx0 = (dx * 2).min(w - 1);
            let sx1 = (dx * 2 + 1).min(w - 1);
            let p00 = &src[(sy0 * w + sx0) * 4..][..4];
            let p10 = &src[(sy0 * w + sx1) * 4..][..4];
            let p01 = &src[(sy1 * w + sx0) * 4..][..4];
            let p11 = &src[(sy1 * w + sx1) * 4..][..4];
            let out = &mut dst[(dy * dst_w + dx) * 4..][..4];
            for c in 0..4 {
                let sum = p00[c] as u16 + p10[c] as u16 + p01[c] as u16 + p11[c] as u16;
                out[c] = ((sum + 2) >> 2) as u8;
            }
        }
    }
    (dst, dst_w as u32, dst_h as u32)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn downsample_2x_avx2(
    src: &[u8],
    src_w: usize,
    dst_w: usize,
    dst_h: usize,
    dst: &mut [u8],
) {
    let stride = src_w * 4;
    let zero = _mm256_setzero_si256();
    let round = _mm256_set1_epi16(2);
    // Reorder [out0, out1, out4, out5 | out2, out3, out6, out7] -> [out0..out7]
    let perm = _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7);

    for dy in 0..dst_h {
        let row0 = src.as_ptr().add(dy * 2 * stride);
        let row1 = src.as_ptr().add((dy * 2 + 1) * stride);
        let out_row = dst.as_mut_ptr().add(dy * dst_w * 4);
        let mut dx = 0usize;
        while dx + 8 <= dst_w {
            // Process 16 source RGBA pixels (64 bytes) -> 8 destination RGBA pixels (32 bytes)
            let a0 = _mm256_loadu_si256(row0.add(dx * 8) as *const __m256i);
            let a1 = _mm256_loadu_si256(row0.add(dx * 8 + 32) as *const __m256i);
            let b0 = _mm256_loadu_si256(row1.add(dx * 8) as *const __m256i);
            let b1 = _mm256_loadu_si256(row1.add(dx * 8 + 32) as *const __m256i);

            let vsum0_lo = _mm256_add_epi16(
                _mm256_unpacklo_epi8(a0, zero),
                _mm256_unpacklo_epi8(b0, zero),
            );
            let vsum0_hi = _mm256_add_epi16(
                _mm256_unpackhi_epi8(a0, zero),
                _mm256_unpackhi_epi8(b0, zero),
            );
            let vsum1_lo = _mm256_add_epi16(
                _mm256_unpacklo_epi8(a1, zero),
                _mm256_unpacklo_epi8(b1, zero),
            );
            let vsum1_hi = _mm256_add_epi16(
                _mm256_unpackhi_epi8(a1, zero),
                _mm256_unpackhi_epi8(b1, zero),
            );

            // In each 128-bit lane, unpacklo holds pixels [0, 1] (or [4, 5]), unpackhi holds [2, 3] (or [6, 7]).
            // Shuffle 64-bit halves to add pixel 2k + pixel 2k+1!
            let pair_sum = |lo: __m256i, hi: __m256i| -> __m256i {
                let even = _mm256_unpacklo_epi64(lo, hi); // px0, px2 | px4, px6
                let odd = _mm256_unpackhi_epi64(lo, hi); // px1, px3 | px5, px7
                _mm256_srli_epi16(_mm256_add_epi16(_mm256_add_epi16(even, odd), round), 2)
            };
            let s0 = pair_sum(vsum0_lo, vsum0_hi); // 4 pixels (16 u16s): px0, px2 across lanes -> wait: [px0, px1 | px2, px3]
            let s1 = pair_sum(vsum1_lo, vsum1_hi);
            let packed = _mm256_packus_epi16(s0, s1);
            let ordered = _mm256_permutevar8x32_epi32(packed, perm);
            _mm256_storeu_si256(out_row.add(dx * 4) as *mut __m256i, ordered);
            dx += 8;
        }
        while dx < dst_w {
            let sx0 = dx * 2;
            let sx1 = dx * 2 + 1;
            for c in 0..4 {
                let sum = *row0.add(sx0 * 4 + c) as u16
                    + *row0.add(sx1 * 4 + c) as u16
                    + *row1.add(sx0 * 4 + c) as u16
                    + *row1.add(sx1 * 4 + c) as u16;
                *out_row.add(dx * 4 + c) = ((sum + 2) >> 2) as u8;
            }
            dx += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn downsample_2x_neon(
    src: &[u8],
    src_w: usize,
    dst_w: usize,
    dst_h: usize,
    dst: &mut [u8],
) {
    let stride = src_w * 4;
    for dy in 0..dst_h {
        let row0 = src.as_ptr().add(dy * 2 * stride);
        let row1 = src.as_ptr().add((dy * 2 + 1) * stride);
        let out_row = dst.as_mut_ptr().add(dy * dst_w * 4);
        let mut dx = 0usize;
        while dx + 4 <= dst_w {
            // Load 8 RGBA pixels (deinterleaved channels) from row0 and row1
            let top = vld4_u8(row0.add(dx * 8));
            let bot = vld4_u8(row1.add(dx * 8));
            let avg_chan = |t: uint8x8_t, b: uint8x8_t| -> [u8; 4] {
                let h_top = vpaddl_u8(t); // 4 u16s: p00+p10
                let h_bot = vpaddl_u8(b); // 4 u16s: p01+p11
                let sum = vadd_u16(h_top, h_bot);
                let shifted = vrshr_n_u16(sum, 2);
                let mut lane = [0u16; 4];
                vst1_u16(lane.as_mut_ptr(), shifted);
                [lane[0] as u8, lane[1] as u8, lane[2] as u8, lane[3] as u8]
            };
            let r = avg_chan(top.0, bot.0);
            let g = avg_chan(top.1, bot.1);
            let b = avg_chan(top.2, bot.2);
            let a = avg_chan(top.3, bot.3);
            for k in 0..4 {
                *out_row.add((dx + k) * 4) = r[k];
                *out_row.add((dx + k) * 4 + 1) = g[k];
                *out_row.add((dx + k) * 4 + 2) = b[k];
                *out_row.add((dx + k) * 4 + 3) = a[k];
            }
            dx += 4;
        }
        while dx < dst_w {
            let sx0 = dx * 2;
            let sx1 = dx * 2 + 1;
            for c in 0..4 {
                let sum = *row0.add(sx0 * 4 + c) as u16
                    + *row0.add(sx1 * 4 + c) as u16
                    + *row1.add(sx0 * 4 + c) as u16
                    + *row1.add(sx1 * 4 + c) as u16;
                *out_row.add(dx * 4 + c) = ((sum + 2) >> 2) as u8;
            }
            dx += 1;
        }
    }
}

/// Generates a full RGBA8 mipmap chain (level 0 + downsampled levels down to 1x1).
pub fn generate_mip_chain_rgba(level0: Vec<u8>, width: u32, height: u32) -> Vec<(Vec<u8>, u32, u32)> {
    let mut chain = Vec::new();
    let mut cur_w = width.max(1);
    let mut cur_h = height.max(1);
    chain.push((level0, cur_w, cur_h));
    while cur_w > 1 || cur_h > 1 {
        let (prev, pw, ph) = chain.last().unwrap();
        let (next, nw, nh) = downsample_rgba_2x(prev, *pw, *ph);
        cur_w = nw;
        cur_h = nh;
        chain.push((next, nw, nh));
    }
    chain
}

// ============================================================================
// 5. SIMD Byte Inspection: Line Count, Binary Detection & Shannon Entropy
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ByteInspection {
    pub total_bytes: usize,
    pub newline_count: usize,
    pub nul_bytes: usize,
    /// Shannon entropy in `[0.0, 8.0]` bits per byte.
    pub entropy_bits: f32,
    pub is_likely_binary: bool,
}

/// Inspects a byte slice with SIMD newline/NUL scanning and 4-way unrolled histogram entropy.
pub fn inspect_bytes_simd(data: &[u8]) -> ByteInspection {
    if data.is_empty() {
        return ByteInspection {
            total_bytes: 0,
            newline_count: 0,
            nul_bytes: 0,
            entropy_bits: 0.0,
            is_likely_binary: false,
        };
    }

    let (newlines, nuls) = count_newlines_and_nuls(data);

    // 4-way unrolled histogram to break store-to-load forwarding stalls on deeply pipelined CPUs
    let mut h0 = [0u32; 256];
    let mut h1 = [0u32; 256];
    let mut h2 = [0u32; 256];
    let mut h3 = [0u32; 256];
    let mut chunks = data.chunks_exact(4);
    for c in &mut chunks {
        h0[c[0] as usize] += 1;
        h1[c[1] as usize] += 1;
        h2[c[2] as usize] += 1;
        h3[c[3] as usize] += 1;
    }
    for &b in chunks.remainder() {
        h0[b as usize] += 1;
    }

    let inv_len = 1.0f64 / (data.len() as f64);
    let mut entropy = 0.0f64;
    let mut control_non_ws = 0usize;
    for byte in 0..256 {
        let count = h0[byte] + h1[byte] + h2[byte] + h3[byte];
        if count > 0 {
            let p = (count as f64) * inv_len;
            entropy -= p * p.log2();
            if byte < 0x20 && !matches!(byte as u8, b'\n' | b'\r' | b'\t' | 0x0c | 0x1b) {
                control_non_ws += count as usize;
            }
        }
    }

    let is_likely_binary = nuls > 0 || (control_non_ws * 20 > data.len());
    ByteInspection {
        total_bytes: data.len(),
        newline_count: newlines,
        nul_bytes: nuls,
        entropy_bits: (entropy as f32).clamp(0.0, 8.0),
        is_likely_binary,
    }
}

fn count_newlines_and_nuls(data: &[u8]) -> (usize, usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && data.len() >= 32 {
            // SAFETY: AVX2 checked at runtime.
            return unsafe { count_nl_nul_avx2(data) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if data.len() >= 16 {
            // SAFETY: NEON standard on aarch64.
            return unsafe { count_nl_nul_neon(data) };
        }
    }
    count_nl_nul_scalar(data)
}

fn count_nl_nul_scalar(data: &[u8]) -> (usize, usize) {
    let mut nl = 0usize;
    let mut nul = 0usize;
    for &b in data {
        nl += usize::from(b == b'\n');
        nul += usize::from(b == 0);
    }
    (nl, nul)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn count_nl_nul_avx2(data: &[u8]) -> (usize, usize) {
    let v_nl = _mm256_set1_epi8(b'\n' as i8);
    let v_zero = _mm256_setzero_si256();
    let mut nl = 0usize;
    let mut nul = 0usize;
    let mut i = 0usize;
    while i + 32 <= data.len() {
        let chunk = _mm256_loadu_si256(data.as_ptr().add(i) as *const __m256i);
        nl += (_mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, v_nl)) as u32).count_ones() as usize;
        nul += (_mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, v_zero)) as u32).count_ones() as usize;
        i += 32;
    }
    let (rem_nl, rem_nul) = count_nl_nul_scalar(&data[i..]);
    (nl + rem_nl, nul + rem_nul)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn count_nl_nul_neon(data: &[u8]) -> (usize, usize) {
    let v_nl = vdupq_n_u8(b'\n');
    let v_zero = vdupq_n_u8(0);
    let one = vdupq_n_u8(1);
    let mut nl = 0usize;
    let mut nul = 0usize;
    let mut i = 0usize;
    while i + 16 <= data.len() {
        let chunk = vld1q_u8(data.as_ptr().add(i));
        let m_nl = vandq_u8(vceqq_u8(chunk, v_nl), one);
        let m_zr = vandq_u8(vceqq_u8(chunk, v_zero), one);
        nl += vaddvq_u8(m_nl) as usize;
        nul += vaddvq_u8(m_zr) as usize;
        i += 16;
    }
    let (rem_nl, rem_nul) = count_nl_nul_scalar(&data[i..]);
    (nl + rem_nl, nul + rem_nul)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simd_ray_pick_matches_scalar_and_honors_audit01_direct_priority() {
        let mut batch = SoaAabbBatch::with_capacity(32);
        let slop = 2.2f32.to_radians();
        // Adjacent coplanar buttons at z = -1.75m:
        // Button 0 (earlier in traversal!) sits just to the left [-0.20, -0.02]
        batch.push([-0.20, -0.05, -1.76], [-0.02, 0.05, -1.74], slop);
        // Button 1 sits immediately adjacent [0.00, 0.18]
        batch.push([0.00, -0.05, -1.76], [0.18, 0.05, -1.74], slop);
        // Add 25 more cards across an arc to exercise multi-chunk SIMD loops
        for k in 0..25 {
            let x = -1.2 + (k as f32) * 0.1;
            let z = -1.8 - (k % 3) as f32 * 0.15;
            batch.push([x, 0.2, z - 0.02], [x + 0.07, 0.32, z + 0.02], slop);
        }
        batch.finalize_padding();

        // Ray aimed 8mm inside Button 1 (x = +0.008m at z = -1.75m), which ALSO falls inside
        // Button 0's 2.2° slop halo (~67mm)! AUDIT-01 requires Button 1 (direct) to win!
        let dir = normalize3([0.008, 0.0, -1.75]).unwrap();
        let simd_hit = batch.pick_ray([0.0, 0.0, 0.0], dir, 0.05, 10.0).unwrap();
        let scalar_hit = batch
            .pick_ray_scalar([0.0, 0.0, 0.0], dir, 0.05, 10.0)
            .unwrap();
        assert_eq!(simd_hit.index, 1, "direct hit must beat earlier slop hit");
        assert!(simd_hit.direct);
        assert_eq!(simd_hit.index, scalar_hit.index);
        assert!((simd_hit.distance - scalar_hit.distance).abs() < 1e-5);

        // Sweep 60 deterministic rays and verify 100% SIMD vs Scalar parity
        for step in 0..60 {
            let rx = -1.35 + (step as f32) * 0.045;
            let ry = -0.10 + ((step % 7) as f32) * 0.07;
            let d = normalize3([rx, ry, -1.75]).unwrap();
            let h_simd = batch.pick_ray([0.02, 0.01, 0.0], d, 0.05, 10.0);
            let h_scal = batch.pick_ray_scalar([0.02, 0.01, 0.0], d, 0.05, 10.0);
            assert_eq!(
                h_simd.map(|h| (h.index, h.direct)),
                h_scal.map(|h| (h.index, h.direct)),
                "SIMD/scalar mismatch on ray step {step}"
            );
        }
    }

    #[test]
    fn simd_case_insensitive_and_fuzzy_search_parity() {
        let long_name = "openKylin_XR_Spatial_Architecture_Specification_2026_v3_Final_Report.markdown";
        for needle in [
            "openkylin",
            "specification",
            "report.markdown",
            "final_report",
            "nonexistent_token",
        ] {
            assert_eq!(
                contains_case_insensitive_prelowered(long_name, needle),
                contains_ci_scalar(long_name.as_bytes(), needle.as_bytes()),
                "mismatch for needle {needle}"
            );
        }

        let files = [
            "README.md",
            "xr_spatial_renderer.rs",
            "xr-cinema-demo.mp4",
            "空间影院演示片.webm",
            "libopenxr_monado.so",
        ];
        let ranked = filter_and_rank_names(&files, "xrsr");
        assert_eq!(ranked[0].0, 1);
        let cjk = filter_and_rank_names(&files, "影院");
        assert_eq!(cjk.len(), 1);
        assert_eq!(cjk[0].0, 3);
    }

    #[test]
    fn premultiply_preserves_buffer_bounds_and_scalar_values() {
        for len in [0, 4, 12, 16, 20, 28, 32, 36, 48, 60, 64, 68] {
            // Keep a readable guard after the slice to detect NEON stores that
            // exceed the requested length, including a partial eight-pixel batch.
            let mut actual: Vec<u8> = (0..len + 32)
                .map(|i| ((i * 37 + 19) & 0xff) as u8)
                .collect();
            let mut expected = actual.clone();
            premultiply_rgba_in_place(&mut actual[..len]);
            premultiply_scalar(&mut expected[..len]);
            assert_eq!(&actual[len..], &expected[len..], "guard overwritten at {len}");
            for (a, b) in actual[..len].iter().zip(&expected[..len]) {
                assert!(a.abs_diff(*b) <= 1, "premultiply diff at length {len}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn simd_rgba_downsample_and_premultiply_match_scalar() {
        let w = 36u32;
        let h = 20u32;
        let mut img = vec![0u8; (w * h * 4) as usize];
        for (i, b) in img.iter_mut().enumerate() {
            *b = ((i * 37 + 19) & 0xff) as u8;
        }
        let mut pm_simd = img.clone();
        let mut pm_scal = img.clone();
        premultiply_rgba_in_place(&mut pm_simd);
        premultiply_scalar(&mut pm_scal);
        for (a, b) in pm_simd.iter().zip(pm_scal.iter()) {
            assert!(a.abs_diff(*b) <= 1, "premultiply diff: {a} vs {b}");
        }

        let (ds_simd, dw1, dh1) = downsample_rgba_2x(&img, w, h);
        let (ds_scal, dw2, dh2) = downsample_rgba_2x_scalar(&img, w, h);
        assert_eq!((dw1, dh1), (dw2, dh2));
        assert_eq!(ds_simd, ds_scal, "SIMD 2x box filter must match scalar bit-for-bit");

        let mips = generate_mip_chain_rgba(img, w, h);
        assert_eq!(mips.last().map(|(_, mw, mh)| (*mw, *mh)), Some((1, 1)));
    }

    #[test]
    fn simd_byte_inspection_counts_lines_and_entropy() {
        let text = "line 1\nline 2\nline 3\n".repeat(40);
        let info = inspect_bytes_simd(text.as_bytes());
        assert_eq!(info.newline_count, 120);
        assert_eq!(info.nul_bytes, 0);
        assert!(!info.is_likely_binary);
        assert!(info.entropy_bits > 2.5 && info.entropy_bits < 4.5);

        let mut bin = vec![0u8; 512];
        for (i, b) in bin.iter_mut().enumerate() {
            *b = (i & 0xff) as u8;
        }
        let bin_info = inspect_bytes_simd(&bin);
        assert!(bin_info.is_likely_binary);
        assert!((bin_info.entropy_bits - 8.0).abs() < 0.01);
    }
}
