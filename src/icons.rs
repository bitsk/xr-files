// SPDX-License-Identifier: GPL-3.0-or-later
//! Procedural 4x4-supersampled UKUI XR "Qingkong" (`晴空`) 2.5D squircle & golden folder icons,
//! plus procedural fallback Qingkong 360° clear-sky panorama generator.

use crate::{
    fs_ops::FileKind,
    preview::MipmappedRgba,
    simd,
};

pub const BUILTIN_ICON_COUNT: u32 = 8;
const ICON_EDGE: u32 = 128;

pub fn icon_slot_for_kind(kind: FileKind) -> u32 {
    match kind {
        FileKind::Directory => 0,
        FileKind::Document => 1,
        FileKind::Video => 2,
        FileKind::Code | FileKind::Executable => 3,
        FileKind::Image => 4,
        FileKind::Model3D => 5,
        FileKind::Archive => 6,
        FileKind::Audio | FileKind::Other => 7,
    }
}

/// Synthesizes all 8 UKUI XR Qingkong 2.5D file/folder icons as premultiplied, SIMD-mipmapped RGBA8 textures.
pub fn build_qingkong_icons() -> Vec<(u32, MipmappedRgba)> {
    (0..BUILTIN_ICON_COUNT)
        .map(|slot| {
            let mut raw = render_icon_supersampled(slot, ICON_EDGE);
            simd::premultiply_rgba_in_place(&mut raw);
            let mips = simd::generate_mip_chain_rgba(raw, ICON_EDGE, ICON_EDGE);
            (
                slot,
                MipmappedRgba {
                    mips,
                    original_dimensions: [ICON_EDGE, ICON_EDGE],
                },
            )
        })
        .collect()
}

fn render_icon_supersampled(slot: u32, edge: u32) -> Vec<u8> {
    let mut out = vec![0u8; (edge * edge * 4) as usize];
    let inv = 1.0 / edge as f32;
    // 3x3 subpixel supersampling for silky-smooth rounded edges and vector glyphs
    let offsets = [-0.33f32, 0.0, 0.33];

    for py in 0..edge {
        for px in 0..edge {
            let mut acc = [0.0f32; 4];
            for &oy in &offsets {
                for &ox in &offsets {
                    let u = (px as f32 + 0.5 + ox) * inv * 2.0 - 1.0;
                    let v = (py as f32 + 0.5 + oy) * inv * 2.0 - 1.0;
                    let c = sample_icon_sdf(slot, u, v);
                    // Accumulate premultiplied in sample space for clean edge anti-aliasing
                    acc[0] += c[0] * c[3];
                    acc[1] += c[1] * c[3];
                    acc[2] += c[2] * c[3];
                    acc[3] += c[3];
                }
            }
            let a = (acc[3] / 9.0).clamp(0.0, 1.0);
            let (r, g, b) = if a > 1e-4 {
                (
                    (acc[0] / 9.0 / a).clamp(0.0, 1.0),
                    (acc[1] / 9.0 / a).clamp(0.0, 1.0),
                    (acc[2] / 9.0 / a).clamp(0.0, 1.0),
                )
            } else {
                (0.0, 0.0, 0.0)
            };
            let idx = ((py * edge + px) * 4) as usize;
            out[idx] = (r * 255.0).round() as u8;
            out[idx + 1] = (g * 255.0).round() as u8;
            out[idx + 2] = (b * 255.0).round() as u8;
            out[idx + 3] = (a * 255.0).round() as u8;
        }
    }
    out
}

/// Evaluates normalized `u, v in [-1, 1]` (`+v` down) for UKUI Qingkong icon `slot`.
fn sample_icon_sdf(slot: u32, u: f32, v: f32) -> [f32; 4] {
    if slot == 0 {
        return sample_golden_folder(u, v);
    }

    // Standard UKUI rounded squircle (`half = 0.84, radius = 0.36`)
    let d_bg = sd_rbox([u, v], [0.84, 0.84], 0.36);
    if d_bg > 0.0 {
        return [0.0, 0.0, 0.0, 0.0];
    }

    let (top_rgb, bot_rgb) = match slot {
        1 => ([1.00, 0.56, 0.16], [0.96, 0.38, 0.08]), // Document warm orange
        2 => ([0.62, 0.44, 1.00], [0.42, 0.25, 0.94]), // Video violet-purple
        3 => ([0.23, 0.31, 0.44], [0.14, 0.20, 0.30]), // Code / Terminal deep slate
        4 => ([0.26, 0.75, 1.00], [0.14, 0.52, 0.98]), // Image azure sky
        5 => ([0.30, 0.60, 1.00], [0.46, 0.32, 0.96]), // 3D Model crystal indigo
        6 => ([1.00, 0.78, 0.22], [0.92, 0.54, 0.10]), // Archive amber gold
        _ => ([0.60, 0.69, 0.80], [0.43, 0.52, 0.66]), // Other periwinkle slate
    };

    let t = (0.5 * (v + 1.0)).clamp(0.0, 1.0);
    let mut rgb = [
        lerp(top_rgb[0], bot_rgb[0], t),
        lerp(top_rgb[1], bot_rgb[1], t),
        lerp(top_rgb[2], bot_rgb[2], t),
    ];

    // Subtle top-left UKUI squircle gloss sheen + soft inner bevel rim
    let gloss = (1.0 - ((u + 0.45).powi(2) + (v + 0.55).powi(2)).sqrt() * 0.95).clamp(0.0, 1.0);
    for c in &mut rgb {
        *c = (*c + gloss * 0.14).clamp(0.0, 1.0);
    }
    if d_bg > -0.045 {
        let rim = ((d_bg + 0.045) / 0.045).clamp(0.0, 1.0) * (0.65 - 0.35 * t);
        for c in &mut rgb {
            *c = lerp(*c, 1.0, rim * 0.42);
        }
    }

    // Foreground vector symbol with subtle drop shadow
    let sym_cov = symbol_coverage(slot, u, v);
    let shd_cov = symbol_coverage(slot, u - 0.02, v - 0.035);
    if shd_cov > 0.01 {
        for c in &mut rgb {
            *c *= 1.0 - 0.18 * shd_cov;
        }
    }
    if sym_cov > 0.005 {
        let fg = if slot == 1 {
            [1.0, 0.96, 0.92]
        } else {
            [0.99, 0.995, 1.0]
        };
        for k in 0..3 {
            rgb[k] = lerp(rgb[k], fg[k], sym_cov);
        }
    }

    [rgb[0], rgb[1], rgb[2], 1.0]
}

/// Sculpted UKUI `#FFC735` golden folder matching `application-icons.png` & `style-reference.png`.
fn sample_golden_folder(u: f32, v: f32) -> [f32; 4] {
    // Soft rounded golden squircle backing (`#FFC735` -> `#F59E0B`)
    let d_bg = sd_rbox([u, v], [0.85, 0.85], 0.35);
    if d_bg > 0.0 {
        return [0.0, 0.0, 0.0, 0.0];
    }
    let t_bg = (0.5 * (v + 1.0)).clamp(0.0, 1.0);
    let mut rgb = [
        lerp(1.00, 0.95, t_bg),
        lerp(0.81, 0.62, t_bg),
        lerp(0.24, 0.09, t_bg),
    ];
    let gloss = (1.0 - ((u + 0.48).powi(2) + (v + 0.52).powi(2)).sqrt()).clamp(0.0, 1.0);
    rgb[0] = (rgb[0] + gloss * 0.10).min(1.0);
    rgb[1] = (rgb[1] + gloss * 0.12).min(1.0);
    rgb[2] = (rgb[2] + gloss * 0.14).min(1.0);

    // Back folder sheet + top-left folder tab
    let d_back = sd_rbox([u, v + 0.03], [0.58, 0.39], 0.11);
    let d_tab = sd_rbox([u + 0.24, v + 0.34], [0.27, 0.16], 0.09);
    let d_rear = d_back.min(d_tab);
    if d_rear <= 0.0 {
        rgb = [1.0, 0.87, 0.42];
    }

    // Soft shadow cast by front pocket onto back sheet
    let d_pocket_shd = sd_rbox([u, v - 0.05], [0.61, 0.35], 0.11);
    if d_pocket_shd <= 0.04 && d_rear <= 0.0 {
        let s = (1.0 - (d_pocket_shd + 0.02).abs() / 0.06).clamp(0.0, 1.0);
        rgb[0] *= 1.0 - 0.10 * s;
        rgb[1] *= 1.0 - 0.12 * s;
        rgb[2] *= 1.0 - 0.14 * s;
    }

    // Warm cream-gold front folder pocket (`#FFF0A6` -> `#FFD75E`)
    let d_front = sd_rbox([u, v - 0.09], [0.61, 0.34], 0.11);
    if d_front <= 0.0 {
        let tf = ((v + 0.25) / 0.68).clamp(0.0, 1.0);
        rgb = [
            lerp(1.00, 1.00, tf),
            lerp(0.95, 0.84, tf),
            lerp(0.64, 0.35, tf),
        ];
        // Crisp top highlight along the folder pocket lip
        if (v + 0.23).abs() < 0.025 {
            rgb = [1.0, 0.98, 0.84];
        }
    }

    [rgb[0], rgb[1], rgb[2], 1.0]
}

fn symbol_coverage(slot: u32, u: f32, v: f32) -> f32 {
    match slot {
        1 => {
            // Document page with 3 horizontal lines cutout
            let page = sd_rbox([u, v], [0.44, 0.52], 0.10);
            if page > 0.0 {
                return 0.0;
            }
            let l1 = sd_rbox([u, v + 0.10], [0.27, 0.038], 0.03);
            let l2 = sd_rbox([u, v - 0.07], [0.27, 0.038], 0.03);
            let l3 = sd_rbox([u + 0.06, v - 0.24], [0.21, 0.038], 0.03);
            if l1 < 0.0 || l2 < 0.0 || l3 < 0.0 {
                0.0
            } else {
                1.0
            }
        }
        2 => {
            // Rounded play triangle centered slightly right (`u - 0.05`)
            let px = u - 0.05;
            let py = v;
            let d1 = -px - 0.28;
            let d2 = 0.5 * px + 0.866 * py - 0.22;
            let d3 = 0.5 * px - 0.866 * py - 0.22;
            let tri = d1.max(d2).max(d3) - 0.06;
            if tri <= 0.0 { 1.0 } else { 0.0 }
        }
        3 => {
            // Terminal `>_` chevron + underscore
            let d_top = sd_segment([u, v], [-0.36, -0.22], [-0.04, 0.04]) - 0.065;
            let d_bot = sd_segment([u, v], [-0.36, 0.30], [-0.04, 0.04]) - 0.065;
            let d_bar = sd_rbox([u - 0.24, v - 0.28], [0.20, 0.055], 0.05);
            if d_top.min(d_bot).min(d_bar) <= 0.0 {
                1.0
            } else {
                0.0
            }
        }
        4 => {
            // Image sun + twin mountain peaks
            let sun = ((u + 0.24).powi(2) + (v + 0.24).powi(2)).sqrt() - 0.13;
            let m1 = (u + 0.14).abs() * 0.95 + (v - 0.05) - 0.0;
            let m2 = (u - 0.22).abs() * 0.85 + (v + 0.05) - 0.0;
            let clip = sd_rbox([u, v], [0.54, 0.44], 0.10);
            if clip <= 0.0 && (sun <= 0.0 || (v > 0.02 && (m1 < 0.32 || m2 < 0.36))) {
                1.0
            } else {
                0.0
            }
        }
        5 => {
            // Isometric 3D wireframe/faceted hexagon cube
            let r = (u * u + v * v).sqrt();
            let hex = sd_rbox([u * 0.866 - v * 0.5, u * 0.5 + v * 0.866], [0.36, 0.36], 0.08);
            let y1 = sd_segment([u, v], [0.0, 0.0], [0.0, 0.42]) - 0.045;
            let y2 = sd_segment([u, v], [0.0, 0.0], [-0.36, -0.20]) - 0.045;
            let y3 = sd_segment([u, v], [0.0, 0.0], [0.36, -0.20]) - 0.045;
            let ring = (hex.abs() - 0.048).min(y1).min(y2).min(y3);
            let _ = r;
            if ring <= 0.0 { 1.0 } else { 0.0 }
        }
        6 => {
            // Archive box + vertical zipper band
            let box_d = sd_rbox([u, v], [0.44, 0.40], 0.10);
            let slot_cut = sd_rbox([u, v + 0.06], [0.08, 0.24], 0.03);
            if box_d <= 0.0 && slot_cut > 0.0 {
                1.0
            } else {
                0.0
            }
        }
        _ => {
            // Clean audio/data bars
            let b1 = sd_rbox([u + 0.24, v], [0.055, 0.20], 0.05);
            let b2 = sd_rbox([u, v], [0.055, 0.36], 0.05);
            let b3 = sd_rbox([u - 0.24, v], [0.055, 0.25], 0.05);
            if b1.min(b2).min(b3) <= 0.0 { 1.0 } else { 0.0 }
        }
    }
}

fn sd_rbox(p: [f32; 2], b: [f32; 2], r: f32) -> f32 {
    let qx = p[0].abs() - b[0] + r;
    let qy = p[1].abs() - b[1] + r;
    let out = (qx.max(0.0).powi(2) + qy.max(0.0).powi(2)).sqrt();
    out + qx.max(qy).min(0.0) - r
}

fn sd_segment(p: [f32; 2], a: [f32; 2], b: [f32; 2]) -> f32 {
    let pa = [p[0] - a[0], p[1] - a[1]];
    let ba = [b[0] - a[0], b[1] - a[1]];
    let h = ((pa[0] * ba[0] + pa[1] * ba[1]) / (ba[0] * ba[0] + ba[1] * ba[1]).max(1e-6))
        .clamp(0.0, 1.0);
    ((pa[0] - ba[0] * h).powi(2) + (pa[1] - ba[1] * h).powi(2)).sqrt()
}

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Generates a procedural fallback `Qingkong` (`晴空`) 360° equirectangular RGBA8 panorama
/// when `/usr/share/xr/environments/qingkong/qingkong-360.png` is not installed on the host.
pub fn synthesize_qingkong_panorama(width: u32, height: u32) -> Vec<u8> {
    let mut pixels = vec![0u8; (width * height * 4) as usize];
    for y in 0..height {
        let v = (y as f32 + 0.5) / height as f32; // 0 = zenith (+Y), 0.5 = horizon, 1 = nadir (-Y)
        for x in 0..width {
            let u = (x as f32 + 0.5) / width as f32;
            let az = u * std::f32::consts::TAU;

            let rgb = if v <= 0.505 {
                // Upper hemisphere: clear sky #6CA8FF zenith -> #9EC8FF mid -> #F3F8FF glowing horizon
                let t = (v / 0.505).clamp(0.0, 1.0);
                let sky_r = lerp(0.42, 0.95, t.powf(1.55));
                let sky_g = lerp(0.66, 0.97, t.powf(1.35));
                let sky_b = lerp(0.99, 1.00, t.powf(0.85));

                // Translucent Qingkong silk-ribbon mountains near horizon (`v in 0.43..0.505`)
                let m_wave1 = 0.042 * (az * 2.0 + 0.8).sin().max(0.0).powf(1.6)
                    + 0.025 * (az * 5.0 - 0.4).cos().abs();
                let m_wave2 = 0.036 * (az * 3.0 - 1.5).sin().max(0.0).powf(1.4);
                let h_dist = (0.503 - v).max(0.0);
                let mut col = [sky_r, sky_g, sky_b];
                if h_dist < m_wave1 {
                    let k = (1.0 - h_dist / m_wave1.max(1e-4)).clamp(0.0, 1.0) * 0.58;
                    col[0] = lerp(col[0], 0.36, k);
                    col[1] = lerp(col[1], 0.78, k);
                    col[2] = lerp(col[2], 0.99, k);
                }
                if h_dist < m_wave2 {
                    let k = (1.0 - h_dist / m_wave2.max(1e-4)).clamp(0.0, 1.0) * 0.48;
                    col[0] = lerp(col[0], 0.74, k);
                    col[1] = lerp(col[1], 0.66, k);
                    col[2] = lerp(col[2], 0.99, k);
                }
                col
            } else {
                // Lower hemisphere: calm misty mirror water #DCE9FF -> #9FBDF2
                let t = ((v - 0.505) / 0.495).clamp(0.0, 1.0);
                let shimmer = 0.02 * (az * 8.0).sin() * (1.0 - t);
                [
                    (lerp(0.89, 0.64, t.powf(0.65)) + shimmer).clamp(0.0, 1.0),
                    (lerp(0.93, 0.75, t.powf(0.68)) + shimmer).clamp(0.0, 1.0),
                    (lerp(0.99, 0.96, t.powf(0.80))).clamp(0.0, 1.0),
                ]
            };

            let idx = ((y * width + x) * 4) as usize;
            pixels[idx] = (rgb[0] * 255.0).round() as u8;
            pixels[idx + 1] = (rgb[1] * 255.0).round() as u8;
            pixels[idx + 2] = (rgb[2] * 255.0).round() as u8;
            pixels[idx + 3] = 255;
        }
    }
    pixels
}
