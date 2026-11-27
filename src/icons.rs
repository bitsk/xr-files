// SPDX-License-Identifier: GPL-3.0-or-later
//! Procedural supersampled UKUI XR "Qingkong" (`晴空`) 2.5D squircle & golden folder icons,
//! tintable crisp vector SDF spatial UI icon atlas (`UiIcon`), and fallback 360° Qingkong panorama.

use crate::{fs_ops::FileKind, preview::MipmappedRgba, simd};

pub const BUILTIN_ICON_COUNT: u32 = 38;
const ICON_EDGE: u32 = 128;

/// Tintable monochrome vector SDF icons (`tex_kind == 5`, rendered via linear-alpha coverage).
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UiIcon {
    Home = 8,
    Desktop = 9,
    Docs = 10,
    Downloads = 11,
    Pictures = 12,
    Videos = 13,
    Music = 14,
    Trash = 15,
    Search = 16,
    GridView = 17,
    Cube3DView = 18,
    NewFolder = 19,
    Sort = 20,
    BackLeft = 21,
    UpArrow = 22,
    Recenter = 23,
    OpenLaunch = 24,
    PlaySolid = 25,
    EditRename = 26,
    CopyDup = 27,
    PasteClip = 28,
    UndoCurved = 29,
    MoreDots = 30,
    CloseCross = 31,
    CheckMark = 32,
    CutScissors = 33,
    EyeHidden = 34,
    ChevronLeft = 35,
    ChevronRight = 36,
    StorageDrive = 37,
}

impl UiIcon {
    #[inline]
    pub fn slot(self) -> f32 {
        (self as u32) as f32
    }

    pub fn from_bookmark_tag(tag: &str) -> Self {
        match tag {
            "HOME" => Self::Home,
            "DESK" => Self::Desktop,
            "DOCS" => Self::Docs,
            "DOWN" => Self::Downloads,
            "PICS" => Self::Pictures,
            "VIDS" => Self::Videos,
            "MUSC" => Self::Music,
            "TRSH" => Self::Trash,
            _ => Self::Home,
        }
    }

    pub fn badge_for_kind(kind: FileKind) -> Self {
        match kind {
            FileKind::Video => Self::PlaySolid,
            FileKind::Model3D => Self::Cube3DView,
            FileKind::Image => Self::Pictures,
            FileKind::Audio => Self::Music,
            FileKind::Document | FileKind::Code => Self::Docs,
            _ => Self::OpenLaunch,
        }
    }
}

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

/// Synthesizes all 8 colorful 2.5D UKUI file/folder icons (`0..8`) and 30 crisp vector SDF UI icons (`8..38`).
pub fn build_qingkong_icons() -> Vec<(u32, MipmappedRgba)> {
    (0..BUILTIN_ICON_COUNT)
        .map(|slot| {
            let mut raw = render_icon_supersampled(slot, ICON_EDGE);
            if slot < 8 {
                simd::premultiply_rgba_in_place(&mut raw);
            }
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
    let offsets = [-0.33f32, 0.0, 0.33];

    if slot >= 8 {
        // Pure linear-alpha coverage mask for tintable SDF UI icons (`tex_kind == 5`)
        for py in 0..edge {
            for px in 0..edge {
                let mut cov = 0.0f32;
                for &oy in &offsets {
                    for &ox in &offsets {
                        let u = (px as f32 + 0.5 + ox) * inv * 2.0 - 1.0;
                        let v = (py as f32 + 0.5 + oy) * inv * 2.0 - 1.0;
                        if eval_ui_icon_sdf(slot, u, v) <= 0.0 {
                            cov += 1.0;
                        }
                    }
                }
                let a = ((cov / 9.0) * 255.0).round() as u8;
                let idx = ((py * edge + px) * 4) as usize;
                out[idx] = 255;
                out[idx + 1] = 255;
                out[idx + 2] = 255;
                out[idx + 3] = a;
            }
        }
        return out;
    }

    for py in 0..edge {
        for px in 0..edge {
            let mut acc = [0.0f32; 4];
            for &oy in &offsets {
                for &ox in &offsets {
                    let u = (px as f32 + 0.5 + ox) * inv * 2.0 - 1.0;
                    let v = (py as f32 + 0.5 + oy) * inv * 2.0 - 1.0;
                    let c = sample_icon_sdf(slot, u, v);
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

/// Signed distance (`<= 0.0` inside stroke/fill) for crisp spatial UI vector icons (`u, v in [-1, 1]`, `+v` down).
fn eval_ui_icon_sdf(slot: u32, u: f32, v: f32) -> f32 {
    let p = [u, v];
    let sw = 0.088f32; // Bold, clean spatial stroke half-width
    match slot {
        8 => {
            // Home: pitched roof + rounded walls + door cutout
            let roof = sd_segment(p, [-0.66, 0.02], [0.0, -0.58])
                .min(sd_segment(p, [0.0, -0.58], [0.66, 0.02]))
                - sw;
            let body_ring = (sd_rbox([u, v - 0.16], [0.46, 0.38], 0.12)).abs() - sw * 0.88;
            let body_clipped = body_ring.max(-(v + 0.10));
            let door = sd_rbox([u, v - 0.34], [0.13, 0.20], 0.06);
            roof.min(body_clipped).min(door)
        }
        9 => {
            // Desktop monitor + stem + base
            let screen = sd_rbox([u, v + 0.10], [0.62, 0.40], 0.13).abs() - sw * 0.88;
            let stem = sd_segment(p, [0.0, 0.32], [0.0, 0.56]) - sw * 0.85;
            let base = sd_segment(p, [-0.30, 0.58], [0.30, 0.58]) - sw * 0.85;
            screen.min(stem).min(base)
        }
        10 => {
            // Document outline + 3 clean lines
            let page = sd_rbox(p, [0.48, 0.58], 0.13).abs() - sw * 0.85;
            let l1 = sd_segment(p, [-0.24, -0.18], [0.24, -0.18]) - sw * 0.72;
            let l2 = sd_segment(p, [-0.24, 0.04], [0.24, 0.04]) - sw * 0.72;
            let l3 = sd_segment(p, [-0.24, 0.26], [0.08, 0.26]) - sw * 0.72;
            page.min(l1).min(l2).min(l3)
        }
        11 => {
            // Downloads: arrow down + open tray
            let shaft = sd_segment(p, [0.0, -0.56], [0.0, 0.18]) - sw;
            let head = sd_segment(p, [-0.30, -0.10], [0.0, 0.22])
                .min(sd_segment(p, [0.30, -0.10], [0.0, 0.22]))
                - sw;
            let tray = sd_segment(p, [-0.54, 0.22], [-0.54, 0.54])
                .min(sd_segment(p, [-0.54, 0.54], [0.54, 0.54]))
                .min(sd_segment(p, [0.54, 0.54], [0.54, 0.22]))
                - sw * 0.88;
            shaft.min(head).min(tray)
        }
        12 => {
            // Pictures: rounded landscape frame + sun + mountain ridge
            let frame = sd_rbox(p, [0.58, 0.48], 0.14).abs() - sw * 0.85;
            let sun = sd_circle(p, [-0.24, -0.16], 0.12);
            let mtn = sd_segment(p, [-0.50, 0.36], [-0.10, -0.02])
                .min(sd_segment(p, [-0.10, -0.02], [0.14, 0.22]))
                .min(sd_segment(p, [0.14, 0.22], [0.32, 0.04]))
                .min(sd_segment(p, [0.32, 0.04], [0.52, 0.32]))
                - sw * 0.78;
            frame.min(sun).min(mtn)
        }
        13 => {
            // Videos: circle ring + solid play triangle
            let ring = (sd_circle(p, [0.0, 0.0], 0.58)).abs() - sw * 0.88;
            let tri = sd_play_triangle([u - 0.05, v], 0.26);
            ring.min(tri)
        }
        14 => {
            // Music: beamed eighth notes
            let n1 = sd_circle(p, [-0.30, 0.38], 0.17);
            let n2 = sd_circle(p, [0.24, 0.28], 0.17);
            let s1 = sd_segment(p, [-0.16, 0.36], [-0.16, -0.42]) - sw * 0.82;
            let s2 = sd_segment(p, [0.38, 0.26], [0.38, -0.54]) - sw * 0.82;
            let beam = sd_segment(p, [-0.16, -0.42], [0.38, -0.54]) - sw * 1.15;
            n1.min(n2).min(s1).min(s2).min(beam)
        }
        15 => {
            // Trash bin: lid + handle + bin outline + 2 vertical ribs
            let lid = sd_segment(p, [-0.52, -0.36], [0.52, -0.36]) - sw * 0.88;
            let handle = sd_rbox([u, v + 0.48], [0.18, 0.08], 0.05).abs() - sw * 0.70;
            let bin = sd_rbox([u, v - 0.14], [0.40, 0.44], 0.12).abs() - sw * 0.85;
            let r1 = sd_segment(p, [-0.14, -0.12], [-0.14, 0.34]) - sw * 0.68;
            let r2 = sd_segment(p, [0.14, -0.12], [0.14, 0.34]) - sw * 0.68;
            lid.min(handle).min(bin).min(r1).min(r2)
        }
        16 => {
            // Magnifying glass search
            let ring = sd_circle(p, [-0.10, -0.10], 0.39).abs() - sw * 0.95;
            let tail = sd_segment(p, [0.19, 0.19], [0.58, 0.58]) - sw * 1.05;
            ring.min(tail)
        }
        17 => {
            // 2x2 Bento Grid View
            let q = [u.abs() - 0.31, v.abs() - 0.31];
            sd_rbox(q, [0.22, 0.22], 0.08)
        }
        18 => {
            // Isometric 3D Volumetric Cube
            let top = [0.0, -0.62];
            let tr = [0.54, -0.30];
            let br = [0.54, 0.30];
            let bot = [0.0, 0.62];
            let bl = [-0.54, 0.30];
            let tl = [-0.54, -0.30];
            let ctr = [0.0, 0.02];
            let hex = sd_segment(p, top, tr)
                .min(sd_segment(p, tr, br))
                .min(sd_segment(p, br, bot))
                .min(sd_segment(p, bot, bl))
                .min(sd_segment(p, bl, tl))
                .min(sd_segment(p, tl, top));
            let ribs = sd_segment(p, tl, ctr)
                .min(sd_segment(p, tr, ctr))
                .min(sd_segment(p, bot, ctr));
            hex.min(ribs) - sw * 0.82
        }
        19 => {
            // New Folder (`folder +`)
            let tab = sd_segment(p, [-0.56, -0.42], [-0.16, -0.42])
                .min(sd_segment(p, [-0.16, -0.42], [-0.04, -0.26]));
            let body = sd_segment(p, [-0.58, -0.42], [-0.58, 0.42])
                .min(sd_segment(p, [-0.58, 0.42], [0.02, 0.42]))
                .min(sd_segment(p, [-0.04, -0.26], [0.46, -0.26]))
                .min(sd_segment(p, [0.46, -0.26], [0.46, -0.08]));
            let plus = sd_segment(p, [0.34, 0.06], [0.34, 0.56])
                .min(sd_segment(p, [0.09, 0.31], [0.59, 0.31]));
            tab.min(body).min(plus) - sw * 0.85
        }
        20 => {
            // Sort (`⇅`)
            let s1 = sd_segment(p, [-0.26, -0.48], [-0.26, 0.46]);
            let h1 = sd_segment(p, [-0.48, -0.24], [-0.26, -0.48])
                .min(sd_segment(p, [-0.04, -0.24], [-0.26, -0.48]));
            let s2 = sd_segment(p, [0.26, -0.46], [0.26, 0.48]);
            let h2 = sd_segment(p, [0.04, 0.24], [0.26, 0.48])
                .min(sd_segment(p, [0.48, 0.24], [0.26, 0.48]));
            s1.min(h1).min(s2).min(h2) - sw * 0.85
        }
        21 => {
            // Back Left Arrow (`←`)
            let shaft = sd_segment(p, [-0.48, 0.0], [0.50, 0.0]);
            let head = sd_segment(p, [-0.12, -0.38], [-0.50, 0.0])
                .min(sd_segment(p, [-0.12, 0.38], [-0.50, 0.0]));
            shaft.min(head) - sw * 0.95
        }
        22 => {
            // Up Arrow (`↑`)
            let shaft = sd_segment(p, [0.0, -0.48], [0.0, 0.50]);
            let head = sd_segment(p, [-0.38, -0.12], [0.0, -0.50])
                .min(sd_segment(p, [0.38, -0.12], [0.0, -0.50]));
            shaft.min(head) - sw * 0.95
        }
        23 => {
            // Spatial Recenter Crosshair (`⌖`)
            let ring = sd_circle(p, [0.0, 0.0], 0.42).abs() - sw * 0.82;
            let dot = sd_circle(p, [0.0, 0.0], 0.11);
            let ticks = sd_segment(p, [0.0, -0.64], [0.0, -0.34])
                .min(sd_segment(p, [0.0, 0.34], [0.0, 0.64]))
                .min(sd_segment(p, [-0.64, 0.0], [-0.34, 0.0]))
                .min(sd_segment(p, [0.34, 0.0], [0.64, 0.0]))
                - sw * 0.80;
            ring.min(dot).min(ticks)
        }
        24 => {
            // Open / Launch (`↗` out of rounded corner)
            let diag = sd_segment(p, [-0.10, 0.10], [0.50, -0.50]);
            let head = sd_segment(p, [0.08, -0.50], [0.50, -0.50])
                .min(sd_segment(p, [0.50, -0.08], [0.50, -0.50]));
            let box_c = sd_segment(p, [-0.14, -0.42], [-0.48, -0.42])
                .min(sd_segment(p, [-0.48, -0.42], [-0.48, 0.48]))
                .min(sd_segment(p, [-0.48, 0.48], [0.42, 0.48]))
                .min(sd_segment(p, [0.42, 0.48], [0.42, 0.14]));
            diag.min(head).min(box_c) - sw * 0.92
        }
        25 => {
            // Solid Play Triangle (`▶`)
            sd_play_triangle([u - 0.06, v], 0.46)
        }
        26 => {
            // Pencil / Rename (`✎`)
            let body = sd_segment(p, [-0.26, 0.26], [0.38, -0.38]) - 0.13;
            let nib = sd_segment(p, [-0.46, 0.46], [-0.26, 0.26]) - 0.065;
            let line = sd_segment(p, [-0.48, 0.54], [0.46, 0.54]) - sw * 0.78;
            body.min(nib).min(line)
        }
        27 => {
            // Copy / Duplicate (`❐`)
            let front = sd_rbox([u - 0.12, v - 0.12], [0.38, 0.38], 0.10).abs() - sw * 0.85;
            let back_full = sd_rbox([u + 0.14, v + 0.14], [0.38, 0.38], 0.10).abs() - sw * 0.85;
            let mask_front = sd_rbox([u - 0.12, v - 0.12], [0.44, 0.44], 0.10);
            front.min(back_full.max(-mask_front))
        }
        28 => {
            // Paste / Clipboard (`📋`)
            let clip = sd_rbox([u, v + 0.46], [0.24, 0.11], 0.06);
            let board = sd_rbox([u, v - 0.06], [0.46, 0.48], 0.12).abs() - sw * 0.85;
            let mask_clip = sd_rbox([u, v + 0.46], [0.30, 0.16], 0.06);
            clip.min(board.max(-mask_clip))
        }
        29 => {
            // Undo Curved Arrow (`↶`)
            let arc = (sd_circle(p, [0.06, 0.10], 0.38).abs() - sw * 0.92).max(v - 0.32);
            let head = sd_segment(p, [-0.32, 0.06], [-0.32, -0.32])
                .min(sd_segment(p, [-0.32, 0.06], [0.06, 0.06]))
                - sw * 0.90;
            arc.min(head)
        }
        30 => {
            // More Dots (`⋯`)
            sd_circle(p, [-0.46, 0.0], 0.115)
                .min(sd_circle(p, [0.0, 0.0], 0.115))
                .min(sd_circle(p, [0.46, 0.0], 0.115))
        }
        31 => {
            // Close Cross (`×`)
            sd_segment(p, [-0.42, -0.42], [0.42, 0.42])
                .min(sd_segment(p, [-0.42, 0.42], [0.42, -0.42]))
                - sw * 0.95
        }
        32 => {
            // Checkmark (`✓`)
            sd_segment(p, [-0.46, 0.04], [-0.12, 0.38])
                .min(sd_segment(p, [-0.12, 0.38], [0.50, -0.34]))
                - sw * 1.0
        }
        33 => {
            // Cut Scissors (`✂`)
            let r1 = sd_circle(p, [-0.32, 0.36], 0.18).abs() - sw * 0.78;
            let r2 = sd_circle(p, [0.32, 0.36], 0.18).abs() - sw * 0.78;
            let b1 = sd_segment(p, [-0.18, 0.24], [0.38, -0.50]) - sw * 0.80;
            let b2 = sd_segment(p, [0.18, 0.24], [-0.38, -0.50]) - sw * 0.80;
            r1.min(r2).min(b1).min(b2)
        }
        34 => {
            // Eye / Dotfiles (`👁`)
            let top_arc = (sd_circle(p, [0.0, 0.46], 0.76).abs() - sw * 0.82).max(v - 0.08);
            let bot_arc = (sd_circle(p, [0.0, -0.46], 0.76).abs() - sw * 0.82).max(-v - 0.08);
            let pupil = sd_circle(p, [0.0, 0.0], 0.17);
            top_arc.min(bot_arc).min(pupil)
        }
        35 => {
            // Chevron Left (`‹`)
            sd_segment(p, [0.18, -0.44], [-0.24, 0.0])
                .min(sd_segment(p, [-0.24, 0.0], [0.18, 0.44]))
                - sw * 0.98
        }
        36 => {
            // Chevron Right (`›`)
            sd_segment(p, [-0.18, -0.44], [0.24, 0.0])
                .min(sd_segment(p, [0.24, 0.0], [-0.18, 0.44]))
                - sw * 0.98
        }
        _ => {
            // Storage Drive (`37`)
            let box_d = sd_rbox(p, [0.56, 0.38], 0.14).abs() - sw * 0.85;
            let mid = sd_segment(p, [-0.52, 0.06], [0.52, 0.06]) - sw * 0.72;
            let led = sd_circle(p, [0.30, 0.22], 0.08);
            box_d.min(mid).min(led)
        }
    }
}

/// Evaluates normalized `u, v in [-1, 1]` (`+v` down) for UKUI Qingkong 2.5D file/folder icon `slot`.
fn sample_icon_sdf(slot: u32, u: f32, v: f32) -> [f32; 4] {
    if slot == 0 {
        return sample_golden_folder(u, v);
    }

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

/// Sculpted UKUI `#FFC735` golden folder with peeking inner photo/document sheets.
fn sample_golden_folder(u: f32, v: f32) -> [f32; 4] {
    let d_back = sd_rbox([u, v + 0.02], [0.76, 0.48], 0.13);
    let d_tab = sd_rbox([u + 0.34, v + 0.42], [0.34, 0.20], 0.11);
    let d_rear = d_back.min(d_tab);

    let u_c1 = u * 0.985 - (v + 0.16) * 0.17;
    let v_c1 = u * 0.17 + (v + 0.16) * 0.985;
    let d_card1 = sd_rbox([u_c1 + 0.08, v_c1], [0.46, 0.28], 0.07);

    let u_c2 = u * 0.99 + (v + 0.12) * 0.12;
    let v_c2 = -u * 0.12 + (v + 0.12) * 0.99;
    let d_card2 = sd_rbox([u_c2 - 0.12, v_c2], [0.44, 0.27], 0.07);

    let d_front = sd_rbox([u, v - 0.13], [0.80, 0.44], 0.14);

    let d_union = d_rear.min(d_card1).min(d_card2).min(d_front);
    if d_union > 0.0 {
        return [0.0, 0.0, 0.0, 0.0];
    }

    let mut rgb = [0.96, 0.67, 0.12];
    if d_rear <= 0.0 {
        let tr = ((v + 0.55) / 1.05).clamp(0.0, 1.0);
        rgb = [lerp(0.99, 0.92, tr), lerp(0.74, 0.58, tr), lerp(0.20, 0.08, tr)];
    }
    if d_card1 <= 0.0 {
        rgb = [0.85, 0.93, 1.0];
    }
    if d_card2 <= 0.0 {
        rgb = [0.98, 0.99, 1.0];
        if v_c2 < -0.12 && v_c2 > -0.20 && (u_c2 - 0.12).abs() < 0.32 {
            rgb = [0.36, 0.66, 0.99];
        }
    }
    if d_front <= 0.04 && (d_card1 <= 0.0 || d_card2 <= 0.0 || d_rear <= 0.0) {
        let shd = (1.0 - (d_front + 0.01).abs() / 0.05).clamp(0.0, 1.0);
        rgb[0] *= 1.0 - 0.12 * shd;
        rgb[1] *= 1.0 - 0.14 * shd;
        rgb[2] *= 1.0 - 0.16 * shd;
    }
    if d_front <= 0.0 {
        let tf = ((v + 0.31) / 0.88).clamp(0.0, 1.0);
        rgb = [
            lerp(1.00, 0.99, tf),
            lerp(0.86, 0.71, tf),
            lerp(0.35, 0.15, tf),
        ];
        if (v + 0.30).abs() < 0.028 {
            rgb = [1.0, 0.96, 0.76];
        }
    }

    [rgb[0], rgb[1], rgb[2], 1.0]
}

fn symbol_coverage(slot: u32, u: f32, v: f32) -> f32 {
    match slot {
        1 => {
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
            if sd_play_triangle([u - 0.05, v], 0.28) <= 0.0 {
                1.0
            } else {
                0.0
            }
        }
        3 => {
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
            let sun = ((u + 0.24).powi(2) + (v + 0.24).powi(2)).sqrt() - 0.13;
            let m1 = (u + 0.14).abs() * 0.95 + (v - 0.05);
            let m2 = (u - 0.22).abs() * 0.85 + (v + 0.05);
            let clip = sd_rbox([u, v], [0.54, 0.44], 0.10);
            if clip <= 0.0 && (sun <= 0.0 || (v > 0.02 && (m1 < 0.32 || m2 < 0.36))) {
                1.0
            } else {
                0.0
            }
        }
        5 => {
            let hex = sd_rbox([u * 0.866 - v * 0.5, u * 0.5 + v * 0.866], [0.36, 0.36], 0.08);
            let y1 = sd_segment([u, v], [0.0, 0.0], [0.0, 0.42]) - 0.045;
            let y2 = sd_segment([u, v], [0.0, 0.0], [-0.36, -0.20]) - 0.045;
            let y3 = sd_segment([u, v], [0.0, 0.0], [0.36, -0.20]) - 0.045;
            let ring = (hex.abs() - 0.048).min(y1).min(y2).min(y3);
            if ring <= 0.0 { 1.0 } else { 0.0 }
        }
        6 => {
            let box_d = sd_rbox([u, v], [0.44, 0.40], 0.10);
            let slot_cut = sd_rbox([u, v + 0.06], [0.08, 0.24], 0.03);
            if box_d <= 0.0 && slot_cut > 0.0 {
                1.0
            } else {
                0.0
            }
        }
        _ => {
            let b1 = sd_rbox([u + 0.24, v], [0.055, 0.20], 0.05);
            let b2 = sd_rbox([u, v], [0.055, 0.36], 0.05);
            let b3 = sd_rbox([u - 0.24, v], [0.055, 0.25], 0.05);
            if b1.min(b2).min(b3) <= 0.0 { 1.0 } else { 0.0 }
        }
    }
}

fn sd_circle(p: [f32; 2], c: [f32; 2], r: f32) -> f32 {
    ((p[0] - c[0]).powi(2) + (p[1] - c[1]).powi(2)).sqrt() - r
}

fn sd_play_triangle(p: [f32; 2], scale: f32) -> f32 {
    let d1 = -p[0] - scale;
    let d2 = 0.5 * p[0] + 0.866 * p[1] - scale * 0.78;
    let d3 = 0.5 * p[0] - 0.866 * p[1] - scale * 0.78;
    d1.max(d2).max(d3) - scale * 0.20
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
        let v = (y as f32 + 0.5) / height as f32;
        for x in 0..width {
            let u = (x as f32 + 0.5) / width as f32;
            let az = u * std::f32::consts::TAU;

            let rgb = if v <= 0.505 {
                let t = (v / 0.505).clamp(0.0, 1.0);
                let sky_r = lerp(0.42, 0.95, t.powf(1.55));
                let sky_g = lerp(0.66, 0.97, t.powf(1.35));
                let sky_b = lerp(0.99, 1.00, t.powf(0.85));

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
