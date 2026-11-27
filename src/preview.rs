// SPDX-License-Identifier: GPL-3.0-or-later
//! Non-blocking spatial content previewer:
//! - SIMD-premultiplied & mipmapped image thumbnails + high-res inspector textures
//! - SIMD file entropy, line count, UTF-8 code/text excerpt & binary hex preview
//! - Pure-Rust 3D mesh loader for `.obj`, ASCII/binary `.stl`, and `.glb` (glTF 2.0)

use crate::{
    fs_ops::{FileEntry, FileKind},
    simd::{self, ByteInspection},
};
use anyhow::{bail, ensure, Context, Result};
use bytemuck::{Pod, Zeroable};
use serde_json::Value;
use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::Read,
    path::{Path, PathBuf},
    sync::mpsc::{self, Receiver, Sender},
    thread,
};

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, Pod, Zeroable)]
pub struct MeshVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
}

#[derive(Debug, Clone)]
pub struct PreviewMesh {
    pub vertices: Vec<MeshVertex>,
    pub triangle_count: usize,
    pub format_label: String,
}

#[derive(Debug, Clone)]
pub struct MipmappedRgba {
    /// Mip levels from level 0 down to 1x1, premultiplied RGBA8.
    pub mips: Vec<(Vec<u8>, u32, u32)>,
    pub original_dimensions: [u32; 2],
}

#[derive(Debug, Clone)]
pub struct InspectorPreview {
    pub path: PathBuf,
    pub inspection: ByteInspection,
    pub text_lines: Vec<String>,
    pub image: Option<MipmappedRgba>,
    pub mesh: Option<PreviewMesh>,
    pub summary_note: String,
}

enum PreviewJob {
    Thumbnail { path: PathBuf, kind: FileKind },
    Inspect { path: PathBuf, kind: FileKind },
}

pub enum PreviewResult {
    Thumbnail {
        path: PathBuf,
        image: Option<MipmappedRgba>,
    },
    Inspected(InspectorPreview),
}

pub struct AsyncPreviewer {
    tx: Sender<PreviewJob>,
    rx: Receiver<PreviewResult>,
    pending_thumbs: HashSet<PathBuf>,
    failed_thumbs: HashSet<PathBuf>,
    pub thumb_cache: HashMap<PathBuf, MipmappedRgba>,
    last_requested_inspect: Option<PathBuf>,
    pub active_inspect: Option<InspectorPreview>,
}

impl AsyncPreviewer {
    pub fn new() -> Self {
        let (job_tx, job_rx) = mpsc::channel::<PreviewJob>();
        let (res_tx, res_rx) = mpsc::channel::<PreviewResult>();
        thread::Builder::new()
            .name("xr-files-preview".into())
            .spawn(move || worker_loop(job_rx, res_tx))
            .expect("spawn xr-files preview worker");
        Self {
            tx: job_tx,
            rx: res_rx,
            pending_thumbs: HashSet::new(),
            failed_thumbs: HashSet::new(),
            thumb_cache: HashMap::new(),
            last_requested_inspect: None,
            active_inspect: None,
        }
    }

    pub fn request_thumbnail(&mut self, entry: &FileEntry) {
        if entry.kind == FileKind::Directory
            || (entry.kind == FileKind::Image && entry.size > 32 * 1024 * 1024)
            || self.thumb_cache.contains_key(&entry.path)
            || self.pending_thumbs.contains(&entry.path)
            || self.failed_thumbs.contains(&entry.path)
        {
            return;
        }
        if self.pending_thumbs.len() >= 48 {
            return;
        }
        self.pending_thumbs.insert(entry.path.clone());
        let _ = self.tx.send(PreviewJob::Thumbnail {
            path: entry.path.clone(),
            kind: entry.kind,
        });
    }

    pub fn request_inspect(&mut self, entry: &FileEntry) {
        if self.last_requested_inspect.as_ref() == Some(&entry.path) {
            return;
        }
        self.last_requested_inspect = Some(entry.path.clone());
        let _ = self.tx.send(PreviewJob::Inspect {
            path: entry.path.clone(),
            kind: entry.kind,
        });
    }

    pub fn clear_inspect(&mut self) {
        self.last_requested_inspect = None;
        self.active_inspect = None;
    }

    /// Polls completed background thumbnails and inspector data; returns newly arrived thumbnails.
    pub fn poll(&mut self) -> Vec<(PathBuf, MipmappedRgba)> {
        let mut new_thumbs = Vec::new();
        while let Ok(msg) = self.rx.try_recv() {
            match msg {
                PreviewResult::Thumbnail { path, image } => {
                    self.pending_thumbs.remove(&path);
                    if let Some(img) = image {
                        self.thumb_cache.insert(path.clone(), img.clone());
                        new_thumbs.push((path, img));
                    } else {
                        self.failed_thumbs.insert(path);
                    }
                }
                PreviewResult::Inspected(preview) => {
                    if self.last_requested_inspect.as_ref() == Some(&preview.path) {
                        self.active_inspect = Some(preview);
                    }
                }
            }
        }
        new_thumbs
    }
}

fn worker_loop(rx: Receiver<PreviewJob>, tx: Sender<PreviewResult>) {
    while let Ok(job) = rx.recv() {
        let reply = match job {
            PreviewJob::Thumbnail { path, kind } => {
                let image = build_rich_card_thumbnail(&path, kind, 160);
                PreviewResult::Thumbnail { path, image }
            }
            PreviewJob::Inspect { path, kind } => {
                PreviewResult::Inspected(build_inspector_preview(&path, kind))
            }
        };
        if tx.send(reply).is_err() {
            break;
        }
    }
}

/// Builds a rich visual `edge × edge` mipmapped card preview for Images, Videos, 3D Models, and Docs/Code.
fn build_rich_card_thumbnail(path: &Path, kind: FileKind, edge: u32) -> Option<MipmappedRgba> {
    match kind {
        FileKind::Image => decode_scaled_rgba_mips(path, edge).ok(),
        FileKind::Model3D => render_mesh_studio_thumbnail(path, edge).ok(),
        FileKind::Video => Some(render_video_cinema_poster(path, edge)),
        FileKind::Code | FileKind::Document => render_document_page_thumbnail(path, kind, edge).ok(),
        FileKind::Archive | FileKind::Audio | FileKind::Executable | FileKind::Other => {
            Some(render_scenic_asset_card(path, kind, edge))
        }
        FileKind::Directory => None,
    }
}

fn hash_seed(path: &Path) -> u32 {
    let mut h = 2166136261u32;
    for &b in path.to_string_lossy().as_bytes() {
        h ^= b as u32;
        h = h.wrapping_mul(16777619);
    }
    h
}

/// Renders a widescreen cinema horizon / planetary poster with frosted glass play emblem for video files.
fn render_video_cinema_poster(path: &Path, edge: u32) -> MipmappedRgba {
    let seed = hash_seed(path);
    let hue_shift = ((seed & 0xff) as f32) / 255.0;
    let mut raw = vec![0u8; (edge * edge * 4) as usize];

    for y in 0..edge {
        let v = y as f32 / edge as f32;
        for x in 0..edge {
            let u = x as f32 / edge as f32;
            // Atmospheric twilight sky + glowing planetary arc + ocean reflection
            let dx = u - (0.62 - 0.24 * hue_shift);
            let dy = v - 0.40;
            let planet_r = (dx * dx + dy * dy).sqrt();

            let mut r = 0.14 + 0.42 * (1.0 - v).powf(1.3) + 0.18 * hue_shift;
            let mut g = 0.30 + 0.46 * (1.0 - (v - 0.48).abs() * 1.6).max(0.0);
            let mut b = 0.64 + 0.34 * (1.0 - v * 0.5);

            // Glowing planet / VR180 sphere + atmospheric crescent rim
            if planet_r < 0.26 {
                let shade = ((dx + 0.12) * 2.8 - dy * 2.2).clamp(0.08, 1.0);
                let rim = (planet_r / 0.26).powf(4.0) * 0.55;
                r = 0.22 + 0.58 * shade + rim;
                g = 0.46 + 0.42 * shade + rim;
                b = 0.88 + 0.12 * shade + rim * 0.5;
            } else if planet_r < 0.33 {
                let glow = (1.0 - (planet_r - 0.26) / 0.07).powi(2) * 0.45;
                r += glow * 0.55;
                g += glow * 0.78;
                b += glow * 1.00;
            }

            // Soft horizon silk wave
            let wave = 0.58 + 0.04 * (u * 7.5 + hue_shift * 5.0).sin();
            if v > wave {
                let wt = ((v - wave) / (1.0 - wave)).clamp(0.0, 1.0);
                let ripple = 0.04 * (v * 48.0 + u * 9.0).sin() * (1.0 - wt);
                r = r * 0.78 + 0.14 + ripple;
                g = g * 0.84 + 0.16 + ripple;
                b = (b * 0.92 + 0.10 + ripple).min(1.0);
            }

            // Frosted translucent center play circle + crisp white play triangle
            let cx = u - 0.5;
            let cy = v - 0.5;
            let cr = (cx * cx + cy * cy).sqrt();
            if cr < 0.18 {
                r = r * 0.45 + 0.52;
                g = g * 0.45 + 0.53;
                b = b * 0.45 + 0.55;
                let px = cx - 0.015;
                let py = cy;
                if px > -0.055 && 0.5 * px + 0.866 * py < 0.048 && 0.5 * px - 0.866 * py < 0.048 {
                    r = 1.0;
                    g = 1.0;
                    b = 1.0;
                }
            }

            let idx = ((y * edge + x) * 4) as usize;
            raw[idx] = (r.clamp(0.0, 1.0) * 255.0) as u8;
            raw[idx + 1] = (g.clamp(0.0, 1.0) * 255.0) as u8;
            raw[idx + 2] = (b.clamp(0.0, 1.0) * 255.0) as u8;
            raw[idx + 3] = 255;
        }
    }
    simd::premultiply_rgba_in_place(&mut raw);
    MipmappedRgba {
        mips: simd::generate_mip_chain_rgba(raw, edge, edge),
        original_dimensions: [1920, 1080],
    }
}

/// Software-rasterizes a smooth shaded 3D studio preview of an `.obj` / `.stl` / `.glb` mesh!
fn render_mesh_studio_thumbnail(path: &Path, edge: u32) -> Result<MipmappedRgba> {
    let mesh = load_3d_mesh_preview(path)?;
    let mut raw = vec![0u8; (edge * edge * 4) as usize];
    let mut zbuf = vec![f32::INFINITY; (edge * edge) as usize];

    // Soft Qingkong studio sky backdrop + subtle ground reflection disc
    for y in 0..edge {
        let v = y as f32 / edge as f32;
        for x in 0..edge {
            let u = x as f32 / edge as f32;
            let rad = ((u - 0.5).powi(2) + (v - 0.55).powi(2)).sqrt();
            let glow = (1.0 - rad * 1.15).clamp(0.0, 1.0);
            let r = 0.78 + 0.20 * glow;
            let g = 0.87 + 0.12 * glow;
            let b = 0.98 + 0.02 * glow;
            let idx = ((y * edge + x) * 4) as usize;
            raw[idx] = (r * 255.0) as u8;
            raw[idx + 1] = (g * 255.0) as u8;
            raw[idx + 2] = (b * 255.0) as u8;
            raw[idx + 3] = 255;
        }
    }

    let yaw = 0.62f32;
    let pitch = 0.36f32;
    let (sy, cy) = yaw.sin_cos();
    let (sp, cp) = pitch.sin_cos();
    let rot = |p: [f32; 3]| -> [f32; 3] {
        let x1 = p[0] * cy + p[2] * sy;
        let z1 = -p[0] * sy + p[2] * cy;
        let y2 = p[1] * cp - z1 * sp;
        let z2 = p[1] * sp + z1 * cp;
        [x1, y2, z2]
    };

    let scale = edge as f32 * 0.74;
    let half = edge as f32 * 0.5;
    for tri in mesh.vertices.chunks_exact(3) {
        let v0 = rot(tri[0].position);
        let v1 = rot(tri[1].position);
        let v2 = rot(tri[2].position);
        let n = rot(tri[0].normal);

        let s0 = [half + v0[0] * scale, half - v0[1] * scale, 2.0 - v0[2]];
        let s1 = [half + v1[0] * scale, half - v1[1] * scale, 2.0 - v1[2]];
        let s2 = [half + v2[0] * scale, half - v2[1] * scale, 2.0 - v2[2]];

        let min_x = s0[0].min(s1[0]).min(s2[0]).floor().max(0.0) as u32;
        let max_x = s0[0].max(s1[0]).max(s2[0]).ceil().min((edge - 1) as f32) as u32;
        let min_y = s0[1].min(s1[1]).min(s2[1]).floor().max(0.0) as u32;
        let max_y = s0[1].max(s1[1]).max(s2[1]).ceil().min((edge - 1) as f32) as u32;

        let area = (s1[0] - s0[0]) * (s2[1] - s0[1]) - (s1[1] - s0[1]) * (s2[0] - s0[0]);
        if area.abs() < 1e-4 {
            continue;
        }
        let inv_area = 1.0 / area;

        let nl = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(1e-5);
        let nn = [n[0] / nl, n[1] / nl, n[2] / nl];
        let sun = (-0.45 * nn[0] + 0.75 * nn[1] + 0.48 * nn[2]).abs();
        let col = [
            (0.24 + 0.52 * sun).clamp(0.0, 1.0),
            (0.46 + 0.46 * sun).clamp(0.0, 1.0),
            (0.88 + 0.12 * sun).clamp(0.0, 1.0),
        ];

        for py in min_y..=max_y {
            for px in min_x..=max_x {
                let p = [px as f32 + 0.5, py as f32 + 0.5];
                let w0 = ((s1[0] - p[0]) * (s2[1] - p[1]) - (s1[1] - p[1]) * (s2[0] - p[0])) * inv_area;
                let w1 = ((s2[0] - p[0]) * (s0[1] - p[1]) - (s2[1] - p[1]) * (s0[0] - p[0])) * inv_area;
                let w2 = 1.0 - w0 - w1;
                if w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0 {
                    let z = w0 * s0[2] + w1 * s1[2] + w2 * s2[2];
                    let zi = (py * edge + px) as usize;
                    if z < zbuf[zi] {
                        zbuf[zi] = z;
                        raw[zi * 4] = (col[0] * 255.0) as u8;
                        raw[zi * 4 + 1] = (col[1] * 255.0) as u8;
                        raw[zi * 4 + 2] = (col[2] * 255.0) as u8;
                    }
                }
            }
        }
    }

    simd::premultiply_rgba_in_place(&mut raw);
    Ok(MipmappedRgba {
        mips: simd::generate_mip_chain_rgba(raw, edge, edge),
        original_dimensions: [edge, edge],
    })
}

/// Renders a miniature document/code page preview from actual lines of the file!
fn render_document_page_thumbnail(path: &Path, kind: FileKind, edge: u32) -> Result<MipmappedRgba> {
    let mut sample = Vec::with_capacity(4096);
    if let Ok(f) = File::open(path) {
        let _ = f.take(4096).read_to_end(&mut sample);
    }
    let text = String::from_utf8_lossy(&sample);
    let lines: Vec<&str> = text.lines().take(12).collect();

    let is_code = kind == FileKind::Code;
    let mut raw = vec![0u8; (edge * edge * 4) as usize];

    for y in 0..edge {
        let v = y as f32 / edge as f32;
        for x in 0..edge {
            let u = x as f32 / edge as f32;
            let paper_shade = 0.012 * u;
            let mut rgb = if is_code {
                [0.92f32 - paper_shade, 0.955 - paper_shade, 0.995]
            } else {
                [0.985f32 - paper_shade, 0.975 - paper_shade, 0.96]
            };
            // Top colorful banner header on miniature page
            if v < 0.22 {
                rgb = if is_code {
                    [0.24, 0.56, 0.96]
                } else {
                    [0.98, 0.54, 0.20]
                };
            }
            let idx = ((y * edge + x) * 4) as usize;
            raw[idx] = (rgb[0] * 255.0) as u8;
            raw[idx + 1] = (rgb[1] * 255.0) as u8;
            raw[idx + 2] = (rgb[2] * 255.0) as u8;
            raw[idx + 3] = 255;
        }
    }

    // Draw crisp miniature syntax/paragraph bars representing real line lengths!
    for (row, line) in lines.iter().enumerate().take(8) {
        let trimmed = line.trim_start();
        if trimmed.is_empty() {
            continue;
        }
        let indent = (line.len() - trimmed.len()).min(12) as u32 * 4;
        let bar_w = ((trimmed.chars().count().min(36) as f32 / 36.0) * (edge as f32 * 0.72))
            .max(14.0) as u32;
        let y0 = (edge * 30 / 100) + (row as u32) * (edge * 8 / 100);
        let y1 = (y0 + (edge * 4 / 100).max(3)).min(edge - 4);
        let x0 = (edge * 12 / 100 + indent).min(edge - 8);
        let x1 = (x0 + bar_w).min(edge - 10);

        let bar_col: [u8; 3] = if trimmed.starts_with('#') || trimmed.starts_with("pub ") || trimmed.starts_with("fn ") {
            [55, 136, 246]
        } else if is_code {
            [72, 96, 132]
        } else {
            [96, 108, 128]
        };
        for py in y0..y1 {
            for px in x0..x1 {
                let idx = ((py * edge + px) * 4) as usize;
                raw[idx..idx + 3].copy_from_slice(&bar_col);
            }
        }
    }

    simd::premultiply_rgba_in_place(&mut raw);
    Ok(MipmappedRgba {
        mips: simd::generate_mip_chain_rgba(raw, edge, edge),
        original_dimensions: [edge, edge],
    })
}

fn render_scenic_asset_card(path: &Path, kind: FileKind, edge: u32) -> MipmappedRgba {
    let seed = hash_seed(path);
    let phase = ((seed & 0xff) as f32) * 0.025;
    let mut raw = vec![0u8; (edge * edge * 4) as usize];
    for y in 0..edge {
        let v = y as f32 / edge as f32;
        for x in 0..edge {
            let u = x as f32 / edge as f32;
            let wave = (u * 4.5 + phase).sin() * 0.08 + (v * 3.0).cos() * 0.05;
            let (r, g, b) = match kind {
                FileKind::Archive => (
                    0.98 - 0.12 * v + wave * 0.2,
                    0.82 - 0.18 * v + wave * 0.2,
                    0.46 + 0.18 * u,
                ),
                _ => (
                    0.76 + 0.18 * u + wave,
                    0.86 + 0.10 * (1.0 - v) + wave,
                    0.99,
                ),
            };
            let idx = ((y * edge + x) * 4) as usize;
            raw[idx] = (r.clamp(0.0, 1.0) * 255.0) as u8;
            raw[idx + 1] = (g.clamp(0.0, 1.0) * 255.0) as u8;
            raw[idx + 2] = (b.clamp(0.0, 1.0) * 255.0) as u8;
            raw[idx + 3] = 255;
        }
    }
    simd::premultiply_rgba_in_place(&mut raw);
    MipmappedRgba {
        mips: simd::generate_mip_chain_rgba(raw, edge, edge),
        original_dimensions: [edge, edge],
    }
}

pub fn decode_scaled_rgba_mips(path: &Path, max_edge: u32) -> Result<MipmappedRgba> {
    let decoded = image::open(path).with_context(|| format!("decode image {}", path.display()))?;
    let orig_w = decoded.width().max(1);
    let orig_h = decoded.height().max(1);
    let resized = if orig_w > max_edge || orig_h > max_edge {
        decoded.thumbnail(max_edge, max_edge)
    } else {
        decoded
    };
    let rgba = resized.to_rgba8();
    let w = rgba.width().max(1);
    let h = rgba.height().max(1);
    let mut raw = rgba.into_raw();
    simd::premultiply_rgba_in_place(&mut raw);
    let mips = simd::generate_mip_chain_rgba(raw, w, h);
    Ok(MipmappedRgba {
        mips,
        original_dimensions: [orig_w, orig_h],
    })
}

pub fn build_inspector_preview(path: &Path, kind: FileKind) -> InspectorPreview {
    if kind == FileKind::Directory {
        let count = std::fs::read_dir(path).map(|rd| rd.count()).unwrap_or(0);
        return InspectorPreview {
            path: path.to_path_buf(),
            inspection: ByteInspection {
                total_bytes: 0,
                newline_count: 0,
                nul_bytes: 0,
                entropy_bits: 0.0,
                is_likely_binary: false,
            },
            text_lines: vec![
                format!("子项目数量: {count}"),
                "双击卡片或扣动扳机进入该空间目录".into(),
                "支持将文件卡片用抓握键直接拖入此目录".into(),
            ],
            image: None,
            mesh: None,
            summary_note: format!("{count}"),
        };
    }

    let mut sample = Vec::with_capacity(65536);
    if let Ok(file) = File::open(path) {
        let _ = file.take(65536).read_to_end(&mut sample);
    }
    let inspection = simd::inspect_bytes_simd(&sample);

    let image = match kind {
        FileKind::Image => decode_scaled_rgba_mips(path, 384).ok(),
        FileKind::Video => Some(render_video_cinema_poster(path, 384)),
        FileKind::Code | FileKind::Document => render_document_page_thumbnail(path, kind, 384).ok(),
        _ => None,
    };

    let mesh = if kind == FileKind::Model3D {
        load_3d_mesh_preview(path).ok()
    } else {
        None
    };

    let text_lines = if let Some(ref m) = mesh {
        vec![
            format!("三维格式: {}", m.format_label),
            format!("三角面片: {} tris", m.triangle_count),
            format!("顶点缓冲: {} verts", m.vertices.len()),
            "全息预览台正在实时双目立体旋转渲染".into(),
        ]
    } else if let Some(ref img) = image {
        vec![
            format!(
                "原始分辨率: {} × {} px",
                img.original_dimensions[0], img.original_dimensions[1]
            ),
            format!("Mipmap 层级: {} 级 (SIMD 预乘 Alpha)", img.mips.len()),
            format!("信息熵: {:.2} bits/B", inspection.entropy_bits),
        ]
    } else if !inspection.is_likely_binary && !sample.is_empty() {
        let text = String::from_utf8_lossy(&sample);
        text.lines()
            .take(18)
            .map(|l| {
                let expanded = l.replace('\t', "    ");
                truncate_chars(&expanded, 44)
            })
            .collect()
    } else if !sample.is_empty() {
        // Formatted hex + ASCII excerpt for binary files
        sample
            .chunks(8)
            .take(12)
            .enumerate()
            .map(|(row, bytes)| {
                let mut hex = String::new();
                let mut asc = String::new();
                for b in bytes {
                    hex.push_str(&format!("{b:02X} "));
                    asc.push(if b.is_ascii_graphic() || *b == b' ' {
                        *b as char
                    } else {
                        '.'
                    });
                }
                format!("{:04X}: {:<24} {asc}", row * 8, hex)
            })
            .collect()
    } else {
        vec!["（空文件）".into()]
    };

    let summary_note = if let Some(ref m) = mesh {
        format!("{} △", m.triangle_count)
    } else if kind == FileKind::Image {
        if let Some(ref img) = image {
            format!("{}×{}", img.original_dimensions[0], img.original_dimensions[1])
        } else {
            String::new()
        }
    } else if kind == FileKind::Video {
        "VR / 2D".into()
    } else if !inspection.is_likely_binary {
        format!(
            "{}L",
            inspection.newline_count + usize::from(!sample.is_empty())
        )
    } else {
        String::new()
    };

    InspectorPreview {
        path: path.to_path_buf(),
        inspection,
        text_lines,
        image,
        mesh,
        summary_note,
    }
}

fn truncate_chars(s: &str, max_chars: usize) -> String {
    let mut out = String::new();
    for (i, ch) in s.chars().enumerate() {
        if i >= max_chars {
            out.push('…');
            break;
        }
        out.push(ch);
    }
    out
}

// ============================================================================
// Pure-Rust 3D Model Parsers (`.obj`, `.stl`, `.glb`) + Normalization
// ============================================================================

pub fn load_3d_mesh_preview(path: &Path) -> Result<PreviewMesh> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default();
    let mut data = Vec::new();
    File::open(path)?
        .take(24 * 1024 * 1024)
        .read_to_end(&mut data)?;
    let (mut vertices, label) = match ext.as_str() {
        "obj" => (parse_obj(&data)?, "Wavefront OBJ"),
        "stl" => (parse_stl(&data)?, "Stereolithography STL"),
        "glb" => (parse_glb(&data)?, "glTF 2.0 Binary (GLB)"),
        _ => bail!("暂不支持直接网格解析的 3D 格式: {ext}"),
    };
    ensure!(!vertices.is_empty(), "3D 模型不包含有效三角面");
    normalize_mesh_unit_box(&mut vertices);
    let tris = vertices.len() / 3;
    Ok(PreviewMesh {
        vertices,
        triangle_count: tris,
        format_label: label.into(),
    })
}

fn parse_obj(bytes: &[u8]) -> Result<Vec<MeshVertex>> {
    let text = String::from_utf8_lossy(bytes);
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut out: Vec<MeshVertex> = Vec::new();

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.split_whitespace();
        match parts.next() {
            Some("v") => {
                let x: f32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
                let y: f32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
                let z: f32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
                positions.push([x, y, z]);
            }
            Some("vn") => {
                let x: f32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
                let y: f32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(1.0);
                let z: f32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
                normals.push([x, y, z]);
            }
            Some("f") => {
                if out.len() >= 180_000 {
                    break;
                }
                let mut face: Vec<([f32; 3], Option<[f32; 3]>)> = Vec::new();
                for tok in parts {
                    let mut segs = tok.split('/');
                    let vi = parse_obj_idx(segs.next().unwrap_or(""), positions.len());
                    let _vt = segs.next();
                    let ni = segs.next().and_then(|s| parse_obj_idx(s, normals.len()));
                    if let Some(p_idx) = vi {
                        let p = positions[p_idx];
                        let n = ni.map(|idx| normals[idx]);
                        face.push((p, n));
                    }
                }
                for k in 1..face.len().saturating_sub(1) {
                    let (p0, n0) = face[0];
                    let (p1, n1) = face[k];
                    let (p2, n2) = face[k + 1];
                    let fnrm = face_normal(p0, p1, p2);
                    out.push(MeshVertex {
                        position: p0,
                        normal: n0.unwrap_or(fnrm),
                    });
                    out.push(MeshVertex {
                        position: p1,
                        normal: n1.unwrap_or(fnrm),
                    });
                    out.push(MeshVertex {
                        position: p2,
                        normal: n2.unwrap_or(fnrm),
                    });
                }
            }
            _ => {}
        }
    }
    Ok(out)
}

fn parse_obj_idx(token: &str, len: usize) -> Option<usize> {
    if token.is_empty() || len == 0 {
        return None;
    }
    let raw: isize = token.parse().ok()?;
    let idx = if raw > 0 {
        (raw as usize).checked_sub(1)?
    } else if raw < 0 {
        len.checked_sub(raw.unsigned_abs())?
    } else {
        return None;
    };
    (idx < len).then_some(idx)
}

fn parse_stl(bytes: &[u8]) -> Result<Vec<MeshVertex>> {
    // Check binary STL header (80-byte header + u32 triangle count + 50 bytes per tri)
    if bytes.len() >= 84 && !bytes.starts_with(b"solid ") {
        let tri_count = u32::from_le_bytes(bytes[80..84].try_into().unwrap()) as usize;
        let expected = 84usize.saturating_add(tri_count.saturating_mul(50));
        if expected <= bytes.len() && tri_count > 0 {
            let capped = tri_count.min(60_000);
            let mut out = Vec::with_capacity(capped * 3);
            for i in 0..capped {
                let off = 84 + i * 50;
                let read_v3 = |o: usize| -> [f32; 3] {
                    [
                        f32::from_le_bytes(bytes[o..o + 4].try_into().unwrap()),
                        f32::from_le_bytes(bytes[o + 4..o + 8].try_into().unwrap()),
                        f32::from_le_bytes(bytes[o + 8..o + 12].try_into().unwrap()),
                    ]
                };
                let n_raw = read_v3(off);
                let p0 = read_v3(off + 12);
                let p1 = read_v3(off + 24);
                let p2 = read_v3(off + 36);
                let n = if n_raw[0].abs() + n_raw[1].abs() + n_raw[2].abs() > 1e-4 {
                    n_raw
                } else {
                    face_normal(p0, p1, p2)
                };
                out.push(MeshVertex { position: p0, normal: n });
                out.push(MeshVertex { position: p1, normal: n });
                out.push(MeshVertex { position: p2, normal: n });
            }
            return Ok(out);
        }
    }

    // ASCII STL fallback
    let text = String::from_utf8_lossy(bytes);
    let mut verts: Vec<[f32; 3]> = Vec::new();
    for line in text.lines() {
        if let Some(rest) = line.trim().strip_prefix("vertex") {
            let mut it = rest.split_whitespace();
            let x: f32 = it.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
            let y: f32 = it.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
            let z: f32 = it.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
            verts.push([x, y, z]);
        }
    }
    let mut out = Vec::with_capacity(verts.len());
    for tri in verts.chunks_exact(3).take(60_000) {
        let n = face_normal(tri[0], tri[1], tri[2]);
        for &p in tri {
            out.push(MeshVertex { position: p, normal: n });
        }
    }
    Ok(out)
}

fn parse_glb(bytes: &[u8]) -> Result<Vec<MeshVertex>> {
    ensure!(bytes.len() >= 20 && &bytes[0..4] == b"glTF", "无效 GLB 魔数");
    let json_len = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
    let json_type = u32::from_le_bytes(bytes[16..20].try_into().unwrap());
    ensure!(json_type == 0x4E4F534A && 20 + json_len <= bytes.len(), "损坏的 GLB JSON 块");
    let gltf: Value = serde_json::from_slice(&bytes[20..20 + json_len])?;

    let bin_offset = 20 + json_len;
    ensure!(bin_offset + 8 <= bytes.len(), "GLB 缺少二进制缓冲块");
    let bin_len = u32::from_le_bytes(bytes[bin_offset..bin_offset + 4].try_into().unwrap()) as usize;
    let bin = &bytes[bin_offset + 8..(bin_offset + 8 + bin_len).min(bytes.len())];

    let accessors = gltf["accessors"].as_array().context("GLB 无 accessors")?;
    let views = gltf["bufferViews"].as_array().context("GLB 无 bufferViews")?;
    let meshes = gltf["meshes"].as_array().context("GLB 无 meshes")?;

    let mut out = Vec::new();
    for mesh in meshes {
        let Some(prims) = mesh["primitives"].as_array() else { continue };
        for prim in prims {
            if prim["mode"].as_u64().unwrap_or(4) != 4 {
                continue;
            }
            let Some(pos_acc_idx) = prim["attributes"]["POSITION"].as_u64() else {
                continue;
            };
            let positions = read_glb_vec3(bin, accessors, views, pos_acc_idx as usize)?;
            let normals = prim["attributes"]["NORMAL"]
                .as_u64()
                .and_then(|idx| read_glb_vec3(bin, accessors, views, idx as usize).ok());

            if let Some(idx_acc) = prim["indices"].as_u64() {
                let indices = read_glb_indices(bin, accessors, views, idx_acc as usize)?;
                for tri in indices.chunks_exact(3).take(60_000) {
                    let (i0, i1, i2) = (tri[0], tri[1], tri[2]);
                    if i0 >= positions.len() || i1 >= positions.len() || i2 >= positions.len() {
                        continue;
                    }
                    let (p0, p1, p2) = (positions[i0], positions[i1], positions[i2]);
                    let fnrm = face_normal(p0, p1, p2);
                    out.push(MeshVertex {
                        position: p0,
                        normal: normals.as_ref().and_then(|n| n.get(i0).copied()).unwrap_or(fnrm),
                    });
                    out.push(MeshVertex {
                        position: p1,
                        normal: normals.as_ref().and_then(|n| n.get(i1).copied()).unwrap_or(fnrm),
                    });
                    out.push(MeshVertex {
                        position: p2,
                        normal: normals.as_ref().and_then(|n| n.get(i2).copied()).unwrap_or(fnrm),
                    });
                }
            } else {
                for (t, tri) in positions.chunks_exact(3).take(60_000).enumerate() {
                    let fnrm = face_normal(tri[0], tri[1], tri[2]);
                    for k in 0..3 {
                        let vi = t * 3 + k;
                        out.push(MeshVertex {
                            position: tri[k],
                            normal: normals.as_ref().and_then(|n| n.get(vi).copied()).unwrap_or(fnrm),
                        });
                    }
                }
            }
        }
    }
    Ok(out)
}

fn read_glb_vec3(
    bin: &[u8],
    accessors: &[Value],
    views: &[Value],
    acc_idx: usize,
) -> Result<Vec<[f32; 3]>> {
    let acc = accessors.get(acc_idx).context("accessor 越界")?;
    ensure!(acc["componentType"].as_u64() == Some(5126) && acc["type"].as_str() == Some("VEC3"), "仅支持 FLOAT VEC3");
    let count = acc["count"].as_u64().unwrap_or(0) as usize;
    let view_idx = acc["bufferView"].as_u64().context("缺少 bufferView")? as usize;
    let view = views.get(view_idx).context("bufferView 越界")?;
    let base = view["byteOffset"].as_u64().unwrap_or(0) as usize
        + acc["byteOffset"].as_u64().unwrap_or(0) as usize;
    let stride = view["byteStride"].as_u64().map(|s| s as usize).unwrap_or(12);
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let off = base + i * stride;
        if off + 12 > bin.len() {
            break;
        }
        out.push([
            f32::from_le_bytes(bin[off..off + 4].try_into().unwrap()),
            f32::from_le_bytes(bin[off + 4..off + 8].try_into().unwrap()),
            f32::from_le_bytes(bin[off + 8..off + 12].try_into().unwrap()),
        ]);
    }
    Ok(out)
}

fn read_glb_indices(
    bin: &[u8],
    accessors: &[Value],
    views: &[Value],
    acc_idx: usize,
) -> Result<Vec<usize>> {
    let acc = accessors.get(acc_idx).context("index accessor 越界")?;
    let comp = acc["componentType"].as_u64().unwrap_or(0);
    let count = acc["count"].as_u64().unwrap_or(0) as usize;
    let view_idx = acc["bufferView"].as_u64().context("缺少 index bufferView")? as usize;
    let view = views.get(view_idx).context("index bufferView 越界")?;
    let base = view["byteOffset"].as_u64().unwrap_or(0) as usize
        + acc["byteOffset"].as_u64().unwrap_or(0) as usize;
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        match comp {
            5121 => {
                let off = base + i;
                if off < bin.len() {
                    out.push(bin[off] as usize);
                }
            }
            5123 => {
                let off = base + i * 2;
                if off + 2 <= bin.len() {
                    out.push(u16::from_le_bytes(bin[off..off + 2].try_into().unwrap()) as usize);
                }
            }
            5125 => {
                let off = base + i * 4;
                if off + 4 <= bin.len() {
                    out.push(u32::from_le_bytes(bin[off..off + 4].try_into().unwrap()) as usize);
                }
            }
            _ => bail!("不支持的索引类型: {comp}"),
        }
    }
    Ok(out)
}

fn face_normal(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> [f32; 3] {
    let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let nx = u[1] * v[2] - u[2] * v[1];
    let ny = u[2] * v[0] - u[0] * v[2];
    let nz = u[0] * v[1] - u[1] * v[0];
    let len = (nx * nx + ny * ny + nz * nz).sqrt();
    if len > 1e-8 {
        [nx / len, ny / len, nz / len]
    } else {
        [0.0, 1.0, 0.0]
    }
}

fn normalize_mesh_unit_box(vertices: &mut [MeshVertex]) {
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for v in vertices.iter() {
        for c in 0..3 {
            if v.position[c].is_finite() {
                min[c] = min[c].min(v.position[c]);
                max[c] = max[c].max(v.position[c]);
            }
        }
    }
    let center = [
        0.5 * (min[0] + max[0]),
        0.5 * (min[1] + max[1]),
        0.5 * (min[2] + max[2]),
    ];
    let span = (max[0] - min[0])
        .max(max[1] - min[1])
        .max(max[2] - min[2])
        .max(1e-5);
    let scale = 1.0 / span;
    for v in vertices.iter_mut() {
        v.position[0] = (v.position[0] - center[0]) * scale;
        v.position[1] = (v.position[1] - center[1]) * scale;
        v.position[2] = (v.position[2] - center[2]) * scale;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn obj_parser_normalizes_unit_hologram_mesh() {
        let obj = b"
            v -2.0 0.0 0.0
            v  2.0 0.0 0.0
            v  0.0 4.0 0.0
            v  0.0 2.0 2.0
            f 1 2 3
            f 1 2 4
        ";
        let mut verts = parse_obj(obj).unwrap();
        assert_eq!(verts.len(), 6);
        normalize_mesh_unit_box(&mut verts);
        for v in &verts {
            assert!(v.position.iter().all(|c| c.abs() <= 0.5001));
        }
    }
}
