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
    Thumbnail(PathBuf),
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
        if entry.kind != FileKind::Image
            || entry.size > 32 * 1024 * 1024
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
        let _ = self.tx.send(PreviewJob::Thumbnail(entry.path.clone()));
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
            PreviewJob::Thumbnail(path) => {
                let image = decode_scaled_rgba_mips(&path, 128).ok();
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
            summary_note: format!("目录 · 共 {count} 个直接子项"),
        };
    }

    let mut sample = Vec::with_capacity(65536);
    if let Ok(file) = File::open(path) {
        let _ = file.take(65536).read_to_end(&mut sample);
    }
    let inspection = simd::inspect_bytes_simd(&sample);

    let image = if kind == FileKind::Image {
        decode_scaled_rgba_mips(path, 384).ok()
    } else {
        None
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
        format!("{} · {} 三角面", m.format_label, m.triangle_count)
    } else if let Some(ref img) = image {
        format!(
            "图像 {}×{} · {} 级 Mip",
            img.original_dimensions[0],
            img.original_dimensions[1],
            img.mips.len()
        )
    } else if inspection.is_likely_binary {
        format!("二进制数据 · 熵 {:.2} bits/B", inspection.entropy_bits)
    } else {
        format!(
            "文本 {} 行(采样) · 熵 {:.2} bits/B",
            inspection.newline_count + usize::from(!sample.is_empty()),
            inspection.entropy_bits
        )
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
