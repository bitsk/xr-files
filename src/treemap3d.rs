// SPDX-License-Identifier: GPL-3.0-or-later
//! 3D Spatial Treemap (Squarified Volumetric Disk Cityscape) for `xr-files`.
//! Maps directory entries onto an ergonomic 3D podium where block footprint reflects
//! relative byte weight and extruded block height (`log2` relief) provides instant
//! stereo depth perception of storage hotspots.

use crate::fs_ops::FileEntry;

#[derive(Debug, Clone)]
pub struct TreemapBlock {
    pub entry_index: usize,
    /// Local 2D podium coordinates in meters (`[-width/2..width/2, -depth/2..depth/2]`).
    pub x_center: f32,
    pub z_center: f32,
    pub width: f32,
    pub depth: f32,
    /// Extruded 3D block height in meters.
    pub height: f32,
    /// Share of displayed directory bytes in `[0.0, 1.0]`.
    pub weight_ratio: f32,
}

#[derive(Debug, Clone, Copy)]
struct Rect2D {
    x: f32,
    z: f32,
    w: f32,
    d: f32,
}

impl Rect2D {
    fn shortest_side(&self) -> f32 {
        self.w.min(self.d).max(1e-4)
    }
}

/// Builds up to `max_blocks` squarified 3D volumetric blocks inside `podium_w × podium_d` meters.
pub fn build_spatial_treemap(
    entries: &[FileEntry],
    podium_w: f32,
    podium_d: f32,
    max_blocks: usize,
) -> Vec<TreemapBlock> {
    if entries.is_empty() || podium_w <= 0.05 || podium_d <= 0.05 {
        return Vec::new();
    }

    // Rank by size descending, keeping original entry indices for selection/navigation
    let mut indexed: Vec<(usize, f32, u64)> = entries
        .iter()
        .enumerate()
        .map(|(idx, e)| {
            // Apply mild perceptual floor so tiny config files remain clickable in 3D space
            let raw = e.size.max(512);
            let area_weight = (raw as f64).powf(0.72) as f32;
            (idx, area_weight.max(1.0), e.size)
        })
        .collect();
    indexed.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    indexed.truncate(max_blocks.max(1));

    let total_bytes: u64 = indexed.iter().map(|(_, _, b)| (*b).max(1)).sum::<u64>().max(1);
    let total_weight: f32 = indexed.iter().map(|(_, w, _)| *w).sum::<f32>().max(1e-4);
    let total_area = podium_w * podium_d;
    let scale = total_area / total_weight;

    let items: Vec<(usize, f32, u64)> = indexed
        .into_iter()
        .map(|(idx, w, bytes)| (idx, (w * scale).max(1e-5), bytes))
        .collect();

    let mut bounds = Rect2D {
        x: -0.5 * podium_w,
        z: -0.5 * podium_d,
        w: podium_w,
        d: podium_d,
    };
    let mut blocks = Vec::with_capacity(items.len());
    let mut row: Vec<(usize, f32, u64)> = Vec::new();

    for item in items {
        if row.is_empty() {
            row.push(item);
            continue;
        }
        let side = bounds.shortest_side();
        let cur_worst = worst_aspect(&row, side);
        row.push(item);
        let next_worst = worst_aspect(&row, side);
        if next_worst > cur_worst {
            let popped = row.pop().unwrap();
            layout_row(&row, &mut bounds, total_bytes, &mut blocks);
            row.clear();
            row.push(popped);
        }
    }
    if !row.is_empty() {
        layout_row(&row, &mut bounds, total_bytes, &mut blocks);
    }
    blocks
}

fn worst_aspect(row: &[(usize, f32, u64)], side: f32) -> f32 {
    let sum: f32 = row.iter().map(|(_, a, _)| *a).sum();
    if sum <= 1e-6 || side <= 1e-6 {
        return f32::INFINITY;
    }
    let min_a = row
        .iter()
        .map(|(_, a, _)| *a)
        .fold(f32::INFINITY, f32::min)
        .max(1e-6);
    let max_a = row.iter().map(|(_, a, _)| *a).fold(0.0f32, f32::max);
    let s2 = side * side;
    let sum2 = sum * sum;
    ((s2 * max_a) / sum2).max(sum2 / (s2 * min_a))
}

fn layout_row(
    row: &[(usize, f32, u64)],
    bounds: &mut Rect2D,
    total_bytes: u64,
    out: &mut Vec<TreemapBlock>,
) {
    let row_area: f32 = row.iter().map(|(_, a, _)| *a).sum();
    if row_area <= 1e-6 || bounds.w <= 1e-4 || bounds.d <= 1e-4 {
        return;
    }
    let gutter = 0.008f32;
    let horizontal_strip = bounds.w >= bounds.d;

    if horizontal_strip {
        let strip_w = (row_area / bounds.d.max(1e-4)).min(bounds.w);
        let mut cursor_z = bounds.z;
        for &(entry_index, area, bytes) in row {
            let seg_d = (area / strip_w.max(1e-4)).min(bounds.z + bounds.d - cursor_z);
            let w = (strip_w - gutter).max(0.014);
            let d = (seg_d - gutter).max(0.014);
            out.push(make_block(
                entry_index,
                bounds.x + 0.5 * strip_w,
                cursor_z + 0.5 * seg_d,
                w,
                d,
                bytes,
                total_bytes,
            ));
            cursor_z += seg_d;
        }
        bounds.x += strip_w;
        bounds.w = (bounds.w - strip_w).max(0.0);
    } else {
        let strip_d = (row_area / bounds.w.max(1e-4)).min(bounds.d);
        let mut cursor_x = bounds.x;
        for &(entry_index, area, bytes) in row {
            let seg_w = (area / strip_d.max(1e-4)).min(bounds.x + bounds.w - cursor_x);
            let w = (seg_w - gutter).max(0.014);
            let d = (strip_d - gutter).max(0.014);
            out.push(make_block(
                entry_index,
                cursor_x + 0.5 * seg_w,
                bounds.z + 0.5 * strip_d,
                w,
                d,
                bytes,
                total_bytes,
            ));
            cursor_x += seg_w;
        }
        bounds.z += strip_d;
        bounds.d = (bounds.d - strip_d).max(0.0);
    }
}

fn make_block(
    entry_index: usize,
    x_center: f32,
    z_center: f32,
    width: f32,
    depth: f32,
    bytes: u64,
    total_bytes: u64,
) -> TreemapBlock {
    let ratio = (bytes.max(1) as f32 / total_bytes.max(1) as f32).clamp(0.0, 1.0);
    let kb_log = (1.0 + (bytes as f32) / 1024.0).log2();
    let height = (0.018 + kb_log * 0.0095 + ratio * 0.06).clamp(0.018, 0.19);
    TreemapBlock {
        entry_index,
        x_center,
        z_center,
        width,
        depth,
        height,
        weight_ratio: ratio,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fs_ops::FileKind;
    use std::path::PathBuf;

    #[test]
    fn squarified_3d_treemap_stays_within_podium_bounds() {
        let entries: Vec<FileEntry> = (0..18)
            .map(|i| FileEntry {
                name: format!("file_{i}.dat"),
                path: PathBuf::from(format!("/tmp/file_{i}.dat")),
                kind: FileKind::Code,
                size: 1024 * (1 << (i % 10)),
                modified_unix: 0,
                readonly: false,
                hidden: false,
                is_symlink: false,
                trash_original: None,
                trash_info_file: None,
            })
            .collect();
        let blocks = build_spatial_treemap(&entries, 1.20, 0.72, 32);
        assert_eq!(blocks.len(), 18);
        for b in &blocks {
            assert!(b.x_center.abs() <= 0.61);
            assert!(b.z_center.abs() <= 0.37);
            assert!(b.height >= 0.018 && b.height <= 0.20);
        }
    }
}
