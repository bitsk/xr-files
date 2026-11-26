// SPDX-License-Identifier: GPL-3.0-or-later
//! Pure-Rust CJK + Latin Dual-Weight (`Regular` + `Bold`) Dynamic Glyph Atlas
//! with 2px padding borders and SIMD-generated mipmap chains (`Constraint 4.2`).

use crate::simd;
use ab_glyph::{Font, FontArc, FontVec, PxScale, ScaleFont};
use std::{collections::HashMap, env, fs, path::Path};

pub const ATLAS_SIZE: u32 = 2048;
const GLYPH_PAD: u32 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FontWeight {
    Regular,
    Bold,
}

#[derive(Debug, Clone, Copy)]
pub struct GlyphEntry {
    pub uv_min: [f32; 2],
    pub uv_max: [f32; 2],
    pub size_px: [f32; 2],
    pub bearing_px: [f32; 2],
    pub advance_px: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct GlyphKey {
    ch: char,
    px_tenths: u16,
    weight: FontWeight,
}

pub struct GlyphAtlas {
    font_regular: Option<FontArc>,
    font_bold: Option<FontArc>,
    pub pixels: Vec<u8>,
    cursor_x: u32,
    cursor_y: u32,
    row_height: u32,
    cache: HashMap<GlyphKey, GlyphEntry>,
    pub dirty: bool,
    pub white_uv: [f32; 2],
}

impl GlyphAtlas {
    pub fn new() -> Self {
        let (font_regular, font_bold) = load_system_fonts();
        let mut pixels = vec![0u8; (ATLAS_SIZE * ATLAS_SIZE * 4) as usize];
        for y in 0..6u32 {
            for x in 0..6u32 {
                let idx = ((y * ATLAS_SIZE + x) * 4) as usize;
                pixels[idx..idx + 4].copy_from_slice(&[255, 255, 255, 255]);
            }
        }
        Self {
            font_regular,
            font_bold,
            pixels,
            cursor_x: 10,
            cursor_y: 2,
            row_height: 10,
            cache: HashMap::with_capacity(1536),
            dirty: true,
            white_uv: [2.5 / ATLAS_SIZE as f32, 2.5 / ATLAS_SIZE as f32],
        }
    }

    pub fn build_mip_chain(&mut self) -> Vec<(Vec<u8>, u32, u32)> {
        self.dirty = false;
        simd::generate_mip_chain_rgba(self.pixels.clone(), ATLAS_SIZE, ATLAS_SIZE)
    }

    pub fn layout_line(
        &mut self,
        text: &str,
        px_size: f32,
        max_width_px: f32,
    ) -> (Vec<(GlyphEntry, f32, f32)>, f32) {
        self.layout_line_weighted(text, px_size, max_width_px, FontWeight::Regular)
    }

    pub fn layout_line_weighted(
        &mut self,
        text: &str,
        px_size: f32,
        max_width_px: f32,
        weight: FontWeight,
    ) -> (Vec<(GlyphEntry, f32, f32)>, f32) {
        let px = px_size.clamp(12.0, 96.0);
        let mut pen_x = 0.0f32;
        let baseline_y = px * 0.80;
        let mut out = Vec::with_capacity(text.len());

        for ch in text.chars() {
            if ch == '\n' || ch == '\r' {
                break;
            }
            if ch == '\t' {
                pen_x += px * 1.6;
                continue;
            }
            let entry = self.get_or_rasterize(ch, px, weight);
            if pen_x + entry.advance_px > max_width_px && max_width_px > 0.0 {
                let dots = self.get_or_rasterize('…', px, weight);
                if pen_x + dots.size_px[0] <= max_width_px + px * 0.5 && dots.size_px[0] > 0.0 {
                    out.push((
                        dots,
                        pen_x + dots.bearing_px[0],
                        baseline_y - dots.bearing_px[1],
                    ));
                    pen_x += dots.advance_px;
                }
                break;
            }
            if entry.size_px[0] > 0.0 && entry.size_px[1] > 0.0 {
                out.push((
                    entry,
                    pen_x + entry.bearing_px[0],
                    baseline_y - entry.bearing_px[1],
                ));
            }
            pen_x += entry.advance_px;
        }
        (out, pen_x)
    }

    fn get_or_rasterize(&mut self, ch: char, px_size: f32, weight: FontWeight) -> GlyphEntry {
        let px_tenths = (px_size * 10.0).round() as u16;
        let key = GlyphKey {
            ch,
            px_tenths,
            weight,
        };
        if let Some(&cached) = self.cache.get(&key) {
            return cached;
        }
        let chosen_font = match weight {
            FontWeight::Bold => self.font_bold.clone().or_else(|| self.font_regular.clone()),
            FontWeight::Regular => self.font_regular.clone().or_else(|| self.font_bold.clone()),
        };
        let entry = if let Some(font) = chosen_font {
            self.rasterize_ab_glyph(&font, ch, px_size)
        } else {
            self.rasterize_fallback(ch, px_size)
        };
        self.cache.insert(key, entry);
        entry
    }

    fn rasterize_ab_glyph(&mut self, font: &FontArc, ch: char, px_size: f32) -> GlyphEntry {
        let scaled = font.as_scaled(PxScale::from(px_size));
        let gid = scaled.glyph_id(ch);
        if gid.0 == 0 && ch != ' ' {
            return self.rasterize_fallback(ch, px_size);
        }
        let advance = scaled.h_advance(gid);
        if ch.is_whitespace() {
            return GlyphEntry {
                uv_min: self.white_uv,
                uv_max: self.white_uv,
                size_px: [0.0, 0.0],
                bearing_px: [0.0, 0.0],
                advance_px: advance.max(px_size * 0.30),
            };
        }
        let glyph = gid.with_scale_and_position(PxScale::from(px_size), ab_glyph::point(0.0, 0.0));
        let Some(outlined) = font.outline_glyph(glyph) else {
            return GlyphEntry {
                uv_min: self.white_uv,
                uv_max: self.white_uv,
                size_px: [0.0, 0.0],
                bearing_px: [0.0, 0.0],
                advance_px: advance.max(px_size * 0.45),
            };
        };
        let bounds = outlined.px_bounds();
        let gw = (bounds.width().ceil() as u32).max(1);
        let gh = (bounds.height().ceil() as u32).max(1);
        let Some((ax, ay)) = self.allocate_rect(gw, gh) else {
            return GlyphEntry {
                uv_min: self.white_uv,
                uv_max: self.white_uv,
                size_px: [0.0, 0.0],
                bearing_px: [0.0, 0.0],
                advance_px: advance,
            };
        };
        outlined.draw(|x, y, cov| {
            if x < gw && y < gh {
                // Mild optical stem contrast boost (`pow(0.84)`) for dark ink text on light frosted glass
                let boosted = cov.clamp(0.0, 1.0).powf(0.84);
                let a = (boosted * 255.0).round() as u8;
                let idx = (((ay + y) * ATLAS_SIZE + (ax + x)) * 4) as usize;
                self.pixels[idx] = a;
                self.pixels[idx + 1] = a;
                self.pixels[idx + 2] = a;
                self.pixels[idx + 3] = a;
            }
        });
        self.dirty = true;
        let inv = 1.0 / ATLAS_SIZE as f32;
        GlyphEntry {
            uv_min: [ax as f32 * inv, ay as f32 * inv],
            uv_max: [(ax + gw) as f32 * inv, (ay + gh) as f32 * inv],
            size_px: [gw as f32, gh as f32],
            bearing_px: [bounds.min.x, -bounds.min.y],
            advance_px: advance,
        }
    }

    fn rasterize_fallback(&mut self, ch: char, px_size: f32) -> GlyphEntry {
        if ch.is_whitespace() {
            return GlyphEntry {
                uv_min: self.white_uv,
                uv_max: self.white_uv,
                size_px: [0.0, 0.0],
                bearing_px: [0.0, 0.0],
                advance_px: px_size * 0.35,
            };
        }
        let scale = (px_size / 14.0).clamp(1.0, 4.0).round() as u32;
        let is_wide = !ch.is_ascii();
        let gw = (if is_wide { 10 } else { 6 }) * scale;
        let gh = 9 * scale;
        let Some((ax, ay)) = self.allocate_rect(gw, gh) else {
            return GlyphEntry {
                uv_min: self.white_uv,
                uv_max: self.white_uv,
                size_px: [0.0, 0.0],
                bearing_px: [0.0, 0.0],
                advance_px: gw as f32 + 2.0,
            };
        };
        for py in 0..gh {
            for px in 0..gw {
                let bx = px / scale;
                let by = py / scale;
                if bx == 0 || bx + 1 == gw / scale || by == 0 || by + 1 == gh / scale {
                    let idx = (((ay + py) * ATLAS_SIZE + (ax + px)) * 4) as usize;
                    self.pixels[idx..idx + 4].copy_from_slice(&[255, 255, 255, 255]);
                }
            }
        }
        self.dirty = true;
        let inv = 1.0 / ATLAS_SIZE as f32;
        GlyphEntry {
            uv_min: [ax as f32 * inv, ay as f32 * inv],
            uv_max: [(ax + gw) as f32 * inv, (ay + gh) as f32 * inv],
            size_px: [gw as f32, gh as f32],
            bearing_px: [0.0, gh as f32 * 0.85],
            advance_px: gw as f32 + 2.0 * scale as f32,
        }
    }

    fn allocate_rect(&mut self, w: u32, h: u32) -> Option<(u32, u32)> {
        let pw = w + GLYPH_PAD * 2;
        let ph = h + GLYPH_PAD * 2;
        if self.cursor_x + pw >= ATLAS_SIZE {
            self.cursor_x = 2;
            self.cursor_y += self.row_height;
            self.row_height = 0;
        }
        if self.cursor_y + ph >= ATLAS_SIZE {
            return None;
        }
        let out = (self.cursor_x + GLYPH_PAD, self.cursor_y + GLYPH_PAD);
        self.cursor_x += pw;
        self.row_height = self.row_height.max(ph);
        Some(out)
    }
}

fn try_load_font(path: &str) -> Option<FontArc> {
    if !Path::new(path).is_file() {
        return None;
    }
    let bytes = fs::read(path).ok()?;
    if let Ok(vec_font) = FontVec::try_from_vec_and_index(bytes.clone(), 2) {
        return Some(FontArc::new(vec_font));
    }
    FontArc::try_from_vec(bytes).ok()
}

fn load_system_fonts() -> (Option<FontArc>, Option<FontArc>) {
    if let Some(custom) = env::var_os("XR_FILES_FONT") {
        let f = try_load_font(&custom.to_string_lossy());
        return (f.clone(), f);
    }
    let regular_paths = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-DemiLight.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ];
    let bold_paths = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Bold.otf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc",
    ];
    let reg = regular_paths.iter().find_map(|p| try_load_font(p));
    let bold = bold_paths
        .iter()
        .find_map(|p| try_load_font(p))
        .or_else(|| reg.clone());
    (reg, bold)
}
