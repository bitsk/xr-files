// SPDX-License-Identifier: GPL-3.0-or-later
//! Spatial File Manager Application State Machine & Icon-First UKUI XR "Qingkong" Scene Builder.
//! Design Philosophy:
//! - Icon-First Zero-Clutter Chrome: Spatial Icon Rail on left, pure SDF vector icon toolbar in header,
//!   4-orb circular spatial action dock on the right preview stage, and 4×2 icon context palette.
//!   Text labels appear only as contextual hover tooltips or concise single-word stems!
//! - Visual Imagery First: Unobstructed 16:9 Panoramic Hero Media Showcase Strip + deduplicated
//!   full-bleed visual cards & golden folders + massive full-bleed Right Visual Preview Stage.

use crate::{
    fs_ops::{
        self, Bookmark, ClipboardOp, FileClipboard, FileEntry, FileKind, SortBy, StorageVolume,
        UndoRecord, TRASH_URI,
    },
    icons::{icon_slot_for_kind, UiIcon, BUILTIN_ICON_COUNT},
    preview::{AsyncPreviewer, MeshVertex, MipmappedRgba},
    renderer::THUMB_LAYERS,
    simd::{self, SoaAabbBatch},
    spatial::{
        arc_panel_pose, center_eye_position, quat_forward, quat_from_yaw_pitch, quat_mul,
        quat_rotate_vec3, user_facing_orientation, vec3_add, vec3_dot, vec3_len, vec3_normalize,
        vec3_scale, vec3_sub, EyeView, Pose3D,
    },
    text::{FontWeight, GlyphAtlas},
    treemap3d::{self, TreemapBlock},
};
use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

pub const CARDS_PER_PAGE: usize = 8; // 3 Hero Media Cards on top + 8 unique Grid Cards (4×2)
const GRID_COLS: usize = 4;
const HIT_SLOP_RAD: f32 = 0.0384; // 2.2° minimum angular target rule

#[inline]
fn srgb_chan_to_linear(u: u8) -> f32 {
    let c = (u as f32) / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

#[inline]
fn qk_rgba(r: u8, g: u8, b: u8, a: f32) -> [f32; 4] {
    [
        srgb_chan_to_linear(r),
        srgb_chan_to_linear(g),
        srgb_chan_to_linear(b),
        a,
    ]
}

fn col_glass_window() -> [f32; 4] {
    qk_rgba(235, 243, 252, 0.58)
}
fn col_content_canvas() -> [f32; 4] {
    qk_rgba(248, 251, 255, 0.62)
}
fn col_card_surface() -> [f32; 4] {
    qk_rgba(255, 255, 255, 0.80)
}
fn col_card_hover() -> [f32; 4] {
    qk_rgba(255, 255, 255, 0.95)
}
fn col_card_selected() -> [f32; 4] {
    qk_rgba(232, 243, 255, 0.94)
}
fn col_ghost_hover() -> [f32; 4] {
    qk_rgba(255, 255, 255, 0.62)
}
fn col_orb_idle() -> [f32; 4] {
    qk_rgba(255, 255, 255, 0.56)
}
fn col_kylin_blue() -> [f32; 4] {
    qk_rgba(55, 144, 250, 0.95)
}
fn col_ink_navy() -> [f32; 4] {
    qk_rgba(32, 45, 68, 1.0)
}
fn col_muted_slate() -> [f32; 4] {
    qk_rgba(88, 108, 138, 0.95)
}
fn col_pure_white() -> [f32; 4] {
    [1.0, 1.0, 1.0, 1.0]
}
fn col_rim_luminous() -> [f32; 4] {
    [1.0, 1.0, 1.0, 0.78]
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ViewMode {
    Workbench,
    #[serde(rename = "treemap3d", alias = "treemap3_d")]
    Treemap3D,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptMode {
    Search,
    NewFolder,
    Rename,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct HandControllerInput {
    pub active: bool,
    pub grip_pose: Pose3D,
    pub aim_origin: [f32; 3],
    pub aim_dir: [f32; 3],
    pub trigger_pressed: bool,
    pub squeeze_pressed: bool,
    pub secondary_just_pressed: bool,
    pub back_just_pressed: bool,
    pub thumb_click_just_pressed: bool,
    pub stick: [f32; 2],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HapticEvent {
    HoverTick,
    PrimaryClick,
    ContextMenu,
    GrabStart,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, Pod, Zeroable)]
pub struct SolidVertex {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 4],
    pub emissive: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, Pod, Zeroable)]
pub struct QuadInstance {
    pub center: [f32; 3],
    pub right_vec: [f32; 3],
    pub up_vec: [f32; 3],
    pub uv_rect: [f32; 4],
    pub fill_color: [f32; 4],
    pub border_color: [f32; 4],
    pub params: [f32; 4],
}

#[derive(Default)]
pub struct FrameScene {
    pub solid_vertices: Vec<SolidVertex>,
    pub quads: Vec<QuadInstance>,
}

#[derive(Debug, Clone)]
enum HitAction {
    NavigateLocation(String),
    ToggleView(ViewMode),
    CycleSort,
    ToggleHidden,
    PageDelta(isize),
    SelectEntry(usize),
    NavigateUp,
    NavigateBack,
    StartPrompt(PromptMode),
    ClearSearch,
    PromptBackspace,
    PromptCommit,
    PromptCancel,
    InspectorOpen { cinema: bool },
    InspectorTrashOrRestore,
    OpenContextMenuHere,
    UndoLast,
    RecenterWorkspace,
    GrabWorkspaceBar,
    ContextAction(ContextMenuOp),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ContextMenuOp {
    Open,
    OpenCinema,
    Copy,
    Cut,
    Paste,
    NewFolder,
    Rename,
    TrashOrRestore,
    Close,
}

#[derive(Debug, Clone)]
struct ClickStamp {
    entry_index: usize,
    ray_dir: [f32; 3],
    at: Instant,
}

#[derive(Debug, Clone)]
struct DragSession {
    hand: usize,
    source_paths: Vec<PathBuf>,
    primary_label: String,
    start_point: [f32; 3],
    active_drag: bool,
}

#[derive(Debug, Clone)]
struct ContextMenuState {
    anchor_world: [f32; 3],
}

#[derive(Debug, Clone)]
struct WorkspaceGrab {
    hand: usize,
    start_ray_dir: [f32; 3],
    base_yaw: f32,
    via_trigger: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct StatusSnapshot {
    pub ready: bool,
    pub simd_backend: String,
    pub current_location: String,
    pub view_mode: ViewMode,
    pub sort_by: SortBy,
    pub ascending: bool,
    pub show_hidden: bool,
    pub search_query: String,
    pub total_entries: usize,
    pub visible_entries: usize,
    pub selected_paths: Vec<String>,
    pub undo_depth: usize,
    pub status_message: String,
    pub submitted_frames: u64,
}

pub struct SpatialFilesApp {
    pub current_location: String,
    back_stack: Vec<String>,
    forward_stack: Vec<String>,
    pub all_entries: Vec<FileEntry>,
    pub filtered_indices: Vec<usize>,
    pub selected_entry_indices: HashSet<usize>,
    pub primary_selected: Option<usize>,
    pub page: usize,

    pub view_mode: ViewMode,
    pub sort_by: SortBy,
    pub ascending: bool,
    pub show_hidden: bool,
    pub search_query: String,
    pub prompt_mode: Option<PromptMode>,
    pub prompt_buffer: String,

    pub bookmarks: Vec<Bookmark>,
    pub volumes: Vec<StorageVolume>,
    pub clipboard: Option<FileClipboard>,
    pub undo_stack: Vec<UndoRecord>,
    pub status_message: String,

    pub previewer: AsyncPreviewer,
    pub thumb_slot_by_path: HashMap<PathBuf, u32>,
    next_photo_slot: u32,
    pub pending_gpu_thumbs: Vec<(u32, MipmappedRgba)>,
    pub pending_gpu_inspect_image: Option<MipmappedRgba>,
    last_uploaded_inspect_path: Option<PathBuf>,

    pub anchor_pos: [f32; 3],
    pub anchor_yaw: f32,
    pub workspace_radius: f32,
    pub card_scale: f32,
    pub embedded_panel: bool,
    anchor_initialized: bool,
    recenter_requested: bool,
    shutdown_flag: bool,
    pub submitted_frames: u64,

    slot_z_cur: Vec<f32>,
    slot_z_tgt: Vec<f32>,
    treemap_lift_cur: Vec<f32>,
    treemap_lift_tgt: Vec<f32>,
    cached_treemap: Vec<TreemapBlock>,
    treemap_dirty: bool,

    hit_batch: SoaAabbBatch,
    hit_actions: Vec<HitAction>,
    hovered_hit: [Option<usize>; 2],
    prev_trigger: [bool; 2],
    prev_squeeze: [bool; 2],
    last_click: [Option<ClickStamp>; 2],
    stick_cooldown_until: [Instant; 2],
    drag_session: Option<DragSession>,
    workspace_grab: Option<WorkspaceGrab>,
    context_menu: Option<ContextMenuState>,
    pending_haptics: [Option<HapticEvent>; 2],

    start_instant: Instant,
    last_step_instant: Instant,
}

impl SpatialFilesApp {
    pub fn new(initial_path: Option<PathBuf>) -> Self {
        let initial_file = initial_path
            .as_ref()
            .filter(|p| p.is_file())
            .cloned();
        let start_loc = initial_path
            .filter(|p| p.exists())
            .map(|p| {
                if p.is_dir() {
                    p
                } else {
                    p.parent().map(Path::to_path_buf).unwrap_or_else(fs_ops::home_dir)
                }
            })
            .unwrap_or_else(fs_ops::home_dir)
            .to_string_lossy()
            .into_owned();

        let now = Instant::now();
        let mut app = Self {
            current_location: start_loc,
            back_stack: Vec::new(),
            forward_stack: Vec::new(),
            all_entries: Vec::new(),
            filtered_indices: Vec::new(),
            selected_entry_indices: HashSet::new(),
            primary_selected: None,
            page: 0,
            view_mode: ViewMode::Workbench,
            sort_by: SortBy::Name,
            ascending: true,
            show_hidden: false,
            search_query: String::new(),
            prompt_mode: None,
            prompt_buffer: String::new(),
            bookmarks: fs_ops::discover_bookmarks(),
            volumes: fs_ops::discover_volumes(),
            clipboard: None,
            undo_stack: Vec::new(),
            status_message: String::new(),
            previewer: AsyncPreviewer::new(),
            thumb_slot_by_path: HashMap::new(),
            next_photo_slot: 0,
            pending_gpu_thumbs: Vec::new(),
            pending_gpu_inspect_image: None,
            last_uploaded_inspect_path: None,
            anchor_pos: [0.0, 1.60, 0.0],
            anchor_yaw: 0.0,
            workspace_radius: 1.24,
            card_scale: 1.0,
            embedded_panel: false,
            anchor_initialized: false,
            recenter_requested: false,
            shutdown_flag: false,
            submitted_frames: 0,
            slot_z_cur: vec![0.010; 64],
            slot_z_tgt: vec![0.010; 64],
            treemap_lift_cur: vec![0.0; 64],
            treemap_lift_tgt: vec![0.0; 64],
            cached_treemap: Vec::new(),
            treemap_dirty: true,
            hit_batch: SoaAabbBatch::with_capacity(192),
            hit_actions: Vec::with_capacity(192),
            hovered_hit: [None, None],
            prev_trigger: [false, false],
            prev_squeeze: [false, false],
            last_click: [None, None],
            stick_cooldown_until: [now, now],
            drag_session: None,
            workspace_grab: None,
            context_menu: None,
            pending_haptics: [None, None],
            start_instant: now,
            last_step_instant: now,
        };
        app.reload_directory();
        if let Some(target_file) = initial_file {
            app.select_by_path(&target_file);
        }
        app
    }

    pub fn request_shutdown(&mut self) {
        self.shutdown_flag = true;
    }

    pub fn shutdown_requested(&self) -> bool {
        self.shutdown_flag
    }

    pub fn request_recenter(&mut self) {
        self.recenter_requested = true;
    }

    pub fn take_recenter_request(&mut self) -> bool {
        std::mem::replace(&mut self.recenter_requested, false)
    }

    pub fn take_haptics(&mut self) -> [Option<HapticEvent>; 2] {
        std::mem::take(&mut self.pending_haptics)
    }

    pub fn set_embedded_panel(&mut self, embedded: bool) {
        self.embedded_panel = embedded;
        if embedded {
            self.anchor_pos = [0.0, 1.60, 0.0];
            self.anchor_yaw = 0.0;
            self.workspace_radius = 1.24;
            self.anchor_initialized = true;
        }
    }

    /// Computes an aspect-preserving camera frustum that frames the coplanar two-pane
    /// Spatial Workbench + Right Preview Stage inside a HyperSpace Wayland surface.
    pub fn embedded_eye(width: u32, height: u32) -> EyeView {
        let aspect = (width.max(1) as f32 / height.max(1) as f32).clamp(0.6, 4.0);
        let tan_x0 = 0.905f32 / 1.24f32;
        let tan_y0 = 0.402f32 / 1.24f32;
        let design_aspect = tan_x0 / tan_y0;
        let (tan_x, tan_y) = if aspect >= design_aspect {
            (tan_y0 * aspect, tan_y0)
        } else {
            (tan_x0, tan_x0 / aspect)
        };
        let hx = tan_x.atan();
        let hy = tan_y.atan();
        EyeView {
            position: [0.0, 1.60, 0.0],
            orientation: [0.0, 0.0, 0.0, 1.0],
            fov: [-hx, hx, hy, -hy],
        }
    }

    /// Returns `true` when background thumbnails, hover springs, or a 3D hologram mesh require active animation frames.
    pub fn has_active_animation(&self) -> bool {
        if !self.pending_gpu_thumbs.is_empty() || self.pending_gpu_inspect_image.is_some() {
            return true;
        }
        if self
            .previewer
            .active_inspect
            .as_ref()
            .is_some_and(|p| p.mesh.is_some())
        {
            return true;
        }
        let spring_moving = self
            .slot_z_cur
            .iter()
            .zip(self.slot_z_tgt.iter())
            .any(|(a, b)| (a - b).abs() > 0.0005)
            || self
                .treemap_lift_cur
                .iter()
                .zip(self.treemap_lift_tgt.iter())
                .any(|(a, b)| (a - b).abs() > 0.0005);
        spring_moving
    }

    pub fn status_snapshot(&self) -> StatusSnapshot {
        let mut selected_paths: Vec<String> = self
            .selected_entry_indices
            .iter()
            .filter_map(|&idx| self.all_entries.get(idx))
            .map(|e| e.path.to_string_lossy().into_owned())
            .collect();
        selected_paths.sort();
        StatusSnapshot {
            ready: true,
            simd_backend: simd::active_simd_backend().into(),
            current_location: self.current_location.clone(),
            view_mode: self.view_mode,
            sort_by: self.sort_by,
            ascending: self.ascending,
            show_hidden: self.show_hidden,
            search_query: self.search_query.clone(),
            total_entries: self.all_entries.len(),
            visible_entries: self.filtered_indices.len(),
            selected_paths,
            undo_depth: self.undo_stack.len(),
            status_message: self.status_message.clone(),
            submitted_frames: self.submitted_frames,
        }
    }

    pub fn navigate_to(&mut self, location: impl Into<String>) {
        let loc = location.into();
        if loc == self.current_location {
            self.reload_directory();
            return;
        }
        self.back_stack.push(self.current_location.clone());
        self.forward_stack.clear();
        self.current_location = loc;
        self.search_query.clear();
        self.page = 0;
        self.context_menu = None;
        self.reload_directory();
    }

    pub fn navigate_up(&mut self) {
        if self.current_location == TRASH_URI {
            self.navigate_to(fs_ops::home_dir().to_string_lossy().into_owned());
            return;
        }
        if let Some(parent) = Path::new(&self.current_location).parent() {
            let parent_str = parent.to_string_lossy().into_owned();
            if !parent_str.is_empty() && parent_str != self.current_location {
                self.navigate_to(parent_str);
            }
        }
    }

    pub fn reload_directory(&mut self) {
        match fs_ops::read_location_entries(&self.current_location, self.show_hidden) {
            Ok(mut entries) => {
                fs_ops::sort_entries(&mut entries, self.sort_by, self.ascending);
                self.all_entries = entries;
                self.selected_entry_indices.clear();
                self.primary_selected = None;
                self.previewer.clear_inspect();
                self.last_uploaded_inspect_path = None;
                self.recompute_filtered();
                let preferred = self
                    .filtered_indices
                    .iter()
                    .copied()
                    .find(|&idx| {
                        matches!(
                            self.all_entries[idx].kind,
                            FileKind::Image | FileKind::Model3D | FileKind::Video
                        )
                    })
                    .or_else(|| self.filtered_indices.first().copied());
                if let Some(idx) = preferred {
                    self.select_single(idx);
                }
                self.status_message = format!("{} 项", self.filtered_indices.len());
            }
            Err(err) => {
                self.status_message = format!("读取失败: {err}");
            }
        }
    }

    pub fn recompute_filtered(&mut self) {
        if self.search_query.trim().is_empty() {
            self.filtered_indices = (0..self.all_entries.len()).collect();
        } else {
            let names: Vec<&str> = self.all_entries.iter().map(|e| e.name.as_str()).collect();
            let ranked = simd::filter_and_rank_names(&names, &self.search_query);
            self.filtered_indices = ranked.into_iter().map(|(pos, _)| pos).collect();
        }
        let max_page = self.page_count().saturating_sub(1);
        if self.page > max_page {
            self.page = max_page;
        }
        self.treemap_dirty = true;
    }

    fn hero_entry_indices(&self) -> Vec<usize> {
        if !self.search_query.is_empty() {
            return Vec::new();
        }
        self.filtered_indices
            .iter()
            .copied()
            .filter(|&idx| self.all_entries[idx].kind != FileKind::Directory)
            .take(3)
            .collect()
    }

    fn grid_entry_indices(&self, hero: &[usize]) -> Vec<usize> {
        if hero.is_empty() {
            return self.filtered_indices.clone();
        }
        let rest: Vec<usize> = self
            .filtered_indices
            .iter()
            .copied()
            .filter(|idx| !hero.contains(idx))
            .collect();
        if rest.is_empty() {
            Vec::new()
        } else {
            rest
        }
    }

    pub fn page_count(&self) -> usize {
        let hero = self.hero_entry_indices();
        let grid_len = self.grid_entry_indices(&hero).len();
        let per_page = if hero.is_empty() {
            GRID_COLS * 3
        } else {
            CARDS_PER_PAGE
        };
        if grid_len == 0 {
            1
        } else {
            (grid_len + per_page - 1) / per_page
        }
    }

    pub fn select_single(&mut self, entry_idx: usize) {
        if entry_idx >= self.all_entries.len() {
            return;
        }
        self.selected_entry_indices.clear();
        self.selected_entry_indices.insert(entry_idx);
        self.primary_selected = Some(entry_idx);
        self.previewer.request_inspect(&self.all_entries[entry_idx]);
    }

    pub fn set_search_query(&mut self, query: String) {
        self.search_query = query;
        self.page = 0;
        self.recompute_filtered();
        if let Some(&first_match) = self.filtered_indices.first() {
            if self
                .primary_selected
                .is_none_or(|sel| !self.filtered_indices.contains(&sel))
            {
                self.select_single(first_match);
            }
        }
    }

    pub fn set_view_mode(&mut self, mode: ViewMode) {
        self.view_mode = mode;
        self.treemap_dirty = true;
    }

    pub fn create_folder_named(&mut self, name: &str) -> Result<PathBuf> {
        anyhow::ensure!(self.current_location != TRASH_URI, "不能在回收站内新建文件夹");
        let (created, undo) = fs_ops::create_directory(Path::new(&self.current_location), name)?;
        self.undo_stack.push(undo);
        self.reload_directory();
        self.select_by_path(&created);
        self.status_message = format!("已新建：{}", name.trim());
        Ok(created)
    }

    pub fn rename_selected(&mut self, new_name: &str) -> Result<PathBuf> {
        let idx = self
            .primary_selected
            .ok_or_else(|| anyhow::anyhow!("请先选择要重命名的项目"))?;
        let src = self.all_entries[idx].path.clone();
        let (dst, undo) = fs_ops::rename_path(&src, new_name)?;
        self.undo_stack.push(undo);
        self.reload_directory();
        self.select_by_path(&dst);
        self.status_message = format!("已重命名为：{}", new_name.trim());
        Ok(dst)
    }

    pub fn copy_selected_to_clipboard(&mut self, op: ClipboardOp) {
        let paths = self.selected_paths();
        if paths.is_empty() {
            return;
        }
        let n = paths.len();
        self.clipboard = Some(FileClipboard { op, paths });
        self.status_message = match op {
            ClipboardOp::Copy => format!("已复制 {n} 项"),
            ClipboardOp::Cut => format!("已剪切 {n} 项"),
        };
    }

    pub fn paste_clipboard_here(&mut self) -> Result<()> {
        anyhow::ensure!(self.current_location != TRASH_URI, "不能粘贴到回收站");
        let Some(clip) = self.clipboard.clone() else {
            anyhow::bail!("剪贴板为空");
        };
        let target_dir = PathBuf::from(&self.current_location);
        let undo = match clip.op {
            ClipboardOp::Copy => fs_ops::copy_paths_to_dir(&clip.paths, &target_dir)?,
            ClipboardOp::Cut => {
                let u = fs_ops::move_paths_to_dir(&clip.paths, &target_dir)?;
                self.clipboard = None;
                u
            }
        };
        let summary = undo.summary();
        self.undo_stack.push(undo);
        self.reload_directory();
        self.status_message = format!("已完成：{summary}");
        Ok(())
    }

    pub fn trash_or_restore_selected(&mut self) -> Result<()> {
        if self.current_location == TRASH_URI {
            let entries: Vec<FileEntry> = self
                .selected_entry_indices
                .iter()
                .filter_map(|&i| self.all_entries.get(i).cloned())
                .collect();
            anyhow::ensure!(!entries.is_empty(), "请先选择要还原的项目");
            let mut count = 0usize;
            for entry in &entries {
                fs_ops::restore_trash_entry(entry)?;
                count += 1;
            }
            self.reload_directory();
            self.status_message = format!("已还原 {count} 项");
            return Ok(());
        }
        let paths = self.selected_paths();
        anyhow::ensure!(!paths.is_empty(), "请先选择项目");
        let n = paths.len();
        let undo = fs_ops::trash_paths(&paths)?;
        self.undo_stack.push(undo);
        self.reload_directory();
        self.status_message = format!("已移入回收站 {n} 项");
        Ok(())
    }

    pub fn undo_last(&mut self) -> Result<String> {
        let Some(record) = self.undo_stack.pop() else {
            anyhow::bail!("没有可撤销的操作");
        };
        let msg = fs_ops::execute_undo(record)?;
        self.reload_directory();
        self.status_message = msg.clone();
        Ok(msg)
    }

    pub fn open_selected(&mut self, force_cinema: bool) {
        let Some(idx) = self.primary_selected else {
            return;
        };
        let Some(entry) = self.all_entries.get(idx).cloned() else {
            return;
        };
        if entry.kind == FileKind::Directory && !force_cinema && self.current_location != TRASH_URI
        {
            self.navigate_to(entry.path.to_string_lossy().into_owned());
            return;
        }
        match fs_ops::open_with_system_or_cinema(&entry, force_cinema) {
            Ok(msg) => self.status_message = msg,
            Err(err) => self.status_message = format!("打开失败: {err}"),
        }
    }

    pub fn handle_key_text(&mut self, text: &str) {
        for ch in text.chars() {
            if ch.is_control() {
                continue;
            }
            if self.prompt_mode.is_none() {
                self.prompt_mode = Some(PromptMode::Search);
                self.prompt_buffer = self.search_query.clone();
            }
            self.prompt_buffer.push(ch);
            if self.prompt_mode == Some(PromptMode::Search) {
                self.set_search_query(self.prompt_buffer.clone());
            }
        }
    }

    pub fn handle_backspace(&mut self) {
        if self.prompt_mode.is_some() {
            self.prompt_buffer.pop();
            if self.prompt_mode == Some(PromptMode::Search) {
                self.set_search_query(self.prompt_buffer.clone());
            }
        } else if !self.search_query.is_empty() {
            let mut q = self.search_query.clone();
            q.pop();
            self.set_search_query(q);
        } else {
            self.navigate_up();
        }
    }

    pub fn commit_prompt(&mut self) {
        let Some(mode) = self.prompt_mode.take() else {
            self.open_selected(false);
            return;
        };
        let text = self.prompt_buffer.trim().to_string();
        match mode {
            PromptMode::Search => self.set_search_query(text),
            PromptMode::NewFolder => {
                let name = if text.is_empty() { "新建文件夹" } else { &text };
                if let Err(e) = self.create_folder_named(name) {
                    self.status_message = format!("新建失败: {e}");
                }
            }
            PromptMode::Rename => {
                if !text.is_empty() {
                    if let Err(e) = self.rename_selected(&text) {
                        self.status_message = format!("重命名失败: {e}");
                    }
                }
            }
        }
    }

    pub fn cancel_modals(&mut self) {
        if self.context_menu.take().is_some() {
            return;
        }
        if self.prompt_mode.take().is_some() {
            return;
        }
        if !self.search_query.is_empty() {
            self.set_search_query(String::new());
            return;
        }
        self.navigate_up();
    }

    fn selected_paths(&self) -> Vec<PathBuf> {
        let mut out: Vec<PathBuf> = self
            .selected_entry_indices
            .iter()
            .filter_map(|&idx| self.all_entries.get(idx).map(|e| e.path.clone()))
            .collect();
        out.sort();
        out
    }

    fn select_by_path(&mut self, target: &Path) {
        if let Some((idx, _)) = self
            .all_entries
            .iter()
            .enumerate()
            .find(|(_, e)| e.path == target)
        {
            self.select_single(idx);
            let hero = self.hero_entry_indices();
            let grid = self.grid_entry_indices(&hero);
            if let Some(pos) = grid.iter().position(|&i| i == idx) {
                self.page = pos / CARDS_PER_PAGE;
            }
        }
    }

    pub fn step_page(&mut self, delta: isize) {
        let total = self.page_count() as isize;
        self.page = (self.page as isize + delta).clamp(0, (total - 1).max(0)) as usize;
    }

    // ========================================================================
    // Per-Frame Simulation & Icon-First Qingkong Scene Builder
    // ========================================================================

    pub fn step_and_build_scene(
        &mut self,
        eyes: &[EyeView; 2],
        hands: &[HandControllerInput; 2],
        glyphs: &mut GlyphAtlas,
    ) -> FrameScene {
        let now = Instant::now();
        let dt = now
            .duration_since(self.last_step_instant)
            .as_secs_f32()
            .clamp(0.001, 0.1);
        self.last_step_instant = now;

        let eye_center = center_eye_position(eyes);
        if !self.anchor_initialized || self.recenter_requested {
            self.anchor_pos = eye_center;
            let fwd = quat_forward(eyes[0].orientation);
            self.anchor_yaw = (-fwd[0]).atan2(-fwd[2]);
            self.anchor_initialized = true;
            self.recenter_requested = false;
        }

        for &idx in self.filtered_indices.iter().take(28) {
            self.previewer.request_thumbnail(&self.all_entries[idx]);
        }

        let photo_cap = (THUMB_LAYERS - BUILTIN_ICON_COUNT).max(1);
        for (path, mips) in self.previewer.poll() {
            let slot = *self.thumb_slot_by_path.entry(path).or_insert_with(|| {
                let s = BUILTIN_ICON_COUNT + (self.next_photo_slot % photo_cap);
                self.next_photo_slot = self.next_photo_slot.wrapping_add(1);
                s
            });
            self.pending_gpu_thumbs.push((slot, mips));
        }
        if let Some(ref inspect) = self.previewer.active_inspect {
            if inspect.image.is_some()
                && self.last_uploaded_inspect_path.as_ref() != Some(&inspect.path)
            {
                self.last_uploaded_inspect_path = Some(inspect.path.clone());
                self.pending_gpu_inspect_image = inspect.image.clone();
            }
        }

        self.process_controller_inputs(hands, now, dt);

        let spring_alpha = (1.0 - (-18.0 * dt).exp()).clamp(0.05, 1.0);
        simd::lerp_f32_slice_in_place(&mut self.slot_z_cur, &self.slot_z_tgt, spring_alpha);
        simd::lerp_f32_slice_in_place(
            &mut self.treemap_lift_cur,
            &self.treemap_lift_tgt,
            spring_alpha,
        );

        let mut scene = FrameScene::default();
        self.hit_batch.clear();
        self.hit_actions.clear();

        if self.embedded_panel {
            self.anchor_pos = [0.0, 1.60, 0.0];
            self.anchor_yaw = 0.0;
            self.workspace_radius = 1.24;
            self.emit_main_qingkong_window(&mut scene, 0.0, eye_center, hands, glyphs);
            self.emit_right_gallery_stage(&mut scene, 0.0, eye_center, glyphs);
        } else {
            let main_yaw_deg = 9.5;
            let insp_yaw_deg = -30.5;
            self.emit_depth_breadcrumbs(&mut scene, main_yaw_deg, eye_center, glyphs);
            self.emit_main_qingkong_window(&mut scene, main_yaw_deg, eye_center, hands, glyphs);
            self.emit_minimal_companion_bar(&mut scene, main_yaw_deg, eye_center, glyphs);
            self.emit_right_gallery_stage(&mut scene, insp_yaw_deg, eye_center, glyphs);
        }
        if self.context_menu.is_some() {
            self.emit_context_menu(&mut scene, eye_center, glyphs);
        }
        self.hit_batch.finalize_padding();

        self.update_hover_and_emit_hands(&mut scene, eye_center, hands, glyphs);

        scene
    }

    fn process_controller_inputs(
        &mut self,
        hands: &[HandControllerInput; 2],
        now: Instant,
        dt: f32,
    ) {
        if self.hit_batch.is_empty() {
            return;
        }
        for h in 0..2 {
            let ctrl = hands[h];
            if !ctrl.active {
                self.prev_trigger[h] = false;
                self.prev_squeeze[h] = false;
                if self.drag_session.as_ref().is_some_and(|d| d.hand == h) {
                    self.drag_session = None;
                }
                if self.workspace_grab.as_ref().is_some_and(|g| g.hand == h) {
                    self.workspace_grab = None;
                }
                continue;
            }

            let pick = self.hit_batch.pick_ray(ctrl.aim_origin, ctrl.aim_dir, 0.06, 8.0);
            let trig_just_down = ctrl.trigger_pressed && !self.prev_trigger[h];
            let trig_just_up = !ctrl.trigger_pressed && self.prev_trigger[h];
            let sqz_just_down = ctrl.squeeze_pressed && !self.prev_squeeze[h];
            let sqz_just_up = !ctrl.squeeze_pressed && self.prev_squeeze[h];
            self.prev_trigger[h] = ctrl.trigger_pressed;
            self.prev_squeeze[h] = ctrl.squeeze_pressed;

            if ctrl.back_just_pressed {
                self.cancel_modals();
            }
            if ctrl.thumb_click_just_pressed {
                self.workspace_radius = 1.24;
                self.card_scale = 1.0;
                self.request_recenter();
            }

            if ctrl.secondary_just_pressed {
                if let Some(p) = pick {
                    if let Some(&HitAction::SelectEntry(idx)) = self.hit_actions.get(p.index) {
                        self.select_single(idx);
                    }
                }
                let anchor_world = pick.map(|p| p.hit_point).unwrap_or_else(|| {
                    vec3_add(ctrl.aim_origin, vec3_scale(ctrl.aim_dir, 1.18))
                });
                self.context_menu = Some(ContextMenuState { anchor_world });
                self.pending_haptics[h] = Some(HapticEvent::ContextMenu);
            }

            if sqz_just_down {
                if let Some(p) = pick {
                    if let Some(&HitAction::SelectEntry(idx)) = self.hit_actions.get(p.index) {
                        if !self.selected_entry_indices.contains(&idx) {
                            self.select_single(idx);
                        }
                        let paths = self.selected_paths();
                        if !paths.is_empty() && self.current_location != TRASH_URI {
                            self.drag_session = Some(DragSession {
                                hand: h,
                                primary_label: short_display_stem(&self.all_entries[idx]),
                                source_paths: paths,
                                start_point: p.hit_point,
                                active_drag: true,
                            });
                            self.pending_haptics[h] = Some(HapticEvent::GrabStart);
                        }
                    } else {
                        self.workspace_grab = Some(WorkspaceGrab {
                            hand: h,
                            start_ray_dir: ctrl.aim_dir,
                            base_yaw: self.anchor_yaw,
                            via_trigger: false,
                        });
                        self.pending_haptics[h] = Some(HapticEvent::GrabStart);
                    }
                } else {
                    self.workspace_grab = Some(WorkspaceGrab {
                        hand: h,
                        start_ray_dir: ctrl.aim_dir,
                        base_yaw: self.anchor_yaw,
                        via_trigger: false,
                    });
                    self.pending_haptics[h] = Some(HapticEvent::GrabStart);
                }
            }

            if let Some(ref grab) = self.workspace_grab {
                let held = if grab.via_trigger {
                    ctrl.trigger_pressed
                } else {
                    ctrl.squeeze_pressed
                };
                if grab.hand == h && held {
                    let yaw0 = (-grab.start_ray_dir[0]).atan2(-grab.start_ray_dir[2]);
                    let yaw1 = (-ctrl.aim_dir[0]).atan2(-ctrl.aim_dir[2]);
                    self.anchor_yaw = grab.base_yaw + (yaw1 - yaw0);
                    if ctrl.stick[1].abs() > 0.18 {
                        self.workspace_radius =
                            (self.workspace_radius - ctrl.stick[1] * 1.15 * dt).clamp(1.00, 2.80);
                    }
                }
            }
            if sqz_just_up {
                if self
                    .workspace_grab
                    .as_ref()
                    .is_some_and(|g| g.hand == h && !g.via_trigger)
                {
                    self.workspace_grab = None;
                }
                if self.drag_session.as_ref().is_some_and(|d| d.hand == h) {
                    self.finish_drag_drop(pick.map(|p| p.index));
                }
            }

            if self.workspace_grab.is_none() && now >= self.stick_cooldown_until[h] {
                let sy = ctrl.stick[1];
                let sx = ctrl.stick[0];
                if sy <= -0.62 || sx >= 0.65 {
                    self.step_page(1);
                    self.stick_cooldown_until[h] = now + Duration::from_millis(260);
                } else if sy >= 0.62 || sx <= -0.65 {
                    self.step_page(-1);
                    self.stick_cooldown_until[h] = now + Duration::from_millis(260);
                }
            }

            if trig_just_down {
                if let Some(p) = pick {
                    if let Some(action) = self.hit_actions.get(p.index).cloned() {
                        self.pending_haptics[h] = Some(HapticEvent::PrimaryClick);
                        self.execute_hit_press(h, action, p.hit_point, ctrl.aim_dir, now);
                    }
                } else {
                    self.context_menu = None;
                }
            } else if ctrl.trigger_pressed {
                if let Some(ref mut drag) = self.drag_session {
                    if drag.hand == h && !drag.active_drag {
                        let cur_pt = pick.map(|p| p.hit_point).unwrap_or_else(|| {
                            vec3_add(ctrl.aim_origin, vec3_scale(ctrl.aim_dir, 1.25))
                        });
                        if vec3_len(vec3_sub(cur_pt, drag.start_point)) > 0.042 {
                            drag.active_drag = true;
                            self.pending_haptics[h] = Some(HapticEvent::GrabStart);
                        }
                    }
                }
            } else if trig_just_up {
                if self
                    .workspace_grab
                    .as_ref()
                    .is_some_and(|g| g.hand == h && g.via_trigger)
                {
                    self.workspace_grab = None;
                }
                if self.drag_session.as_ref().is_some_and(|d| d.hand == h) {
                    self.finish_drag_drop(pick.map(|p| p.index));
                }
            }
        }
    }

    fn execute_hit_press(
        &mut self,
        hand: usize,
        action: HitAction,
        hit_point: [f32; 3],
        ray_dir: [f32; 3],
        now: Instant,
    ) {
        match action {
            HitAction::GrabWorkspaceBar => {
                self.workspace_grab = Some(WorkspaceGrab {
                    hand,
                    start_ray_dir: ray_dir,
                    base_yaw: self.anchor_yaw,
                    via_trigger: true,
                });
            }
            HitAction::SelectEntry(entry_idx) => {
                self.context_menu = None;
                let is_double = self.last_click[hand].as_ref().is_some_and(|prev| {
                    let dt = now.duration_since(prev.at);
                    let cos_ang = vec3_dot(prev.ray_dir, ray_dir).clamp(-1.0, 1.0);
                    prev.entry_index == entry_idx
                        && dt >= Duration::from_millis(35)
                        && dt <= Duration::from_millis(450)
                        && cos_ang.acos() <= 1.6f32.to_radians()
                });
                if is_double {
                    self.last_click[hand] = None;
                    self.drag_session = None;
                    self.select_single(entry_idx);
                    self.open_selected(false);
                    return;
                }
                self.last_click[hand] = Some(ClickStamp {
                    entry_index: entry_idx,
                    ray_dir,
                    at: now,
                });
                self.select_single(entry_idx);
                if self.current_location != TRASH_URI {
                    let paths = self.selected_paths();
                    if !paths.is_empty() {
                        self.drag_session = Some(DragSession {
                            hand,
                            primary_label: short_display_stem(&self.all_entries[entry_idx]),
                            source_paths: paths,
                            start_point: hit_point,
                            active_drag: false,
                        });
                    }
                }
            }
            HitAction::NavigateLocation(loc) => self.navigate_to(loc),
            HitAction::ToggleView(mode) => self.set_view_mode(mode),
            HitAction::CycleSort => {
                self.sort_by = self.sort_by.next();
                self.reload_directory();
            }
            HitAction::ToggleHidden => {
                self.show_hidden = !self.show_hidden;
                self.reload_directory();
            }
            HitAction::PageDelta(d) => self.step_page(d),
            HitAction::NavigateUp => self.navigate_up(),
            HitAction::NavigateBack => {
                if let Some(prev) = self.back_stack.pop() {
                    self.forward_stack.push(self.current_location.clone());
                    self.current_location = prev;
                    self.page = 0;
                    self.reload_directory();
                }
            }
            HitAction::StartPrompt(mode) => {
                self.prompt_mode = Some(mode);
                self.prompt_buffer = match mode {
                    PromptMode::Search => self.search_query.clone(),
                    PromptMode::NewFolder => "新建文件夹".into(),
                    PromptMode::Rename => self
                        .primary_selected
                        .and_then(|i| self.all_entries.get(i))
                        .map(|e| e.name.clone())
                        .unwrap_or_default(),
                };
            }
            HitAction::ClearSearch => {
                self.prompt_mode = None;
                self.prompt_buffer.clear();
                self.set_search_query(String::new());
            }
            HitAction::PromptBackspace => self.handle_backspace(),
            HitAction::PromptCommit => self.commit_prompt(),
            HitAction::PromptCancel => self.prompt_mode = None,
            HitAction::InspectorOpen { cinema } => self.open_selected(cinema),
            HitAction::InspectorTrashOrRestore => {
                let _ = self.trash_or_restore_selected();
            }
            HitAction::OpenContextMenuHere => {
                self.context_menu = Some(ContextMenuState {
                    anchor_world: hit_point,
                });
            }
            HitAction::UndoLast => {
                let _ = self.undo_last();
            }
            HitAction::RecenterWorkspace => self.request_recenter(),
            HitAction::ContextAction(op) => {
                self.context_menu = None;
                match op {
                    ContextMenuOp::Open => self.open_selected(false),
                    ContextMenuOp::OpenCinema => self.open_selected(true),
                    ContextMenuOp::Copy => self.copy_selected_to_clipboard(ClipboardOp::Copy),
                    ContextMenuOp::Cut => self.copy_selected_to_clipboard(ClipboardOp::Cut),
                    ContextMenuOp::Paste => {
                        let _ = self.paste_clipboard_here();
                    }
                    ContextMenuOp::NewFolder => {
                        let _ = self.create_folder_named("新建文件夹");
                    }
                    ContextMenuOp::Rename => {
                        if let Some(idx) = self.primary_selected {
                            self.prompt_mode = Some(PromptMode::Rename);
                            self.prompt_buffer = self.all_entries[idx].name.clone();
                        }
                    }
                    ContextMenuOp::TrashOrRestore => {
                        let _ = self.trash_or_restore_selected();
                    }
                    ContextMenuOp::Close => {}
                }
            }
        }
    }

    fn finish_drag_drop(&mut self, drop_hit_idx: Option<usize>) {
        let Some(drag) = self.drag_session.take() else {
            return;
        };
        if !drag.active_drag {
            return;
        }
        let Some(hit_idx) = drop_hit_idx else {
            return;
        };
        let Some(action) = self.hit_actions.get(hit_idx).cloned() else {
            return;
        };
        let target_result = match action {
            HitAction::SelectEntry(idx) => self
                .all_entries
                .get(idx)
                .filter(|e| e.kind == FileKind::Directory)
                .map(|e| e.path.to_string_lossy().into_owned()),
            HitAction::NavigateLocation(loc) => Some(loc),
            _ => None,
        };
        if let Some(dest) = target_result {
            if dest == TRASH_URI {
                if let Ok(undo) = fs_ops::trash_paths(&drag.source_paths) {
                    self.undo_stack.push(undo);
                    self.reload_directory();
                }
            } else if let Ok(undo) = fs_ops::move_paths_to_dir(&drag.source_paths, Path::new(&dest))
            {
                self.undo_stack.push(undo);
                self.reload_directory();
            }
        }
    }

    // ========================================================================
    // Icon-First + Visual Gallery Scene Builders
    // ========================================================================

    fn register_hit(&mut self, center: [f32; 3], half_ext: [f32; 3], action: HitAction) -> usize {
        let min = [
            center[0] - half_ext[0],
            center[1] - half_ext[1],
            center[2] - half_ext[2],
        ];
        let max = [
            center[0] + half_ext[0],
            center[1] + half_ext[1],
            center[2] + half_ext[2],
        ];
        let idx = self.hit_batch.push(min, max, HIT_SLOP_RAD);
        self.hit_actions.push(action);
        idx
    }

    fn is_hit_hovered(&self, hit_index: usize) -> bool {
        self.hovered_hit[0] == Some(hit_index) || self.hovered_hit[1] == Some(hit_index)
    }

    /// Top Parent Depth Orb: pure `[ ↑ ]` icon pill peeking gently behind top edge;
    /// parent name appears ONLY on hover!
    fn emit_depth_breadcrumbs(
        &mut self,
        scene: &mut FrameScene,
        yaw_deg: f32,
        eye_center: [f32; 3],
        glyphs: &mut GlyphAtlas,
    ) {
        if self.current_location == TRASH_URI {
            return;
        }
        let Some(parent) = Path::new(&self.current_location).parent() else {
            return;
        };
        let pose = arc_panel_pose(
            self.anchor_pos,
            self.anchor_yaw,
            yaw_deg,
            0.398,
            self.workspace_radius + 0.08,
            eye_center,
        );
        let parent_name = parent
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "/".into());
        self.emit_icon_orb(
            scene,
            &pose,
            [0.0, 0.0, 0.0],
            0.036,
            UiIcon::UpArrow,
            HitAction::NavigateLocation(parent.to_string_lossy().into_owned()),
            false,
            true,
            Some(&parent_name),
            glyphs,
        );
    }

    fn emit_main_qingkong_window(
        &mut self,
        scene: &mut FrameScene,
        yaw_deg: f32,
        eye_center: [f32; 3],
        hands: &[HandControllerInput; 2],
        glyphs: &mut GlyphAtlas,
    ) {
        let win = if self.embedded_panel {
            Pose3D {
                position: [-0.242, 1.60, -1.24],
                orientation: [0.0, 0.0, 0.0, 1.0],
            }
        } else {
            arc_panel_pose(
                self.anchor_pos,
                self.anchor_yaw,
                yaw_deg,
                -0.01,
                self.workspace_radius,
                eye_center,
            )
        };
        let win_w = 1.28f32;
        let win_h = 0.76f32;

        if !self.embedded_panel {
            push_panel_shadow(
                &mut scene.quads,
                &win,
                [0.0, -0.010, -0.012],
                [win_w, win_h],
                0.034,
                0.036,
                0.075,
            );
        }
        push_panel_glass(
            &mut scene.quads,
            &win,
            [0.0, 0.0, 0.0],
            [win_w, win_h],
            0.034,
            if self.embedded_panel {
                qk_rgba(236, 244, 252, 0.84)
            } else {
                col_glass_window()
            },
            col_rim_luminous(),
        );

        // ====================================================================
        // Icon-First Minimalist Header Row (Zero text-button clutter!)
        // ====================================================================
        if self.embedded_panel {
            // In HyperSpace coexisting mode, Shell owns the bottom companion bar,
            // so place crisp `←` Back and `↑` Parent orbs right in the header!
            self.emit_icon_orb(
                scene,
                &win,
                [-0.590, 0.328, 0.006],
                0.032,
                UiIcon::BackLeft,
                HitAction::NavigateBack,
                false,
                false,
                Some("返回"),
                glyphs,
            );
            self.emit_icon_orb(
                scene,
                &win,
                [-0.550, 0.328, 0.006],
                0.032,
                UiIcon::UpArrow,
                HitAction::NavigateUp,
                false,
                false,
                Some("上层"),
                glyphs,
            );
            if self.page_count() > 1 {
                self.emit_icon_orb(
                    scene,
                    &win,
                    [0.240, 0.328, 0.006],
                    0.030,
                    UiIcon::ChevronLeft,
                    HitAction::PageDelta(-1),
                    false,
                    false,
                    None,
                    glyphs,
                );
                self.emit_icon_orb(
                    scene,
                    &win,
                    [0.276, 0.328, 0.006],
                    0.030,
                    UiIcon::ChevronRight,
                    HitAction::PageDelta(1),
                    false,
                    false,
                    None,
                    glyphs,
                );
            }
        }
        let header_folder_x = if self.embedded_panel { -0.506 } else { -0.578 };
        let header_title_x = if self.embedded_panel { -0.472 } else { -0.542 };
        push_textured_quad(
            &mut scene.quads,
            &win,
            [header_folder_x, 0.328, 0.005],
            [0.036, 0.036],
            0.006,
            2,
            0.0,
        );
        let dir_short = if self.current_location == TRASH_URI {
            "回收站".to_string()
        } else {
            Path::new(&self.current_location)
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "/".into())
        };
        push_panel_text(
            &mut scene.quads,
            &win,
            [header_title_x, 0.320, 0.006],
            &dir_short,
            0.021,
            0.19,
            col_ink_navy(),
            FontWeight::Bold,
            glyphs,
        );

        // Sleek Icon-First Search Capsule (`🔍` icon only when idle — zero placeholder prose!)
        let search_cx = -0.045f32;
        let search_w = if self.prompt_mode.is_some() || !self.search_query.is_empty() {
            0.40f32
        } else {
            0.22f32
        };
        let search_hit = self.register_hit(
            vec3_add(
                win.position,
                quat_rotate_vec3(win.orientation, [search_cx, 0.328, 0.006]),
            ),
            [search_w * 0.5, 0.020, 0.02],
            HitAction::StartPrompt(PromptMode::Search),
        );
        let search_hov = self.is_hit_hovered(search_hit);
        push_panel_glass(
            &mut scene.quads,
            &win,
            [search_cx, 0.328, 0.005],
            [search_w, 0.036],
            0.018,
            if self.prompt_mode.is_some() || search_hov {
                qk_rgba(255, 255, 255, 0.88)
            } else {
                qk_rgba(255, 255, 255, 0.46)
            },
            [1.0, 1.0, 1.0, 0.55],
        );

        let prompt_icon = match self.prompt_mode {
            Some(PromptMode::NewFolder) => UiIcon::NewFolder,
            Some(PromptMode::Rename) => UiIcon::EditRename,
            _ => UiIcon::Search,
        };
        push_ui_icon(
            &mut scene.quads,
            &win,
            [search_cx - search_w * 0.5 + 0.022, 0.328, 0.008],
            0.019,
            prompt_icon,
            if self.prompt_mode.is_some() || !self.search_query.is_empty() {
                col_kylin_blue()
            } else {
                col_muted_slate()
            },
        );

        if self.prompt_mode.is_some() {
            push_panel_text(
                &mut scene.quads,
                &win,
                [search_cx - search_w * 0.5 + 0.040, 0.321, 0.008],
                &format!("{}▌", self.prompt_buffer),
                0.0155,
                search_w - 0.055,
                col_ink_navy(),
                FontWeight::Bold,
                glyphs,
            );
            // Pure icon orbs for Backspace, Check, Cancel (zero words!)
            self.emit_icon_orb(
                scene,
                &win,
                [search_cx + search_w * 0.5 + 0.026, 0.328, 0.007],
                0.032,
                UiIcon::BackLeft,
                HitAction::PromptBackspace,
                false,
                false,
                None,
                glyphs,
            );
            self.emit_icon_orb(
                scene,
                &win,
                [search_cx + search_w * 0.5 + 0.064, 0.328, 0.007],
                0.032,
                UiIcon::CheckMark,
                HitAction::PromptCommit,
                true,
                true,
                None,
                glyphs,
            );
            self.emit_icon_orb(
                scene,
                &win,
                [search_cx + search_w * 0.5 + 0.102, 0.328, 0.007],
                0.032,
                UiIcon::CloseCross,
                HitAction::PromptCancel,
                false,
                false,
                None,
                glyphs,
            );
        } else if !self.search_query.is_empty() {
            push_panel_text(
                &mut scene.quads,
                &win,
                [search_cx - search_w * 0.5 + 0.040, 0.321, 0.008],
                &self.search_query,
                0.0155,
                search_w - 0.080,
                col_ink_navy(),
                FontWeight::Bold,
                glyphs,
            );
            self.emit_icon_orb(
                scene,
                &win,
                [search_cx + search_w * 0.5 - 0.020, 0.328, 0.008],
                0.026,
                UiIcon::CloseCross,
                HitAction::ClearSearch,
                false,
                false,
                None,
                glyphs,
            );
        }

        // Pure-Icon View Switcher Pill (`⊞` Gallery | `⬡` 3D Treemap) + Pure-Icon Action Strip!
        push_panel_glass(
            &mut scene.quads,
            &win,
            [0.346, 0.328, 0.005],
            [0.088, 0.036],
            0.018,
            qk_rgba(255, 255, 255, 0.44),
            [1.0, 1.0, 1.0, 0.50],
        );
        self.emit_icon_orb(
            scene,
            &win,
            [0.325, 0.328, 0.007],
            0.032,
            UiIcon::GridView,
            HitAction::ToggleView(ViewMode::Workbench),
            self.view_mode == ViewMode::Workbench,
            false,
            Some("画廊"),
            glyphs,
        );
        self.emit_icon_orb(
            scene,
            &win,
            [0.367, 0.328, 0.007],
            0.032,
            UiIcon::Cube3DView,
            HitAction::ToggleView(ViewMode::Treemap3D),
            self.view_mode == ViewMode::Treemap3D,
            false,
            Some("3D 星图"),
            glyphs,
        );

        self.emit_icon_orb(
            scene,
            &win,
            [0.432, 0.328, 0.006],
            0.034,
            UiIcon::NewFolder,
            HitAction::StartPrompt(PromptMode::NewFolder),
            false,
            false,
            Some("新建"),
            glyphs,
        );
        self.emit_icon_orb(
            scene,
            &win,
            [0.480, 0.328, 0.006],
            0.034,
            UiIcon::Sort,
            HitAction::CycleSort,
            false,
            false,
            Some(self.sort_by.label_zh()),
            glyphs,
        );
        if !self.undo_stack.is_empty() {
            self.emit_icon_orb(
                scene,
                &win,
                [0.528, 0.328, 0.006],
                0.034,
                UiIcon::UndoCurved,
                HitAction::UndoLast,
                true,
                false,
                Some("撤销"),
                glyphs,
            );
        } else {
            self.emit_icon_orb(
                scene,
                &win,
                [0.528, 0.328, 0.006],
                0.034,
                UiIcon::EyeHidden,
                HitAction::ToggleHidden,
                self.show_hidden,
                false,
                Some("隐藏项"),
                glyphs,
            );
        }
        self.emit_icon_orb(
            scene,
            &win,
            [0.576, 0.328, 0.006],
            0.034,
            UiIcon::MoreDots,
            HitAction::OpenContextMenuHere,
            false,
            false,
            Some("更多"),
            glyphs,
        );

        // ====================================================================
        // Left Spatial Icon Dock (`x = -0.575` — Pure Icons, Zero Text Column!)
        // ====================================================================
        let bookmarks = self.bookmarks.clone();
        for (i, bm) in bookmarks.iter().take(8).enumerate() {
            let ly = 0.228 - (i as f32) * 0.058;
            let active = self.current_location == bm.target;
            let ui_icon = UiIcon::from_bookmark_tag(&bm.icon_tag);
            self.emit_sidebar_icon_dock_slot(
                scene,
                &win,
                [-0.575, ly, 0.006],
                ui_icon,
                &bm.label,
                HitAction::NavigateLocation(bm.target.clone()),
                active,
                glyphs,
            );
        }

        // Storage Drive Icon + Sleek Capacity Gauge (Zero permanent text!)
        if let Some(vol) = self.volumes.first().cloned() {
            let used = vol.total_bytes.saturating_sub(vol.free_bytes);
            let ratio = (used as f32 / vol.total_bytes.max(1) as f32).clamp(0.10, 1.0);
            push_ui_icon(
                &mut scene.quads,
                &win,
                [-0.575, -0.275, 0.006],
                0.021,
                UiIcon::StorageDrive,
                col_muted_slate(),
            );
            push_panel_glass(
                &mut scene.quads,
                &win,
                [-0.575, -0.304, 0.005],
                [0.046, 0.0055],
                0.0028,
                qk_rgba(205, 218, 236, 0.65),
                [0.0; 4],
            );
            let fw = 0.046 * ratio;
            push_panel_glass(
                &mut scene.quads,
                &win,
                [-0.598 + 0.5 * fw, -0.304, 0.007],
                [fw, 0.0055],
                0.0028,
                col_kylin_blue(),
                [0.0; 4],
            );
        }

        // ====================================================================
        // Expanded Warm Visual Content Canvas (`x in -0.515..+0.615`, `well_w = 1.13m`)
        // ====================================================================
        let well_cx = 0.050f32;
        let well_cy = -0.025f32;
        let well_w = 1.13f32;
        let well_h = 0.65f32;
        push_panel_glass(
            &mut scene.quads,
            &win,
            [well_cx, well_cy, 0.002],
            [well_w, well_h],
            0.026,
            col_content_canvas(),
            [1.0, 1.0, 1.0, 0.42],
        );

        match self.view_mode {
            ViewMode::Workbench => {
                self.emit_visual_gallery_workbench(scene, &win, well_cx, hands, glyphs);
            }
            ViewMode::Treemap3D => {
                self.emit_qingkong_treemap_3d(scene, &win, well_cx, eye_center, glyphs);
            }
        }
    }

    /// Renders:
    /// 1. Unobstructed **16:9 Panoramic Hero Media Showcase Strip** (corner icon badges; text only on select/hover!)
    /// 2. Deduplicated **4 × 2 Visual File & Golden Folder Grid** (concise single-word stems, zero subtitle prose!)
    fn emit_visual_gallery_workbench(
        &mut self,
        scene: &mut FrameScene,
        win: &Pose3D,
        well_cx: f32,
        hands: &[HandControllerInput; 2],
        glyphs: &mut GlyphAtlas,
    ) {
        if self.filtered_indices.is_empty() {
            push_ui_icon(
                &mut scene.quads,
                win,
                [well_cx, 0.01, 0.012],
                0.055,
                UiIcon::Search,
                qk_rgba(120, 140, 170, 0.55),
            );
            return;
        }

        let hero_entries = self.hero_entry_indices();
        let grid_entries = self.grid_entry_indices(&hero_entries);
        let has_hero = !hero_entries.is_empty();

        if has_hero {
            let hero_w = 0.344f32;
            let hero_h = 0.186f32;
            let hero_y = if grid_entries.is_empty() {
                0.04f32
            } else {
                0.188f32
            };

            for (hi, &entry_idx) in hero_entries.iter().enumerate() {
                let entry = self.all_entries[entry_idx].clone();
                let lx = well_cx + (hi as f32 - (hero_entries.len() as f32 - 1.0) * 0.5) * 0.362;
                let slot_anim = 24 + hi;
                let z_lift = self.slot_z_cur.get(slot_anim).copied().unwrap_or(0.014);

                let world_c = vec3_add(
                    win.position,
                    quat_rotate_vec3(win.orientation, [lx, hero_y, 0.006 + z_lift]),
                );
                let hit_idx = self.register_hit(
                    world_c,
                    [hero_w * 0.5, hero_h * 0.5, 0.022],
                    HitAction::SelectEntry(entry_idx),
                );
                let hovered = self.is_hit_hovered(hit_idx);
                let pressed = hovered && (hands[0].trigger_pressed || hands[1].trigger_pressed);
                if slot_anim < self.slot_z_tgt.len() {
                    self.slot_z_tgt[slot_anim] = if pressed {
                        0.004
                    } else if hovered {
                        0.026
                    } else {
                        0.014
                    };
                }

                let selected = self.selected_entry_indices.contains(&entry_idx);
                let cz = 0.006 + z_lift;

                push_panel_shadow(
                    &mut scene.quads,
                    win,
                    [lx, hero_y - 0.005, 0.004],
                    [hero_w, hero_h],
                    0.022,
                    0.020,
                    if hovered { 0.11 } else { 0.06 },
                );

                // 100% Edge-to-Edge Unobstructed 16:9 Artwork!
                let tex_layer = self
                    .thumb_slot_by_path
                    .get(&entry.path)
                    .copied()
                    .unwrap_or_else(|| icon_slot_for_kind(entry.kind));
                push_textured_quad(
                    &mut scene.quads,
                    win,
                    [lx, hero_y, cz],
                    [hero_w, hero_h],
                    0.022,
                    2,
                    tex_layer as f32,
                );

                if selected || hovered {
                    push_panel_glass(
                        &mut scene.quads,
                        win,
                        [lx, hero_y, cz + 0.001],
                        [hero_w, hero_h],
                        0.022,
                        [1.0, 1.0, 1.0, 0.0],
                        if selected {
                            col_kylin_blue()
                        } else {
                            col_pure_white()
                        },
                    );
                }

                // Pure-Icon Type Badge Pill in Top-Right Corner of 16:9 Artwork!
                let bx = lx + hero_w * 0.5 - 0.026;
                let by = hero_y + hero_h * 0.5 - 0.026;
                push_panel_glass(
                    &mut scene.quads,
                    win,
                    [bx, by, cz + 0.003],
                    [0.030, 0.030],
                    0.015,
                    qk_rgba(250, 252, 255, 0.86),
                    [1.0, 1.0, 1.0, 0.65],
                );
                push_ui_icon(
                    &mut scene.quads,
                    win,
                    [bx, by, cz + 0.006],
                    0.016,
                    UiIcon::badge_for_kind(entry.kind),
                    col_ink_navy(),
                );

                // Show concise stem pill ONLY when selected or hovered — unselected Hero cards are 100% pure imagery!
                if selected || hovered {
                    let stem = short_display_stem(&entry);
                    let pill_w = (0.045 + stem.chars().count() as f32 * 0.017).clamp(0.10, hero_w - 0.03);
                    let px = lx - hero_w * 0.5 + 0.016 + pill_w * 0.5;
                    let py = hero_y - hero_h * 0.5 + 0.026;
                    push_panel_glass(
                        &mut scene.quads,
                        win,
                        [px, py, cz + 0.003],
                        [pill_w, 0.032],
                        0.016,
                        qk_rgba(248, 251, 255, 0.90),
                        [1.0, 1.0, 1.0, 0.65],
                    );
                    push_panel_text(
                        &mut scene.quads,
                        win,
                        [px - pill_w * 0.5 + 0.014, py - 0.0065, cz + 0.006],
                        &stem,
                        0.0145,
                        pill_w - 0.024,
                        col_ink_navy(),
                        FontWeight::Bold,
                        glyphs,
                    );
                }
            }
        }

        if grid_entries.is_empty() {
            return;
        }

        let rows_avail = if has_hero { 2usize } else { 3usize };
        let per_page = GRID_COLS * rows_avail;
        let start = (self.page * per_page).min(grid_entries.len());
        let end = (start + per_page).min(grid_entries.len());

        let card_w = 0.252f32;
        let card_h = 0.172f32;
        let top_row_y = if has_hero { -0.028f32 } else { 0.165f32 };

        for slot in 0..(end.saturating_sub(start)) {
            let entry_idx = grid_entries[start + slot];
            let entry = self.all_entries[entry_idx].clone();
            let col = slot % GRID_COLS;
            let row = slot / GRID_COLS;

            let lx = well_cx + (col as f32 - 1.5) * 0.270;
            let ly = top_row_y - (row as f32) * 0.190;
            let z_lift = self.slot_z_cur.get(slot).copied().unwrap_or(0.010);

            let world_c = vec3_add(
                win.position,
                quat_rotate_vec3(win.orientation, [lx, ly, 0.006 + z_lift]),
            );
            let hit_idx = self.register_hit(
                world_c,
                [card_w * 0.5, card_h * 0.5, 0.022],
                HitAction::SelectEntry(entry_idx),
            );
            let hovered = self.is_hit_hovered(hit_idx);
            let pressed = hovered && (hands[0].trigger_pressed || hands[1].trigger_pressed);
            if slot < self.slot_z_tgt.len() {
                self.slot_z_tgt[slot] = if pressed {
                    0.003
                } else if hovered {
                    0.022
                } else {
                    0.010
                };
            }

            let selected = self.selected_entry_indices.contains(&entry_idx);
            let cz = 0.006 + z_lift;

            push_panel_shadow(
                &mut scene.quads,
                win,
                [lx, ly - 0.004, 0.004],
                [card_w, card_h],
                0.020,
                0.016,
                if hovered { 0.085 } else { 0.04 },
            );

            let border = if selected {
                col_kylin_blue()
            } else if hovered {
                col_pure_white()
            } else {
                [1.0, 1.0, 1.0, 0.45]
            };

            if let Some(&visual_layer) = self.thumb_slot_by_path.get(&entry.path) {
                // Rich visual card: top 76% is edge-to-edge imagery + corner category icon badge!
                push_panel_glass(
                    &mut scene.quads,
                    win,
                    [lx, ly, cz],
                    [card_w, card_h],
                    0.020,
                    if selected {
                        col_card_selected()
                    } else {
                        col_card_surface()
                    },
                    border,
                );
                push_textured_quad(
                    &mut scene.quads,
                    win,
                    [lx, ly + 0.018, cz + 0.002],
                    [card_w - 0.012, card_h * 0.74],
                    0.016,
                    2,
                    visual_layer as f32,
                );
                let bx = lx + card_w * 0.5 - 0.022;
                let by = ly + card_h * 0.5 - 0.022;
                push_panel_glass(
                    &mut scene.quads,
                    win,
                    [bx, by, cz + 0.004],
                    [0.024, 0.024],
                    0.012,
                    qk_rgba(250, 252, 255, 0.86),
                    [0.0; 4],
                );
                push_ui_icon(
                    &mut scene.quads,
                    win,
                    [bx, by, cz + 0.006],
                    0.0135,
                    UiIcon::badge_for_kind(entry.kind),
                    col_ink_navy(),
                );
            } else {
                // Sculpted Golden Folder card: airy frosted glass + large golden folder with peeking inner pages!
                push_panel_glass(
                    &mut scene.quads,
                    win,
                    [lx, ly, cz],
                    [card_w, card_h],
                    0.020,
                    if selected {
                        col_card_selected()
                    } else if hovered {
                        col_card_hover()
                    } else {
                        qk_rgba(255, 255, 255, 0.62)
                    },
                    border,
                );
                let icon_slot = icon_slot_for_kind(entry.kind) as f32;
                push_textured_quad(
                    &mut scene.quads,
                    win,
                    [lx, ly + 0.018, cz + 0.003],
                    [0.106, 0.106],
                    0.004,
                    2,
                    icon_slot,
                );
            }

            // ONLY a single clean stem title (zero 2nd line of "空间目录" / "66.1 KB" text!)
            let stem = short_display_stem(&entry);
            push_panel_text(
                &mut scene.quads,
                win,
                [lx - card_w * 0.42, ly - 0.064, cz + 0.004],
                &stem,
                0.0155,
                card_w * 0.84,
                col_ink_navy(),
                FontWeight::Bold,
                glyphs,
            );
        }
    }

    fn emit_qingkong_treemap_3d(
        &mut self,
        scene: &mut FrameScene,
        win: &Pose3D,
        well_cx: f32,
        eye_center: [f32; 3],
        glyphs: &mut GlyphAtlas,
    ) {
        let podium_w = 0.96f32;
        let podium_d = 0.50f32;
        if self.treemap_dirty {
            let visible_entries: Vec<FileEntry> = self
                .filtered_indices
                .iter()
                .filter_map(|&i| self.all_entries.get(i).cloned())
                .collect();
            let mut blocks =
                treemap3d::build_spatial_treemap(&visible_entries, podium_w, podium_d, 36);
            for b in &mut blocks {
                b.entry_index = self.filtered_indices[b.entry_index];
            }
            self.cached_treemap = blocks;
            self.treemap_dirty = false;
        }

        let tray_center = vec3_add(
            win.position,
            quat_rotate_vec3(win.orientation, [well_cx, -0.055, 0.16]),
        );
        let podium_pose = Pose3D {
            position: tray_center,
            orientation: quat_mul(
                win.orientation,
                quat_from_yaw_pitch(0.0, 28.0f32.to_radians()),
            ),
        };

        push_oriented_box(
            &mut scene.solid_vertices,
            &podium_pose,
            [0.0, -0.012, 0.0],
            [podium_w * 0.52, 0.010, podium_d * 0.53],
            qk_rgba(235, 243, 252, 1.0),
            0.04,
        );

        let blocks = self.cached_treemap.clone();
        for (bi, block) in blocks.iter().enumerate() {
            let entry = self.all_entries[block.entry_index].clone();
            let lift = self.treemap_lift_cur.get(bi).copied().unwrap_or(0.0);
            let h_scaled = block.height * 0.82;
            let local_center = [block.x_center, 0.5 * h_scaled + lift, block.z_center];
            let world_center = vec3_add(
                podium_pose.position,
                quat_rotate_vec3(podium_pose.orientation, local_center),
            );
            let hit_idx = self.register_hit(
                world_center,
                [
                    (0.5 * block.width).max(0.022),
                    (0.5 * h_scaled + 0.01).max(0.022),
                    (0.5 * block.depth).max(0.022),
                ],
                HitAction::SelectEntry(block.entry_index),
            );
            let hovered = self.is_hit_hovered(hit_idx);
            let selected = self.selected_entry_indices.contains(&block.entry_index);
            if bi < self.treemap_lift_tgt.len() {
                self.treemap_lift_tgt[bi] = if hovered {
                    0.020
                } else if selected {
                    0.008
                } else {
                    0.0
                };
            }

            let accent = entry.kind.accent_rgba();
            push_oriented_box(
                &mut scene.solid_vertices,
                &podium_pose,
                local_center,
                [0.5 * block.width, 0.5 * h_scaled, 0.5 * block.depth],
                accent,
                if hovered { 0.32 } else { 0.05 },
            );

            if hovered || self.primary_selected == Some(block.entry_index) {
                let clamped_local = [
                    block.x_center.clamp(-0.24, 0.20),
                    0.5 * h_scaled + 0.11,
                    block.z_center,
                ];
                let raw_pos = vec3_add(
                    podium_pose.position,
                    quat_rotate_vec3(podium_pose.orientation, clamped_local),
                );
                let to_eye = vec3_normalize(vec3_sub(eye_center, raw_pos));
                let banner_pos = vec3_add(raw_pos, vec3_scale(to_eye, 0.08));
                let banner_pose = Pose3D {
                    position: banner_pos,
                    orientation: user_facing_orientation(banner_pos, eye_center),
                };
                push_panel_glass(
                    &mut scene.quads,
                    &banner_pose,
                    [0.0, 0.0, 0.0],
                    [0.32, 0.042],
                    0.021,
                    qk_rgba(255, 255, 255, 0.92),
                    [1.0, 1.0, 1.0, 0.85],
                );
                push_textured_quad(
                    &mut scene.quads,
                    &banner_pose,
                    [-0.132, 0.0, 0.005],
                    [0.026, 0.026],
                    0.006,
                    2,
                    icon_slot_for_kind(entry.kind) as f32,
                );
                let info = format!(
                    "{} · {:.0}%",
                    short_display_stem(&entry),
                    block.weight_ratio * 100.0
                );
                push_panel_text(
                    &mut scene.quads,
                    &banner_pose,
                    [-0.108, -0.0068, 0.006],
                    &info,
                    0.0155,
                    0.25,
                    col_ink_navy(),
                    FontWeight::Bold,
                    glyphs,
                );
            }
        }
    }

    /// Minimalist HyperSpace Bottom Companion Handle:
    /// Crisp vector SDF icon orbs (`←`, `↑`, `⌖`, `‹`, `›`) + slender HyperSpace Grab Capsule!
    fn emit_minimal_companion_bar(
        &mut self,
        scene: &mut FrameScene,
        yaw_deg: f32,
        eye_center: [f32; 3],
        glyphs: &mut GlyphAtlas,
    ) {
        let bar = arc_panel_pose(
            self.anchor_pos,
            self.anchor_yaw,
            yaw_deg,
            -0.435,
            self.workspace_radius - 0.01,
            eye_center,
        );

        self.emit_icon_orb(
            scene,
            &bar,
            [-0.155, 0.0, 0.0],
            0.034,
            UiIcon::BackLeft,
            HitAction::NavigateBack,
            false,
            true,
            None,
            glyphs,
        );
        self.emit_icon_orb(
            scene,
            &bar,
            [-0.112, 0.0, 0.0],
            0.034,
            UiIcon::UpArrow,
            HitAction::NavigateUp,
            false,
            true,
            None,
            glyphs,
        );

        let grab_hit = self.register_hit(
            bar.position,
            [0.095, 0.018, 0.02],
            HitAction::GrabWorkspaceBar,
        );
        let grab_hov = self.is_hit_hovered(grab_hit);
        push_panel_glass(
            &mut scene.quads,
            &bar,
            [0.0, 0.0, if grab_hov { 0.006 } else { 0.0 }],
            [0.175, 0.024],
            0.012,
            qk_rgba(248, 251, 255, if grab_hov { 0.94 } else { 0.72 }),
            col_rim_luminous(),
        );
        push_panel_glass(
            &mut scene.quads,
            &bar,
            [0.0, 0.0, if grab_hov { 0.009 } else { 0.003 }],
            [0.076, 0.005],
            0.0025,
            col_kylin_blue(),
            [0.0; 4],
        );

        if self.page_count() > 1 {
            self.emit_icon_orb(
                scene,
                &bar,
                [0.112, 0.0, 0.0],
                0.034,
                UiIcon::ChevronLeft,
                HitAction::PageDelta(-1),
                false,
                true,
                None,
                glyphs,
            );
            self.emit_icon_orb(
                scene,
                &bar,
                [0.155, 0.0, 0.0],
                0.034,
                UiIcon::ChevronRight,
                HitAction::PageDelta(1),
                false,
                true,
                None,
                glyphs,
            );
        } else {
            self.emit_icon_orb(
                scene,
                &bar,
                [0.120, 0.0, 0.0],
                0.034,
                UiIcon::Recenter,
                HitAction::RecenterWorkspace,
                false,
                true,
                None,
                glyphs,
            );
        }
    }

    /// Right Floating Visual Gallery Stage (`空间预览`):
    /// - Top 70%: Massive full-bleed artwork or rotating 3D stereo hologram
    /// - Middle: Colorful 2.5D category icon + short name + numeric chip (zero prose!)
    /// - Bottom: 4 Pure Circular Spatial Icon Orbs (`[▶/↗]` `[✎]` `[❐]` `[🗑]`, zero text buttons!)
    fn emit_right_gallery_stage(
        &mut self,
        scene: &mut FrameScene,
        yaw_deg: f32,
        eye_center: [f32; 3],
        glyphs: &mut GlyphAtlas,
    ) {
        let panel = if self.embedded_panel {
            Pose3D {
                position: [0.653, 1.60, -1.24],
                orientation: [0.0, 0.0, 0.0, 1.0],
            }
        } else {
            arc_panel_pose(
                self.anchor_pos,
                self.anchor_yaw,
                yaw_deg,
                -0.01,
                self.workspace_radius,
                eye_center,
            )
        };
        let pw = 0.45f32;
        let ph = 0.76f32;

        if !self.embedded_panel {
            push_panel_shadow(
                &mut scene.quads,
                &panel,
                [0.0, -0.010, -0.012],
                [pw, ph],
                0.032,
                0.034,
                0.075,
            );
        }
        push_panel_glass(
            &mut scene.quads,
            &panel,
            [0.0, 0.0, 0.0],
            [pw, ph],
            0.032,
            if self.embedded_panel {
                qk_rgba(236, 244, 252, 0.84)
            } else {
                col_glass_window()
            },
            col_rim_luminous(),
        );

        let Some(sel_idx) = self.primary_selected.filter(|&i| i < self.all_entries.len()) else {
            return;
        };
        let entry = self.all_entries[sel_idx].clone();

        // Massive Edge-to-Edge Visual Canvas (`0.414m × 0.500m`)
        let stage_y = 0.105f32;
        let stage_w = 0.414f32;
        let stage_h = 0.500f32;

        let has_inspect = self
            .previewer
            .active_inspect
            .as_ref()
            .is_some_and(|p| p.path == entry.path);

        let mut compact_stat = fs_ops::format_bytes(entry.size);
        if has_inspect {
            let inspect = self.previewer.active_inspect.as_ref().unwrap().clone();
            if !inspect.summary_note.is_empty() {
                compact_stat = format!("{} · {}", compact_stat, inspect.summary_note);
            }
            if let Some(ref mesh) = inspect.mesh {
                push_panel_glass(
                    &mut scene.quads,
                    &panel,
                    [0.0, stage_y, 0.004],
                    [stage_w, stage_h],
                    0.024,
                    qk_rgba(232, 243, 255, 0.78),
                    [1.0, 1.0, 1.0, 0.65],
                );
                let spin = self.start_instant.elapsed().as_secs_f32() * 0.85;
                let holo_world = vec3_add(
                    panel.position,
                    quat_rotate_vec3(panel.orientation, [0.0, stage_y, 0.110]),
                );
                let holo_rot = quat_mul(panel.orientation, quat_from_yaw_pitch(spin, 0.26));
                push_hologram_mesh(
                    &mut scene.solid_vertices,
                    holo_world,
                    holo_rot,
                    0.255,
                    &mesh.vertices,
                    qk_rgba(68, 136, 252, 1.0),
                );
            } else if inspect.image.is_some() {
                push_textured_quad(
                    &mut scene.quads,
                    &panel,
                    [0.0, stage_y, 0.006],
                    [stage_w, stage_h],
                    0.024,
                    3,
                    0.0,
                );
            } else if let Some(&card_tex) = self.thumb_slot_by_path.get(&entry.path) {
                push_textured_quad(
                    &mut scene.quads,
                    &panel,
                    [0.0, stage_y, 0.006],
                    [stage_w, stage_h],
                    0.024,
                    2,
                    card_tex as f32,
                );
            } else {
                push_panel_glass(
                    &mut scene.quads,
                    &panel,
                    [0.0, stage_y, 0.004],
                    [stage_w, stage_h],
                    0.024,
                    col_content_canvas(),
                    [1.0, 1.0, 1.0, 0.50],
                );
                push_textured_quad(
                    &mut scene.quads,
                    &panel,
                    [0.0, stage_y + 0.01, 0.008],
                    [0.165, 0.165],
                    0.008,
                    2,
                    icon_slot_for_kind(entry.kind) as f32,
                );
            }
        }

        // Single Iconic Title Row (`[Category Squircle Icon]  Name` + tiny numeric tag)
        push_textured_quad(
            &mut scene.quads,
            &panel,
            [-0.180, -0.192, 0.006],
            [0.036, 0.036],
            0.008,
            2,
            icon_slot_for_kind(entry.kind) as f32,
        );
        push_panel_text(
            &mut scene.quads,
            &panel,
            [-0.150, -0.193, 0.006],
            &short_display_stem(&entry),
            0.0185,
            0.24,
            col_ink_navy(),
            FontWeight::Bold,
            glyphs,
        );
        push_panel_text(
            &mut scene.quads,
            &panel,
            [-0.150, -0.216, 0.006],
            &compact_stat,
            0.0125,
            0.32,
            col_muted_slate(),
            FontWeight::Regular,
            glyphs,
        );

        // 4 Pure Circular Spatial Icon Orbs (`[▶/↗]` `[✎]` `[❐]` `[🗑]`, ZERO text buttons!)
        let is_trash = self.current_location == TRASH_URI;
        let is_video = entry.kind == FileKind::Video;
        let (cta_icon, cta_tip) = if is_trash {
            (UiIcon::UndoCurved, "还原")
        } else if is_video {
            (UiIcon::PlaySolid, "影院播放")
        } else {
            (UiIcon::OpenLaunch, "打开")
        };
        let cta_action = if is_trash {
            HitAction::InspectorTrashOrRestore
        } else {
            HitAction::InspectorOpen { cinema: is_video }
        };

        let dock_y = -0.296f32;
        if is_trash {
            self.emit_icon_orb(
                scene,
                &panel,
                [0.0, dock_y, 0.008],
                0.054,
                cta_icon,
                cta_action,
                true,
                true,
                Some(cta_tip),
                glyphs,
            );
        } else {
            self.emit_icon_orb(
                scene,
                &panel,
                [-0.138, dock_y, 0.008],
                0.052,
                cta_icon,
                cta_action,
                true,
                true,
                Some(cta_tip),
                glyphs,
            );
            self.emit_icon_orb(
                scene,
                &panel,
                [-0.046, dock_y, 0.008],
                0.048,
                UiIcon::EditRename,
                HitAction::StartPrompt(PromptMode::Rename),
                false,
                true,
                Some("重命名"),
                glyphs,
            );
            self.emit_icon_orb(
                scene,
                &panel,
                [0.046, dock_y, 0.008],
                0.048,
                UiIcon::CopyDup,
                HitAction::ContextAction(ContextMenuOp::Copy),
                false,
                true,
                Some("复制"),
                glyphs,
            );
            self.emit_icon_orb(
                scene,
                &panel,
                [0.138, dock_y, 0.008],
                0.048,
                UiIcon::Trash,
                HitAction::InspectorTrashOrRestore,
                false,
                true,
                Some("回收站"),
                glyphs,
            );
        }
    }

    /// Context Menu redesigned as a sleek **4 × 2 Floating Spatial Icon Palette** (zero text wall!).
    fn emit_context_menu(
        &mut self,
        scene: &mut FrameScene,
        eye_center: [f32; 3],
        glyphs: &mut GlyphAtlas,
    ) {
        let Some(menu) = self.context_menu.clone() else {
            return;
        };
        let to_eye = vec3_normalize(vec3_sub(eye_center, menu.anchor_world));
        let pos = vec3_add(menu.anchor_world, vec3_scale(to_eye, 0.09));
        let pose = Pose3D {
            position: pos,
            orientation: user_facing_orientation(pos, eye_center),
        };

        let items: &[(UiIcon, &str, ContextMenuOp)] = if self.current_location == TRASH_URI {
            &[
                (UiIcon::UndoCurved, "还原", ContextMenuOp::TrashOrRestore),
                (UiIcon::CloseCross, "关闭", ContextMenuOp::Close),
            ]
        } else {
            &[
                (UiIcon::OpenLaunch, "打开", ContextMenuOp::Open),
                (UiIcon::PlaySolid, "影院", ContextMenuOp::OpenCinema),
                (UiIcon::NewFolder, "新建", ContextMenuOp::NewFolder),
                (UiIcon::CopyDup, "复制", ContextMenuOp::Copy),
                (UiIcon::CutScissors, "剪切", ContextMenuOp::Cut),
                (UiIcon::PasteClip, "粘贴", ContextMenuOp::Paste),
                (UiIcon::EditRename, "重命名", ContextMenuOp::Rename),
                (UiIcon::Trash, "回收站", ContextMenuOp::TrashOrRestore),
            ]
        };

        let cols = items.len().min(4);
        let rows = (items.len() + 3) / 4;
        let pw = 0.036 + (cols as f32) * 0.056;
        let ph = 0.036 + (rows as f32) * 0.056;

        push_panel_shadow(
            &mut scene.quads,
            &pose,
            [0.0, -0.008, -0.008],
            [pw, ph],
            0.026,
            0.024,
            0.10,
        );
        push_panel_glass(
            &mut scene.quads,
            &pose,
            [0.0, 0.0, 0.0],
            [pw, ph],
            0.026,
            qk_rgba(244, 248, 254, 0.92),
            col_rim_luminous(),
        );

        for (i, (icon, tip, op)) in items.iter().enumerate() {
            let c = i % 4;
            let r = i / 4;
            let lx = (c as f32 - (cols as f32 - 1.0) * 0.5) * 0.056;
            let ly = ((rows as f32 - 1.0) * 0.5 - r as f32) * 0.056;
            self.emit_icon_orb(
                scene,
                &pose,
                [lx, ly, 0.006],
                0.044,
                *icon,
                HitAction::ContextAction(*op),
                i == 0,
                true,
                Some(tip),
                glyphs,
            );
        }
    }

    /// Left Dock Icon Slot: renders ONLY a crisp SDF vector icon (`#3790FA` Kylin-Blue orb when active);
    /// pops out a floating frosted tooltip pill ONLY when pointed at!
    #[allow(clippy::too_many_arguments)]
    fn emit_sidebar_icon_dock_slot(
        &mut self,
        scene: &mut FrameScene,
        panel: &Pose3D,
        local_offset: [f32; 3],
        icon: UiIcon,
        tooltip: &str,
        action: HitAction,
        active: bool,
        glyphs: &mut GlyphAtlas,
    ) {
        let world_center = vec3_add(
            panel.position,
            quat_rotate_vec3(panel.orientation, local_offset),
        );
        let hit_idx = self.register_hit(world_center, [0.026, 0.026, 0.02], action);
        let hovered = self.is_hit_hovered(hit_idx);
        let z = local_offset[2] + if hovered { 0.006 } else { 0.0 };

        if active || hovered {
            push_panel_glass(
                &mut scene.quads,
                panel,
                [local_offset[0], local_offset[1], z],
                [0.046, 0.046],
                0.023,
                if active {
                    col_kylin_blue()
                } else {
                    col_ghost_hover()
                },
                [1.0, 1.0, 1.0, if active { 0.45 } else { 0.65 }],
            );
        }

        push_ui_icon(
            &mut scene.quads,
            panel,
            [local_offset[0], local_offset[1], z + 0.004],
            0.024,
            icon,
            if active {
                col_pure_white()
            } else {
                col_ink_navy()
            },
        );

        if hovered {
            let tw = (0.036 + tooltip.chars().count() as f32 * 0.017).clamp(0.072, 0.16);
            let tx = local_offset[0] + 0.036 + tw * 0.5;
            push_panel_glass(
                &mut scene.quads,
                panel,
                [tx, local_offset[1], z + 0.018],
                [tw, 0.030],
                0.015,
                qk_rgba(250, 252, 255, 0.94),
                col_rim_luminous(),
            );
            push_panel_text(
                &mut scene.quads,
                panel,
                [tx - tw * 0.5 + 0.014, local_offset[1] - 0.0062, z + 0.022],
                tooltip,
                0.014,
                tw - 0.02,
                col_ink_navy(),
                FontWeight::Bold,
                glyphs,
            );
        }
    }

    /// Circular Spatial Icon Orb (`diameter_m`):
    /// - Zero permanent text!
    /// - Optional `show_orb_idle` frosted glass circle (like UKUI `control-library.png` 04)
    /// - Optional floating hover tooltip pill right above the orb when pointed at!
    #[allow(clippy::too_many_arguments)]
    fn emit_icon_orb(
        &mut self,
        scene: &mut FrameScene,
        panel: &Pose3D,
        local_offset: [f32; 3],
        diameter_m: f32,
        icon: UiIcon,
        action: HitAction,
        active_primary: bool,
        show_orb_idle: bool,
        hover_tooltip: Option<&str>,
        glyphs: &mut GlyphAtlas,
    ) {
        let world_center = vec3_add(
            panel.position,
            quat_rotate_vec3(panel.orientation, local_offset),
        );
        let r = diameter_m * 0.5;
        let hit_idx = self.register_hit(world_center, [r.max(0.020), r.max(0.020), 0.022], action);
        let hovered = self.is_hit_hovered(hit_idx);
        let z = local_offset[2] + if hovered { 0.006 } else { 0.0 };

        if active_primary || hovered || show_orb_idle {
            let fill = if active_primary {
                col_kylin_blue()
            } else if hovered {
                col_card_hover()
            } else {
                col_orb_idle()
            };
            push_panel_glass(
                &mut scene.quads,
                panel,
                [local_offset[0], local_offset[1], z],
                [diameter_m, diameter_m],
                r,
                fill,
                [1.0, 1.0, 1.0, if active_primary { 0.45 } else { 0.65 }],
            );
        }

        push_ui_icon(
            &mut scene.quads,
            panel,
            [local_offset[0], local_offset[1], z + 0.004],
            diameter_m * 0.52,
            icon,
            if active_primary {
                col_pure_white()
            } else {
                col_ink_navy()
            },
        );

        if hovered {
            if let Some(tip) = hover_tooltip {
                let tw = (0.034 + tip.chars().count() as f32 * 0.0165).clamp(0.068, 0.16);
                let ty = local_offset[1] + r + 0.022;
                push_panel_glass(
                    &mut scene.quads,
                    panel,
                    [local_offset[0], ty, z + 0.016],
                    [tw, 0.028],
                    0.014,
                    qk_rgba(250, 252, 255, 0.94),
                    col_rim_luminous(),
                );
                push_panel_text(
                    &mut scene.quads,
                    panel,
                    [local_offset[0] - tw * 0.5 + 0.013, ty - 0.006, z + 0.020],
                    tip,
                    0.0135,
                    tw - 0.02,
                    col_ink_navy(),
                    FontWeight::Bold,
                    glyphs,
                );
            }
        }
    }

    fn update_hover_and_emit_hands(
        &mut self,
        scene: &mut FrameScene,
        eye_center: [f32; 3],
        hands: &[HandControllerInput; 2],
        glyphs: &mut GlyphAtlas,
    ) {
        for h in 0..2 {
            let ctrl = hands[h];
            if !ctrl.active {
                self.hovered_hit[h] = None;
                continue;
            }
            let pick = self.hit_batch.pick_ray(ctrl.aim_origin, ctrl.aim_dir, 0.06, 8.0);
            let new_hover = pick.map(|p| p.index);
            if new_hover.is_some() && new_hover != self.hovered_hit[h] {
                if self.pending_haptics[h].is_none() {
                    self.pending_haptics[h] = Some(HapticEvent::HoverTick);
                }
            }
            self.hovered_hit[h] = new_hover;

            let hit_dist = pick.map(|p| p.distance).unwrap_or(1.85);
            let ray_start = vec3_add(ctrl.aim_origin, vec3_scale(ctrl.aim_dir, 0.14));
            let ray_end = vec3_add(ctrl.aim_origin, vec3_scale(ctrl.aim_dir, hit_dist));
            if !self.embedded_panel {
                push_beam_segment(
                    &mut scene.solid_vertices,
                    ray_start,
                    ray_end,
                    0.0014,
                    0.0006,
                    qk_rgba(255, 255, 255, 0.72),
                    0.35,
                );
            }

            if let Some(p) = pick {
                let normal = vec3_normalize(vec3_sub(eye_center, p.hit_point));
                let reticle_pos = vec3_add(p.hit_point, vec3_scale(normal, 0.004));
                let reticle_pose = Pose3D {
                    position: reticle_pos,
                    orientation: user_facing_orientation(reticle_pos, eye_center),
                };
                let r_size = if ctrl.trigger_pressed { 0.014 } else { 0.018 };
                push_panel_glass(
                    &mut scene.quads,
                    &reticle_pose,
                    [0.0, 0.0, 0.0],
                    [r_size, r_size],
                    r_size * 0.5,
                    col_pure_white(),
                    col_kylin_blue(),
                );
            }

            if let Some(ref drag) = self.drag_session {
                if drag.hand == h && drag.active_drag {
                    let ghost_pos = vec3_add(ray_end, [0.0, 0.045, 0.0]);
                    let ghost_pose = Pose3D {
                        position: ghost_pos,
                        orientation: user_facing_orientation(ghost_pos, eye_center),
                    };
                    push_panel_glass(
                        &mut scene.quads,
                        &ghost_pose,
                        [0.0, 0.0, 0.02],
                        [0.18, 0.044],
                        0.022,
                        col_kylin_blue(),
                        col_pure_white(),
                    );
                    push_ui_icon(
                        &mut scene.quads,
                        &ghost_pose,
                        [-0.064, 0.0, 0.025],
                        0.020,
                        UiIcon::OpenLaunch,
                        col_pure_white(),
                    );
                    push_panel_text(
                        &mut scene.quads,
                        &ghost_pose,
                        [-0.044, -0.0068, 0.025],
                        &drag.primary_label,
                        0.015,
                        0.12,
                        col_pure_white(),
                        FontWeight::Bold,
                        glyphs,
                    );
                }
            }
        }
    }
}

/// Strips verbose file extensions and underscores so card labels stay short, scannable, and effortless!
fn short_display_stem(entry: &FileEntry) -> String {
    if entry.kind == FileKind::Directory {
        return entry.name.clone();
    }
    let stem = Path::new(&entry.name)
        .file_stem()
        .and_then(|s| s.to_str())
        .filter(|s| !s.is_empty())
        .unwrap_or(&entry.name);
    stem.replace('_', " ")
}

fn push_panel_shadow(
    quads: &mut Vec<QuadInstance>,
    panel: &Pose3D,
    local_offset: [f32; 3],
    panel_size_m: [f32; 2],
    radius_m: f32,
    spread_m: f32,
    alpha: f32,
) {
    let total_w = panel_size_m[0] + spread_m * 2.0;
    let total_h = panel_size_m[1] + spread_m * 2.0;
    let center = vec3_add(
        panel.position,
        quat_rotate_vec3(panel.orientation, local_offset),
    );
    let right_vec = quat_rotate_vec3(panel.orientation, [0.5 * total_w, 0.0, 0.0]);
    let up_vec = quat_rotate_vec3(panel.orientation, [0.0, 0.5 * total_h, 0.0]);
    quads.push(QuadInstance {
        center,
        right_vec,
        up_vec,
        uv_rect: [0.0, 0.0, 0.002, 0.002],
        fill_color: qk_rgba(28, 48, 84, alpha),
        border_color: [0.0; 4],
        params: [radius_m, spread_m, 4.0, 0.0],
    });
}

fn push_panel_glass(
    quads: &mut Vec<QuadInstance>,
    panel: &Pose3D,
    local_offset: [f32; 3],
    size_m: [f32; 2],
    radius_m: f32,
    fill_color: [f32; 4],
    border_color: [f32; 4],
) {
    let center = vec3_add(
        panel.position,
        quat_rotate_vec3(panel.orientation, local_offset),
    );
    let right_vec = quat_rotate_vec3(panel.orientation, [0.5 * size_m[0], 0.0, 0.0]);
    let up_vec = quat_rotate_vec3(panel.orientation, [0.0, 0.5 * size_m[1], 0.0]);
    quads.push(QuadInstance {
        center,
        right_vec,
        up_vec,
        uv_rect: [0.0, 0.0, 0.002, 0.002],
        fill_color,
        border_color,
        params: [radius_m, 0.0014, 0.0, 0.0],
    });
}

fn push_textured_quad(
    quads: &mut Vec<QuadInstance>,
    panel: &Pose3D,
    local_offset: [f32; 3],
    size_m: [f32; 2],
    radius_m: f32,
    tex_kind: i32,
    tex_layer: f32,
) {
    let center = vec3_add(
        panel.position,
        quat_rotate_vec3(panel.orientation, local_offset),
    );
    let right_vec = quat_rotate_vec3(panel.orientation, [0.5 * size_m[0], 0.0, 0.0]);
    let up_vec = quat_rotate_vec3(panel.orientation, [0.0, 0.5 * size_m[1], 0.0]);
    quads.push(QuadInstance {
        center,
        right_vec,
        up_vec,
        uv_rect: [0.0, 0.0, 1.0, 1.0],
        fill_color: [1.0, 1.0, 1.0, 1.0],
        border_color: [1.0, 1.0, 1.0, 0.0],
        params: [radius_m, 0.0, tex_kind as f32, tex_layer],
    });
}

fn push_ui_icon(
    quads: &mut Vec<QuadInstance>,
    panel: &Pose3D,
    local_offset: [f32; 3],
    edge_m: f32,
    icon: UiIcon,
    tint_color: [f32; 4],
) {
    let center = vec3_add(
        panel.position,
        quat_rotate_vec3(panel.orientation, local_offset),
    );
    let right_vec = quat_rotate_vec3(panel.orientation, [0.5 * edge_m, 0.0, 0.0]);
    let up_vec = quat_rotate_vec3(panel.orientation, [0.0, 0.5 * edge_m, 0.0]);
    quads.push(QuadInstance {
        center,
        right_vec,
        up_vec,
        uv_rect: [0.0, 0.0, 1.0, 1.0],
        fill_color: tint_color,
        border_color: [0.0; 4],
        params: [0.002, 0.0, 5.0, icon.slot()],
    });
}

#[allow(clippy::too_many_arguments)]
fn push_panel_text(
    quads: &mut Vec<QuadInstance>,
    panel: &Pose3D,
    local_origin: [f32; 3],
    text: &str,
    height_m: f32,
    max_width_m: f32,
    color: [f32; 4],
    weight: FontWeight,
    glyphs: &mut GlyphAtlas,
) {
    let raster_px = (height_m * 2400.0).clamp(24.0, 68.0);
    let m_per_px = height_m / raster_px;
    let max_px = (max_width_m / m_per_px).max(16.0);
    let (laid_out, _) = glyphs.layout_line_weighted(text, raster_px, max_px, weight);

    let right_unit = quat_rotate_vec3(panel.orientation, [1.0, 0.0, 0.0]);
    let up_unit = quat_rotate_vec3(panel.orientation, [0.0, 1.0, 0.0]);
    let base_world = vec3_add(
        panel.position,
        quat_rotate_vec3(panel.orientation, local_origin),
    );
    let top_y_m = height_m * 0.74;

    for (g, gx_px, gy_px) in laid_out {
        let gw_m = g.size_px[0] * m_per_px;
        let gh_m = g.size_px[1] * m_per_px;
        let cx_m = gx_px * m_per_px + 0.5 * gw_m;
        let cy_m = top_y_m - (gy_px * m_per_px + 0.5 * gh_m);

        let center = vec3_add(
            base_world,
            vec3_add(vec3_scale(right_unit, cx_m), vec3_scale(up_unit, cy_m)),
        );
        quads.push(QuadInstance {
            center,
            right_vec: vec3_scale(right_unit, 0.5 * gw_m),
            up_vec: vec3_scale(up_unit, 0.5 * gh_m),
            uv_rect: [g.uv_min[0], g.uv_min[1], g.uv_max[0], g.uv_max[1]],
            fill_color: color,
            border_color: [0.0; 4],
            params: [0.0, 0.0, 1.0, 0.0],
        });
    }
}

fn push_oriented_box(
    verts: &mut Vec<SolidVertex>,
    pose: &Pose3D,
    local_center: [f32; 3],
    half_ext: [f32; 3],
    color: [f32; 4],
    emissive: f32,
) {
    let faces: &[([f32; 3], [[f32; 3]; 4])] = &[
        (
            [0.0, 0.0, 1.0],
            [[-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 1.0, 1.0]],
        ),
        (
            [0.0, 0.0, -1.0],
            [[1.0, -1.0, -1.0], [-1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [1.0, 1.0, -1.0]],
        ),
        (
            [0.0, 1.0, 0.0],
            [[-1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, -1.0], [-1.0, 1.0, -1.0]],
        ),
        (
            [0.0, -1.0, 0.0],
            [[-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [1.0, -1.0, 1.0], [-1.0, -1.0, 1.0]],
        ),
        (
            [1.0, 0.0, 0.0],
            [[1.0, -1.0, 1.0], [1.0, -1.0, -1.0], [1.0, 1.0, -1.0], [1.0, 1.0, 1.0]],
        ),
        (
            [-1.0, 0.0, 0.0],
            [[-1.0, -1.0, -1.0], [-1.0, -1.0, 1.0], [-1.0, 1.0, 1.0], [-1.0, 1.0, -1.0]],
        ),
    ];

    for (ln, corners) in faces {
        let wn = quat_rotate_vec3(pose.orientation, *ln);
        let wp = corners.map(|c| {
            let lp = [
                local_center[0] + c[0] * half_ext[0],
                local_center[1] + c[1] * half_ext[1],
                local_center[2] + c[2] * half_ext[2],
            ];
            vec3_add(pose.position, quat_rotate_vec3(pose.orientation, lp))
        });
        for idx in [0usize, 1, 2, 0, 2, 3] {
            verts.push(SolidVertex {
                pos: wp[idx],
                normal: wn,
                color,
                emissive,
            });
        }
    }
}

fn push_beam_segment(
    verts: &mut Vec<SolidVertex>,
    start: [f32; 3],
    end: [f32; 3],
    r_start: f32,
    r_end: f32,
    color: [f32; 4],
    emissive: f32,
) {
    let delta = vec3_sub(end, start);
    let len = vec3_len(delta);
    if len < 1e-4 {
        return;
    }
    let dir = vec3_scale(delta, 1.0 / len);
    let up_ref = if dir[1].abs() < 0.92 {
        [0.0, 1.0, 0.0]
    } else {
        [1.0, 0.0, 0.0]
    };
    let side = vec3_normalize(crate::spatial::vec3_cross(dir, up_ref));
    let perp = vec3_normalize(crate::spatial::vec3_cross(side, dir));

    for axis in [side, perp] {
        let s0 = vec3_add(start, vec3_scale(axis, -r_start));
        let s1 = vec3_add(start, vec3_scale(axis, r_start));
        let e1 = vec3_add(end, vec3_scale(axis, r_end));
        let e0 = vec3_add(end, vec3_scale(axis, -r_end));
        for p in [s0, s1, e1, s0, e1, e0] {
            verts.push(SolidVertex {
                pos: p,
                normal: perp,
                color,
                emissive,
            });
        }
    }
}

fn push_hologram_mesh(
    verts: &mut Vec<SolidVertex>,
    world_center: [f32; 3],
    rotation: [f32; 4],
    scale: f32,
    mesh: &[MeshVertex],
    color: [f32; 4],
) {
    for mv in mesh {
        let p = vec3_add(
            world_center,
            quat_rotate_vec3(rotation, vec3_scale(mv.position, scale)),
        );
        let n = quat_rotate_vec3(rotation, mv.normal);
        verts.push(SolidVertex {
            pos: p,
            normal: n,
            color,
            emissive: 0.16,
        });
    }
}
