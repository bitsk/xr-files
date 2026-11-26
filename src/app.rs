// SPDX-License-Identifier: GPL-3.0-or-later
//! Spatial File Manager Application State Machine & UKUI XR "Qingkong" (`ukui.spatial.qingkong.light`)
//! Spatial Window + Floating Hologram Inspector + Companion Grab Bar Scene Builder.

use crate::{
    fs_ops::{
        self, Bookmark, ClipboardOp, FileClipboard, FileEntry, FileKind, SortBy, StorageVolume,
        UndoRecord, TRASH_URI,
    },
    icons::{icon_slot_for_kind, BUILTIN_ICON_COUNT},
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

pub const CARDS_PER_PAGE: usize = 12; // 4 cols × 3 rows
const GRID_COLS: usize = 4;
const HIT_SLOP_RAD: f32 = 0.0384; // 2.2° minimum angular target rule

// ============================================================================
// Exact UKUI XR "Qingkong" (`ukui.spatial.qingkong.light`) Colorimetric Tokens
// ============================================================================

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

// Official tokens from `xr-spatial-kit/styles/ukui.json` & `style-reference.png`:
fn col_glass_panel() -> [f32; 4] {
    qk_rgba(232, 240, 248, 0.76) // #E8F0F8 Frosted Silver Mist Glass
}
fn col_content_well() -> [f32; 4] {
    qk_rgba(245, 247, 250, 0.88) // #F5F7FA Warm Clear Content Surface
}
fn col_card_idle() -> [f32; 4] {
    qk_rgba(255, 255, 255, 0.90) // Crisp Floating File Card
}
fn col_card_hover() -> [f32; 4] {
    qk_rgba(255, 255, 255, 0.98)
}
fn col_card_selected() -> [f32; 4] {
    qk_rgba(226, 240, 255, 0.96) // Soft Kylin Azure Tint
}
fn col_pill_idle() -> [f32; 4] {
    qk_rgba(245, 247, 250, 0.86) // #F5F7FA Button / Pill
}
fn col_pill_hover() -> [f32; 4] {
    qk_rgba(255, 255, 255, 0.98)
}
fn col_kylin_blue() -> [f32; 4] {
    qk_rgba(55, 144, 250, 0.98) // #3790FA Kylin Focus Blue
}
fn col_kylin_blue_hover() -> [f32; 4] {
    qk_rgba(78, 158, 255, 1.0)
}
fn col_ink_navy() -> [f32; 4] {
    qk_rgba(35, 48, 71, 1.0) // #233047 Primary Ink Navy Text
}
fn col_muted_slate() -> [f32; 4] {
    qk_rgba(87, 108, 139, 1.0) // #576C8B Secondary Muted Caption
}
fn col_pure_white() -> [f32; 4] {
    [1.0, 1.0, 1.0, 1.0]
}
fn col_rim_light() -> [f32; 4] {
    [1.0, 1.0, 1.0, 0.72] // Luminous Specular Glass Edge
}
fn col_rim_subtle() -> [f32; 4] {
    [1.0, 1.0, 1.0, 0.50]
}
fn col_danger_soft() -> [f32; 4] {
    qk_rgba(252, 236, 238, 0.92)
}
fn col_danger_text() -> [f32; 4] {
    qk_rgba(218, 58, 74, 1.0)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ViewMode {
    Workbench,
    #[serde(rename = "treemap3d", alias = "treemap3_d")]
    Treemap3D,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CategoryFilter {
    All,
    Folders,
    Media,
    DocsCode,
    Models3D,
}

impl CategoryFilter {
    pub fn matches(self, kind: FileKind) -> bool {
        match self {
            Self::All => true,
            Self::Folders => kind == FileKind::Directory,
            Self::Media => matches!(kind, FileKind::Image | FileKind::Video | FileKind::Audio),
            Self::DocsCode => matches!(kind, FileKind::Code | FileKind::Document),
            Self::Models3D => kind == FileKind::Model3D,
        }
    }

    pub fn label_zh(self) -> &'static str {
        match self {
            Self::All => "全部",
            Self::Folders => "文件夹",
            Self::Media => "媒体",
            Self::DocsCode => "文档代码",
            Self::Models3D => "3D 资产",
        }
    }
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
    /// `[corner_radius_m, border_or_spread_m, tex_kind (0=glass, 1=glyph, 2=thumb/icon, 3=inspect, 4=shadow), tex_layer]`
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
    ToggleSortOrder,
    ToggleHidden,
    SetFilter(CategoryFilter),
    PageDelta(isize),
    SelectEntry(usize),
    NavigateUp,
    NavigateBack,
    NavigateForward,
    StartPrompt(PromptMode),
    ClearSearch,
    PromptAppend(&'static str),
    PromptBackspace,
    PromptCommit,
    PromptCancel,
    InspectorOpen { cinema: bool },
    InspectorCopy,
    InspectorCut,
    PasteHere,
    InspectorTrashOrRestore,
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
    entry_index: Option<usize>,
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
    pub category_filter: CategoryFilter,
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
            category_filter: CategoryFilter::All,
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
            workspace_radius: 1.26,
            card_scale: 1.0,
            anchor_initialized: false,
            recenter_requested: false,
            shutdown_flag: false,
            submitted_frames: 0,
            slot_z_cur: vec![0.008; 64],
            slot_z_tgt: vec![0.008; 64],
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
                if let Some(&first_idx) = self.filtered_indices.first() {
                    self.select_single(first_idx);
                }
                self.status_message = format!("共 {} 项", self.filtered_indices.len());
            }
            Err(err) => {
                self.status_message = format!("读取失败: {err}");
            }
        }
    }

    pub fn recompute_filtered(&mut self) {
        let candidates: Vec<(usize, &str)> = self
            .all_entries
            .iter()
            .enumerate()
            .filter(|(_, e)| self.category_filter.matches(e.kind))
            .map(|(i, e)| (i, e.name.as_str()))
            .collect();

        if self.search_query.trim().is_empty() {
            self.filtered_indices = candidates.into_iter().map(|(i, _)| i).collect();
        } else {
            let names: Vec<&str> = candidates.iter().map(|(_, n)| *n).collect();
            let ranked = simd::filter_and_rank_names(&names, &self.search_query);
            self.filtered_indices = ranked.into_iter().map(|(pos, _)| candidates[pos].0).collect();
        }
        let max_page = self.page_count().saturating_sub(1);
        if self.page > max_page {
            self.page = max_page;
        }
        self.treemap_dirty = true;
    }

    pub fn page_count(&self) -> usize {
        if self.filtered_indices.is_empty() {
            1
        } else {
            (self.filtered_indices.len() + CARDS_PER_PAGE - 1) / CARDS_PER_PAGE
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
            if let Some(pos) = self.filtered_indices.iter().position(|&i| i == idx) {
                self.page = pos / CARDS_PER_PAGE;
            }
        }
    }

    pub fn step_page(&mut self, delta: isize) {
        let total = self.page_count() as isize;
        self.page = (self.page as isize + delta).clamp(0, (total - 1).max(0)) as usize;
    }

    // ========================================================================
    // Per-Frame Simulation, SIMD Hit-Testing & UKUI Qingkong Scene Builder
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

        // Poll async thumbnails into slots `BUILTIN_ICON_COUNT..THUMB_LAYERS`
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

        let page_start = self.page * CARDS_PER_PAGE;
        let page_end = (page_start + CARDS_PER_PAGE).min(self.filtered_indices.len());
        for pos in page_start..page_end {
            let entry_idx = self.filtered_indices[pos];
            self.previewer.request_thumbnail(&self.all_entries[entry_idx]);
        }

        self.process_controller_inputs(hands, now, dt);

        // SIMD spring smoothing for spatial Z-lift (`+8mm` rest, `+18mm` hover, `+2mm` press)
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

        // Slightly shift main spatial window left (`+6.5°` yaw) when Inspector is visible on right (`-26.5°`)
        // so the full ensemble sits symmetrically balanced in the user's comfortable horizon FOV!
        let main_yaw_deg = 9.2;
        let insp_yaw_deg = -30.5;

        self.emit_depth_breadcrumbs(&mut scene, main_yaw_deg, eye_center, glyphs);
        self.emit_main_qingkong_window(&mut scene, main_yaw_deg, eye_center, hands, glyphs);
        self.emit_companion_window_bar(&mut scene, main_yaw_deg, eye_center, glyphs);
        self.emit_right_inspector_card(&mut scene, insp_yaw_deg, eye_center, glyphs);
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
                self.workspace_radius = 1.26;
                self.card_scale = 1.0;
                self.request_recenter();
            }

            if ctrl.secondary_just_pressed {
                let entry_idx = pick
                    .and_then(|p| self.hit_actions.get(p.index))
                    .and_then(|act| match act {
                        HitAction::SelectEntry(idx) => Some(*idx),
                        _ => None,
                    });
                if let Some(idx) = entry_idx {
                    self.select_single(idx);
                }
                let anchor_world = pick.map(|p| p.hit_point).unwrap_or_else(|| {
                    vec3_add(ctrl.aim_origin, vec3_scale(ctrl.aim_dir, 1.38))
                });
                self.context_menu = Some(ContextMenuState {
                    anchor_world,
                    entry_index: entry_idx.or(self.primary_selected),
                });
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
                                primary_label: if paths.len() == 1 {
                                    self.all_entries[idx].name.clone()
                                } else {
                                    format!("{} ({} 项)", self.all_entries[idx].name, paths.len())
                                },
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
                            (self.workspace_radius - ctrl.stick[1] * 1.15 * dt).clamp(1.05, 3.00);
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
                            vec3_add(ctrl.aim_origin, vec3_scale(ctrl.aim_dir, 1.40))
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
                            primary_label: self.all_entries[entry_idx].name.clone(),
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
            HitAction::ToggleSortOrder => {
                self.ascending = !self.ascending;
                self.reload_directory();
            }
            HitAction::ToggleHidden => {
                self.show_hidden = !self.show_hidden;
                self.reload_directory();
            }
            HitAction::SetFilter(f) => {
                self.category_filter = f;
                self.page = 0;
                self.recompute_filtered();
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
            HitAction::NavigateForward => {
                if let Some(next) = self.forward_stack.pop() {
                    self.back_stack.push(self.current_location.clone());
                    self.current_location = next;
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
            HitAction::PromptAppend(s) => self.handle_key_text(s),
            HitAction::PromptBackspace => self.handle_backspace(),
            HitAction::PromptCommit => self.commit_prompt(),
            HitAction::PromptCancel => self.prompt_mode = None,
            HitAction::InspectorOpen { cinema } => self.open_selected(cinema),
            HitAction::InspectorCopy => self.copy_selected_to_clipboard(ClipboardOp::Copy),
            HitAction::InspectorCut => self.copy_selected_to_clipboard(ClipboardOp::Cut),
            HitAction::PasteHere => {
                if let Err(e) = self.paste_clipboard_here() {
                    self.status_message = format!("粘贴失败: {e}");
                }
            }
            HitAction::InspectorTrashOrRestore => {
                if let Err(e) = self.trash_or_restore_selected() {
                    self.status_message = format!("操作失败: {e}");
                }
            }
            HitAction::UndoLast => {
                if let Err(e) = self.undo_last() {
                    self.status_message = format!("{e}");
                }
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
                    let n = drag.source_paths.len();
                    self.undo_stack.push(undo);
                    self.reload_directory();
                    self.status_message = format!("已将 {n} 项移入回收站");
                }
            } else if let Ok(undo) = fs_ops::move_paths_to_dir(&drag.source_paths, Path::new(&dest))
            {
                let summary = undo.summary();
                self.undo_stack.push(undo);
                self.reload_directory();
                self.status_message = format!("已移动：{summary}");
            }
        }
    }

    // ========================================================================
    // UKUI XR "Qingkong" Spatial Window, Cards, 3D Treemap & Companion Bar
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
        let mut ancestors: Vec<PathBuf> = Vec::new();
        let mut cur = Path::new(&self.current_location).parent();
        while let Some(p) = cur {
            ancestors.push(p.to_path_buf());
            if ancestors.len() >= 2 {
                break;
            }
            cur = p.parent();
        }
        // Draw furthest ancestor first so nearer glass layers blend cleanly in front
        for (depth_idx, anc) in ancestors.into_iter().enumerate().rev() {
            let z_push = (depth_idx + 1) as f32 * 0.11;
            let y_lift = 0.375 + depth_idx as f32 * 0.036;
            let pose = arc_panel_pose(
                self.anchor_pos,
                self.anchor_yaw,
                yaw_deg,
                y_lift,
                self.workspace_radius + z_push,
                eye_center,
            );
            let title = if anc == Path::new("/") {
                "上层空间  ›  系统根目录 /".to_string()
            } else {
                format!(
                    "上层空间  ›  {}",
                    anc.file_name()
                        .map(|s| s.to_string_lossy().into_owned())
                        .unwrap_or_else(|| anc.display().to_string())
                )
            };
            let alpha = if depth_idx == 0 { 0.62 } else { 0.42 };
            push_panel_glass(
                &mut scene.quads,
                &pose,
                [0.0, 0.0, 0.0],
                [0.86, 0.046],
                0.022,
                qk_rgba(236, 243, 252, alpha),
                [1.0, 1.0, 1.0, alpha * 0.85],
            );
            push_panel_text(
                &mut scene.quads,
                &pose,
                [-0.39, -0.008, 0.004],
                &title,
                0.017,
                0.78,
                qk_rgba(55, 76, 108, alpha + 0.25),
                FontWeight::Regular,
                glyphs,
            );
            self.register_hit(
                pose.position,
                [0.43, 0.024, 0.02],
                HitAction::NavigateLocation(anc.to_string_lossy().into_owned()),
            );
        }
    }

    fn emit_main_qingkong_window(
        &mut self,
        scene: &mut FrameScene,
        yaw_deg: f32,
        eye_center: [f32; 3],
        hands: &[HandControllerInput; 2],
        glyphs: &mut GlyphAtlas,
    ) {
        let win = arc_panel_pose(
            self.anchor_pos,
            self.anchor_yaw,
            yaw_deg,
            -0.02,
            self.workspace_radius,
            eye_center,
        );
        let win_w = 1.28f32;
        let win_h = 0.74f32;

        // 1. Exterior-only soft Qingkong drop shadow (`xr-shell` material==4 style)
        push_panel_shadow(
            &mut scene.quads,
            &win,
            [0.0, -0.010, -0.012],
            [win_w, win_h],
            0.032,
            0.036,
            0.085,
        );

        // 2. Main Frosted Silver-Mist Glass Window (`#E8F0F8`)
        push_panel_glass(
            &mut scene.quads,
            &win,
            [0.0, 0.0, 0.0],
            [win_w, win_h],
            0.032,
            col_glass_panel(),
            col_rim_light(),
        );

        // 3. Header Bar (`个人文件` + Golden Folder emblem + Search/Breadcrumb Pill + Segmented View Switcher)
        push_textured_quad(
            &mut scene.quads,
            &win,
            [-0.585, 0.318, 0.005],
            [0.040, 0.040],
            0.010,
            2,
            0.0, // Golden Folder icon
        );
        push_panel_text(
            &mut scene.quads,
            &win,
            [-0.552, 0.310, 0.006],
            "个人文件",
            0.024,
            0.16,
            col_ink_navy(),
            FontWeight::Bold,
            glyphs,
        );

        // Breadcrumb / Search Pill
        let search_text = if let Some(mode) = self.prompt_mode {
            let prefix = match mode {
                PromptMode::Search => "搜索",
                PromptMode::NewFolder => "新建文件夹",
                PromptMode::Rename => "重命名",
            };
            format!("{prefix}:  {}▌", self.prompt_buffer)
        } else if !self.search_query.is_empty() {
            format!(
                "{}   ·  筛选: {}",
                self.current_location, self.search_query
            )
        } else {
            self.current_location.clone()
        };
        let search_active = self.prompt_mode.is_some() || !self.search_query.is_empty();
        self.emit_qingkong_pill(
            scene,
            &win,
            [-0.145, 0.318, 0.006],
            [0.52, 0.042],
            &search_text,
            HitAction::StartPrompt(PromptMode::Search),
            search_active,
            false,
            glyphs,
        );

        // Segmented View Mode Toggle (`工作台` | `3D 星图`)
        self.emit_qingkong_pill(
            scene,
            &win,
            [0.195, 0.318, 0.006],
            [0.105, 0.040],
            "工作台",
            HitAction::ToggleView(ViewMode::Workbench),
            self.view_mode == ViewMode::Workbench,
            false,
            glyphs,
        );
        self.emit_qingkong_pill(
            scene,
            &win,
            [0.308, 0.318, 0.006],
            [0.105, 0.040],
            "3D 星图",
            HitAction::ToggleView(ViewMode::Treemap3D),
            self.view_mode == ViewMode::Treemap3D,
            false,
            glyphs,
        );

        // Quick Header Actions (`+ 新建` / `粘贴` / `撤销`)
        self.emit_qingkong_pill(
            scene,
            &win,
            [0.428, 0.318, 0.006],
            [0.088, 0.040],
            "+ 新建",
            HitAction::StartPrompt(PromptMode::NewFolder),
            false,
            false,
            glyphs,
        );
        self.emit_qingkong_pill(
            scene,
            &win,
            [0.518, 0.318, 0.006],
            [0.072, 0.040],
            "粘贴",
            HitAction::PasteHere,
            self.clipboard.is_some(),
            false,
            glyphs,
        );
        self.emit_qingkong_pill(
            scene,
            &win,
            [0.594, 0.318, 0.006],
            [0.072, 0.040],
            "撤销",
            HitAction::UndoLast,
            !self.undo_stack.is_empty(),
            false,
            glyphs,
        );

        // 4. Left Embedded Qingkong Sidebar (`x in -0.62..-0.36`, matching `style-reference.png` 04!)
        let bookmarks = self.bookmarks.clone();
        for (i, bm) in bookmarks.iter().take(7).enumerate() {
            let ly = 0.228 - (i as f32) * 0.052;
            let active = self.current_location == bm.target;
            self.emit_qingkong_pill(
                scene,
                &win,
                [-0.492, ly, 0.006],
                [0.235, 0.042],
                &bm.label,
                HitAction::NavigateLocation(bm.target.clone()),
                active,
                false,
                glyphs,
            );
        }

        // Sort & Order & Hidden toggles in Sidebar
        let arrow = if self.ascending { "↑" } else { "↓" };
        self.emit_qingkong_pill(
            scene,
            &win,
            [-0.556, -0.155, 0.006],
            [0.108, 0.038],
            self.sort_by.label_zh(),
            HitAction::CycleSort,
            false,
            false,
            glyphs,
        );
        self.emit_qingkong_pill(
            scene,
            &win,
            [-0.478, -0.155, 0.006],
            [0.038, 0.038],
            arrow,
            HitAction::ToggleSortOrder,
            false,
            false,
            glyphs,
        );
        self.emit_qingkong_pill(
            scene,
            &win,
            [-0.415, -0.155, 0.006],
            [0.072, 0.038],
            "隐藏项",
            HitAction::ToggleHidden,
            self.show_hidden,
            false,
            glyphs,
        );

        // Storage Volume Kylin-Blue Progress Gauge at Sidebar Bottom
        if let Some(vol) = self.volumes.first().cloned() {
            let used = vol.total_bytes.saturating_sub(vol.free_bytes);
            let ratio = (used as f32 / vol.total_bytes.max(1) as f32).clamp(0.05, 1.0);
            let cap_label = format!("存储剩余 {}", fs_ops::format_bytes(vol.free_bytes));
            push_panel_text(
                &mut scene.quads,
                &win,
                [-0.602, -0.235, 0.006],
                &cap_label,
                0.0155,
                0.22,
                col_muted_slate(),
                FontWeight::Regular,
                glyphs,
            );
            // Track
            push_panel_glass(
                &mut scene.quads,
                &win,
                [-0.492, -0.268, 0.006],
                [0.225, 0.010],
                0.005,
                qk_rgba(205, 218, 236, 0.85),
                [0.0; 4],
            );
            // Active Kylin-Blue Fill (`#3790FA`)
            let fill_w = 0.225 * ratio;
            push_panel_glass(
                &mut scene.quads,
                &win,
                [-0.6045 + 0.5 * fill_w, -0.268, 0.008],
                [fill_w, 0.010],
                0.005,
                col_kylin_blue(),
                [0.0; 4],
            );
        }

        // 5. Right Warm-White Content Surface Well (`#F5F7FA`, `x in -0.35..+0.62`)
        let well_cx = 0.135f32;
        let well_cy = -0.028f32;
        let well_w = 0.965f32;
        let well_h = 0.625f32;
        push_panel_glass(
            &mut scene.quads,
            &win,
            [well_cx, well_cy, 0.003],
            [well_w, well_h],
            0.024,
            col_content_well(),
            col_rim_subtle(),
        );

        // Top Filter Bar inside Content Well (or Prompt Quick Tokens)
        if self.prompt_mode.is_some() {
            let tokens: &[(&str, HitAction)] = &[
                ("doc", HitAction::PromptAppend("doc")),
                ("xr", HitAction::PromptAppend("xr")),
                (".rs", HitAction::PromptAppend(".rs")),
                (".mp4", HitAction::PromptAppend(".mp4")),
                ("退格", HitAction::PromptBackspace),
                ("确认", HitAction::PromptCommit),
                ("取消", HitAction::PromptCancel),
            ];
            for (i, (lbl, act)) in tokens.iter().enumerate() {
                let lx = -0.28 + (i as f32) * 0.122;
                self.emit_qingkong_pill(
                    scene,
                    &win,
                    [lx, 0.244, 0.008],
                    [0.108, 0.036],
                    lbl,
                    act.clone(),
                    *lbl == "确认",
                    false,
                    glyphs,
                );
            }
        } else {
            let filters = [
                CategoryFilter::All,
                CategoryFilter::Folders,
                CategoryFilter::Media,
                CategoryFilter::DocsCode,
                CategoryFilter::Models3D,
            ];
            for (i, f) in filters.into_iter().enumerate() {
                let lx = -0.275 + (i as f32) * 0.118;
                self.emit_qingkong_pill(
                    scene,
                    &win,
                    [lx, 0.244, 0.008],
                    [0.106, 0.035],
                    f.label_zh(),
                    HitAction::SetFilter(f),
                    self.category_filter == f,
                    false,
                    glyphs,
                );
            }
            if !self.search_query.is_empty() {
                self.emit_qingkong_pill(
                    scene,
                    &win,
                    [0.335, 0.244, 0.008],
                    [0.098, 0.035],
                    "清除筛选",
                    HitAction::ClearSearch,
                    true,
                    false,
                    glyphs,
                );
            }
            push_panel_text(
                &mut scene.quads,
                &win,
                [0.445, 0.237, 0.008],
                &self.status_message,
                0.0155,
                0.16,
                col_muted_slate(),
                FontWeight::Regular,
                glyphs,
            );
        }

        match self.view_mode {
            ViewMode::Workbench => {
                self.emit_qingkong_file_grid(scene, &win, well_cx, hands, glyphs);
            }
            ViewMode::Treemap3D => {
                self.emit_qingkong_treemap_3d(scene, &win, well_cx, eye_center, glyphs);
            }
        }
    }

    fn emit_qingkong_file_grid(
        &mut self,
        scene: &mut FrameScene,
        win: &Pose3D,
        well_cx: f32,
        hands: &[HandControllerInput; 2],
        glyphs: &mut GlyphAtlas,
    ) {
        let start = self.page * CARDS_PER_PAGE;
        let end = (start + CARDS_PER_PAGE).min(self.filtered_indices.len());
        let card_w = 0.214f32;
        let card_h = 0.158f32;

        if self.filtered_indices.is_empty() {
            push_panel_text(
                &mut scene.quads,
                win,
                [well_cx - 0.14, -0.03, 0.012],
                "当前文件夹为空或无匹配结果",
                0.021,
                0.42,
                col_muted_slate(),
                FontWeight::Regular,
                glyphs,
            );
            return;
        }

        for slot in 0..(end.saturating_sub(start)) {
            let entry_idx = self.filtered_indices[start + slot];
            let entry = self.all_entries[entry_idx].clone();
            let col = slot % GRID_COLS;
            let row = slot / GRID_COLS;

            let lx = well_cx + (col as f32 - 1.5) * 0.232;
            let ly = 0.118 - (row as f32) * 0.176;
            let z_lift = self.slot_z_cur.get(slot).copied().unwrap_or(0.010);

            let card_world = vec3_add(
                win.position,
                quat_rotate_vec3(win.orientation, [lx, ly, 0.006 + z_lift]),
            );
            let hit_idx = self.register_hit(
                card_world,
                [card_w * 0.50, card_h * 0.50, 0.022],
                HitAction::SelectEntry(entry_idx),
            );
            let hovered = self.is_hit_hovered(hit_idx);
            let pressed = hovered && (hands[0].trigger_pressed || hands[1].trigger_pressed);
            if slot < self.slot_z_tgt.len() {
                // Constraint 2.4: 3D Z-travel (+10mm rest, +22mm hover, +3mm press)
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

            // Delicate exterior card shadow so lifted cards cast depth onto `#F5F7FA` content well
            push_panel_shadow(
                &mut scene.quads,
                win,
                [lx, ly - 0.004, 0.005],
                [card_w, card_h],
                0.020,
                0.018,
                if hovered { 0.09 } else { 0.045 },
            );

            let fill = if selected {
                col_card_selected()
            } else if hovered {
                col_card_hover()
            } else {
                col_card_idle()
            };
            let border = if selected {
                col_kylin_blue()
            } else if hovered {
                [1.0, 1.0, 1.0, 0.95]
            } else {
                qk_rgba(214, 226, 242, 0.65)
            };

            push_panel_glass(
                &mut scene.quads,
                win,
                [lx, ly, cz],
                [card_w, card_h],
                0.020,
                fill,
                border,
            );

            // Visual Icon / Photo Preview
            if let Some(&photo_layer) = self.thumb_slot_by_path.get(&entry.path) {
                push_textured_quad(
                    &mut scene.quads,
                    win,
                    [lx, ly + 0.024, cz + 0.003],
                    [card_w - 0.022, 0.086],
                    0.014,
                    2,
                    photo_layer as f32,
                );
            } else {
                let icon_slot = icon_slot_for_kind(entry.kind) as f32;
                push_textured_quad(
                    &mut scene.quads,
                    win,
                    [lx, ly + 0.026, cz + 0.003],
                    [0.074, 0.074],
                    0.014,
                    2,
                    icon_slot,
                );
            }

            // Crisp Ink-Navy `#233047` Filename + Muted `#576C8B` Metadata
            push_panel_text(
                &mut scene.quads,
                win,
                [lx - card_w * 0.42, ly - 0.036, cz + 0.004],
                &entry.name,
                0.0175,
                card_w * 0.84,
                col_ink_navy(),
                FontWeight::Bold,
                glyphs,
            );
            let sub = if entry.kind == FileKind::Directory {
                "文件夹".to_string()
            } else {
                format!("{} · {}", entry.kind.label_zh(), fs_ops::format_bytes(entry.size))
            };
            push_panel_text(
                &mut scene.quads,
                win,
                [lx - card_w * 0.42, ly - 0.059, cz + 0.004],
                &sub,
                0.0135,
                card_w * 0.84,
                col_muted_slate(),
                FontWeight::Regular,
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
        let podium_w = 0.86f32;
        let podium_d = 0.46f32;
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

        // Angled frosted-pearl 3D tray projecting gracefully out in front of the content well
        let tray_center = vec3_add(
            win.position,
            quat_rotate_vec3(win.orientation, [well_cx, -0.075, 0.16]),
        );
        let podium_pose = Pose3D {
            position: tray_center,
            orientation: quat_mul(
                win.orientation,
                quat_from_yaw_pitch(0.0, 28.0f32.to_radians()),
            ),
        };

        // Pearl-white 3D podium base slab
        push_oriented_box(
            &mut scene.solid_vertices,
            &podium_pose,
            [0.0, -0.012, 0.0],
            [podium_w * 0.52, 0.011, podium_d * 0.53],
            qk_rgba(232, 240, 250, 1.0),
            0.05,
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
                if hovered {
                    0.35
                } else if selected {
                    0.18
                } else {
                    0.04
                },
            );

            if hovered || self.primary_selected == Some(block.entry_index) {
                let clamped_local = [
                    block.x_center.clamp(-0.20, 0.16),
                    0.5 * h_scaled + 0.115,
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
                let info = format!(
                    "{}  ·  {} ({:.1}%)",
                    entry.name,
                    fs_ops::format_bytes(entry.size),
                    block.weight_ratio * 100.0
                );
                push_panel_glass(
                    &mut scene.quads,
                    &banner_pose,
                    [0.0, 0.0, 0.0],
                    [0.44, 0.048],
                    0.024,
                    qk_rgba(255, 255, 255, 0.94),
                    col_kylin_blue(),
                );
                push_panel_text(
                    &mut scene.quads,
                    &banner_pose,
                    [-0.195, -0.007, 0.005],
                    &info,
                    0.017,
                    0.39,
                    col_ink_navy(),
                    FontWeight::Bold,
                    glyphs,
                );
            }
        }
    }

    fn emit_companion_window_bar(
        &mut self,
        scene: &mut FrameScene,
        yaw_deg: f32,
        eye_center: [f32; 3],
        glyphs: &mut GlyphAtlas,
    ) {
        // Signature HyperSpace Qingkong Bottom Window Companion Controls (`control-library.png` 04!)
        let bar = arc_panel_pose(
            self.anchor_pos,
            self.anchor_yaw,
            yaw_deg,
            -0.445,
            self.workspace_radius - 0.02,
            eye_center,
        );

        let left_orbs = [
            ("←", -0.295, HitAction::NavigateBack),
            ("→", -0.240, HitAction::NavigateForward),
            ("↑", -0.185, HitAction::NavigateUp),
            ("⟳", -0.130, HitAction::RecenterWorkspace),
        ];
        for (lbl, lx, act) in left_orbs {
            self.emit_qingkong_pill(
                scene,
                &bar,
                [lx, 0.0, 0.0],
                [0.044, 0.044],
                lbl,
                act,
                false,
                false,
                glyphs,
            );
        }

        // Center Window Grab Handle Capsule (`#F5F7FA` outer capsule + `#3790FA` Kylin-Blue core bar!)
        let grab_world = bar.position;
        let grab_hit = self.register_hit(
            grab_world,
            [0.105, 0.022, 0.02],
            HitAction::GrabWorkspaceBar,
        );
        let grab_hov = self.is_hit_hovered(grab_hit);
        push_panel_shadow(
            &mut scene.quads,
            &bar,
            [0.0, -0.004, -0.006],
            [0.20, 0.036],
            0.018,
            0.014,
            0.07,
        );
        push_panel_glass(
            &mut scene.quads,
            &bar,
            [0.0, 0.0, if grab_hov { 0.008 } else { 0.0 }],
            [0.20, 0.036],
            0.018,
            if grab_hov {
                col_pill_hover()
            } else {
                col_pill_idle()
            },
            col_rim_light(),
        );
        push_panel_glass(
            &mut scene.quads,
            &bar,
            [0.0, 0.0, if grab_hov { 0.011 } else { 0.003 }],
            [0.092, 0.007],
            0.0035,
            col_kylin_blue(),
            [0.0; 4],
        );

        // Right Page Stepper Pills (`◀`  `1/1`  `▶`)
        self.emit_qingkong_pill(
            scene,
            &bar,
            [0.145, 0.0, 0.0],
            [0.044, 0.044],
            "◀",
            HitAction::PageDelta(-1),
            false,
            false,
            glyphs,
        );
        let ptxt = format!("{}/{}", self.page + 1, self.page_count());
        push_panel_glass(
            &mut scene.quads,
            &bar,
            [0.205, 0.0, 0.0],
            [0.062, 0.038],
            0.019,
            col_glass_panel(),
            col_rim_subtle(),
        );
        push_panel_text(
            &mut scene.quads,
            &bar,
            [0.187, -0.007, 0.004],
            &ptxt,
            0.016,
            0.05,
            col_ink_navy(),
            FontWeight::Bold,
            glyphs,
        );
        self.emit_qingkong_pill(
            scene,
            &bar,
            [0.265, 0.0, 0.0],
            [0.044, 0.044],
            "▶",
            HitAction::PageDelta(1),
            false,
            false,
            glyphs,
        );
    }

    fn emit_right_inspector_card(
        &mut self,
        scene: &mut FrameScene,
        yaw_deg: f32,
        eye_center: [f32; 3],
        glyphs: &mut GlyphAtlas,
    ) {
        let panel = arc_panel_pose(
            self.anchor_pos,
            self.anchor_yaw,
            yaw_deg,
            -0.02,
            self.workspace_radius,
            eye_center,
        );
        let pw = 0.44f32;
        let ph = 0.74f32;

        push_panel_shadow(
            &mut scene.quads,
            &panel,
            [0.0, -0.010, -0.012],
            [pw, ph],
            0.030,
            0.034,
            0.085,
        );
        push_panel_glass(
            &mut scene.quads,
            &panel,
            [0.0, 0.0, 0.0],
            [pw, ph],
            0.030,
            col_glass_panel(),
            col_rim_light(),
        );

        push_panel_text(
            &mut scene.quads,
            &panel,
            [-0.185, 0.312, 0.006],
            "空间全息检视",
            0.022,
            0.36,
            col_ink_navy(),
            FontWeight::Bold,
            glyphs,
        );

        let Some(sel_idx) = self.primary_selected.filter(|&i| i < self.all_entries.len()) else {
            push_panel_text(
                &mut scene.quads,
                &panel,
                [-0.15, 0.02, 0.006],
                "选择文件卡片查看全息预览",
                0.018,
                0.34,
                col_muted_slate(),
                FontWeight::Regular,
                glyphs,
            );
            return;
        };

        let entry = self.all_entries[sel_idx].clone();
        // Header icon + filename + metadata
        let icon_layer = self
            .thumb_slot_by_path
            .get(&entry.path)
            .copied()
            .unwrap_or_else(|| icon_slot_for_kind(entry.kind));
        push_textured_quad(
            &mut scene.quads,
            &panel,
            [-0.162, 0.252, 0.006],
            [0.048, 0.048],
            0.012,
            2,
            icon_layer as f32,
        );
        push_panel_text(
            &mut scene.quads,
            &panel,
            [-0.125, 0.258, 0.006],
            &entry.name,
            0.019,
            0.30,
            col_ink_navy(),
            FontWeight::Bold,
            glyphs,
        );
        let meta_line = format!(
            "{}  ·  {}  ·  {}",
            entry.kind.label_zh(),
            fs_ops::format_bytes(entry.size),
            fs_ops::format_timestamp_ymd_hm(entry.modified_unix)
        );
        push_panel_text(
            &mut scene.quads,
            &panel,
            [-0.125, 0.232, 0.006],
            &meta_line,
            0.0135,
            0.30,
            col_muted_slate(),
            FontWeight::Regular,
            glyphs,
        );

        // Warm-White Preview Stage (`#F5F7FA`)
        push_panel_glass(
            &mut scene.quads,
            &panel,
            [0.0, 0.038, 0.004],
            [0.385, 0.285],
            0.020,
            col_content_well(),
            col_rim_subtle(),
        );

        let has_inspect = self
            .previewer
            .active_inspect
            .as_ref()
            .is_some_and(|p| p.path == entry.path);
        if has_inspect {
            let inspect = self.previewer.active_inspect.as_ref().unwrap().clone();
            push_panel_text(
                &mut scene.quads,
                &panel,
                [-0.182, 0.196, 0.006],
                &inspect.summary_note,
                0.0135,
                0.36,
                col_muted_slate(),
                FontWeight::Regular,
                glyphs,
            );

            // Sleek Kylin-Blue Shannon Entropy Gauge
            let ent_ratio = (inspect.inspection.entropy_bits / 8.0).clamp(0.0, 1.0);
            if ent_ratio > 0.0 {
                push_panel_glass(
                    &mut scene.quads,
                    &panel,
                    [0.0, -0.120, 0.006],
                    [0.38, 0.007],
                    0.0035,
                    qk_rgba(208, 220, 238, 0.85),
                    [0.0; 4],
                );
                let ew = 0.38 * ent_ratio;
                push_panel_glass(
                    &mut scene.quads,
                    &panel,
                    [-0.19 + 0.5 * ew, -0.120, 0.008],
                    [ew, 0.007],
                    0.0035,
                    col_kylin_blue(),
                    [0.0; 4],
                );
            }

            if let Some(ref mesh) = inspect.mesh {
                let spin = self.start_instant.elapsed().as_secs_f32() * 0.85;
                let holo_world = vec3_add(
                    panel.position,
                    quat_rotate_vec3(panel.orientation, [0.0, 0.038, 0.095]),
                );
                let holo_rot = quat_mul(panel.orientation, quat_from_yaw_pitch(spin, 0.24));
                push_hologram_mesh(
                    &mut scene.solid_vertices,
                    holo_world,
                    holo_rot,
                    0.185,
                    &mesh.vertices,
                    qk_rgba(72, 138, 252, 1.0),
                );
            } else if inspect.image.is_some() {
                push_textured_quad(
                    &mut scene.quads,
                    &panel,
                    [0.0, 0.038, 0.008],
                    [0.355, 0.255],
                    0.016,
                    3,
                    0.0,
                );
            } else {
                for (li, line) in inspect.text_lines.iter().take(10).enumerate() {
                    let ly = 0.148 - (li as f32) * 0.023;
                    push_panel_text(
                        &mut scene.quads,
                        &panel,
                        [-0.175, ly, 0.008],
                        line,
                        0.0135,
                        0.35,
                        col_ink_navy(),
                        FontWeight::Regular,
                        glyphs,
                    );
                }
            }
        }

        // Inspector Action Pills (`#3790FA` Primary CTA + `#F5F7FA` Secondary + Soft Danger)
        let is_trash = self.current_location == TRASH_URI;
        let is_video = entry.kind == FileKind::Video;
        let open_label = if is_video {
            "▶ 空间影院播放"
        } else if entry.kind == FileKind::Directory {
            "进入文件夹"
        } else {
            "打开文件"
        };
        if !is_trash {
            self.emit_qingkong_pill(
                scene,
                &panel,
                [-0.098, -0.172, 0.008],
                [0.185, 0.044],
                open_label,
                HitAction::InspectorOpen { cinema: is_video },
                true,
                false,
                glyphs,
            );
            self.emit_qingkong_pill(
                scene,
                &panel,
                [0.098, -0.172, 0.008],
                [0.185, 0.044],
                "重命名",
                HitAction::StartPrompt(PromptMode::Rename),
                false,
                false,
                glyphs,
            );
            self.emit_qingkong_pill(
                scene,
                &panel,
                [-0.098, -0.228, 0.008],
                [0.185, 0.042],
                "复制",
                HitAction::InspectorCopy,
                false,
                false,
                glyphs,
            );
            self.emit_qingkong_pill(
                scene,
                &panel,
                [0.098, -0.228, 0.008],
                [0.185, 0.042],
                "剪切",
                HitAction::InspectorCut,
                false,
                false,
                glyphs,
            );
        }
        let trash_label = if is_trash {
            "还原到原位置"
        } else {
            "移入回收站"
        };
        self.emit_qingkong_pill(
            scene,
            &panel,
            [0.0, -0.286, 0.008],
            [0.382, 0.044],
            trash_label,
            HitAction::InspectorTrashOrRestore,
            is_trash,
            !is_trash,
            glyphs,
        );
    }

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
        let pos = vec3_add(menu.anchor_world, vec3_scale(to_eye, 0.08));
        let pose = Pose3D {
            position: pos,
            orientation: user_facing_orientation(pos, eye_center),
        };
        let items: &[(&str, ContextMenuOp)] = if self.current_location == TRASH_URI {
            &[
                ("还原所选项目", ContextMenuOp::TrashOrRestore),
                ("关闭菜单", ContextMenuOp::Close),
            ]
        } else {
            &[
                ("打开", ContextMenuOp::Open),
                ("空间影院播放", ContextMenuOp::OpenCinema),
                ("复制", ContextMenuOp::Copy),
                ("剪切", ContextMenuOp::Cut),
                ("粘贴到此处", ContextMenuOp::Paste),
                ("重命名", ContextMenuOp::Rename),
                ("移入回收站", ContextMenuOp::TrashOrRestore),
                ("关闭", ContextMenuOp::Close),
            ]
        };
        let mh = 0.032 + (items.len() as f32) * 0.044;
        push_panel_shadow(
            &mut scene.quads,
            &pose,
            [0.0, -0.008, -0.008],
            [0.25, mh],
            0.022,
            0.024,
            0.10,
        );
        push_panel_glass(
            &mut scene.quads,
            &pose,
            [0.0, 0.0, 0.0],
            [0.25, mh],
            0.022,
            qk_rgba(240, 246, 253, 0.94),
            col_rim_light(),
        );
        for (i, (label, op)) in items.iter().enumerate() {
            let ly = 0.5 * mh - 0.033 - (i as f32) * 0.043;
            self.emit_qingkong_pill(
                scene,
                &pose,
                [0.0, ly, 0.006],
                [0.222, 0.037],
                label,
                HitAction::ContextAction(*op),
                i == 0,
                *op == ContextMenuOp::TrashOrRestore && self.current_location != TRASH_URI,
                glyphs,
            );
        }
        let _ = menu.entry_index;
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_qingkong_pill(
        &mut self,
        scene: &mut FrameScene,
        panel: &Pose3D,
        local_offset: [f32; 3],
        size_m: [f32; 2],
        label: &str,
        action: HitAction,
        primary_active: bool,
        danger: bool,
        glyphs: &mut GlyphAtlas,
    ) {
        let world_center = vec3_add(
            panel.position,
            quat_rotate_vec3(panel.orientation, local_offset),
        );
        let hit_idx = self.register_hit(
            world_center,
            [size_m[0] * 0.5, size_m[1] * 0.5, 0.02],
            action,
        );
        let hovered = self.is_hit_hovered(hit_idx);
        let z_bump = if hovered { 0.007 } else { 0.0 };
        let bumped = [local_offset[0], local_offset[1], local_offset[2] + z_bump];

        let (fill, border, text_col, weight) = if primary_active {
            (
                if hovered {
                    col_kylin_blue_hover()
                } else {
                    col_kylin_blue()
                },
                [1.0, 1.0, 1.0, 0.45],
                col_pure_white(),
                FontWeight::Bold,
            )
        } else if danger {
            (
                if hovered {
                    qk_rgba(255, 224, 228, 0.98)
                } else {
                    col_danger_soft()
                },
                qk_rgba(246, 150, 160, 0.55),
                col_danger_text(),
                FontWeight::Bold,
            )
        } else {
            (
                if hovered {
                    col_pill_hover()
                } else {
                    col_pill_idle()
                },
                if hovered {
                    col_kylin_blue()
                } else {
                    qk_rgba(210, 222, 238, 0.65)
                },
                col_ink_navy(),
                FontWeight::Regular,
            )
        };

        push_panel_glass(
            &mut scene.quads,
            panel,
            bumped,
            size_m,
            (size_m[1] * 0.44).min(0.018),
            fill,
            border,
        );
        push_panel_text(
            &mut scene.quads,
            panel,
            [
                bumped[0] - size_m[0] * 0.40,
                bumped[1] - 0.0072,
                bumped[2] + 0.003,
            ],
            label,
            0.0165,
            size_m[0] * 0.82,
            text_col,
            weight,
            glyphs,
        );
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

            // Sleek pearl-white 6DoF controller silhouette when tracked away from head
            if vec3_len(vec3_sub(ctrl.grip_pose.position, eye_center)) > 0.12 {
                push_oriented_box(
                    &mut scene.solid_vertices,
                    &ctrl.grip_pose,
                    [0.0, -0.012, 0.042],
                    [0.015, 0.013, 0.052],
                    qk_rgba(242, 246, 252, 1.0),
                    0.12,
                );
            }

            // Delicate Qingkong Kylin-Blue -> White tapered light ray (`Constraint 3.2`)
            let hit_dist = pick.map(|p| p.distance).unwrap_or(1.95);
            let ray_start = vec3_add(ctrl.aim_origin, vec3_scale(ctrl.aim_dir, 0.04));
            let ray_end = vec3_add(ctrl.aim_origin, vec3_scale(ctrl.aim_dir, hit_dist));
            let ray_col = if ctrl.trigger_pressed {
                qk_rgba(55, 144, 250, 0.96)
            } else {
                qk_rgba(245, 250, 255, 0.86)
            };
            push_beam_segment(
                &mut scene.solid_vertices,
                ray_start,
                ray_end,
                0.0022,
                0.0009,
                ray_col,
                0.45,
            );

            if let Some(p) = pick {
                let normal = vec3_normalize(vec3_sub(eye_center, p.hit_point));
                let reticle_pos = vec3_add(p.hit_point, vec3_scale(normal, 0.004));
                let reticle_pose = Pose3D {
                    position: reticle_pos,
                    orientation: user_facing_orientation(reticle_pos, eye_center),
                };
                let r_size = if ctrl.trigger_pressed { 0.016 } else { 0.022 };
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
                        [0.24, 0.052],
                        0.026,
                        col_kylin_blue(),
                        col_pure_white(),
                    );
                    push_panel_text(
                        &mut scene.quads,
                        &ghost_pose,
                        [-0.102, -0.007, 0.024],
                        &format!("移动: {}", drag.primary_label),
                        0.016,
                        0.20,
                        col_pure_white(),
                        FontWeight::Bold,
                        glyphs,
                    );
                }
            }
        }
    }
}

// ============================================================================
// Low-Level Qingkong Geometry Builders
// ============================================================================

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
        params: [radius_m, 0.0016, 0.0, 0.0],
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
