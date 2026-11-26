// SPDX-License-Identifier: GPL-3.0-or-later
//! Spatial File Manager Application State Machine, Dual-View Scene Builder,
//! 6DoF Dual-Hand Ray Routing, Double-Click Stabilizer, and Spatial Drag-and-Drop.

use crate::{
    fs_ops::{
        self, Bookmark, ClipboardOp, FileClipboard, FileEntry, FileKind, SortBy, StorageVolume,
        UndoRecord, TRASH_URI,
    },
    preview::{AsyncPreviewer, MeshVertex, MipmappedRgba},
    simd::{self, SoaAabbBatch},
    spatial::{
        arc_panel_pose, center_eye_position, quat_forward, quat_from_yaw_pitch, quat_mul,
        quat_rotate_vec3, user_facing_orientation, vec3_add, vec3_dot, vec3_len, vec3_normalize,
        vec3_scale, vec3_sub, EyeView, Pose3D,
    },
    text::GlyphAtlas,
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
const GRID_ROWS: usize = 3;
/// 2.2° minimum angular hit slop (`Constraint 2.3`).
const HIT_SLOP_RAD: f32 = 0.0384; // 2.2 deg in radians

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
            Self::Models3D => "3D资产",
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

// ============================================================================
// Renderable Scene Primitives Output per Frame
// ============================================================================

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
    /// `[corner_radius_m, border_width_m, tex_kind (0=sdf, 1=glyph, 2=thumb, 3=inspect), tex_layer]`
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
    base_radius: f32,
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
    /// Indices into `all_entries` after SIMD search & category filtering.
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
    next_thumb_slot: u32,
    pub max_thumb_slots: u32,
    pub pending_gpu_thumbs: Vec<(u32, MipmappedRgba)>,
    pub pending_gpu_inspect_image: Option<MipmappedRgba>,
    last_uploaded_inspect_path: Option<PathBuf>,

    // Spatial workspace anchor & ergonomic bounds
    pub anchor_pos: [f32; 3],
    pub anchor_yaw: f32,
    pub workspace_radius: f32,
    pub card_scale: f32,
    anchor_initialized: bool,
    recenter_requested: bool,
    shutdown_flag: bool,
    pub submitted_frames: u64,

    // Per-visible-slot SIMD smoothed Z-offset (meters) & hover glow
    slot_z_cur: Vec<f32>,
    slot_z_tgt: Vec<f32>,
    treemap_lift_cur: Vec<f32>,
    treemap_lift_tgt: Vec<f32>,
    cached_treemap: Vec<TreemapBlock>,
    treemap_dirty: bool,

    // Interaction & ray hitboxes rebuilt each frame
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
            status_message: format!("空间文件已就绪 · SIMD 后端: {}", simd::active_simd_backend()),
            previewer: AsyncPreviewer::new(),
            thumb_slot_by_path: HashMap::new(),
            next_thumb_slot: 0,
            max_thumb_slots: 64,
            pending_gpu_thumbs: Vec::new(),
            pending_gpu_inspect_image: None,
            last_uploaded_inspect_path: None,
            anchor_pos: [0.0, 1.60, 0.0],
            anchor_yaw: 0.0,
            workspace_radius: 1.48,
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
                self.status_message = format!(
                    "{} · {} 项",
                    self.display_location_title(),
                    self.filtered_indices.len()
                );
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

    pub fn toggle_select(&mut self, entry_idx: usize) {
        if entry_idx >= self.all_entries.len() {
            return;
        }
        if !self.selected_entry_indices.remove(&entry_idx) {
            self.selected_entry_indices.insert(entry_idx);
            self.primary_selected = Some(entry_idx);
            self.previewer.request_inspect(&self.all_entries[entry_idx]);
        } else {
            self.primary_selected = self.selected_entry_indices.iter().next().copied();
        }
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
        self.status_message = format!("已新建文件夹：{}", name.trim());
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
            self.status_message = "未选择任何文件".into();
            return;
        }
        let n = paths.len();
        self.clipboard = Some(FileClipboard { op, paths });
        self.status_message = match op {
            ClipboardOp::Copy => format!("已复制 {n} 项到空间剪贴板"),
            ClipboardOp::Cut => format!("已剪切 {n} 项，前往目标目录点「粘贴」移动"),
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
            self.status_message = format!("已从回收站还原 {count} 项");
            return Ok(());
        }
        let paths = self.selected_paths();
        anyhow::ensure!(!paths.is_empty(), "请先选择要移入回收站的项目");
        let n = paths.len();
        let undo = fs_ops::trash_paths(&paths)?;
        self.undo_stack.push(undo);
        self.reload_directory();
        self.status_message = format!("已移入回收站 {n} 项（可点击「撤销」恢复）");
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
            PromptMode::Search => {
                self.set_search_query(text);
            }
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

    fn display_location_title(&self) -> String {
        if self.current_location == TRASH_URI {
            return "回收站 (trash://)".into();
        }
        if self.current_location == "/" {
            return "/".into();
        }
        Path::new(&self.current_location)
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| self.current_location.clone())
    }

    // ========================================================================
    // Per-Frame Simulation, SIMD Hit-Testing, Controller Input & Scene Build
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

        // Poll non-blocking thumbnails and inspector results
        for (path, mips) in self.previewer.poll() {
            let slot = *self.thumb_slot_by_path.entry(path).or_insert_with(|| {
                let s = self.next_thumb_slot % self.max_thumb_slots.max(1);
                self.next_thumb_slot = self.next_thumb_slot.wrapping_add(1);
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

        // Request thumbnails for visible page entries
        let page_start = self.page * CARDS_PER_PAGE;
        let page_end = (page_start + CARDS_PER_PAGE).min(self.filtered_indices.len());
        for pos in page_start..page_end {
            let entry_idx = self.filtered_indices[pos];
            self.previewer.request_thumbnail(&self.all_entries[entry_idx]);
        }

        // Process inputs against previous frame's hitboxes (1-frame stable or immediate rebuild)
        self.process_controller_inputs(hands, now, dt);

        // Smooth Z-travel (`+8mm` rest, `+18mm` hover, `+2mm` press) via SIMD lerp!
        let spring_alpha = (1.0 - (-18.0 * dt).exp()).clamp(0.05, 1.0);
        simd::lerp_f32_slice_in_place(&mut self.slot_z_cur, &self.slot_z_tgt, spring_alpha);
        simd::lerp_f32_slice_in_place(
            &mut self.treemap_lift_cur,
            &self.treemap_lift_tgt,
            spring_alpha,
        );

        // Rebuild scene & SIMD SoA hitboxes for current pose
        let mut scene = FrameScene::default();
        self.hit_batch.clear();
        self.hit_actions.clear();

        self.emit_environment_reference(&mut scene);
        self.emit_top_command_bar(&mut scene, eye_center, glyphs);
        self.emit_left_navigation_wing(&mut scene, eye_center, glyphs);
        match self.view_mode {
            ViewMode::Workbench => {
                self.emit_depth_breadcrumbs(&mut scene, eye_center, glyphs);
                self.emit_center_workbench_cards(&mut scene, eye_center, hands, glyphs);
            }
            ViewMode::Treemap3D => {
                self.emit_center_treemap_3d(&mut scene, eye_center, glyphs);
            }
        }
        self.emit_right_inspector_wing(&mut scene, eye_center, glyphs);
        if self.context_menu.is_some() {
            self.emit_context_menu(&mut scene, eye_center, glyphs);
        }
        self.hit_batch.finalize_padding();

        // Update hovered targets & draw 6DoF controllers + rays + normal-aligned reticles + drag ghost
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
                self.workspace_radius = 1.48;
                self.card_scale = 1.0;
                self.request_recenter();
            }

            // Secondary button (A / X or right-click) opens spatial context menu
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
                    vec3_add(ctrl.aim_origin, vec3_scale(ctrl.aim_dir, 1.55))
                });
                self.context_menu = Some(ContextMenuState {
                    anchor_world,
                    entry_index: entry_idx.or(self.primary_selected),
                });
                self.pending_haptics[h] = Some(HapticEvent::ContextMenu);
            }

            // Squeeze (Grip): either grab a hovered file card for 3D drag-and-drop,
            // or grab the workspace horizon arc to rotate & push/pull depth (`Constraint 2.1`)!
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
                            base_radius: self.workspace_radius,
                        });
                        self.pending_haptics[h] = Some(HapticEvent::GrabStart);
                    }
                } else {
                    self.workspace_grab = Some(WorkspaceGrab {
                        hand: h,
                        start_ray_dir: ctrl.aim_dir,
                        base_yaw: self.anchor_yaw,
                        base_radius: self.workspace_radius,
                    });
                    self.pending_haptics[h] = Some(HapticEvent::GrabStart);
                }
            }

            if let Some(ref grab) = self.workspace_grab {
                if grab.hand == h && ctrl.squeeze_pressed {
                    let yaw0 = (-grab.start_ray_dir[0]).atan2(-grab.start_ray_dir[2]);
                    let yaw1 = (-ctrl.aim_dir[0]).atan2(-ctrl.aim_dir[2]);
                    self.anchor_yaw = grab.base_yaw + (yaw1 - yaw0);
                    if ctrl.stick[1].abs() > 0.18 {
                        self.workspace_radius =
                            (self.workspace_radius - ctrl.stick[1] * 1.15 * dt).clamp(1.05, 3.20);
                    }
                    let _ = grab.base_radius;
                }
            }
            if sqz_just_up {
                if self.workspace_grab.as_ref().is_some_and(|g| g.hand == h) {
                    self.workspace_grab = None;
                }
                if self.drag_session.as_ref().is_some_and(|d| d.hand == h) {
                    self.finish_drag_drop(pick.map(|p| p.index));
                }
            }

            // Thumbstick pagination when not grabbing workspace
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

            // Primary Trigger: click, spatial double-click stabilizer, or threshold drag-and-drop
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
                            vec3_add(ctrl.aim_origin, vec3_scale(ctrl.aim_dir, 1.65))
                        });
                        if vec3_len(vec3_sub(cur_pt, drag.start_point)) > 0.045 {
                            drag.active_drag = true;
                            self.pending_haptics[h] = Some(HapticEvent::GrabStart);
                        }
                    }
                }
            } else if trig_just_up {
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
            HitAction::SelectEntry(entry_idx) => {
                self.context_menu = None;
                // Spatial Double-Click Stabilizer (`40ms..440ms`, ray angle `<= 1.6°` -> open!)
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
                    PromptMode::NewFolder => "新建空间目录".into(),
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
            HitAction::PromptCancel => {
                self.prompt_mode = None;
            }
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
            HitAction::SelectEntry(idx) => {
                if let Some(entry) = self.all_entries.get(idx) {
                    if entry.kind == FileKind::Directory {
                        Some(entry.path.to_string_lossy().into_owned())
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            HitAction::NavigateLocation(loc) => Some(loc),
            _ => None,
        };
        if let Some(dest) = target_result {
            if dest == TRASH_URI {
                match fs_ops::trash_paths(&drag.source_paths) {
                    Ok(undo) => {
                        let n = drag.source_paths.len();
                        self.undo_stack.push(undo);
                        self.reload_directory();
                        self.status_message = format!("空间拖拽：已将 {n} 项移入回收站");
                    }
                    Err(e) => self.status_message = format!("拖入回收站失败: {e}"),
                }
            } else {
                let dest_path = PathBuf::from(&dest);
                match fs_ops::move_paths_to_dir(&drag.source_paths, &dest_path) {
                    Ok(undo) => {
                        let summary = undo.summary();
                        self.undo_stack.push(undo);
                        self.reload_directory();
                        self.status_message =
                            format!("空间拖拽完成：{summary} -> {}", dest_path.display());
                    }
                    Err(e) => self.status_message = format!("空间拖拽移动失败: {e}"),
                }
            }
        }
    }

    pub fn step_page(&mut self, delta: isize) {
        let total = self.page_count() as isize;
        self.page = (self.page as isize + delta).clamp(0, (total - 1).max(0)) as usize;
    }

    // ========================================================================
    // Spatial Geometry Emission Helpers
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

    fn emit_environment_reference(&self, scene: &mut FrameScene) {
        // Calm circular spatial horizon reference rings below floor (`y = anchor_pos.y - 1.55m`)
        // to give stable peripheral orientation reference during head turns (`Constraint 6 / P1-7`).
        let floor_y = self.anchor_pos[1] - 1.52;
        for ring in [1.45f32, 2.35, 3.40] {
            let segs = 48usize;
            for i in 0..segs {
                let a0 = (i as f32 / segs as f32) * std::f32::consts::TAU;
                let a1 = ((i + 1) as f32 / segs as f32) * std::f32::consts::TAU;
                let p0 = [
                    self.anchor_pos[0] + a0.cos() * ring,
                    floor_y,
                    self.anchor_pos[2] + a0.sin() * ring,
                ];
                let p1 = [
                    self.anchor_pos[0] + a1.cos() * ring,
                    floor_y,
                    self.anchor_pos[2] + a1.sin() * ring,
                ];
                push_beam_segment(
                    &mut scene.solid_vertices,
                    p0,
                    p1,
                    0.0025,
                    0.0025,
                    [0.14, 0.28, 0.48, 0.42],
                    0.15,
                );
            }
        }
    }

    fn emit_top_command_bar(
        &mut self,
        scene: &mut FrameScene,
        eye_center: [f32; 3],
        glyphs: &mut GlyphAtlas,
    ) {
        // Top Command Bar at Pitch ~ +6.5° (elevation +0.20m at R=1.72m), well within [-25°, +10°]
        let panel = arc_panel_pose(
            self.anchor_pos,
            self.anchor_yaw,
            0.0,
            0.29,
            self.workspace_radius - 0.02,
            eye_center,
        );
        let pw = 1.24;
        let ph = 0.13;
        push_panel_glass(
            &mut scene.quads,
            &panel,
            [0.0, 0.0, 0.0],
            [pw, ph],
            0.022,
            [0.055, 0.085, 0.145, 0.84],
            [0.36, 0.62, 0.96, 0.34],
        );

        // Navigation buttons: Back, Forward, Up
        let navs = [
            ("←", -0.555, HitAction::NavigateBack),
            ("→", -0.495, HitAction::NavigateForward),
            ("↑上级", -0.415, HitAction::NavigateUp),
        ];
        for (label, lx, act) in navs {
            let bw = if label.len() > 3 { 0.082 } else { 0.048 };
            self.emit_panel_button(
                scene,
                &panel,
                [lx, 0.028, 0.010],
                [bw, 0.044],
                label,
                act,
                false,
                glyphs,
            );
        }

        // Breadcrumb / Active Prompt Display
        let path_display = if let Some(mode) = self.prompt_mode {
            let tag = match mode {
                PromptMode::Search => "SIMD 搜索",
                PromptMode::NewFolder => "新建文件夹",
                PromptMode::Rename => "重命名",
            };
            format!("{tag}: {}▌", self.prompt_buffer)
        } else if !self.search_query.is_empty() {
            format!(
                "{}  [筛选: {}]",
                self.current_location, self.search_query
            )
        } else {
            self.current_location.clone()
        };
        let search_active = self.prompt_mode.is_some() || !self.search_query.is_empty();
        self.emit_panel_button(
            scene,
            &panel,
            [-0.055, 0.028, 0.010],
            [0.58, 0.044],
            &path_display,
            HitAction::StartPrompt(PromptMode::Search),
            search_active,
            glyphs,
        );

        // Top-right quick actions: Search Clear / New Folder / Paste / Undo / Recenter
        self.emit_panel_button(
            scene,
            &panel,
            [0.295, 0.028, 0.010],
            [0.095, 0.044],
            "+新建夹",
            HitAction::StartPrompt(PromptMode::NewFolder),
            false,
            glyphs,
        );
        let has_clip = self.clipboard.is_some();
        self.emit_panel_button(
            scene,
            &panel,
            [0.400, 0.028, 0.010],
            [0.082, 0.044],
            "粘贴",
            HitAction::PasteHere,
            has_clip,
            glyphs,
        );
        let has_undo = !self.undo_stack.is_empty();
        self.emit_panel_button(
            scene,
            &panel,
            [0.492, 0.028, 0.010],
            [0.082, 0.044],
            "撤销",
            HitAction::UndoLast,
            has_undo,
            glyphs,
        );
        self.emit_panel_button(
            scene,
            &panel,
            [0.575, 0.028, 0.010],
            [0.062, 0.044],
            "归中",
            HitAction::RecenterWorkspace,
            false,
            glyphs,
        );

        // Bottom row of Command Bar: Category Filter Chips OR Prompt Quick-Input Bar
        if self.prompt_mode.is_some() {
            let tokens: &[(&str, HitAction)] = &[
                ("doc", HitAction::PromptAppend("doc")),
                ("xr", HitAction::PromptAppend("xr")),
                (".rs", HitAction::PromptAppend(".rs")),
                (".mp4", HitAction::PromptAppend(".mp4")),
                ("_2", HitAction::PromptAppend("_2")),
                ("退格", HitAction::PromptBackspace),
                ("确认", HitAction::PromptCommit),
                ("取消", HitAction::PromptCancel),
            ];
            for (i, (lbl, act)) in tokens.iter().enumerate() {
                let lx = -0.49 + (i as f32) * 0.138;
                self.emit_panel_button(
                    scene,
                    &panel,
                    [lx, -0.032, 0.010],
                    [0.122, 0.038],
                    lbl,
                    act.clone(),
                    *lbl == "确认",
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
                let lx = -0.50 + (i as f32) * 0.135;
                self.emit_panel_button(
                    scene,
                    &panel,
                    [lx, -0.032, 0.010],
                    [0.122, 0.036],
                    f.label_zh(),
                    HitAction::SetFilter(f),
                    self.category_filter == f,
                    glyphs,
                );
            }
            if !self.search_query.is_empty() {
                self.emit_panel_button(
                    scene,
                    &panel,
                    [0.19, -0.032, 0.010],
                    [0.105, 0.036],
                    "清除搜索",
                    HitAction::ClearSearch,
                    true,
                    glyphs,
                );
            }
            // Live status message readout
            push_panel_text(
                &mut scene.quads,
                &panel,
                [0.27, -0.038, 0.012],
                &self.status_message,
                0.017,
                0.33,
                [0.72, 0.86, 1.0, 0.94],
                glyphs,
            );
        }
    }

    fn emit_left_navigation_wing(
        &mut self,
        scene: &mut FrameScene,
        eye_center: [f32; 3],
        glyphs: &mut GlyphAtlas,
    ) {
        // Left Wing at +34.5° arc yaw (>3° angular gap from Center Stage!)
        let panel = arc_panel_pose(
            self.anchor_pos,
            self.anchor_yaw,
            34.5,
            -0.06,
            self.workspace_radius,
            eye_center,
        );
        let pw = 0.42;
        let ph = 0.66;
        push_panel_glass(
            &mut scene.quads,
            &panel,
            [0.0, 0.0, 0.0],
            [pw, ph],
            0.024,
            [0.05, 0.08, 0.14, 0.84],
            [0.32, 0.58, 0.94, 0.30],
        );

        push_panel_text(
            &mut scene.quads,
            &panel,
            [-0.18, 0.285, 0.008],
            "空间导航与视图",
            0.021,
            0.36,
            [0.90, 0.95, 1.0, 1.0],
            glyphs,
        );

        // Dual Spatial View Switcher
        self.emit_panel_button(
            scene,
            &panel,
            [-0.095, 0.235, 0.010],
            [0.175, 0.042],
            "弧面工作台",
            HitAction::ToggleView(ViewMode::Workbench),
            self.view_mode == ViewMode::Workbench,
            glyphs,
        );
        self.emit_panel_button(
            scene,
            &panel,
            [0.095, 0.235, 0.010],
            [0.175, 0.042],
            "3D 体积星图",
            HitAction::ToggleView(ViewMode::Treemap3D),
            self.view_mode == ViewMode::Treemap3D,
            glyphs,
        );

        // Sort & Hidden controls
        let order_arrow = if self.ascending { "↑" } else { "↓" };
        self.emit_panel_button(
            scene,
            &panel,
            [-0.095, 0.182, 0.010],
            [0.175, 0.038],
            &format!("{}{order_arrow}", self.sort_by.label_zh()),
            HitAction::CycleSort,
            false,
            glyphs,
        );
        self.emit_panel_button(
            scene,
            &panel,
            [0.042, 0.182, 0.010],
            [0.075, 0.038],
            order_arrow,
            HitAction::ToggleSortOrder,
            false,
            glyphs,
        );
        self.emit_panel_button(
            scene,
            &panel,
            [0.135, 0.182, 0.010],
            [0.095, 0.038],
            ".隐藏",
            HitAction::ToggleHidden,
            self.show_hidden,
            glyphs,
        );

        // Bookmarks (also act as spatial drag-and-drop targets!)
        let bookmarks = self.bookmarks.clone();
        for (i, bm) in bookmarks.iter().take(6).enumerate() {
            let ly = 0.118 - (i as f32) * 0.048;
            let active = self.current_location == bm.target;
            let label = format!("[{}] {}", bm.icon_tag, bm.label);
            self.emit_panel_button(
                scene,
                &panel,
                [0.0, ly, 0.010],
                [0.365, 0.040],
                &label,
                HitAction::NavigateLocation(bm.target.clone()),
                active,
                glyphs,
            );
        }

        // Storage Volumes with capacity bar
        push_panel_text(
            &mut scene.quads,
            &panel,
            [-0.18, -0.185, 0.008],
            "存储卷容量",
            0.018,
            0.34,
            [0.68, 0.80, 0.96, 0.92],
            glyphs,
        );
        let vols = self.volumes.clone();
        for (i, vol) in vols.iter().take(2).enumerate() {
            let ly = -0.232 - (i as f32) * 0.058;
            let used = vol.total_bytes.saturating_sub(vol.free_bytes);
            let ratio = (used as f32 / vol.total_bytes.max(1) as f32).clamp(0.02, 1.0);
            let label = format!(
                "{} · 剩余 {}",
                vol.name,
                fs_ops::format_bytes(vol.free_bytes)
            );
            self.emit_panel_button(
                scene,
                &panel,
                [0.0, ly, 0.010],
                [0.365, 0.044],
                &label,
                HitAction::NavigateLocation(vol.mount_point.to_string_lossy().into_owned()),
                self.current_location == vol.mount_point.to_string_lossy(),
                glyphs,
            );
            // Capacity gauge bar
            let bar_w = 0.34 * ratio;
            push_panel_glass(
                &mut scene.quads,
                &panel,
                [-0.17 + 0.5 * bar_w, ly - 0.017, 0.015],
                [bar_w, 0.005],
                0.002,
                [0.24, 0.72, 0.98, 0.95],
                [0.0, 0.0, 0.0, 0.0],
            );
        }
    }

    fn emit_depth_breadcrumbs(
        &mut self,
        scene: &mut FrameScene,
        eye_center: [f32; 3],
        glyphs: &mut GlyphAtlas,
    ) {
        if self.current_location == TRASH_URI {
            return;
        }
        // Receding Z-depth parent cards above/behind the active grid for true spatial hierarchy
        let mut ancestors: Vec<PathBuf> = Vec::new();
        let mut cur = Path::new(&self.current_location).parent();
        while let Some(p) = cur {
            ancestors.push(p.to_path_buf());
            if ancestors.len() >= 2 {
                break;
            }
            cur = p.parent();
        }
        for (depth_idx, anc) in ancestors.into_iter().enumerate() {
            let z_push = (depth_idx + 1) as f32 * 0.14;
            let y_lift = 0.185 + depth_idx as f32 * 0.032;
            let pose = arc_panel_pose(
                self.anchor_pos,
                self.anchor_yaw,
                0.0,
                y_lift,
                self.workspace_radius + z_push,
                eye_center,
            );
            let title = if anc == Path::new("/") {
                "上层根空间 /".to_string()
            } else {
                format!(
                    "上层空间: {}",
                    anc.file_name()
                        .map(|s| s.to_string_lossy().into_owned())
                        .unwrap_or_else(|| anc.display().to_string())
                )
            };
            let fade = if depth_idx == 0 { 0.56 } else { 0.36 };
            push_panel_glass(
                &mut scene.quads,
                &pose,
                [0.0, 0.0, 0.0],
                [0.64, 0.042],
                0.014,
                [0.07, 0.12, 0.22, fade],
                [0.35, 0.65, 0.98, fade * 0.6],
            );
            push_panel_text(
                &mut scene.quads,
                &pose,
                [-0.29, -0.007, 0.004],
                &title,
                0.017,
                0.58,
                [0.78, 0.90, 1.0, fade + 0.25],
                glyphs,
            );
            self.register_hit(
                pose.position,
                [0.32, 0.022, 0.02],
                HitAction::NavigateLocation(anc.to_string_lossy().into_owned()),
            );
        }
    }

    fn emit_center_workbench_cards(
        &mut self,
        scene: &mut FrameScene,
        eye_center: [f32; 3],
        hands: &[HandControllerInput; 2],
        glyphs: &mut GlyphAtlas,
    ) {
        let start = self.page * CARDS_PER_PAGE;
        let end = (start + CARDS_PER_PAGE).min(self.filtered_indices.len());
        let scale = self.card_scale.clamp(0.75, 1.40);
        let card_w = 0.245 * scale;
        let card_h = 0.155 * scale;

        if self.filtered_indices.is_empty() {
            let empty_pose = arc_panel_pose(
                self.anchor_pos,
                self.anchor_yaw,
                0.0,
                -0.06,
                self.workspace_radius,
                eye_center,
            );
            push_panel_glass(
                &mut scene.quads,
                &empty_pose,
                [0.0, 0.0, 0.0],
                [0.68, 0.16],
                0.024,
                [0.06, 0.09, 0.16, 0.78],
                [0.32, 0.55, 0.88, 0.28],
            );
            push_panel_text(
                &mut scene.quads,
                &empty_pose,
                [-0.24, 0.005, 0.008],
                "当前目录或筛选条件为空",
                0.024,
                0.52,
                [0.84, 0.92, 1.0, 0.95],
                glyphs,
            );
        }

        for slot in 0..(end.saturating_sub(start)) {
            let entry_idx = self.filtered_indices[start + slot];
            let entry = self.all_entries[entry_idx].clone();
            let col = slot % GRID_COLS;
            let row = slot / GRID_COLS;

            // Place each card along a smooth cylindrical arc with >= 2.2° gap between cards
            let col_yaw_deg = (1.5 - col as f32) * 11.2;
            let row_y = 0.095 - (row as f32) * 0.182;
            let z_lift = self.slot_z_cur.get(slot).copied().unwrap_or(0.008);

            let card_pose = arc_panel_pose(
                self.anchor_pos,
                self.anchor_yaw,
                col_yaw_deg,
                row_y,
                self.workspace_radius - z_lift,
                eye_center,
            );

            let hit_idx = self.register_hit(
                card_pose.position,
                [card_w * 0.50, card_h * 0.50, 0.025],
                HitAction::SelectEntry(entry_idx),
            );
            let hovered = self.is_hit_hovered(hit_idx);
            let any_pressed = hovered && (hands[0].trigger_pressed || hands[1].trigger_pressed);
            // Constraint 2.4: +8mm rest, +18mm hover, +2mm press!
            if slot < self.slot_z_tgt.len() {
                self.slot_z_tgt[slot] = if any_pressed {
                    0.002
                } else if hovered {
                    0.018
                } else {
                    0.008
                };
            }

            let selected = self.selected_entry_indices.contains(&entry_idx);
            let accent = entry.kind.accent_rgba();

            // 3D volumetric card back-slab for genuine binocular edge thickness
            push_oriented_box(
                &mut scene.solid_vertices,
                &card_pose,
                [0.0, 0.0, -0.008],
                [card_w * 0.485, card_h * 0.485, 0.006],
                if selected {
                    [accent[0] * 0.45, accent[1] * 0.45, accent[2] * 0.55, 1.0]
                } else {
                    [0.045, 0.070, 0.125, 1.0]
                },
                if hovered || selected { 0.25 } else { 0.04 },
            );

            let fill = if selected {
                [0.10, 0.20, 0.36, 0.90]
            } else if hovered {
                [0.09, 0.15, 0.26, 0.88]
            } else {
                [0.055, 0.090, 0.155, 0.84]
            };
            let border = if selected {
                [accent[0], accent[1], accent[2], 0.95]
            } else if hovered {
                [0.52, 0.80, 1.0, 0.72]
            } else {
                [0.28, 0.46, 0.74, 0.26]
            };
            push_panel_glass(
                &mut scene.quads,
                &card_pose,
                [0.0, 0.0, 0.0],
                [card_w, card_h],
                0.018,
                fill,
                border,
            );

            // Top color-coded badge or SIMD mipmapped image thumbnail
            if let Some(&thumb_layer) = self.thumb_slot_by_path.get(&entry.path) {
                push_textured_quad(
                    &mut scene.quads,
                    &card_pose,
                    [0.0, 0.022, 0.004],
                    [card_w - 0.024, card_h * 0.52],
                    0.010,
                    2,
                    thumb_layer as f32,
                );
            } else {
                // Volumetric 3D emblem block + badge tag
                push_oriented_box(
                    &mut scene.solid_vertices,
                    &card_pose,
                    [-card_w * 0.31, 0.030, 0.007],
                    [0.024, 0.020, 0.006],
                    accent,
                    if entry.kind == FileKind::Directory {
                        0.32
                    } else {
                        0.18
                    },
                );
                push_panel_text(
                    &mut scene.quads,
                    &card_pose,
                    [-card_w * 0.16, 0.022, 0.006],
                    entry.kind.label_zh(),
                    0.016,
                    card_w * 0.60,
                    [accent[0].max(0.45), accent[1].max(0.65), accent[2].max(0.75), 0.95],
                    glyphs,
                );
            }

            // Filename (Angular typography >= 20 dmm -> ~0.020m at 1.7m)
            push_panel_text(
                &mut scene.quads,
                &card_pose,
                [-card_w * 0.43, -0.028, 0.006],
                &entry.name,
                0.020,
                card_w * 0.86,
                [0.95, 0.97, 1.0, 1.0],
                glyphs,
            );

            // Size + Date subtitle
            let sub = if entry.kind == FileKind::Directory {
                format!("目录 · {}", fs_ops::format_timestamp_ymd_hm(entry.modified_unix))
            } else {
                format!(
                    "{} · {}",
                    fs_ops::format_bytes(entry.size),
                    fs_ops::format_timestamp_ymd_hm(entry.modified_unix)
                )
            };
            push_panel_text(
                &mut scene.quads,
                &card_pose,
                [-card_w * 0.43, -0.056, 0.006],
                &sub,
                0.0145,
                card_w * 0.86,
                [0.64, 0.76, 0.92, 0.88],
                glyphs,
            );
        }

        // Bottom Pagination Capsule
        let pager_pose = arc_panel_pose(
            self.anchor_pos,
            self.anchor_yaw,
            0.0,
            -0.37,
            self.workspace_radius - 0.02,
            eye_center,
        );
        push_panel_glass(
            &mut scene.quads,
            &pager_pose,
            [0.0, 0.0, 0.0],
            [0.46, 0.056],
            0.024,
            [0.05, 0.08, 0.15, 0.86],
            [0.34, 0.60, 0.94, 0.30],
        );
        self.emit_panel_button(
            scene,
            &pager_pose,
            [-0.155, 0.0, 0.008],
            [0.105, 0.038],
            "◀ 上一页",
            HitAction::PageDelta(-1),
            false,
            glyphs,
        );
        let page_str = format!("第 {} / {} 页", self.page + 1, self.page_count());
        push_panel_text(
            &mut scene.quads,
            &pager_pose,
            [-0.068, -0.007, 0.008],
            &page_str,
            0.017,
            0.16,
            [0.86, 0.93, 1.0, 0.95],
            glyphs,
        );
        self.emit_panel_button(
            scene,
            &pager_pose,
            [0.155, 0.0, 0.008],
            [0.105, 0.038],
            "下一页 ▶",
            HitAction::PageDelta(1),
            false,
            glyphs,
        );
        let _ = GRID_ROWS;
    }

    fn emit_center_treemap_3d(
        &mut self,
        scene: &mut FrameScene,
        eye_center: [f32; 3],
        glyphs: &mut GlyphAtlas,
    ) {
        let podium_w = 1.02f32;
        let podium_d = 0.62f32;
        if self.treemap_dirty {
            let visible_entries: Vec<FileEntry> = self
                .filtered_indices
                .iter()
                .filter_map(|&i| self.all_entries.get(i).cloned())
                .collect();
            let mut blocks = treemap3d::build_spatial_treemap(&visible_entries, podium_w, podium_d, 40);
            // Remap block entry_index from `visible_entries` index to `all_entries` index
            for b in &mut blocks {
                b.entry_index = self.filtered_indices[b.entry_index];
            }
            self.cached_treemap = blocks;
            self.treemap_dirty = false;
        }

        // Angled 3D Podium tilted 22° toward the viewer for effortless stereo overview
        let (sy, cy) = self.anchor_yaw.sin_cos();
        let dist = self.workspace_radius - 0.14;
        let podium_center = [
            self.anchor_pos[0] - sy * dist,
            self.anchor_pos[1] - 0.24,
            self.anchor_pos[2] - cy * dist,
        ];
        let podium_pose = Pose3D {
            position: podium_center,
            orientation: quat_mul(
                quat_from_yaw_pitch(self.anchor_yaw, 0.0),
                quat_from_yaw_pitch(0.0, 22.0f32.to_radians()),
            ),
        };

        // Podium base slab
        push_oriented_box(
            &mut scene.solid_vertices,
            &podium_pose,
            [0.0, -0.016, 0.0],
            [podium_w * 0.52, 0.014, podium_d * 0.53],
            [0.05, 0.085, 0.15, 1.0],
            0.08,
        );

        let blocks = self.cached_treemap.clone();
        for (bi, block) in blocks.iter().enumerate() {
            let entry = self.all_entries[block.entry_index].clone();
            let lift = self.treemap_lift_cur.get(bi).copied().unwrap_or(0.0);
            let local_center = [block.x_center, 0.5 * block.height + lift, block.z_center];
            let world_center = vec3_add(
                podium_pose.position,
                quat_rotate_vec3(podium_pose.orientation, local_center),
            );
            let half_ext = [
                0.5 * block.width,
                0.5 * block.height + 0.01,
                0.5 * block.depth,
            ];
            let hit_idx = self.register_hit(
                world_center,
                [
                    half_ext[0].max(0.025),
                    half_ext[1].max(0.025),
                    half_ext[2].max(0.025),
                ],
                HitAction::SelectEntry(block.entry_index),
            );
            let hovered = self.is_hit_hovered(hit_idx);
            let selected = self.selected_entry_indices.contains(&block.entry_index);
            if bi < self.treemap_lift_tgt.len() {
                self.treemap_lift_tgt[bi] = if hovered {
                    0.022
                } else if selected {
                    0.010
                } else {
                    0.0
                };
            }

            let accent = entry.kind.accent_rgba();
            push_oriented_box(
                &mut scene.solid_vertices,
                &podium_pose,
                local_center,
                [0.5 * block.width, 0.5 * block.height, 0.5 * block.depth],
                accent,
                if hovered {
                    0.55
                } else if selected {
                    0.35
                } else {
                    0.10
                },
            );

            // Floating callout elevated clearly above all podium blocks and pulled toward eye
            if hovered || self.primary_selected == Some(block.entry_index) {
                let raw_pos = [
                    world_center[0],
                    world_center[1] + 0.5 * block.height + 0.165,
                    world_center[2],
                ];
                let to_eye = vec3_normalize(vec3_sub(eye_center, raw_pos));
                let banner_pos = vec3_add(raw_pos, vec3_scale(to_eye, 0.10));
                let banner_pose = Pose3D {
                    position: banner_pos,
                    orientation: user_facing_orientation(banner_pos, eye_center),
                };
                let info = format!(
                    "{} ({} · {:.1}%)",
                    entry.name,
                    fs_ops::format_bytes(entry.size),
                    block.weight_ratio * 100.0
                );
                push_panel_glass(
                    &mut scene.quads,
                    &banner_pose,
                    [0.0, 0.0, 0.0],
                    [0.48, 0.054],
                    0.016,
                    [0.05, 0.10, 0.19, 0.94],
                    [accent[0], accent[1], accent[2], 0.90],
                );
                push_panel_text(
                    &mut scene.quads,
                    &banner_pose,
                    [-0.22, -0.008, 0.006],
                    &info,
                    0.020,
                    0.44,
                    [0.96, 0.98, 1.0, 1.0],
                    glyphs,
                );
            }
        }
    }

    fn emit_right_inspector_wing(
        &mut self,
        scene: &mut FrameScene,
        eye_center: [f32; 3],
        glyphs: &mut GlyphAtlas,
    ) {
        // Right Wing at -34.5° arc yaw (>3° angular gap from Center Stage)
        let panel = arc_panel_pose(
            self.anchor_pos,
            self.anchor_yaw,
            -34.5,
            -0.06,
            self.workspace_radius,
            eye_center,
        );
        let pw = 0.46;
        let ph = 0.66;
        push_panel_glass(
            &mut scene.quads,
            &panel,
            [0.0, 0.0, 0.0],
            [pw, ph],
            0.024,
            [0.05, 0.08, 0.14, 0.84],
            [0.32, 0.58, 0.94, 0.30],
        );

        push_panel_text(
            &mut scene.quads,
            &panel,
            [-0.20, 0.285, 0.008],
            "空间全息检视台",
            0.021,
            0.40,
            [0.90, 0.95, 1.0, 1.0],
            glyphs,
        );

        let Some(sel_idx) = self.primary_selected.filter(|&i| i < self.all_entries.len()) else {
            push_panel_text(
                &mut scene.quads,
                &panel,
                [-0.16, 0.02, 0.008],
                "指向并选择任意空间卡片以预览",
                0.018,
                0.36,
                [0.65, 0.78, 0.94, 0.85],
                glyphs,
            );
            return;
        };

        let entry = self.all_entries[sel_idx].clone();
        let accent = entry.kind.accent_rgba();
        push_panel_text(
            &mut scene.quads,
            &panel,
            [-0.20, 0.242, 0.008],
            &entry.name,
            0.020,
            0.40,
            [0.98, 0.99, 1.0, 1.0],
            glyphs,
        );
        let meta_line = format!(
            "{} · {} · {}",
            entry.kind.label_zh(),
            fs_ops::format_bytes(entry.size),
            fs_ops::format_timestamp_ymd_hm(entry.modified_unix)
        );
        push_panel_text(
            &mut scene.quads,
            &panel,
            [-0.20, 0.210, 0.008],
            &meta_line,
            0.015,
            0.40,
            [accent[0].max(0.5), accent[1].max(0.7), accent[2].max(0.8), 0.95],
            glyphs,
        );

        // Live Preview Region: 3D Rotating Hologram Mesh, High-Res Image, or Code/Hex Excerpt
        let has_matching_inspect = self
            .previewer
            .active_inspect
            .as_ref()
            .is_some_and(|p| p.path == entry.path);

        if has_matching_inspect {
            let inspect = self.previewer.active_inspect.as_ref().unwrap().clone();
            // SIMD Entropy Gauge
            let ent_ratio = (inspect.inspection.entropy_bits / 8.0).clamp(0.0, 1.0);
            push_panel_text(
                &mut scene.quads,
                &panel,
                [-0.20, 0.178, 0.008],
                &inspect.summary_note,
                0.0145,
                0.40,
                [0.72, 0.85, 0.98, 0.92],
                glyphs,
            );
            if ent_ratio > 0.0 {
                push_panel_glass(
                    &mut scene.quads,
                    &panel,
                    [-0.20 + 0.20 * ent_ratio, 0.162, 0.010],
                    [0.40 * ent_ratio, 0.005],
                    0.002,
                    accent,
                    [0.0, 0.0, 0.0, 0.0],
                );
            }

            if let Some(ref mesh) = inspect.mesh {
                // Render real 3D stereo rotating hologram mesh floating +9cm in front of the Inspector!
                let spin = self.start_instant.elapsed().as_secs_f32() * 0.85;
                let holo_world = vec3_add(
                    panel.position,
                    quat_rotate_vec3(panel.orientation, [0.0, 0.035, 0.095]),
                );
                let holo_rot = quat_mul(panel.orientation, quat_from_yaw_pitch(spin, 0.22));
                push_hologram_mesh(
                    &mut scene.solid_vertices,
                    holo_world,
                    holo_rot,
                    0.18,
                    &mesh.vertices,
                    accent,
                );
            } else if inspect.image.is_some() {
                push_textured_quad(
                    &mut scene.quads,
                    &panel,
                    [0.0, 0.035, 0.014],
                    [0.38, 0.22],
                    0.012,
                    3,
                    0.0,
                );
            } else {
                // Code / Text / Hex preview box
                push_panel_glass(
                    &mut scene.quads,
                    &panel,
                    [0.0, 0.032, 0.008],
                    [0.41, 0.23],
                    0.012,
                    [0.03, 0.05, 0.09, 0.88],
                    [0.22, 0.38, 0.64, 0.28],
                );
                for (li, line) in inspect.text_lines.iter().take(9).enumerate() {
                    let ly = 0.120 - (li as f32) * 0.022;
                    push_panel_text(
                        &mut scene.quads,
                        &panel,
                        [-0.192, ly, 0.012],
                        line,
                        0.0135,
                        0.385,
                        [0.82, 0.91, 0.98, 0.94],
                        glyphs,
                    );
                }
            }
        }

        // Action Buttons at bottom of Inspector
        let is_trash = self.current_location == TRASH_URI;
        let is_video = entry.kind == FileKind::Video;
        let open_label = if is_video {
            "空间影院播放"
        } else if entry.kind == FileKind::Directory {
            "进入文件夹"
        } else {
            "打开项目"
        };
        if !is_trash {
            self.emit_panel_button(
                scene,
                &panel,
                [-0.10, -0.145, 0.010],
                [0.19, 0.044],
                open_label,
                HitAction::InspectorOpen { cinema: is_video },
                true,
                glyphs,
            );
            self.emit_panel_button(
                scene,
                &panel,
                [0.10, -0.145, 0.010],
                [0.19, 0.044],
                "重命名",
                HitAction::StartPrompt(PromptMode::Rename),
                false,
                glyphs,
            );
            self.emit_panel_button(
                scene,
                &panel,
                [-0.10, -0.200, 0.010],
                [0.19, 0.042],
                "复制",
                HitAction::InspectorCopy,
                false,
                glyphs,
            );
            self.emit_panel_button(
                scene,
                &panel,
                [0.10, -0.200, 0.010],
                [0.19, 0.042],
                "剪切",
                HitAction::InspectorCut,
                false,
                glyphs,
            );
        }
        let trash_label = if is_trash {
            "还原到原位置"
        } else {
            "移入回收站"
        };
        self.emit_panel_button(
            scene,
            &panel,
            [0.0, -0.258, 0.010],
            [0.39, 0.044],
            trash_label,
            HitAction::InspectorTrashOrRestore,
            is_trash,
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
        // Pull context menu 8cm closer toward eye_center so it cleanly floats above cards
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
                ("用空间影院播放", ContextMenuOp::OpenCinema),
                ("复制", ContextMenuOp::Copy),
                ("剪切", ContextMenuOp::Cut),
                ("粘贴到当前目录", ContextMenuOp::Paste),
                ("重命名", ContextMenuOp::Rename),
                ("移入回收站", ContextMenuOp::TrashOrRestore),
                ("关闭菜单", ContextMenuOp::Close),
            ]
        };
        let mh = 0.030 + (items.len() as f32) * 0.043;
        push_panel_glass(
            &mut scene.quads,
            &pose,
            [0.0, 0.0, 0.0],
            [0.26, mh],
            0.018,
            [0.04, 0.07, 0.13, 0.94],
            [0.42, 0.72, 1.0, 0.55],
        );
        for (i, (label, op)) in items.iter().enumerate() {
            let ly = 0.5 * mh - 0.032 - (i as f32) * 0.042;
            self.emit_panel_button(
                scene,
                &pose,
                [0.0, ly, 0.008],
                [0.235, 0.036],
                label,
                HitAction::ContextAction(*op),
                false,
                glyphs,
            );
        }
        let _ = menu.entry_index;
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_panel_button(
        &mut self,
        scene: &mut FrameScene,
        panel: &Pose3D,
        local_offset: [f32; 3],
        size_m: [f32; 2],
        label: &str,
        action: HitAction,
        highlighted: bool,
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
        let z_bump = if hovered { 0.008 } else { 0.0 };
        let bumped = [local_offset[0], local_offset[1], local_offset[2] + z_bump];

        let fill = if highlighted {
            [0.16, 0.42, 0.82, 0.92]
        } else if hovered {
            [0.14, 0.24, 0.42, 0.90]
        } else {
            [0.08, 0.13, 0.22, 0.82]
        };
        let border = if highlighted || hovered {
            [0.52, 0.82, 1.0, 0.82]
        } else {
            [0.28, 0.48, 0.76, 0.28]
        };
        push_panel_glass(
            &mut scene.quads,
            panel,
            bumped,
            size_m,
            (size_m[1] * 0.36).min(0.014),
            fill,
            border,
        );
        push_panel_text(
            &mut scene.quads,
            panel,
            [
                bumped[0] - size_m[0] * 0.43,
                bumped[1] - 0.0075,
                bumped[2] + 0.003,
            ],
            label,
            0.019,
            size_m[0] * 0.88,
            [0.95, 0.98, 1.0, 1.0],
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

            // 6DoF Controller 3D handle body (when grip_pose is away from camera eye)
            if vec3_len(vec3_sub(ctrl.grip_pose.position, eye_center)) > 0.12 {
                push_oriented_box(
                    &mut scene.solid_vertices,
                    &ctrl.grip_pose,
                    [0.0, -0.012, 0.035],
                    [0.019, 0.016, 0.058],
                    [0.14, 0.20, 0.30, 1.0],
                    if ctrl.trigger_pressed || ctrl.squeeze_pressed {
                        0.45
                    } else {
                        0.12
                    },
                );
            }

            // Tapered 3D Volumetric Ray + Surface-Normal-Aligned Reticle (`Constraint 3.2`)
            let hit_dist = pick.map(|p| p.distance).unwrap_or(2.2);
            let ray_end = vec3_add(ctrl.aim_origin, vec3_scale(ctrl.aim_dir, hit_dist));
            let ray_color = if ctrl.trigger_pressed {
                [0.35, 0.92, 1.0, 0.95]
            } else if pick.is_some() {
                [0.28, 0.72, 1.0, 0.85]
            } else {
                [0.24, 0.52, 0.85, 0.45]
            };
            push_beam_segment(
                &mut scene.solid_vertices,
                vec3_add(ctrl.aim_origin, vec3_scale(ctrl.aim_dir, 0.03)),
                ray_end,
                0.0028,
                0.0010,
                ray_color,
                0.65,
            );

            if let Some(p) = pick {
                let normal = vec3_normalize(vec3_sub(eye_center, p.hit_point));
                let reticle_pos = vec3_add(p.hit_point, vec3_scale(normal, 0.004));
                let reticle_pose = Pose3D {
                    position: reticle_pos,
                    orientation: user_facing_orientation(reticle_pos, eye_center),
                };
                let r_size = if ctrl.trigger_pressed { 0.018 } else { 0.024 };
                push_panel_glass(
                    &mut scene.quads,
                    &reticle_pose,
                    [0.0, 0.0, 0.0],
                    [r_size, r_size],
                    r_size * 0.5,
                    [0.40, 0.88, 1.0, 0.92],
                    [1.0, 1.0, 1.0, 0.95],
                );
            }

            // Floating 3D Ghost Card during active spatial Drag-and-Drop
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
                        [0.24, 0.056],
                        0.016,
                        [0.14, 0.42, 0.86, 0.88],
                        [0.65, 0.92, 1.0, 0.95],
                    );
                    push_panel_text(
                        &mut scene.quads,
                        &ghost_pose,
                        [-0.105, -0.007, 0.024],
                        &format!("移动: {}", drag.primary_label),
                        0.016,
                        0.21,
                        [1.0, 1.0, 1.0, 1.0],
                        glyphs,
                    );
                }
            }
        }
    }
}

// ============================================================================
// Low-Level Geometry Builders
// ============================================================================

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
        params: [radius_m, 0.0018, 0.0, 0.0],
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
    glyphs: &mut GlyphAtlas,
) {
    // Rasterize at crisp 42px per 20mm (~2100 px/m) for sharp VR readability
    let raster_px = (height_m * 1900.0).clamp(18.0, 56.0);
    let m_per_px = height_m / raster_px;
    let max_px = (max_width_m / m_per_px).max(16.0);
    let (laid_out, _) = glyphs.layout_line(text, raster_px, max_px);

    let right_unit = quat_rotate_vec3(panel.orientation, [1.0, 0.0, 0.0]);
    let up_unit = quat_rotate_vec3(panel.orientation, [0.0, 1.0, 0.0]);
    let base_world = vec3_add(
        panel.position,
        quat_rotate_vec3(panel.orientation, local_origin),
    );
    // Align vertical origin near text cap-center
    let top_y_m = height_m * 0.72;

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
            border_color: [0.0, 0.0, 0.0, 0.0],
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
            emissive: 0.32,
        });
    }
}
