// SPDX-License-Identifier: GPL-3.0-or-later
//! Windowless spatial shelf, magnetic input, marking menu, pocket and Quick Look.

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

pub const CARDS_PER_PAGE: usize = 6;
const GRID_COLS: usize = 3;
pub const CARD_TITLE_M: f32 = 0.033;
pub const HIT_SLOP_RAD: f32 = 0.056; // 3.2 degree magnetic capture cone

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

fn col_card_surface() -> [f32; 4] {
    qk_rgba(255, 255, 255, 0.80)
}
fn col_card_hover() -> [f32; 4] {
    qk_rgba(255, 255, 255, 0.95)
}
fn col_card_selected() -> [f32; 4] {
    qk_rgba(232, 243, 255, 0.94)
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
    pub secondary_pressed: bool,
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

#[derive(Debug, Clone, PartialEq)]
enum HitAction {
    NavigateLocation(String),
    ToggleView(ViewMode),
    PageDelta(isize),
    SelectEntry(usize),
    QuickLookEntry(usize),
    NavigateBack,
    StartPrompt(PromptMode),
    PromptBackspace,
    PromptCommit,
    PromptCancel,
    InspectorOpen,
    InspectorTrashOrRestore,
    UndoLast,
    StashSelected,
    GrabPocket,
    DeployPocket,
    ClearPocket,
    QuickLookClose,
    OrbitModel,
    RadialSector(usize),
    DismissRadial,
}

#[derive(Debug, Clone)]
struct StableHover {
    index: usize,
    action: HitAction,
    point: [f32; 3],
    ray_dir: [f32; 3],
    at: Instant,
}
#[derive(Debug, Clone)]
struct PressHold {
    at: Instant,
    point: [f32; 3],
    action: Option<HitAction>,
    fired: bool,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InputLayer {
    Shelf,
    QuickLook,
    Radial,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RadialSource {
    Secondary,
    LongPress,
}
#[derive(Debug, Clone)]
struct RadialMenuState {
    hand: usize,
    anchor_world: [f32; 3],
    center_ray_dir: [f32; 3],
    opened_at: Instant,
    latched_open: bool,
    active_sector: Option<usize>,
    orientation: [f32; 4],
    source: RadialSource,
    on_entry: bool,
    sort_hidden: bool,
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
    via_squeeze: bool,
    from_pocket: bool,
}

#[derive(Debug, Clone)]
struct WorkspaceGrab {
    hand: usize,
    start_ray_dir: [f32; 3],
    base_yaw: f32,
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
    pub quick_look_open: bool,
    pub radial_menu_open: bool,
    pub radial_active_sector: Option<usize>,
    pub drop_pocket: Vec<String>,
    pub holo_yaw_pitch: [f32; 2],
    pub cards_per_page: usize,
    pub press_lock_recoveries: u64,
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
    radial_menu: Option<RadialMenuState>,
    pub drop_pocket: Vec<PathBuf>,
    pub quick_look_open: bool,
    pub holo_yaw_pitch: [f32; 2],
    pub holo_orbit_drag: Option<(usize, [f32; 3], [f32; 2])>,
    pub page_scroll_offset: f32,
    pub press_lock_recoveries: u64,
    holo_user_rotated: bool,
    prev_secondary: [bool; 2],
    recent_stable_hover: [Option<StableHover>; 2],
    press_hold: [Option<PressHold>; 2],
    hit_layer: InputLayer,
    control_lifts: Vec<f32>,
    path_pill_width: f32,
    pending_haptics: [Option<HapticEvent>; 2],

    start_instant: Instant,
    last_step_instant: Instant,
}

impl SpatialFilesApp {
    pub fn new(initial_path: Option<PathBuf>) -> Self {
        let initial_file = initial_path.as_ref().filter(|p| p.is_file()).cloned();
        let start_loc = initial_path
            .filter(|p| p.exists())
            .map(|p| {
                if p.is_dir() {
                    p
                } else {
                    p.parent()
                        .map(Path::to_path_buf)
                        .unwrap_or_else(fs_ops::home_dir)
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
            radial_menu: None,
            drop_pocket: Vec::new(),
            quick_look_open: false,
            holo_yaw_pitch: [0.0; 2],
            holo_orbit_drag: None,
            page_scroll_offset: 0.0,
            holo_user_rotated: false,
            prev_secondary: [false; 2],
            recent_stable_hover: [None, None],
            press_hold: [None, None],
            hit_layer: InputLayer::Shelf,
            control_lifts: Vec::new(),
            path_pill_width: 0.60,
            press_lock_recoveries: 0,
            pending_haptics: [None, None],
            start_instant: now,
            last_step_instant: now,
        };
        app.reload_directory();
        if let Some(target_file) = initial_file {
            app.select_by_path(&target_file);
            app.quick_look_open = true;
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

    /// Aspect-preserving virtual view of the shelf and its independent ornaments.
    pub fn embedded_eye(width: u32, height: u32) -> EyeView {
        let aspect = (width.max(1) as f32 / height.max(1) as f32).clamp(0.6, 4.0);
        let tan_x0 = 0.84f32 / 1.24f32;
        let tan_y0 = 0.49f32 / 1.24f32;
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
        if self.quick_look_open
            && self
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
            || self.page_scroll_offset.abs() > 0.0002
            || self.prev_trigger.iter().any(|p| *p)
            || self.prev_secondary.iter().any(|p| *p)
            || self.radial_menu.is_some()
            || self.hovered_hit.iter().any(Option::is_some)
            || self.control_lifts.iter().any(|lift| *lift > 0.0002)
            || (self.path_pill_width
                - if self.prompt_mode.is_some() {
                    0.68
                } else {
                    0.60
                })
            .abs()
                > 0.0002
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
            quick_look_open: self.quick_look_open,
            radial_menu_open: self.radial_menu.is_some(),
            radial_active_sector: self.radial_menu.as_ref().and_then(|m| m.active_sector),
            drop_pocket: self
                .drop_pocket
                .iter()
                .map(|p| p.to_string_lossy().into_owned())
                .collect(),
            holo_yaw_pitch: self.holo_yaw_pitch,
            cards_per_page: CARDS_PER_PAGE,
            press_lock_recoveries: self.press_lock_recoveries,
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
        self.radial_menu = None;
        self.quick_look_open = false;
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
        self.radial_menu = None;
        self.quick_look_open = false;
        self.holo_orbit_drag = None;
        self.recent_stable_hover = [None, None];
        self.last_click = [None, None];
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

    pub fn page_count(&self) -> usize {
        self.filtered_indices.len().div_ceil(CARDS_PER_PAGE).max(1)
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
        anyhow::ensure!(
            self.current_location != TRASH_URI,
            "不能在回收站内新建文件夹"
        );
        let (created, undo) = fs_ops::create_directory(Path::new(&self.current_location), name)?;
        self.undo_stack.push(undo);
        self.reload_directory();
        self.select_by_path(&created);
        self.status_message = format!("已新建：{}", name.trim());
        Ok(created)
    }

    pub fn rename_selected(&mut self, new_name: &str) -> Result<PathBuf> {
        anyhow::ensure!(self.current_location != TRASH_URI, "请先还原再重命名");
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
                let name = if text.is_empty() {
                    "新建文件夹"
                } else {
                    &text
                };
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
        if self.radial_menu.take().is_some() {
            return;
        }
        if self.quick_look_open {
            self.quick_look_open = false;
            self.holo_orbit_drag = None;
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
        let page = (self.page as isize + delta).clamp(0, (total - 1).max(0)) as usize;
        if page != self.page {
            self.page_scroll_offset = if page > self.page { 0.28 } else { -0.28 };
            self.page = page;
            self.treemap_dirty = true;
            self.last_click = [None, None];
        }
    }

    // ========================================================================
    // Per-Frame Simulation & Icon-First Qingkong Scene Builder
    // ========================================================================

    pub fn toggle_quick_look(&mut self) {
        if self.primary_selected.is_some() {
            self.quick_look_open = !self.quick_look_open;
            self.radial_menu = None;
            self.holo_orbit_drag = None;
        }
    }

    pub fn stash_selected(&mut self) {
        if self.current_location == TRASH_URI {
            self.status_message = "请先还原回收站中的项目".into();
            return;
        }
        self.stash_paths(self.selected_paths());
    }

    fn stash_paths(&mut self, paths: Vec<PathBuf>) {
        for path in paths {
            if path.exists() && !self.drop_pocket.contains(&path) {
                self.drop_pocket.push(path);
            }
        }
        self.status_message = format!("暂存 {} 项", self.drop_pocket.len());
    }

    pub fn deploy_pocket(&mut self) -> Result<()> {
        anyhow::ensure!(self.current_location != TRASH_URI, "不能投放到回收站");
        anyhow::ensure!(!self.drop_pocket.is_empty(), "暂存囊为空");
        let undo = fs_ops::move_paths_to_dir(&self.drop_pocket, Path::new(&self.current_location))?;
        self.undo_stack.push(undo);
        self.drop_pocket.clear();
        self.reload_directory();
        self.status_message = "已投放，可撤销".into();
        Ok(())
    }

    fn main_pose(&self, eyes: &[EyeView; 2]) -> Pose3D {
        if self.embedded_panel {
            Pose3D {
                position: [0.0, 1.60, -1.24],
                orientation: [0.0, 0.0, 0.0, 1.0],
            }
        } else {
            arc_panel_pose(
                self.anchor_pos,
                self.anchor_yaw,
                0.0,
                0.0,
                self.workspace_radius,
                center_eye_position(eyes),
            )
        }
    }

    pub fn step_and_build_scene(
        &mut self,
        eyes: &[EyeView; 2],
        hands: &[HandControllerInput; 2],
        glyphs: &mut GlyphAtlas,
    ) -> FrameScene {
        self.step_at(eyes, hands, glyphs, Instant::now())
    }

    fn step_at(
        &mut self,
        eyes: &[EyeView; 2],
        hands: &[HandControllerInput; 2],
        glyphs: &mut GlyphAtlas,
        now: Instant,
    ) -> FrameScene {
        let dt = now
            .saturating_duration_since(self.last_step_instant)
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
        for &idx in self
            .filtered_indices
            .iter()
            .skip(self.page * CARDS_PER_PAGE)
            .take(CARDS_PER_PAGE)
        {
            self.previewer.request_thumbnail(&self.all_entries[idx]);
        }
        for (path, mips) in self.previewer.poll() {
            let slot = if let Some(&slot) = self.thumb_slot_by_path.get(&path) {
                slot
            } else {
                let cap = (THUMB_LAYERS - BUILTIN_ICON_COUNT).max(1);
                let slot = BUILTIN_ICON_COUNT + self.next_photo_slot % cap;
                self.next_photo_slot = self.next_photo_slot.wrapping_add(1);
                self.thumb_slot_by_path.retain(|_, value| *value != slot);
                self.thumb_slot_by_path.insert(path, slot);
                slot
            };
            self.pending_gpu_thumbs.push((slot, mips));
        }
        if let Some(inspect) = &self.previewer.active_inspect {
            if inspect.image.is_some()
                && self.last_uploaded_inspect_path.as_ref() != Some(&inspect.path)
            {
                self.last_uploaded_inspect_path = Some(inspect.path.clone());
                self.pending_gpu_inspect_image = inspect.image.clone();
            }
        }
        let pose = self.main_pose(eyes);
        self.process_controller_inputs(hands, &pose, eye_center, now, dt);
        self.page_scroll_offset *= (-16.0 * dt).exp();
        if self.page_scroll_offset.abs() < 0.0002 {
            self.page_scroll_offset = 0.0;
        }
        let spring = (1.0 - (-18.0 * dt).exp()).clamp(0.01, 1.0);
        simd::lerp_f32_slice_in_place(&mut self.slot_z_cur, &self.slot_z_tgt, spring);
        simd::lerp_f32_slice_in_place(&mut self.treemap_lift_cur, &self.treemap_lift_tgt, spring);
        for (i, lift) in self.control_lifts.iter_mut().enumerate() {
            let target = if self.hovered_hit.contains(&Some(i)) {
                0.028
            } else {
                0.0
            };
            *lift += (target - *lift) * spring;
            if (*lift - target).abs() < 0.0002 {
                *lift = target;
            }
        }
        let path_width = if self.prompt_mode.is_some() {
            0.68
        } else {
            0.60
        };
        self.path_pill_width += (path_width - self.path_pill_width) * spring;
        self.hit_batch.clear();
        self.hit_actions.clear();
        self.hit_layer = InputLayer::Shelf;
        let mut scene = FrameScene::default();
        self.emit_ornaments(&mut scene, &pose, glyphs);
        let mut shelf = pose;
        shelf.position = local_world(
            &pose,
            [
                self.page_scroll_offset,
                0.0,
                if self.quick_look_open { -0.045 } else { 0.0 },
            ],
        );
        match self.view_mode {
            ViewMode::Workbench => self.emit_spatial_shelf(&mut scene, &shelf, glyphs),
            ViewMode::Treemap3D => {
                self.emit_spatial_treemap(&mut scene, &shelf, eye_center, glyphs)
            }
        }
        self.emit_pocket(&mut scene, &pose, glyphs);
        if self.quick_look_open {
            self.hit_layer = InputLayer::QuickLook;
            self.emit_quick_look(&mut scene, &pose, glyphs);
        }
        if self.radial_menu.is_some() {
            self.hit_layer = InputLayer::Radial;
            self.emit_radial_menu(&mut scene, glyphs);
        }
        self.hit_batch.finalize_padding();
        self.update_hover_and_emit_hands(&mut scene, eye_center, hands, glyphs);
        scene
    }

    fn process_controller_inputs(
        &mut self,
        hands: &[HandControllerInput; 2],
        pose: &Pose3D,
        eye: [f32; 3],
        now: Instant,
        dt: f32,
    ) {
        for (h, ctrl) in hands.iter().copied().enumerate() {
            let down = ctrl.trigger_pressed && !self.prev_trigger[h];
            let up = !ctrl.trigger_pressed && self.prev_trigger[h];
            let secondary_down = ctrl.secondary_pressed && !self.prev_secondary[h];
            let secondary_up = !ctrl.secondary_pressed && self.prev_secondary[h];
            let grip_down = ctrl.squeeze_pressed && !self.prev_squeeze[h];
            let grip_up = !ctrl.squeeze_pressed && self.prev_squeeze[h];
            self.prev_trigger[h] = ctrl.active && ctrl.trigger_pressed;
            self.prev_secondary[h] = ctrl.active && ctrl.secondary_pressed;
            self.prev_squeeze[h] = ctrl.active && ctrl.squeeze_pressed;
            if !ctrl.active {
                self.recent_stable_hover[h] = None;
                self.press_hold[h] = None;
                if self.drag_session.as_ref().is_some_and(|d| d.hand == h) {
                    self.drag_session = None;
                }
                if self.radial_menu.as_ref().is_some_and(|m| m.hand == h) {
                    self.radial_menu = None;
                }
                if self.holo_orbit_drag.as_ref().is_some_and(|d| d.0 == h) {
                    self.holo_orbit_drag = None;
                }
                continue;
            }
            if ctrl.back_just_pressed {
                self.cancel_modals();
                continue;
            }
            if ctrl.thumb_click_just_pressed {
                self.request_recenter();
            }
            let mut pick = self
                .hit_batch
                .pick_ray(ctrl.aim_origin, ctrl.aim_dir, 0.06, 8.0);
            if down && pick.is_none_or(|p| !p.direct) {
                if let Some(stable) = &self.recent_stable_hover[h] {
                    if now.saturating_duration_since(stable.at) <= Duration::from_millis(120)
                        && angle_between(stable.ray_dir, ctrl.aim_dir) <= 2.4f32.to_radians()
                        && self.hit_actions.get(stable.index) == Some(&stable.action)
                    {
                        self.press_lock_recoveries += 1;
                        pick = Some(simd::RayPickHit {
                            index: stable.index,
                            distance: vec3_len(vec3_sub(stable.point, ctrl.aim_origin)),
                            hit_point: stable.point,
                            direct: false,
                            border_distance: 0.0,
                        });
                    }
                }
            }
            let action = pick.and_then(|p| self.hit_actions.get(p.index).cloned());
            if !ctrl.trigger_pressed && !ctrl.secondary_pressed && self.radial_menu.is_none() {
                if let Some(p) = pick {
                    if let Some(action) = self.hit_actions.get(p.index).cloned() {
                        self.recent_stable_hover[h] = Some(StableHover {
                            index: p.index,
                            action,
                            point: p.hit_point,
                            ray_dir: ctrl.aim_dir,
                            at: now,
                        });
                    }
                }
            }
            if self.radial_menu.is_some() {
                let (owner, latched, source) = {
                    let menu = self.radial_menu.as_ref().unwrap();
                    (menu.hand, menu.latched_open, menu.source)
                };
                if latched && down {
                    if let Some(HitAction::RadialSector(sector)) = action {
                        let menu = self.radial_menu.take().unwrap();
                        self.execute_radial_sector(sector, menu.on_entry, false);
                    } else {
                        self.radial_menu = None;
                    }
                    self.press_hold[h] = None;
                    continue;
                }
                if owner == h && !latched {
                    let menu = self.radial_menu.as_mut().unwrap();
                    let rot = menu.orientation;
                    let right = quat_rotate_vec3(rot, [1.0, 0.0, 0.0]);
                    let up_axis = quat_rotate_vec3(rot, [0.0, 1.0, 0.0]);
                    let normal = quat_rotate_vec3(rot, [0.0, 0.0, 1.0]);
                    let center = ray_plane_point(
                        ctrl.aim_origin,
                        menu.center_ray_dir,
                        menu.anchor_world,
                        normal,
                    )
                    .unwrap_or(menu.anchor_world);
                    let offset =
                        ray_plane_point(ctrl.aim_origin, ctrl.aim_dir, menu.anchor_world, normal)
                            .map(|p| vec3_sub(p, center))
                            .unwrap_or([0.0; 3]);
                    let xy = if ctrl.stick[0].hypot(ctrl.stick[1]) > 0.24 {
                        [ctrl.stick[0] * 0.115, ctrl.stick[1] * 0.115]
                    } else {
                        [vec3_dot(offset, right), vec3_dot(offset, up_axis)]
                    };
                    let sector = radial_sector(xy);
                    if sector != menu.active_sector {
                        self.pending_haptics[h] = Some(HapticEvent::HoverTick);
                        menu.active_sector = sector;
                    }
                    menu.sort_hidden = menu.active_sector == Some(4)
                        && now.saturating_duration_since(menu.opened_at)
                            >= Duration::from_millis(700);
                    let released = match source {
                        RadialSource::Secondary => secondary_up,
                        RadialSource::LongPress => up,
                    };
                    if released {
                        let mut menu = self.radial_menu.take().unwrap();
                        if menu.active_sector.is_none()
                            && source == RadialSource::Secondary
                            && now.saturating_duration_since(menu.opened_at)
                                < Duration::from_millis(240)
                        {
                            menu.latched_open = true;
                            self.radial_menu = Some(menu);
                        } else if let Some(sector) = menu.active_sector {
                            self.execute_radial_sector(sector, menu.on_entry, menu.sort_hidden);
                        }
                        self.press_hold[h] = None;
                    }
                }
                continue;
            }
            if let Some((owner, start_dir, base)) = self.holo_orbit_drag {
                if owner == h {
                    if ctrl.trigger_pressed {
                        self.holo_yaw_pitch = [
                            base[0] + ray_yaw_delta(start_dir, ctrl.aim_dir) * 4.0,
                            base[1]
                                + (ctrl.aim_dir[1].clamp(-1.0, 1.0).asin()
                                    - start_dir[1].clamp(-1.0, 1.0).asin())
                                    * 4.0,
                        ];
                        self.holo_user_rotated = true;
                    } else {
                        self.holo_orbit_drag = None;
                    }
                    continue;
                }
            }
            if secondary_down {
                self.open_radial(
                    h,
                    ctrl,
                    pick.map(|p| p.hit_point),
                    action.as_ref(),
                    eye,
                    now,
                    RadialSource::Secondary,
                );
                continue;
            }
            let point = ray_plane_point(
                ctrl.aim_origin,
                ctrl.aim_dir,
                pose.position,
                quat_rotate_vec3(pose.orientation, [0.0, 0.0, 1.0]),
            );
            if down {
                let eligible = !self.quick_look_open
                    && (action.is_none() || matches!(action, Some(HitAction::SelectEntry(_))));
                self.press_hold[h] = if eligible {
                    point.map(|point| PressHold {
                        at: now,
                        point,
                        action: action.clone(),
                        fired: false,
                    })
                } else {
                    None
                };
                if let Some(action) = action.clone() {
                    self.execute_hit_press(h, action, pick.unwrap().hit_point, ctrl.aim_dir, now);
                }
            }
            if ctrl.trigger_pressed {
                if let (Some(hold), Some(point)) = (&mut self.press_hold[h], point) {
                    let displacement = vec3_sub(point, hold.point);
                    if !hold.fired
                        && vec3_len(displacement) < 0.018
                        && now.saturating_duration_since(hold.at) >= Duration::from_millis(340)
                    {
                        let action = hold.action.clone();
                        hold.fired = true;
                        self.open_radial(
                            h,
                            ctrl,
                            Some(point),
                            action.as_ref(),
                            eye,
                            now,
                            RadialSource::LongPress,
                        );
                        continue;
                    }
                    if hold.action.is_none() && !hold.fired {
                        let dx = vec3_dot(
                            displacement,
                            quat_rotate_vec3(pose.orientation, [1.0, 0.0, 0.0]),
                        );
                        if dx.abs() > 0.09 {
                            hold.fired = true;
                            self.step_page(if dx < 0.0 { 1 } else { -1 });
                            self.pending_haptics[h] = Some(HapticEvent::HoverTick);
                        }
                    }
                }
                if let (Some(drag), Some(point)) = (&mut self.drag_session, point) {
                    if drag.hand == h
                        && !drag.via_squeeze
                        && vec3_len(vec3_sub(point, drag.start_point)) > 0.042
                    {
                        drag.active_drag = true;
                    }
                }
            }
            if up {
                if self
                    .drag_session
                    .as_ref()
                    .is_some_and(|d| d.hand == h && !d.via_squeeze)
                {
                    self.finish_drag_drop(action.clone());
                }
                self.press_hold[h] = None;
            }
            if grip_down {
                match action {
                    Some(HitAction::SelectEntry(idx)) => {
                        if !self.selected_entry_indices.contains(&idx) {
                            self.select_single(idx);
                        }
                        self.begin_drag(h, pick.unwrap().hit_point, true, false);
                    }
                    Some(HitAction::GrabPocket) => {
                        self.begin_drag(h, pick.unwrap().hit_point, true, true)
                    }
                    _ if !self.embedded_panel => {
                        self.workspace_grab = Some(WorkspaceGrab {
                            hand: h,
                            start_ray_dir: ctrl.aim_dir,
                            base_yaw: self.anchor_yaw,
                        })
                    }
                    _ => {}
                }
            }
            if let Some(grab) = &self.workspace_grab {
                if grab.hand == h && ctrl.squeeze_pressed {
                    self.anchor_yaw =
                        grab.base_yaw + ray_yaw_delta(grab.start_ray_dir, ctrl.aim_dir);
                    self.workspace_radius =
                        (self.workspace_radius - ctrl.stick[1] * dt).clamp(1.0, 2.8);
                }
            }
            if grip_up {
                if self
                    .drag_session
                    .as_ref()
                    .is_some_and(|d| d.hand == h && d.via_squeeze)
                {
                    self.finish_drag_drop(action.clone());
                }
                self.workspace_grab = None;
            }
            if self.workspace_grab.is_none() && now >= self.stick_cooldown_until[h] {
                let d = if ctrl.stick[1] < -0.62 || ctrl.stick[0] > 0.65 {
                    1
                } else if ctrl.stick[1] > 0.62 || ctrl.stick[0] < -0.65 {
                    -1
                } else {
                    0
                };
                if d != 0 {
                    self.step_page(d);
                    self.stick_cooldown_until[h] = now + Duration::from_millis(260);
                }
            }
        }
    }

    fn begin_drag(&mut self, hand: usize, point: [f32; 3], via_squeeze: bool, from_pocket: bool) {
        let paths = if from_pocket {
            self.drop_pocket.clone()
        } else {
            self.selected_paths()
        };
        if paths.is_empty() || self.current_location == TRASH_URI {
            return;
        }
        self.drag_session = Some(DragSession {
            hand,
            primary_label: format!("{} 项", paths.len()),
            source_paths: paths,
            start_point: point,
            active_drag: via_squeeze,
            via_squeeze,
            from_pocket,
        });
    }

    fn open_radial(
        &mut self,
        hand: usize,
        ctrl: HandControllerInput,
        hit: Option<[f32; 3]>,
        action: Option<&HitAction>,
        eye: [f32; 3],
        now: Instant,
        source: RadialSource,
    ) {
        let on_entry = match action {
            Some(HitAction::SelectEntry(idx) | HitAction::QuickLookEntry(idx)) => {
                self.select_single(*idx);
                true
            }
            _ => self.quick_look_open && self.primary_selected.is_some(),
        };
        if !on_entry {
            self.selected_entry_indices.clear();
            self.primary_selected = None;
        }
        let distance = hit
            .map(|p| vec3_len(vec3_sub(p, ctrl.aim_origin)))
            .unwrap_or(1.24);
        let anchor = vec3_add(
            ctrl.aim_origin,
            vec3_scale(ctrl.aim_dir, (distance - 0.12).max(0.35)),
        );
        self.radial_menu = Some(RadialMenuState {
            hand,
            anchor_world: anchor,
            center_ray_dir: ctrl.aim_dir,
            opened_at: now,
            latched_open: false,
            active_sector: None,
            orientation: user_facing_orientation(anchor, eye),
            source,
            on_entry,
            sort_hidden: false,
        });
        self.drag_session = None;
        self.pending_haptics[hand] = Some(HapticEvent::ContextMenu);
    }

    fn radial_sector_enabled(&self, sector: usize) -> bool {
        self.current_location != TRASH_URI || matches!(sector, 1 | 3 | 4)
    }

    fn execute_radial_sector(&mut self, sector: usize, on_entry: bool, hidden: bool) {
        if !self.radial_sector_enabled(sector) {
            return;
        }
        match sector {
            0 => self.open_selected(
                self.primary_selected
                    .is_some_and(|i| self.all_entries[i].kind == FileKind::Video),
            ),
            1 => {
                self.quick_look_open = self.primary_selected.is_some();
            }
            2 if on_entry => self.stash_selected(),
            2 => {
                if let Err(e) = self.deploy_pocket() {
                    self.status_message = e.to_string();
                }
            }
            3 => {
                if let Err(e) = self.trash_or_restore_selected() {
                    self.status_message = e.to_string();
                }
            }
            4 => {
                if hidden {
                    self.show_hidden = !self.show_hidden;
                } else {
                    self.sort_by = self.sort_by.next();
                }
                self.reload_directory();
            }
            5 => self.start_prompt(if on_entry {
                PromptMode::Rename
            } else {
                PromptMode::NewFolder
            }),
            _ => {}
        }
    }

    fn start_prompt(&mut self, mode: PromptMode) {
        self.quick_look_open = false;
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

    fn execute_hit_press(
        &mut self,
        hand: usize,
        action: HitAction,
        point: [f32; 3],
        dir: [f32; 3],
        now: Instant,
    ) {
        self.pending_haptics[hand] = Some(HapticEvent::PrimaryClick);
        match action {
            HitAction::SelectEntry(idx) => {
                let double = self.last_click[hand].as_ref().is_some_and(|last| {
                    last.entry_index == idx
                        && now.saturating_duration_since(last.at) >= Duration::from_millis(35)
                        && now.saturating_duration_since(last.at) <= Duration::from_millis(480)
                        && angle_between(last.ray_dir, dir) <= 2.5f32.to_radians()
                });
                self.select_single(idx);
                if double {
                    self.last_click[hand] = None;
                    self.drag_session = None;
                    self.open_selected(false);
                } else {
                    self.last_click[hand] = Some(ClickStamp {
                        entry_index: idx,
                        ray_dir: dir,
                        at: now,
                    });
                    self.begin_drag(hand, point, false, false);
                }
            }
            HitAction::QuickLookEntry(idx) => {
                self.select_single(idx);
                self.quick_look_open = true;
            }
            HitAction::NavigateLocation(path) => self.navigate_to(path),
            HitAction::NavigateBack => {
                if let Some(previous) = self.back_stack.pop() {
                    self.forward_stack.push(self.current_location.clone());
                    self.current_location = previous;
                    self.quick_look_open = false;
                    self.page = 0;
                    self.reload_directory();
                } else {
                    self.navigate_up();
                }
            }
            HitAction::ToggleView(mode) => self.set_view_mode(mode),
            HitAction::PageDelta(delta) => self.step_page(delta),
            HitAction::StartPrompt(mode) => self.start_prompt(mode),
            HitAction::PromptBackspace => self.handle_backspace(),
            HitAction::PromptCommit => self.commit_prompt(),
            HitAction::PromptCancel => {
                self.prompt_mode = None;
            }
            HitAction::InspectorOpen => self.open_selected(
                self.primary_selected
                    .is_some_and(|i| self.all_entries[i].kind == FileKind::Video),
            ),
            HitAction::InspectorTrashOrRestore => {
                let _ = self.trash_or_restore_selected();
            }
            HitAction::UndoLast => {
                let _ = self.undo_last();
            }
            HitAction::StashSelected => self.stash_selected(),
            HitAction::GrabPocket => self.begin_drag(hand, point, false, true),
            HitAction::DeployPocket => {
                if let Err(e) = self.deploy_pocket() {
                    self.status_message = e.to_string();
                }
            }
            HitAction::ClearPocket => {
                self.drop_pocket.clear();
            }
            HitAction::QuickLookClose => {
                self.quick_look_open = false;
                self.holo_orbit_drag = None;
            }
            HitAction::OrbitModel => {
                self.holo_orbit_drag = Some((hand, dir, self.holo_yaw_pitch));
            }
            HitAction::RadialSector(sector) => {
                if let Some(menu) = self.radial_menu.take() {
                    self.execute_radial_sector(sector, menu.on_entry, menu.sort_hidden);
                }
            }
            HitAction::DismissRadial => {
                self.radial_menu = None;
            }
        }
    }

    fn finish_drag_drop(&mut self, action: Option<HitAction>) {
        let Some(drag) = self.drag_session.take().filter(|d| d.active_drag) else {
            return;
        };
        if drag.from_pocket {
            if matches!(
                action,
                None | Some(HitAction::SelectEntry(_) | HitAction::DeployPocket)
            ) {
                if let Err(e) = self.deploy_pocket() {
                    self.status_message = e.to_string();
                }
            }
            return;
        }
        if matches!(
            action,
            Some(HitAction::GrabPocket | HitAction::DeployPocket)
        ) {
            self.stash_paths(drag.source_paths);
            return;
        }
        let destination = match action {
            Some(HitAction::SelectEntry(i)) => self
                .all_entries
                .get(i)
                .filter(|e| e.kind == FileKind::Directory)
                .map(|e| e.path.clone()),
            Some(HitAction::NavigateLocation(path)) if path != TRASH_URI => {
                Some(PathBuf::from(path))
            }
            _ => None,
        };
        if let Some(dest) = destination {
            match fs_ops::move_paths_to_dir(&drag.source_paths, &dest) {
                Ok(undo) => {
                    self.undo_stack.push(undo);
                    self.reload_directory();
                }
                Err(e) => self.status_message = e.to_string(),
            }
        }
    }

    fn register_hit(&mut self, center: [f32; 3], half: [f32; 3], action: HitAction) -> usize {
        if (self.radial_menu.is_some() && self.hit_layer != InputLayer::Radial)
            || (self.quick_look_open && self.hit_layer == InputLayer::Shelf)
        {
            return usize::MAX;
        }
        let min = vec3_sub(center, half);
        let max = vec3_add(center, half);
        let index = self.hit_batch.push(min, max, HIT_SLOP_RAD);
        self.hit_actions.push(action);
        index
    }

    fn is_hit_hovered(&self, index: usize) -> bool {
        self.hovered_hit.contains(&Some(index))
    }

    fn emit_orb(
        &mut self,
        scene: &mut FrameScene,
        panel: &Pose3D,
        at: [f32; 3],
        diameter: f32,
        icon: UiIcon,
        action: HitAction,
        active: bool,
    ) -> usize {
        let index = self.register_hit(
            local_world(panel, at),
            [diameter * 0.5, diameter * 0.5, 0.022],
            action,
        );
        let hover = self.is_hit_hovered(index);
        let mut position = at;
        if index != usize::MAX {
            self.control_lifts
                .resize(self.control_lifts.len().max(index + 1), 0.0);
            position[2] += self.control_lifts[index];
        }
        if hover {
            push_panel_glass(
                &mut scene.quads,
                panel,
                [position[0], position[1], position[2] - 0.003],
                [diameter + 0.014; 2],
                (diameter + 0.014) * 0.5,
                qk_rgba(55, 144, 250, 0.10),
                qk_rgba(
                    55,
                    144,
                    250,
                    0.46 + 0.12 * (self.start_instant.elapsed().as_secs_f32() * 3.0).sin(),
                ),
            );
        }
        push_panel_glass(
            &mut scene.quads,
            panel,
            position,
            [diameter; 2],
            diameter * 0.5,
            if active {
                col_kylin_blue()
            } else if hover {
                col_card_hover()
            } else {
                col_orb_idle()
            },
            col_rim_luminous(),
        );
        position[2] += 0.005;
        push_ui_icon(
            &mut scene.quads,
            panel,
            position,
            diameter * 0.50,
            icon,
            if active {
                col_pure_white()
            } else {
                col_ink_navy()
            },
        );
        index
    }

    fn emit_ornaments(&mut self, scene: &mut FrameScene, panel: &Pose3D, glyphs: &mut GlyphAtlas) {
        push_panel_glass(
            &mut scene.quads,
            panel,
            [-0.68, 0.02, 0.026],
            [0.104, 0.52],
            0.052,
            qk_rgba(239, 247, 255, 0.65),
            qk_rgba(255, 255, 255, 0.35),
        );
        let nav: Vec<_> = ["HOME", "DOCS", "PICS", "VIDS", "DOWN", "TRSH"]
            .iter()
            .filter_map(|tag| self.bookmarks.iter().find(|b| b.icon_tag == *tag).cloned())
            .collect();
        for (i, bookmark) in nav.iter().enumerate() {
            let y = 0.22 - i as f32 * 0.080;
            let index = self.emit_orb(
                scene,
                panel,
                [-0.68, y, 0.032],
                0.072,
                UiIcon::from_bookmark_tag(&bookmark.icon_tag),
                HitAction::NavigateLocation(bookmark.target.clone()),
                self.current_location == bookmark.target,
            );
            if self.is_hit_hovered(index) {
                push_panel_glass(
                    &mut scene.quads,
                    panel,
                    [-0.50, y, 0.075],
                    [0.23, 0.055],
                    0.0275,
                    col_card_hover(),
                    col_rim_luminous(),
                );
                push_panel_text_centered(
                    &mut scene.quads,
                    panel,
                    [-0.50, y - 0.009, 0.079],
                    &bookmark.label,
                    0.026,
                    0.21,
                    col_ink_navy(),
                    FontWeight::Bold,
                    glyphs,
                );
            }
        }
        self.emit_orb(
            scene,
            panel,
            [-0.55, 0.345, 0.024],
            0.068,
            UiIcon::BackLeft,
            HitAction::NavigateBack,
            false,
        );
        let editing = self.prompt_mode.is_some();
        let width = self.path_pill_width;
        push_panel_glass(
            &mut scene.quads,
            panel,
            [0.005, 0.345, 0.024],
            [width, 0.068],
            0.034,
            col_card_surface(),
            qk_rgba(255, 255, 255, 0.50),
        );
        if editing {
            push_panel_text(
                &mut scene.quads,
                panel,
                [-0.30, 0.333, 0.030],
                &format!("{}│", self.prompt_buffer),
                0.036,
                0.43,
                col_ink_navy(),
                FontWeight::Bold,
                glyphs,
            );
            for (x, icon, action) in [
                (0.19, UiIcon::BackLeft, HitAction::PromptBackspace),
                (0.275, UiIcon::CheckMark, HitAction::PromptCommit),
                (0.36, UiIcon::CloseCross, HitAction::PromptCancel),
            ] {
                self.emit_orb(scene, panel, [x, 0.345, 0.036], 0.068, icon, action, false);
            }
        } else {
            let title = if self.current_location == TRASH_URI {
                "回收站".into()
            } else {
                Path::new(&self.current_location)
                    .file_name()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "文件".into())
            };
            push_textured_quad(
                &mut scene.quads,
                panel,
                [-0.30, 0.345, 0.032],
                [0.043, 0.043],
                0.008,
                2,
                0.0,
            );
            push_panel_text(
                &mut scene.quads,
                panel,
                [-0.263, 0.333, 0.032],
                &title,
                0.038,
                0.49,
                col_ink_navy(),
                FontWeight::Bold,
                glyphs,
            );
            self.emit_orb(
                scene,
                panel,
                [0.31, 0.345, 0.034],
                0.068,
                UiIcon::Search,
                HitAction::StartPrompt(PromptMode::Search),
                false,
            );
            let (icon, next) = match self.view_mode {
                ViewMode::Workbench => (UiIcon::Cube3DView, ViewMode::Treemap3D),
                ViewMode::Treemap3D => (UiIcon::GridView, ViewMode::Workbench),
            };
            self.emit_orb(
                scene,
                panel,
                [0.46, 0.345, 0.024],
                0.068,
                icon,
                HitAction::ToggleView(next),
                false,
            );
            if !self.undo_stack.is_empty() {
                self.emit_orb(
                    scene,
                    panel,
                    [0.55, 0.345, 0.024],
                    0.068,
                    UiIcon::UndoCurved,
                    HitAction::UndoLast,
                    false,
                );
            }
        }
        let pages = self.page_count();
        push_panel_glass(
            &mut scene.quads,
            panel,
            [0.02, -0.355, 0.022],
            [0.28, 0.068],
            0.034,
            qk_rgba(245, 250, 255, 0.45),
            qk_rgba(255, 255, 255, 0.30),
        );
        self.register_hit(
            local_world(panel, [-0.05, -0.355, 0.025]),
            [0.07, 0.034, 0.015],
            HitAction::PageDelta(-1),
        );
        self.register_hit(
            local_world(panel, [0.09, -0.355, 0.025]),
            [0.07, 0.034, 0.015],
            HitAction::PageDelta(1),
        );
        let visible = pages.min(9);
        let first = self
            .page
            .saturating_sub(4)
            .min(pages.saturating_sub(visible));
        for i in 0..visible {
            let current = first + i == self.page;
            let x = 0.02 + (i as f32 - (visible - 1) as f32 * 0.5) * 0.024;
            push_panel_glass(
                &mut scene.quads,
                panel,
                [x, -0.355, 0.030],
                [if current { 0.036 } else { 0.012 }, 0.012],
                0.006,
                if current {
                    col_kylin_blue()
                } else {
                    qk_rgba(95, 120, 150, 0.38)
                },
                [0.0; 4],
            );
        }
    }

    fn emit_spatial_shelf(
        &mut self,
        scene: &mut FrameScene,
        panel: &Pose3D,
        glyphs: &mut GlyphAtlas,
    ) {
        let indices: Vec<_> = self
            .filtered_indices
            .iter()
            .skip(self.page * CARDS_PER_PAGE)
            .take(CARDS_PER_PAGE)
            .copied()
            .collect();
        if indices.is_empty() {
            push_panel_text_centered(
                &mut scene.quads,
                panel,
                [0.0, 0.0, 0.006],
                if self.search_query.is_empty() {
                    "此文件夹为空"
                } else {
                    "没有匹配的文件"
                },
                0.033,
                0.8,
                col_ink_navy(),
                FontWeight::Bold,
                glyphs,
            );
        }
        for (slot, idx) in indices.into_iter().enumerate() {
            let entry = self.all_entries[idx].clone();
            let x = (slot % GRID_COLS) as f32 * 0.396 - 0.396;
            let y = 0.15 - (slot / GRID_COLS) as f32 * 0.301;
            let lift = self.slot_z_cur[slot];
            let hit = self.register_hit(
                local_world(panel, [x, y, lift]),
                [0.180, 0.1325, 0.022],
                HitAction::SelectEntry(idx),
            );
            let hover = self.is_hit_hovered(hit);
            let selected = self.selected_entry_indices.contains(&idx);
            self.slot_z_tgt[slot] = if hover {
                0.028
            } else if selected {
                0.010
            } else {
                0.0
            };
            if hover {
                let pulse = 0.46 + 0.12 * (self.start_instant.elapsed().as_secs_f32() * 3.0).sin();
                push_panel_glass(
                    &mut scene.quads,
                    panel,
                    [x, y, lift - 0.004],
                    [0.374, 0.279],
                    0.031,
                    qk_rgba(55, 144, 250, 0.07),
                    qk_rgba(55, 144, 250, pulse),
                );
            }
            push_panel_glass(
                &mut scene.quads,
                panel,
                [x, y, lift],
                [0.360, 0.265],
                0.026,
                if selected {
                    col_card_selected()
                } else {
                    col_card_surface()
                },
                if hover {
                    col_kylin_blue()
                } else {
                    qk_rgba(255, 255, 255, 0.40)
                },
            );
            if entry.kind == FileKind::Directory {
                push_textured_quad(
                    &mut scene.quads,
                    panel,
                    [x, y + 0.018, lift + 0.009],
                    [0.156, 0.156],
                    0.012,
                    2,
                    0.0,
                );
                if hover {
                    self.previewer.request_directory_peek(&entry.path);
                    let children = self
                        .previewer
                        .dir_peek_cache
                        .get(&entry.path)
                        .cloned()
                        .unwrap_or_default();
                    for (i, (path, kind)) in children.iter().enumerate() {
                        let fan = Pose3D {
                            position: local_world(
                                panel,
                                [
                                    x + (i as f32 - 1.0) * 0.052,
                                    y + 0.070,
                                    lift + 0.018 + i as f32 * 0.008,
                                ],
                            ),
                            orientation: quat_mul(
                                panel.orientation,
                                quat_from_yaw_pitch((i as f32 - 1.0) * 0.16, 0.0),
                            ),
                        };
                        push_panel_glass(
                            &mut scene.quads,
                            &fan,
                            [0.0; 3],
                            [0.096, 0.076],
                            0.012,
                            col_card_hover(),
                            col_rim_luminous(),
                        );
                        let texture = self
                            .thumb_slot_by_path
                            .get(path)
                            .copied()
                            .unwrap_or_else(|| icon_slot_for_kind(*kind));
                        push_textured_quad(
                            &mut scene.quads,
                            &fan,
                            [0.0, 0.0, 0.003],
                            [0.092, 0.072],
                            0.010,
                            2,
                            texture as f32,
                        );
                    }
                }
            } else {
                let texture = self
                    .thumb_slot_by_path
                    .get(&entry.path)
                    .copied()
                    .unwrap_or_else(|| icon_slot_for_kind(entry.kind));
                push_textured_quad(
                    &mut scene.quads,
                    panel,
                    [x, y + 0.034, lift + 0.006],
                    [0.352, 0.196],
                    0.026,
                    2,
                    texture as f32,
                );
                push_panel_glass(
                    &mut scene.quads,
                    panel,
                    [x, y - 0.099, lift + 0.008],
                    [0.354, 0.063],
                    0.020,
                    qk_rgba(255, 255, 255, 0.94),
                    [0.0; 4],
                );
            }
            push_panel_text_centered(
                &mut scene.quads,
                panel,
                [x, y - 0.111, lift + 0.012],
                &short_display_stem(&entry),
                CARD_TITLE_M,
                0.316,
                col_ink_navy(),
                FontWeight::Bold,
                glyphs,
            );
            push_ui_icon(
                &mut scene.quads,
                panel,
                [x - 0.144, y + 0.095, lift + 0.013],
                0.040,
                UiIcon::badge_for_kind(entry.kind),
                col_ink_navy(),
            );
            self.emit_orb(
                scene,
                panel,
                [x + 0.137, y + 0.091, lift + 0.025],
                0.068,
                if entry.kind == FileKind::Directory {
                    UiIcon::OpenLaunch
                } else {
                    UiIcon::Cube3DView
                },
                HitAction::QuickLookEntry(idx),
                false,
            );
        }
    }

    fn emit_spatial_treemap(
        &mut self,
        scene: &mut FrameScene,
        panel: &Pose3D,
        eye: [f32; 3],
        glyphs: &mut GlyphAtlas,
    ) {
        if self.treemap_dirty {
            let selected: Vec<_> = self
                .filtered_indices
                .iter()
                .skip(self.page * CARDS_PER_PAGE)
                .take(CARDS_PER_PAGE)
                .copied()
                .collect();
            let entries: Vec<_> = selected
                .iter()
                .map(|i| self.all_entries[*i].clone())
                .collect();
            let mut blocks = treemap3d::build_spatial_treemap(&entries, 1.05, 0.50, CARDS_PER_PAGE);
            for block in &mut blocks {
                block.entry_index = selected[block.entry_index];
            }
            self.cached_treemap = blocks;
            self.treemap_dirty = false;
        }
        let podium = Pose3D {
            position: local_world(panel, [0.0, -0.10, 0.16]),
            orientation: quat_mul(
                panel.orientation,
                quat_from_yaw_pitch(0.0, 28.0f32.to_radians()),
            ),
        };
        push_oriented_box(
            &mut scene.solid_vertices,
            &podium,
            [0.0, -0.012, 0.0],
            [0.56, 0.010, 0.27],
            qk_rgba(235, 243, 252, 1.0),
            0.04,
        );
        for block in self.cached_treemap.clone() {
            let local = [block.x_center, block.height * 0.41, block.z_center];
            let center = local_world(&podium, local);
            let index = self.register_hit(
                center,
                [
                    (block.width * 0.5).max(0.034),
                    (block.height * 0.41).max(0.034),
                    (block.depth * 0.5).max(0.034),
                ],
                HitAction::SelectEntry(block.entry_index),
            );
            let hover = self.is_hit_hovered(index);
            push_oriented_box(
                &mut scene.solid_vertices,
                &podium,
                local,
                [block.width * 0.5, block.height * 0.41, block.depth * 0.5],
                self.all_entries[block.entry_index].kind.accent_rgba(),
                if hover { 0.32 } else { 0.05 },
            );
            if hover || self.primary_selected == Some(block.entry_index) {
                let position = vec3_add(center, [0.0, block.height * 0.41 + 0.07, 0.03]);
                let label = Pose3D {
                    position,
                    orientation: user_facing_orientation(position, eye),
                };
                push_panel_glass(
                    &mut scene.quads,
                    &label,
                    [0.0; 3],
                    [0.40, 0.068],
                    0.034,
                    col_card_hover(),
                    col_rim_luminous(),
                );
                push_panel_text_centered(
                    &mut scene.quads,
                    &label,
                    [0.0, -0.011, 0.006],
                    &short_display_stem(&self.all_entries[block.entry_index]),
                    0.033,
                    0.37,
                    col_ink_navy(),
                    FontWeight::Bold,
                    glyphs,
                );
            }
        }
    }

    fn emit_pocket(&mut self, scene: &mut FrameScene, panel: &Pose3D, glyphs: &mut GlyphAtlas) {
        let active = !self.drop_pocket.is_empty()
            || self.drag_session.as_ref().is_some_and(|d| d.active_drag);
        let scale = if active { 1.10 } else { 1.0 };
        self.register_hit(
            local_world(panel, [-0.68, -0.31, 0.030]),
            [0.075 * scale, 0.060 * scale, 0.024],
            HitAction::GrabPocket,
        );
        push_panel_glass(
            &mut scene.quads,
            panel,
            [-0.68, -0.31, 0.030],
            [0.15 * scale, 0.12 * scale],
            0.028,
            if active {
                col_card_selected()
            } else {
                col_orb_idle()
            },
            if active {
                col_kylin_blue()
            } else {
                col_rim_luminous()
            },
        );
        for i in 0..self.drop_pocket.len().min(3) {
            push_panel_glass(
                &mut scene.quads,
                panel,
                [
                    -0.706 + i as f32 * 0.012,
                    -0.293 + i as f32 * 0.007,
                    0.037 + i as f32 * 0.004,
                ],
                [0.045, 0.035],
                0.007,
                col_pure_white(),
                col_kylin_blue(),
            );
        }
        push_panel_text_centered(
            &mut scene.quads,
            panel,
            [-0.68, -0.346, 0.048],
            "暂存囊",
            0.025,
            0.14,
            col_ink_navy(),
            FontWeight::Bold,
            glyphs,
        );
        if !self.drop_pocket.is_empty() {
            push_panel_glass(
                &mut scene.quads,
                panel,
                [-0.72, -0.268, 0.060],
                [0.055, 0.042],
                0.021,
                col_kylin_blue(),
                col_pure_white(),
            );
            push_panel_text_centered(
                &mut scene.quads,
                panel,
                [-0.72, -0.276, 0.065],
                &self.drop_pocket.len().to_string(),
                0.025,
                0.05,
                col_pure_white(),
                FontWeight::Bold,
                glyphs,
            );
            self.emit_orb(
                scene,
                panel,
                [-0.597, -0.273, 0.060],
                0.068,
                UiIcon::CloseCross,
                HitAction::ClearPocket,
                false,
            );
            self.emit_orb(
                scene,
                panel,
                [-0.597, -0.351, 0.060],
                0.068,
                UiIcon::PasteClip,
                HitAction::DeployPocket,
                true,
            );
        }
    }

    fn emit_radial_menu(&mut self, scene: &mut FrameScene, glyphs: &mut GlyphAtlas) {
        let menu = self.radial_menu.as_ref().unwrap();
        let pose = Pose3D {
            position: menu.anchor_world,
            orientation: menu.orientation,
        };
        let active = menu.active_sector;
        let on_entry = menu.on_entry;
        let hidden = menu.sort_hidden;
        self.register_hit(
            local_world(&pose, [0.0, 0.0, -0.025]),
            [0.28, 0.26, 0.010],
            HitAction::DismissRadial,
        );
        push_panel_glass(
            &mut scene.quads,
            &pose,
            [0.0, 0.0, -0.015],
            [0.32; 2],
            0.16,
            qk_rgba(236, 245, 255, 0.74),
            col_rim_luminous(),
        );
        let icons = [
            UiIcon::OpenLaunch,
            UiIcon::Cube3DView,
            UiIcon::PasteClip,
            UiIcon::Trash,
            UiIcon::Sort,
            if on_entry {
                UiIcon::EditRename
            } else {
                UiIcon::NewFolder
            },
        ];
        for (sector, icon) in icons.into_iter().enumerate() {
            if !self.radial_sector_enabled(sector) {
                continue;
            }
            let theta = std::f32::consts::FRAC_PI_2 - sector as f32 * std::f32::consts::PI / 3.0;
            let selected = active == Some(sector);
            self.emit_orb(
                scene,
                &pose,
                [theta.cos() * 0.115, theta.sin() * 0.115, 0.012],
                0.076 * if selected { 1.22 } else { 1.0 },
                icon,
                HitAction::RadialSector(sector),
                selected,
            );
        }
        let label = active
            .filter(|sector| self.radial_sector_enabled(*sector))
            .map(|sector| match sector {
                0 => "打开 / 播放",
                1 => "Quick Look",
                2 if on_entry => "放入暂存囊",
                2 => "投放到此",
                3 if self.current_location == TRASH_URI => "还原",
                3 => "移入回收站",
                4 if hidden && self.show_hidden => "隐藏项目",
                4 if hidden => "显示隐藏项",
                4 => "排序 · 长按显隐",
                5 if on_entry => "重命名",
                _ => "新建文件夹",
            })
            .unwrap_or("取消");
        push_panel_text_centered(
            &mut scene.quads,
            &pose,
            [0.0, -0.009, 0.068],
            label,
            0.029,
            0.27,
            col_ink_navy(),
            FontWeight::Bold,
            glyphs,
        );
    }

    fn emit_quick_look(&mut self, scene: &mut FrameScene, stage: &Pose3D, glyphs: &mut GlyphAtlas) {
        let Some(index) = self
            .primary_selected
            .filter(|i| *i < self.all_entries.len())
        else {
            self.quick_look_open = false;
            return;
        };
        let entry = self.all_entries[index].clone();
        self.register_hit(
            local_world(stage, [0.0, 0.0, 0.035]),
            [0.78, 0.43, 0.008],
            HitAction::QuickLookClose,
        );
        push_panel_glass(
            &mut scene.quads,
            stage,
            [0.0, 0.0, 0.035],
            [1.56, 0.86],
            0.080,
            qk_rgba(20, 31, 47, 0.18),
            [0.0; 4],
        );
        let panel = Pose3D {
            position: local_world(stage, [0.0, 0.0, 0.11]),
            orientation: stage.orientation,
        };
        push_panel_glass(
            &mut scene.quads,
            &panel,
            [0.0; 3],
            [0.88, 0.60],
            0.040,
            qk_rgba(240, 247, 255, 0.94),
            col_rim_luminous(),
        );
        self.register_hit(
            local_world(&panel, [0.0, 0.020, 0.015]),
            [0.415, 0.20, 0.022],
            HitAction::OrbitModel,
        );
        push_panel_text_centered(
            &mut scene.quads,
            &panel,
            [0.0, 0.250, 0.012],
            &short_display_stem(&entry),
            0.036,
            0.77,
            col_ink_navy(),
            FontWeight::Bold,
            glyphs,
        );
        let inspect = self
            .previewer
            .active_inspect
            .as_ref()
            .filter(|p| p.path == entry.path);
        if let Some(inspect) = inspect {
            if let Some(mesh) = &inspect.mesh {
                let auto = if self.holo_user_rotated {
                    0.0
                } else {
                    self.start_instant.elapsed().as_secs_f32() * 0.35
                };
                push_hologram_mesh(
                    &mut scene.solid_vertices,
                    local_world(&panel, [0.0, 0.030, 0.22]),
                    quat_mul(
                        panel.orientation,
                        quat_from_yaw_pitch(
                            auto + self.holo_yaw_pitch[0],
                            0.22 + self.holo_yaw_pitch[1],
                        ),
                    ),
                    0.34,
                    &mesh.vertices,
                    qk_rgba(68, 136, 252, 1.0),
                );
            } else if matches!(entry.kind, FileKind::Image | FileKind::Video)
                && inspect.image.is_some()
            {
                push_textured_quad(
                    &mut scene.quads,
                    &panel,
                    [0.0, 0.018, 0.018],
                    [0.82, 0.405],
                    0.026,
                    3,
                    0.0,
                );
            } else {
                for (row, text) in inspect.text_lines.iter().take(10).enumerate() {
                    push_panel_text(
                        &mut scene.quads,
                        &panel,
                        [-0.386, 0.180 - row as f32 * 0.033, 0.022],
                        text,
                        0.025,
                        0.77,
                        col_ink_navy(),
                        FontWeight::Regular,
                        glyphs,
                    );
                }
                push_panel_glass(
                    &mut scene.quads,
                    &panel,
                    [0.0, -0.174, 0.026],
                    [0.79, 0.045],
                    0.0225,
                    col_card_selected(),
                    [0.0; 4],
                );
                push_panel_text_centered(
                    &mut scene.quads,
                    &panel,
                    [0.0, -0.182, 0.032],
                    &inspect.summary_note,
                    0.025,
                    0.75,
                    col_muted_slate(),
                    FontWeight::Bold,
                    glyphs,
                );
            }
        } else {
            push_panel_text_centered(
                &mut scene.quads,
                &panel,
                [0.0, 0.0, 0.025],
                "正在载入预览…",
                0.033,
                0.7,
                col_ink_navy(),
                FontWeight::Bold,
                glyphs,
            );
        }
        self.emit_orb(
            scene,
            &panel,
            [0.438, 0.296, 0.020],
            0.076,
            UiIcon::CloseCross,
            HitAction::QuickLookClose,
            false,
        );
        for (x, icon, action) in [
            (
                -0.27,
                if entry.kind == FileKind::Video {
                    UiIcon::PlaySolid
                } else {
                    UiIcon::OpenLaunch
                },
                HitAction::InspectorOpen,
            ),
            (-0.09, UiIcon::PasteClip, HitAction::StashSelected),
            (
                0.09,
                UiIcon::EditRename,
                HitAction::StartPrompt(PromptMode::Rename),
            ),
            (
                0.27,
                if self.current_location == TRASH_URI {
                    UiIcon::UndoCurved
                } else {
                    UiIcon::Trash
                },
                HitAction::InspectorTrashOrRestore,
            ),
        ] {
            if self.current_location == TRASH_URI
                && matches!(
                    action,
                    HitAction::StashSelected | HitAction::StartPrompt(_) | HitAction::InspectorOpen
                )
            {
                continue;
            }
            self.emit_orb(
                scene,
                &panel,
                [x, -0.245, 0.020],
                0.076,
                icon,
                action,
                false,
            );
        }
    }

    fn update_hover_and_emit_hands(
        &mut self,
        scene: &mut FrameScene,
        eye: [f32; 3],
        hands: &[HandControllerInput; 2],
        glyphs: &mut GlyphAtlas,
    ) {
        for (hand, ctrl) in hands.iter().enumerate() {
            if !ctrl.active {
                self.hovered_hit[hand] = None;
                continue;
            }
            let pick = self
                .hit_batch
                .pick_ray(ctrl.aim_origin, ctrl.aim_dir, 0.06, 8.0);
            let hover = pick.map(|p| p.index);
            if hover.is_some() && hover != self.hovered_hit[hand] {
                self.pending_haptics[hand].get_or_insert(HapticEvent::HoverTick);
            }
            self.hovered_hit[hand] = hover;
            let distance = pick.map(|p| p.distance).unwrap_or(1.85);
            let end = vec3_add(ctrl.aim_origin, vec3_scale(ctrl.aim_dir, distance));
            if !self.embedded_panel {
                push_beam_segment(
                    &mut scene.solid_vertices,
                    vec3_add(ctrl.aim_origin, vec3_scale(ctrl.aim_dir, 0.14)),
                    end,
                    0.0014,
                    0.0006,
                    qk_rgba(255, 255, 255, 0.72),
                    0.35,
                );
            }
            let dragging = self
                .drag_session
                .as_ref()
                .is_some_and(|d| d.hand == hand && d.active_drag);
            if pick.is_none() || dragging {
                let reticle = Pose3D {
                    position: end,
                    orientation: user_facing_orientation(end, eye),
                };
                push_panel_glass(
                    &mut scene.quads,
                    &reticle,
                    [0.0; 3],
                    [0.014; 2],
                    0.007,
                    qk_rgba(255, 255, 255, 0.48),
                    [0.0; 4],
                );
            }
            if let Some(drag) = &self.drag_session {
                if drag.hand == hand && drag.active_drag {
                    let ghost = Pose3D {
                        position: vec3_add(end, [0.0, 0.06, 0.0]),
                        orientation: user_facing_orientation(end, eye),
                    };
                    push_panel_glass(
                        &mut scene.quads,
                        &ghost,
                        [0.0; 3],
                        [0.22, 0.068],
                        0.034,
                        col_kylin_blue(),
                        col_pure_white(),
                    );
                    push_panel_text_centered(
                        &mut scene.quads,
                        &ghost,
                        [0.0, -0.011, 0.006],
                        &drag.primary_label,
                        0.029,
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

fn local_world(pose: &Pose3D, offset: [f32; 3]) -> [f32; 3] {
    vec3_add(pose.position, quat_rotate_vec3(pose.orientation, offset))
}

fn angle_between(a: [f32; 3], b: [f32; 3]) -> f32 {
    vec3_dot(vec3_normalize(a), vec3_normalize(b))
        .clamp(-1.0, 1.0)
        .acos()
}

fn ray_yaw_delta(start: [f32; 3], end: [f32; 3]) -> f32 {
    let delta = end[0].atan2(-end[2]) - start[0].atan2(-start[2]);
    delta.sin().atan2(delta.cos())
}

fn ray_plane_point(
    origin: [f32; 3],
    dir: [f32; 3],
    point: [f32; 3],
    normal: [f32; 3],
) -> Option<[f32; 3]> {
    let denominator = vec3_dot(dir, normal);
    if denominator.abs() < 1e-6 {
        return None;
    }
    let t = vec3_dot(vec3_sub(point, origin), normal) / denominator;
    (t > 0.0 && t.is_finite()).then(|| vec3_add(origin, vec3_scale(dir, t)))
}

fn radial_sector(offset: [f32; 2]) -> Option<usize> {
    if offset[0].hypot(offset[1]) < 0.032 {
        return None;
    }
    let angle = offset[1].atan2(offset[0]);
    Some(
        ((std::f32::consts::FRAC_PI_2 - angle) / (std::f32::consts::PI / 3.0))
            .round()
            .rem_euclid(6.0) as usize,
    )
}

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
fn push_panel_text_centered(
    quads: &mut Vec<QuadInstance>,
    panel: &Pose3D,
    mut origin: [f32; 3],
    text: &str,
    height: f32,
    width: f32,
    color: [f32; 4],
    weight: FontWeight,
    glyphs: &mut GlyphAtlas,
) {
    let raster = (height * 2600.0).clamp(28.0, 96.0);
    let m_per_px = height / raster;
    let (_, advance) = glyphs.layout_line_weighted(text, raster, width / m_per_px, weight);
    origin[0] -= advance * m_per_px * 0.5;
    push_panel_text(
        quads, panel, origin, text, height, width, color, weight, glyphs,
    );
}

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
    let raster_px = (height_m * 2600.0).clamp(28.0, 96.0);
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
            [
                [-1.0, -1.0, 1.0],
                [1.0, -1.0, 1.0],
                [1.0, 1.0, 1.0],
                [-1.0, 1.0, 1.0],
            ],
        ),
        (
            [0.0, 0.0, -1.0],
            [
                [1.0, -1.0, -1.0],
                [-1.0, -1.0, -1.0],
                [-1.0, 1.0, -1.0],
                [1.0, 1.0, -1.0],
            ],
        ),
        (
            [0.0, 1.0, 0.0],
            [
                [-1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, -1.0],
                [-1.0, 1.0, -1.0],
            ],
        ),
        (
            [0.0, -1.0, 0.0],
            [
                [-1.0, -1.0, -1.0],
                [1.0, -1.0, -1.0],
                [1.0, -1.0, 1.0],
                [-1.0, -1.0, 1.0],
            ],
        ),
        (
            [1.0, 0.0, 0.0],
            [
                [1.0, -1.0, 1.0],
                [1.0, -1.0, -1.0],
                [1.0, 1.0, -1.0],
                [1.0, 1.0, 1.0],
            ],
        ),
        (
            [-1.0, 0.0, 0.0],
            [
                [-1.0, -1.0, -1.0],
                [-1.0, -1.0, 1.0],
                [-1.0, 1.0, 1.0],
                [-1.0, 1.0, -1.0],
            ],
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

#[cfg(test)]
mod xrui_tests {
    use super::*;
    use std::{
        fs,
        time::{SystemTime, UNIX_EPOCH},
    };

    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "xr-files-spatial-{}-{}",
                std::process::id(),
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }
        fn app(&self) -> SpatialFilesApp {
            let mut app = SpatialFilesApp::new(Some(self.0.clone()));
            app.set_embedded_panel(true);
            app
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    fn eyes() -> [EyeView; 2] {
        [SpatialFilesApp::embedded_eye(1600, 900); 2]
    }
    fn hand(point: [f32; 3]) -> HandControllerInput {
        let origin = [0.0, 1.6, 0.0];
        HandControllerInput {
            active: true,
            aim_origin: origin,
            aim_dir: vec3_normalize(vec3_sub(point, origin)),
            ..Default::default()
        }
    }
    fn tick(
        app: &mut SpatialFilesApp,
        input: HandControllerInput,
        glyphs: &mut GlyphAtlas,
        base: Instant,
        ms: u64,
    ) -> FrameScene {
        app.step_at(
            &eyes(),
            &[input, HandControllerInput::default()],
            glyphs,
            base + Duration::from_millis(ms),
        )
    }
    fn card(slot: usize) -> [f32; 3] {
        [
            (slot % 3) as f32 * 0.396 - 0.396,
            1.75 - (slot / 3) as f32 * 0.301,
            -1.24,
        ]
    }

    #[test]
    fn radial_marking_menu_flick_and_release_executes_target_sector() {
        let fixture = Fixture::new();
        fs::write(
            fixture.0.join("model.obj"),
            "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
        )
        .unwrap();
        let mut app = fixture.app();
        let mut glyphs = GlyphAtlas::new();
        let base = Instant::now();
        tick(
            &mut app,
            HandControllerInput::default(),
            &mut glyphs,
            base,
            0,
        );
        let mut input = hand(card(0));
        input.secondary_pressed = true;
        tick(&mut app, input, &mut glyphs, base, 10);
        let menu = app.radial_menu.as_ref().unwrap().clone();
        let point = local_world(
            &Pose3D {
                position: menu.anchor_world,
                orientation: menu.orientation,
            },
            [0.068, 0.040, 0.0],
        );
        input.aim_dir = vec3_normalize(vec3_sub(point, input.aim_origin));
        tick(&mut app, input, &mut glyphs, base, 80);
        assert_eq!(app.radial_menu.as_ref().unwrap().active_sector, Some(1));
        input.secondary_pressed = false;
        tick(&mut app, input, &mut glyphs, base, 120);
        assert!(app.quick_look_open && app.radial_menu.is_none());
        app.cancel_modals();
        tick(
            &mut app,
            HandControllerInput::default(),
            &mut glyphs,
            base,
            140,
        );
        input = hand(card(0));
        input.secondary_pressed = true;
        tick(&mut app, input, &mut glyphs, base, 160);
        input.secondary_pressed = false;
        tick(&mut app, input, &mut glyphs, base, 210);
        assert!(app.radial_menu.as_ref().unwrap().latched_open);
        assert!(app.undo_stack.is_empty() && fixture.0.join("model.obj").exists());
    }

    #[test]
    fn stationary_primary_pinch_uses_the_same_radial_menu_and_cancel_is_safe() {
        let fixture = Fixture::new();
        fs::write(fixture.0.join("a.txt"), "hello").unwrap();
        let mut app = fixture.app();
        let mut glyphs = GlyphAtlas::new();
        let base = Instant::now();
        tick(
            &mut app,
            HandControllerInput::default(),
            &mut glyphs,
            base,
            0,
        );
        let mut input = hand(card(0));
        input.trigger_pressed = true;
        tick(&mut app, input, &mut glyphs, base, 10);
        tick(&mut app, input, &mut glyphs, base, 351);
        assert_eq!(
            app.radial_menu.as_ref().unwrap().source,
            RadialSource::LongPress
        );
        input.trigger_pressed = false;
        tick(&mut app, input, &mut glyphs, base, 380);
        assert!(app.radial_menu.is_none() && fixture.0.join("a.txt").exists());
    }

    #[test]
    fn magnetic_press_lock_recovers_from_trigger_pull_wrist_dip() {
        let fixture = Fixture::new();
        for n in 0..6 {
            fs::write(fixture.0.join(format!("{n}.txt")), "text").unwrap();
        }
        let mut app = fixture.app();
        let mut glyphs = GlyphAtlas::new();
        let base = Instant::now();
        tick(
            &mut app,
            HandControllerInput::default(),
            &mut glyphs,
            base,
            0,
        );
        let mut input = hand([-0.396, 1.6 + 0.15 - 0.1325 + 0.008, -1.24]);
        tick(&mut app, input, &mut glyphs, base, 10);
        let stable = app.recent_stable_hover[0].as_ref().unwrap().clone();
        assert_eq!(stable.action, HitAction::SelectEntry(0));
        let mut direction = input.aim_dir;
        direction[1] -= 1.8f32.to_radians().tan();
        input.aim_dir = vec3_normalize(direction);
        input.trigger_pressed = true;
        tick(&mut app, input, &mut glyphs, base, 40);
        assert_eq!(app.primary_selected, Some(0));
        assert_eq!(app.press_lock_recoveries, 1);
    }

    fn drag(
        app: &mut SpatialFilesApp,
        from: [f32; 3],
        to: [f32; 3],
        glyphs: &mut GlyphAtlas,
        base: Instant,
        ms: u64,
    ) {
        tick(app, hand(from), glyphs, base, ms);
        let mut input = hand(from);
        input.trigger_pressed = true;
        tick(app, input, glyphs, base, ms + 10);
        input = hand(to);
        input.trigger_pressed = true;
        tick(app, input, glyphs, base, ms + 50);
        input.trigger_pressed = false;
        tick(app, input, glyphs, base, ms + 70);
    }

    #[test]
    fn spatial_drop_pocket_moves_stashed_files_across_directories_with_undo() {
        let fixture = Fixture::new();
        let a = fixture.0.join("A");
        let b = fixture.0.join("B");
        fs::create_dir_all(&a).unwrap();
        fs::create_dir_all(&b).unwrap();
        fs::write(a.join("one.txt"), "one").unwrap();
        fs::write(a.join("two.txt"), "two").unwrap();
        let mut app = SpatialFilesApp::new(Some(a.clone()));
        app.set_embedded_panel(true);
        let mut glyphs = GlyphAtlas::new();
        let base = Instant::now();
        tick(
            &mut app,
            HandControllerInput::default(),
            &mut glyphs,
            base,
            0,
        );
        let pocket = [-0.68, 1.29, -1.21];
        drag(&mut app, card(0), pocket, &mut glyphs, base, 10);
        drag(&mut app, card(1), pocket, &mut glyphs, base, 180);
        assert_eq!(app.drop_pocket.len(), 2);
        assert!(a.join("one.txt").exists() && a.join("two.txt").exists());
        app.navigate_to(b.to_string_lossy().into_owned());
        tick(
            &mut app,
            HandControllerInput::default(),
            &mut glyphs,
            base,
            300,
        );
        drag(&mut app, pocket, [0.0, 1.6, -1.24], &mut glyphs, base, 340);
        assert!(app.drop_pocket.is_empty());
        assert!(!a.join("one.txt").exists() && !a.join("two.txt").exists());
        assert!(b.join("one.txt").exists() && b.join("two.txt").exists());
        app.undo_last().unwrap();
        assert!(a.join("one.txt").exists() && a.join("two.txt").exists());
        assert!(!b.join("one.txt").exists() && !b.join("two.txt").exists());
    }

    #[test]
    fn folder_hover_requests_directory_peek_fanout() {
        let fixture = Fixture::new();
        let child = fixture.0.join("folder");
        fs::create_dir_all(&child).unwrap();
        for n in 0..4 {
            fs::write(child.join(format!("{n}.txt")), "document preview").unwrap();
        }
        let mut app = fixture.app();
        let mut glyphs = GlyphAtlas::new();
        let base = Instant::now();
        tick(
            &mut app,
            HandControllerInput::default(),
            &mut glyphs,
            base,
            0,
        );
        tick(&mut app, hand(card(0)), &mut glyphs, base, 10);
        let mut fan_count = 0;
        for step in 2..300 {
            let scene = tick(&mut app, hand(card(0)), &mut glyphs, base, step * 10);
            fan_count = scene
                .quads
                .iter()
                .filter(|q| {
                    (vec3_len(q.right_vec) * 2.0 - 0.092).abs() < 0.0001
                        && (vec3_len(q.up_vec) * 2.0 - 0.072).abs() < 0.0001
                        && q.params[2] == 2.0
                })
                .count();
            if fan_count == 3 {
                break;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(app.previewer.dir_peek_cache.get(&child).unwrap().len(), 3);
        assert_eq!(fan_count, 3);
    }

    #[test]
    fn quick_look_file_start_and_wrist_orbit_keep_model_orientation() {
        let fixture = Fixture::new();
        let file = fixture.0.join("prism.obj");
        fs::write(&file, "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n").unwrap();
        let mut app = SpatialFilesApp::new(Some(file));
        app.set_embedded_panel(true);
        assert!(app.quick_look_open);
        let mut glyphs = GlyphAtlas::new();
        let base = Instant::now();
        tick(
            &mut app,
            HandControllerInput::default(),
            &mut glyphs,
            base,
            0,
        );
        let mut input = hand([0.0, 1.62, -1.10]);
        input.trigger_pressed = true;
        tick(&mut app, input, &mut glyphs, base, 10);
        assert!(app.holo_orbit_drag.is_some());
        input.aim_dir = vec3_normalize([0.25, 0.12, -1.0]);
        tick(&mut app, input, &mut glyphs, base, 30);
        assert!(app.holo_yaw_pitch[0].abs() > 0.5 && app.holo_yaw_pitch[1].abs() > 0.1);
        input.trigger_pressed = false;
        tick(&mut app, input, &mut glyphs, base, 50);
        let orientation = app.holo_yaw_pitch;
        tick(
            &mut app,
            HandControllerInput::default(),
            &mut glyphs,
            base,
            80,
        );
        assert_eq!(app.holo_yaw_pitch, orientation);
        app.cancel_modals();
        assert!(!app.quick_look_open);
    }

    #[test]
    fn shelf_metrics_pagination_and_hot_targets_meet_optical_contract() {
        let fixture = Fixture::new();
        for n in 0..7 {
            fs::write(fixture.0.join(format!("{n}.txt")), "text").unwrap();
        }
        let mut app = fixture.app();
        let mut glyphs = GlyphAtlas::new();
        let base = Instant::now();
        tick(
            &mut app,
            HandControllerInput::default(),
            &mut glyphs,
            base,
            0,
        );
        assert_eq!(app.page_count(), 2);
        assert_eq!(
            app.hit_actions
                .iter()
                .filter(|a| matches!(a, HitAction::SelectEntry(_)))
                .count(),
            6
        );
        assert!(CARD_TITLE_M / 1.24 * 1000.0 >= 26.0);
        for i in 0..app.hit_actions.len() {
            let (min, max) = app.hit_batch.aabb_bounds(i).unwrap();
            assert!(max[0] - min[0] >= 0.0679 && max[1] - min[1] >= 0.0679);
        }
        app.step_page(1);
        assert_eq!(app.page_scroll_offset, 0.28);
        tick(
            &mut app,
            HandControllerInput::default(),
            &mut glyphs,
            base,
            50,
        );
        assert!(app.page_scroll_offset > 0.0 && app.page_scroll_offset < 0.28);
    }
}
