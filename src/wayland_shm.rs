// SPDX-License-Identifier: GPL-3.0-or-later
//! Native GPU-to-Wayland `wl_shm` (`Argb8888` premultiplied) Client Backend for `xr-files`.
//!
//! Designed specifically for seamless coexistence inside **HyperSpace (`xr-shell` + `xr-window-server`)**:
//! - Preserves HyperSpace's 360° Qingkong background, 3-island Dock (`xr-dock`), Control Center, and
//!   other open windows without taking exclusive ownership of the OpenXR session.
//! - Uses `xdg_toplevel` (`app_id = "org.openkylin.XRFiles"`) + premultiplied `wl_shm::Format::Argb8888`
//!   so `xr-shell` automatically provides unified spatial window drag, corner/two-hand resize,
//!   cylindrical curve (`弯折`), minimize, and close controls!
//! - Renders hardware-accelerated 3D volumetric treemap, 3D rotating `.obj/.stl/.glb` holograms,
//!   frosted Qingkong glass, and supersampled SDF icons on Vulkan (`Turnip Adreno 650` / desktop GPU).

use crate::{
    app::{HandControllerInput, PromptMode, SpatialFilesApp, ViewMode},
    control::ControlServer,
    fs_ops::ClipboardOp,
    renderer::WgpuSpatialRenderer,
    spatial::{ray_from_window_ndc, Pose3D},
    text::GlyphAtlas,
};
use anyhow::{ensure, Context, Result};
use std::{
    ffi::CString,
    io,
    os::fd::{AsRawFd, BorrowedFd, FromRawFd, OwnedFd},
    ptr,
    time::{Duration, Instant},
};
use wayland_client::{
    delegate_noop,
    protocol::{
        wl_buffer, wl_compositor, wl_keyboard, wl_pointer, wl_registry, wl_seat, wl_shm,
        wl_shm_pool, wl_surface,
    },
    Connection, Dispatch, EventQueue, QueueHandle, WEnum,
};
use wayland_protocols::xdg::{
    decoration::zv1::client::{zxdg_decoration_manager_v1, zxdg_toplevel_decoration_v1},
    shell::client::{xdg_surface, xdg_toplevel, xdg_wm_base},
};

const DEFAULT_WIDTH: u32 = 1600;
const DEFAULT_HEIGHT: u32 = 900;

const BTN_LEFT: u32 = 0x110;
const BTN_RIGHT: u32 = 0x111;
const BTN_MIDDLE: u32 = 0x112;
const BTN_SIDE: u32 = 0x113;
const BTN_EXTRA: u32 = 0x114;

struct ShmSlot {
    buffer: wl_buffer::WlBuffer,
    busy: bool,
}

struct ShmDoublePool {
    fd: OwnedFd,
    map_ptr: *mut u8,
    map_len: usize,
    pool: wl_shm_pool::WlShmPool,
    slots: [ShmSlot; 2],
    width: u32,
    height: u32,
    next_idx: usize,
}

impl ShmDoublePool {
    fn create(
        shm: &wl_shm::WlShm,
        qh: &QueueHandle<WaylandState>,
        width: u32,
        height: u32,
    ) -> Result<Self> {
        let width = width.clamp(320, 4096);
        let height = height.clamp(240, 4096);
        let stride = (width as usize) * 4;
        let frame_bytes = stride * (height as usize);
        let total_bytes = frame_bytes * 2;

        let name = CString::new("xr-files-wl-shm")?;
        // SAFETY: Creates an anonymous close-on-exec shared-memory file sized to two RGBA frames.
        let raw_fd = unsafe { libc::memfd_create(name.as_ptr(), libc::MFD_CLOEXEC) };
        if raw_fd < 0 {
            return Err(io::Error::last_os_error()).context("memfd_create 失败");
        }
        // SAFETY: `raw_fd` was just returned by `memfd_create`.
        let fd = unsafe { OwnedFd::from_raw_fd(raw_fd) };
        if unsafe { libc::ftruncate(fd.as_raw_fd(), total_bytes as libc::off_t) } < 0 {
            return Err(io::Error::last_os_error()).context("ftruncate wl_shm 池失败");
        }
        // SAFETY: Maps `total_bytes` of our private memfd for CPU write + compositor `wl_shm` read.
        let map_ptr = unsafe {
            libc::mmap(
                ptr::null_mut(),
                total_bytes,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd.as_raw_fd(),
                0,
            )
        };
        if map_ptr == libc::MAP_FAILED {
            return Err(io::Error::last_os_error()).context("mmap wl_shm 池失败");
        }

        let borrowed = unsafe { BorrowedFd::borrow_raw(fd.as_raw_fd()) };
        let pool = shm.create_pool(borrowed, total_bytes as i32, qh, ());
        let b0 = pool.create_buffer(
            0,
            width as i32,
            height as i32,
            stride as i32,
            wl_shm::Format::Argb8888,
            qh,
            0usize,
        );
        let b1 = pool.create_buffer(
            frame_bytes as i32,
            width as i32,
            height as i32,
            stride as i32,
            wl_shm::Format::Argb8888,
            qh,
            1usize,
        );

        Ok(Self {
            fd,
            map_ptr: map_ptr as *mut u8,
            map_len: total_bytes,
            pool,
            slots: [
                ShmSlot {
                    buffer: b0,
                    busy: false,
                },
                ShmSlot {
                    buffer: b1,
                    busy: false,
                },
            ],
            width,
            height,
            next_idx: 0,
        })
    }

    fn acquire_slot(&mut self) -> (usize, &wl_buffer::WlBuffer, &mut [u8]) {
        let idx = if !self.slots[self.next_idx].busy {
            self.next_idx
        } else if !self.slots[1 - self.next_idx].busy {
            1 - self.next_idx
        } else {
            // `xr-wayland` copies `wl_shm` bytes synchronously before returning `release`,
            // so fallback to `next_idx` remains safe even if release is still in flight.
            self.next_idx
        };
        self.next_idx = 1 - idx;
        self.slots[idx].busy = true;
        let frame_bytes = (self.width as usize) * (self.height as usize) * 4;
        let offset = idx * frame_bytes;
        // SAFETY: `map_ptr` points to `2 * frame_bytes` valid mapped bytes for the lifetime of `Self`.
        let slice =
            unsafe { std::slice::from_raw_parts_mut(self.map_ptr.add(offset), frame_bytes) };
        (idx, &self.slots[idx].buffer, slice)
    }
}

impl Drop for ShmDoublePool {
    fn drop(&mut self) {
        self.slots[0].buffer.destroy();
        self.slots[1].buffer.destroy();
        self.pool.destroy();
        if !self.map_ptr.is_null() && self.map_len > 0 {
            // SAFETY: Unmaps the exact region mapped in `ShmDoublePool::create`.
            unsafe {
                let _ = libc::munmap(self.map_ptr as *mut _, self.map_len);
            }
        }
        let _ = &self.fd;
    }
}

struct WaylandState {
    compositor: Option<wl_compositor::WlCompositor>,
    shm: Option<wl_shm::WlShm>,
    wm_base: Option<xdg_wm_base::XdgWmBase>,
    decoration_mgr: Option<zxdg_decoration_manager_v1::ZxdgDecorationManagerV1>,
    seat: Option<wl_seat::WlSeat>,
    pointer: Option<wl_pointer::WlPointer>,
    keyboard: Option<wl_keyboard::WlKeyboard>,

    surface: Option<wl_surface::WlSurface>,
    xdg_surface: Option<xdg_surface::XdgSurface>,
    toplevel: Option<xdg_toplevel::XdgToplevel>,
    decoration: Option<zxdg_toplevel_decoration_v1::ZxdgToplevelDecorationV1>,
    pool: Option<ShmDoublePool>,

    configured: bool,
    closed: bool,
    target_width: u32,
    target_height: u32,
    pending_cfg_size: Option<(u32, u32)>,
    needs_redraw: bool,

    pointer_inside: bool,
    cursor_px: [f32; 2],
    left_down: bool,
    right_down: bool,
    back_just_clicked: bool,
    scroll_stick_y: f32,
    ctrl_down: bool,
    shift_down: bool,
    pending_keys: Vec<u32>,
}

impl WaylandState {
    fn new() -> Self {
        Self {
            compositor: None,
            shm: None,
            wm_base: None,
            decoration_mgr: None,
            seat: None,
            pointer: None,
            keyboard: None,
            surface: None,
            xdg_surface: None,
            toplevel: None,
            decoration: None,
            pool: None,
            configured: false,
            closed: false,
            target_width: DEFAULT_WIDTH,
            target_height: DEFAULT_HEIGHT,
            pending_cfg_size: None,
            needs_redraw: true,
            pointer_inside: false,
            cursor_px: [DEFAULT_WIDTH as f32 * 0.5, DEFAULT_HEIGHT as f32 * 0.5],
            left_down: false,
            right_down: false,
            back_just_clicked: false,
            scroll_stick_y: 0.0,
            ctrl_down: false,
            shift_down: false,
            pending_keys: Vec::new(),
        }
    }
}

pub fn run_wayland_shm(
    app: &mut SpatialFilesApp,
    control: Option<&ControlServer>,
    max_seconds: Option<u64>,
) -> Result<()> {
    app.set_embedded_panel(true);

    let conn = Connection::connect_to_env().context("连接 Wayland 会话 (WAYLAND_DISPLAY) 失败")?;
    let mut event_queue: EventQueue<WaylandState> = conn.new_event_queue();
    let qh = event_queue.handle();

    let mut state = WaylandState::new();
    let _registry = conn.display().get_registry(&qh, ());
    event_queue.roundtrip(&mut state)?;

    let compositor = state
        .compositor
        .clone()
        .context("Wayland 合成器缺少 wl_compositor")?;
    let shm = state.shm.clone().context("Wayland 合成器缺少 wl_shm")?;
    let wm_base = state
        .wm_base
        .clone()
        .context("Wayland 合成器缺少 xdg_wm_base")?;

    let surface = compositor.create_surface(&qh, ());
    let xdg_surface = wm_base.get_xdg_surface(&surface, &qh, ());
    let toplevel = xdg_surface.get_toplevel(&qh, ());
    toplevel.set_app_id("org.openkylin.XRFiles".into());
    toplevel.set_title("空间文件".into());
    toplevel.set_min_size(720, 440);

    if let Some(ref mgr) = state.decoration_mgr {
        let dec = mgr.get_toplevel_decoration(&toplevel, &qh, ());
        dec.set_mode(zxdg_toplevel_decoration_v1::Mode::ServerSide);
        state.decoration = Some(dec);
    }

    // Standard `xdg_shell` handshake: commit empty surface first, wait for `Configure`.
    surface.commit();
    state.surface = Some(surface);
    state.xdg_surface = Some(xdg_surface);
    state.toplevel = Some(toplevel);

    for _ in 0..8 {
        event_queue.roundtrip(&mut state)?;
        if state.configured {
            break;
        }
    }
    ensure!(
        state.configured,
        "Wayland xdg_surface 未完成初始 Configure 握手"
    );

    let mut renderer = WgpuSpatialRenderer::new_headless()
        .context("初始化 Vulkan 空间渲染器 (WgpuSpatialRenderer) 失败")?;
    let mut glyphs = GlyphAtlas::new();

    state.pool = Some(ShmDoublePool::create(
        &shm,
        &qh,
        state.target_width,
        state.target_height,
    )?);

    let deadline = max_seconds
        .filter(|&s| s > 0)
        .map(|s| Instant::now() + Duration::from_secs(s));

    let feedback = crate::desktop_feedback::DesktopFeedback::connect();
    let preview_fd = app.previewer.wake.enable()?;
    let control_fd = control
        .map(|ctl| ctl.wake.enable())
        .transpose()?
        .unwrap_or(-1);

    while !state.closed && !app.shutdown_requested() {
        if deadline.is_some_and(|dl| Instant::now() >= dl) {
            break;
        }

        if let Some(ctl) = control {
            state.needs_redraw |= ctl.poll_and_apply(app);
        }
        if app.shutdown_requested() {
            break;
        }

        // Wait for input, worker completion or the duration limit while the window is still.
        conn.flush()?;
        if let Some(guard) = conn.prepare_read() {
            let poll_ms = if state.needs_redraw || app.has_active_animation() {
                0
            } else {
                deadline.map_or(-1, |dl| {
                    dl.saturating_duration_since(Instant::now())
                        .as_millis()
                        .min(i32::MAX as u128) as i32
                })
            };
            let mut fds = [guard.connection_fd().as_raw_fd(), preview_fd, control_fd].map(|fd| {
                libc::pollfd {
                    fd,
                    events: libc::POLLIN,
                    revents: 0,
                }
            });
            let ready = unsafe { libc::poll(fds.as_mut_ptr(), fds.len() as _, poll_ms) };
            if ready < 0 {
                let error = std::io::Error::last_os_error();
                if error.kind() == std::io::ErrorKind::Interrupted {
                    continue;
                }
                return Err(error.into());
            }
            if fds[0].revents != 0 {
                guard.read()?;
            }
            if fds[1].revents != 0 {
                app.previewer.wake.drain();
                state.needs_redraw = true;
            }
            if fds[2].revents != 0 {
                if let Some(ctl) = control {
                    ctl.wake.drain();
                    state.needs_redraw |= ctl.poll_and_apply(app);
                }
            }
        }
        event_queue.dispatch_pending(&mut state)?;

        if state.closed || app.shutdown_requested() {
            break;
        }

        // Apply any Linux evdev keyboard events received from `xr-wayland`
        if !state.pending_keys.is_empty() {
            let keys = std::mem::take(&mut state.pending_keys);
            for keycode in keys {
                apply_evdev_key(app, keycode, state.ctrl_down, state.shift_down);
            }
            state.needs_redraw = true;
        }

        // Recreate SHM pool if `xr-shell` / compositor resized the window
        if state
            .pool
            .as_ref()
            .is_none_or(|p| p.width != state.target_width || p.height != state.target_height)
        {
            state.pool = Some(ShmDoublePool::create(
                &shm,
                &qh,
                state.target_width,
                state.target_height,
            )?);
            state.needs_redraw = true;
        }

        if state.needs_redraw || app.has_active_animation() {
            let w = state.target_width;
            let h = state.target_height;
            let eye = SpatialFilesApp::embedded_eye(w, h);

            let hand0 = if state.pointer_inside
                || state.left_down
                || state.right_down
                || state.back_just_clicked
                || state.scroll_stick_y.abs() > 0.01
            {
                let ndc_x = (state.cursor_px[0] / w.max(1) as f32) * 2.0 - 1.0;
                let ndc_y = 1.0 - (state.cursor_px[1] / h.max(1) as f32) * 2.0;
                let (ray_orig, ray_dir) = ray_from_window_ndc(&eye, ndc_x, ndc_y);
                HandControllerInput {
                    active: true,
                    grip_pose: Pose3D {
                        position: [0.0, 1.40, 0.0],
                        orientation: [0.0, 0.0, 0.0, 1.0],
                    },
                    aim_origin: ray_orig,
                    aim_dir: ray_dir,
                    trigger_pressed: state.left_down,
                    squeeze_pressed: false,
                    secondary_pressed: state.right_down,
                    back_just_pressed: std::mem::replace(&mut state.back_just_clicked, false),
                    thumb_click_just_pressed: false,
                    stick: [0.0, std::mem::replace(&mut state.scroll_stick_y, 0.0)],
                }
            } else {
                HandControllerInput::default()
            };

            let hands = [hand0, HandControllerInput::default()];
            let eyes = [eye, eye];
            let scene = app.step_and_build_scene(&eyes, &hands, &mut glyphs);
            feedback.emit(app.take_haptics());

            renderer.sync_textures(
                &mut glyphs,
                &mut app.pending_gpu_thumbs,
                &mut app.pending_gpu_inspect_image,
            );

            let pool = state.pool.as_mut().unwrap();
            let (_slot_idx, wl_buf, dst_bytes) = pool.acquire_slot();
            renderer.render_offscreen_wayland_argb8888(w, h, eye, &scene, dst_bytes)?;

            let wl_surf = state.surface.as_ref().unwrap();
            state
                .xdg_surface
                .as_ref()
                .unwrap()
                .set_window_geometry(0, 0, w as i32, h as i32);
            wl_surf.attach(Some(wl_buf), 0, 0);
            wl_surf.damage_buffer(0, 0, w as i32, h as i32);
            wl_surf.commit();
            conn.flush()?;

            app.submitted_frames += 1;
            state.needs_redraw = false;
            if app.has_active_animation() {
                std::thread::sleep(Duration::from_millis(16));
            }
        }
    }

    renderer.device.poll(wgpu::Maintain::Wait);
    if let Some(s) = state.surface.take() {
        s.attach(None, 0, 0);
        s.commit();
    }
    let _ = conn.flush();
    Ok(())
}

fn apply_evdev_key(app: &mut SpatialFilesApp, key: u32, ctrl: bool, shift: bool) {
    if ctrl {
        match key {
            46 => {
                app.copy_selected_to_clipboard(ClipboardOp::Copy);
                return;
            } // C
            45 => {
                app.copy_selected_to_clipboard(ClipboardOp::Cut);
                return;
            } // X
            47 => {
                let _ = app.paste_clipboard_here();
                return;
            } // V
            44 => {
                let _ = app.undo_last();
                return;
            } // Z
            49 => {
                let _ = app.create_folder_named("新建文件夹");
                return;
            } // N
            35 => {
                app.show_hidden = !app.show_hidden;
                app.reload_directory();
                return;
            } // H
            16 => {
                app.request_shutdown();
                return;
            } // Q
            _ => {}
        }
    }

    match key {
        1 => app.cancel_modals(),                                   // KEY_ESC
        14 => app.handle_backspace(),                               // KEY_BACKSPACE
        57 if app.prompt_mode.is_none() => app.toggle_quick_look(), // Space
        15 => {
            let next = match app.view_mode {
                ViewMode::Workbench => ViewMode::Treemap3D,
                ViewMode::Treemap3D => ViewMode::Workbench,
            };
            app.set_view_mode(next);
        } // KEY_TAB
        28 | 96 => app.commit_prompt(),                             // KEY_ENTER / KPENTER
        60 => {
            if let Some(idx) = app.primary_selected {
                app.prompt_mode = Some(PromptMode::Rename);
                app.prompt_buffer = app.all_entries[idx].name.clone();
            }
        } // KEY_F2
        104 => app.step_page(-1),                                   // KEY_PAGEUP
        109 => app.step_page(1),                                    // KEY_PAGEDOWN
        111 => {
            let _ = app.trash_or_restore_selected();
        } // KEY_DELETE
        _ => {
            if let Some(ch) = evdev_to_ascii(key, shift) {
                let mut buf = [0u8; 4];
                app.handle_key_text(ch.encode_utf8(&mut buf));
            }
        }
    }
}

fn evdev_to_ascii(key: u32, shift: bool) -> Option<char> {
    let base = match key {
        2 => ('1', '!'),
        3 => ('2', '@'),
        4 => ('3', '#'),
        5 => ('4', '$'),
        6 => ('5', '%'),
        7 => ('6', '^'),
        8 => ('7', '&'),
        9 => ('8', '*'),
        10 => ('9', '('),
        11 => ('0', ')'),
        12 => ('-', '_'),
        13 => ('=', '+'),
        16 => ('q', 'Q'),
        17 => ('w', 'W'),
        18 => ('e', 'E'),
        19 => ('r', 'R'),
        20 => ('t', 'T'),
        21 => ('y', 'Y'),
        22 => ('u', 'U'),
        23 => ('i', 'I'),
        24 => ('o', 'O'),
        25 => ('p', 'P'),
        30 => ('a', 'A'),
        31 => ('s', 'S'),
        32 => ('d', 'D'),
        33 => ('f', 'F'),
        34 => ('g', 'G'),
        35 => ('h', 'H'),
        36 => ('j', 'J'),
        37 => ('k', 'K'),
        38 => ('l', 'L'),
        44 => ('z', 'Z'),
        45 => ('x', 'X'),
        46 => ('c', 'C'),
        47 => ('v', 'V'),
        48 => ('b', 'B'),
        49 => ('n', 'N'),
        50 => ('m', 'M'),
        52 => ('.', '>'),
        53 => ('/', '?'),
        57 => (' ', ' '),
        _ => return None,
    };
    Some(if shift { base.1 } else { base.0 })
}

impl Dispatch<wl_registry::WlRegistry, ()> for WaylandState {
    fn event(
        state: &mut Self,
        registry: &wl_registry::WlRegistry,
        event: wl_registry::Event,
        _: &(),
        _: &Connection,
        qh: &QueueHandle<Self>,
    ) {
        if let wl_registry::Event::Global {
            name,
            interface,
            version,
        } = event
        {
            match interface.as_str() {
                "wl_compositor" => {
                    state.compositor = Some(registry.bind(name, version.min(6), qh, ()));
                }
                "wl_shm" => {
                    state.shm = Some(registry.bind(name, version.min(1), qh, ()));
                }
                "xdg_wm_base" => {
                    state.wm_base = Some(registry.bind(name, version.min(6), qh, ()));
                }
                "zxdg_decoration_manager_v1" => {
                    state.decoration_mgr = Some(registry.bind(name, version.min(1), qh, ()));
                }
                "wl_seat" => {
                    let seat: wl_seat::WlSeat = registry.bind(name, version.min(8), qh, ());
                    state.seat = Some(seat);
                }
                _ => {}
            }
        }
    }
}

impl Dispatch<xdg_wm_base::XdgWmBase, ()> for WaylandState {
    fn event(
        _: &mut Self,
        wm_base: &xdg_wm_base::XdgWmBase,
        event: xdg_wm_base::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let xdg_wm_base::Event::Ping { serial } = event {
            wm_base.pong(serial);
        }
    }
}

impl Dispatch<xdg_surface::XdgSurface, ()> for WaylandState {
    fn event(
        state: &mut Self,
        xdg_surface: &xdg_surface::XdgSurface,
        event: xdg_surface::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let xdg_surface::Event::Configure { serial } = event {
            xdg_surface.ack_configure(serial);
            if let Some((w, h)) = state.pending_cfg_size.take() {
                // `xr-window-server` sends a generic `(800, 600)` placeholder on the initial
                // bufferless configure before adopting the client's committed widescreen SHM size;
                // honor explicit resizes after initial map (or non-placeholder initial sizes).
                if w > 0 && h > 0 && (state.configured || (w, h) != (800, 600)) {
                    state.target_width = w.clamp(480, 4096);
                    state.target_height = h.clamp(320, 4096);
                }
            }
            state.configured = true;
            state.needs_redraw = true;
        }
    }
}

impl Dispatch<xdg_toplevel::XdgToplevel, ()> for WaylandState {
    fn event(
        state: &mut Self,
        _: &xdg_toplevel::XdgToplevel,
        event: xdg_toplevel::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        match event {
            xdg_toplevel::Event::Configure { width, height, .. } => {
                if width > 0 && height > 0 {
                    state.pending_cfg_size = Some((width as u32, height as u32));
                }
            }
            xdg_toplevel::Event::Close => {
                state.closed = true;
            }
            _ => {}
        }
    }
}

impl Dispatch<wl_seat::WlSeat, ()> for WaylandState {
    fn event(
        state: &mut Self,
        seat: &wl_seat::WlSeat,
        event: wl_seat::Event,
        _: &(),
        _: &Connection,
        qh: &QueueHandle<Self>,
    ) {
        if let wl_seat::Event::Capabilities {
            capabilities: WEnum::Value(caps),
        } = event
        {
            if caps.contains(wl_seat::Capability::Pointer) && state.pointer.is_none() {
                state.pointer = Some(seat.get_pointer(qh, ()));
            }
            if caps.contains(wl_seat::Capability::Keyboard) && state.keyboard.is_none() {
                state.keyboard = Some(seat.get_keyboard(qh, ()));
            }
        }
    }
}

impl Dispatch<wl_pointer::WlPointer, ()> for WaylandState {
    fn event(
        state: &mut Self,
        _: &wl_pointer::WlPointer,
        event: wl_pointer::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        match event {
            wl_pointer::Event::Enter {
                surface_x,
                surface_y,
                ..
            } => {
                state.pointer_inside = true;
                state.cursor_px = [surface_x as f32, surface_y as f32];
                state.needs_redraw = true;
            }
            wl_pointer::Event::Leave { .. } => {
                state.pointer_inside = false;
                state.left_down = false;
                state.right_down = false;
                state.needs_redraw = true;
            }
            wl_pointer::Event::Motion {
                surface_x,
                surface_y,
                ..
            } => {
                state.pointer_inside = true;
                state.cursor_px = [surface_x as f32, surface_y as f32];
                state.needs_redraw = true;
            }
            wl_pointer::Event::Button {
                button,
                state: WEnum::Value(btn_state),
                ..
            } => {
                let down = btn_state == wl_pointer::ButtonState::Pressed;
                match button {
                    BTN_LEFT => state.left_down = down,
                    BTN_RIGHT | BTN_MIDDLE => state.right_down = down,
                    BTN_SIDE | BTN_EXTRA => {
                        if down {
                            state.back_just_clicked = true;
                        }
                    }
                    _ => {}
                }
                state.needs_redraw = true;
            }
            wl_pointer::Event::Axis {
                axis: WEnum::Value(wl_pointer::Axis::VerticalScroll),
                value,
                ..
            } => {
                if value.abs() > 0.2 {
                    state.scroll_stick_y = -(value.signum() as f32) * 0.85;
                    state.needs_redraw = true;
                }
            }
            _ => {}
        }
    }
}

impl Dispatch<wl_keyboard::WlKeyboard, ()> for WaylandState {
    fn event(
        state: &mut Self,
        _: &wl_keyboard::WlKeyboard,
        event: wl_keyboard::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        match event {
            wl_keyboard::Event::Key {
                key,
                state: WEnum::Value(kstate),
                ..
            } => {
                let down = kstate == wl_keyboard::KeyState::Pressed;
                match key {
                    29 | 97 => state.ctrl_down = down,
                    42 | 54 => state.shift_down = down,
                    _ if down => state.pending_keys.push(key),
                    _ => {}
                }
            }
            wl_keyboard::Event::Modifiers { mods_depressed, .. } => {
                state.shift_down = (mods_depressed & 1) != 0;
                state.ctrl_down = (mods_depressed & 4) != 0;
            }
            wl_keyboard::Event::Leave { .. } => {
                state.ctrl_down = false;
                state.shift_down = false;
            }
            _ => {}
        }
    }
}

impl Dispatch<wl_buffer::WlBuffer, usize> for WaylandState {
    fn event(
        state: &mut Self,
        _: &wl_buffer::WlBuffer,
        event: wl_buffer::Event,
        slot_idx: &usize,
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let wl_buffer::Event::Release = event {
            if let Some(ref mut pool) = state.pool {
                if *slot_idx < pool.slots.len() {
                    pool.slots[*slot_idx].busy = false;
                }
            }
        }
    }
}

impl Dispatch<zxdg_toplevel_decoration_v1::ZxdgToplevelDecorationV1, ()> for WaylandState {
    fn event(
        _: &mut Self,
        _: &zxdg_toplevel_decoration_v1::ZxdgToplevelDecorationV1,
        _: zxdg_toplevel_decoration_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}

delegate_noop!(WaylandState: ignore wl_compositor::WlCompositor);
delegate_noop!(WaylandState: ignore wl_shm::WlShm);
delegate_noop!(WaylandState: ignore wl_shm_pool::WlShmPool);
delegate_noop!(WaylandState: ignore wl_surface::WlSurface);
delegate_noop!(WaylandState: ignore zxdg_decoration_manager_v1::ZxdgDecorationManagerV1);
