// SPDX-License-Identifier: GPL-3.0-or-later
//! Interactive Wayland / X11 Window Backend (`wgpu` + `winit`) for `xr-files`.
//! Runs smoothly inside HyperSpace `xr-wayland` (multi-window spatial collaboration)
//! as well as on standalone Linux desktop sessions.

use crate::{
    app::{HandControllerInput, SpatialFilesApp, ViewMode},
    control::ControlServer,
    fs_ops::ClipboardOp,
    renderer::WgpuSpatialRenderer,
    spatial::{quat_from_yaw_pitch, ray_from_window_ndc, EyeView, Pose3D},
    text::GlyphAtlas,
};
use anyhow::{Context, Result};
use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{Key, ModifiersState, NamedKey},
    window::{Window, WindowAttributes, WindowId},
};

pub fn run_window(app: &mut SpatialFilesApp, control: Option<&ControlServer>) -> Result<()> {
    app.set_embedded_panel(true);
    let event_loop = EventLoop::new().context("创建 Wayland/X11 事件循环失败")?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut state = WindowRunner::new(app, control);
    event_loop.run_app(&mut state)?;
    state.init_error.map_or(Ok(()), Err)
}

struct GpuSurfaceState {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    renderer: WgpuSpatialRenderer,
    glyphs: GlyphAtlas,
}

struct WindowRunner<'a> {
    app: &'a mut SpatialFilesApp,
    control: Option<&'a ControlServer>,
    gpu: Option<GpuSurfaceState>,
    init_error: Option<anyhow::Error>,
    cursor_px: [f32; 2],
    left_down: bool,
    middle_orbit_down: bool,
    last_orbit_px: [f32; 2],
    right_just_clicked: bool,
    back_just_clicked: bool,
    scroll_stick_y: f32,
    cam_yaw: f32,
    cam_pitch: f32,
    modifiers: ModifiersState,
}

impl<'a> WindowRunner<'a> {
    fn new(app: &'a mut SpatialFilesApp, control: Option<&'a ControlServer>) -> Self {
        Self {
            app,
            control,
            gpu: None,
            init_error: None,
            cursor_px: [640.0, 400.0],
            left_down: false,
            middle_orbit_down: false,
            last_orbit_px: [640.0, 400.0],
            right_just_clicked: false,
            back_just_clicked: false,
            scroll_stick_y: 0.0,
            cam_yaw: 0.0,
            cam_pitch: -0.03,
            modifiers: ModifiersState::empty(),
        }
    }

    fn initialize_gpu(&mut self, event_loop: &ActiveEventLoop) -> Result<()> {
        let attrs = WindowAttributes::default()
            .with_title("空间文件 · openKylin XR Files")
            .with_inner_size(PhysicalSize::new(1560, 720));
        let window = Arc::new(event_loop.create_window(attrs)?);
        let size = window.inner_size();

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::GL,
            flags: wgpu::InstanceFlags::ALLOW_UNDERLYING_NONCOMPLIANT_ADAPTER,
            ..Default::default()
        });
        let surface = instance.create_surface(Arc::clone(&window))?;
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .context("未找到兼容当前窗口的 wgpu 适配器")?;
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("xr-files-window-device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))?;

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let renderer = WgpuSpatialRenderer::new(device, queue, format)?;
        self.cursor_px = [config.width as f32 * 0.5, config.height as f32 * 0.5];
        self.gpu = Some(GpuSurfaceState {
            window,
            surface,
            config,
            renderer,
            glyphs: GlyphAtlas::new(),
        });
        Ok(())
    }

    fn current_eye(&self, width: u32, height: u32) -> EyeView {
        let aspect = (width.max(1) as f32 / height.max(1) as f32).clamp(0.5, 3.5);
        let half_v = 33.0f32.to_radians();
        let half_h = (half_v.tan() * aspect).atan();
        EyeView {
            position: [0.0, 1.60, 0.0],
            orientation: quat_from_yaw_pitch(self.cam_yaw, self.cam_pitch),
            fov: [-half_h, half_h, half_v, -half_v],
        }
    }

    fn render_frame(&mut self, event_loop: &ActiveEventLoop) {
        if let Some(ctl) = self.control {
            ctl.poll_and_apply(self.app);
        }
        if self.app.shutdown_requested() {
            event_loop.exit();
            return;
        }
        if self.app.take_recenter_request() {
            self.cam_yaw = 0.0;
            self.cam_pitch = -0.03;
        }
        let Some(ref mut gpu) = self.gpu else { return };

        let frame = match gpu.surface.get_current_texture() {
            Ok(tex) => tex,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                gpu.surface.configure(&gpu.renderer.device, &gpu.config);
                return;
            }
            Err(_) => return,
        };

        let w = gpu.config.width.max(1);
        let h = gpu.config.height.max(1);
        let eye = SpatialFilesApp::embedded_eye(w, h);

        let ndc_x = (self.cursor_px[0] / w as f32) * 2.0 - 1.0;
        let ndc_y = 1.0 - (self.cursor_px[1] / h as f32) * 2.0;
        let (ray_orig, ray_dir) = ray_from_window_ndc(&eye, ndc_x, ndc_y);

        let hand0 = HandControllerInput {
            active: true,
            grip_pose: Pose3D {
                position: [0.0, 1.40, 0.0],
                orientation: [0.0, 0.0, 0.0, 1.0],
            },
            aim_origin: ray_orig,
            aim_dir: ray_dir,
            trigger_pressed: self.left_down,
            squeeze_pressed: false,
            secondary_just_pressed: std::mem::replace(&mut self.right_just_clicked, false),
            back_just_pressed: std::mem::replace(&mut self.back_just_clicked, false),
            thumb_click_just_pressed: false,
            stick: [0.0, std::mem::replace(&mut self.scroll_stick_y, 0.0)],
        };
        let hands = [hand0, HandControllerInput::default()];
        let eyes = [eye, eye];

        let scene = self
            .app
            .step_and_build_scene(&eyes, &hands, &mut gpu.glyphs);
        let _ = self.app.take_haptics();

        gpu.renderer.sync_textures(
            &mut gpu.glyphs,
            &mut self.app.pending_gpu_thumbs,
            &mut self.app.pending_gpu_inspect_image,
        );

        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        gpu.renderer.render_views(&[(&view, w, h, eye)], &scene, 1.0);
        frame.present();
        self.app.submitted_frames += 1;
        gpu.window.request_redraw();
    }
}

impl ApplicationHandler for WindowRunner<'_> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.gpu.is_none() {
            if let Err(err) = self.initialize_gpu(event_loop) {
                self.init_error = Some(err);
                event_loop.exit();
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(new_size) => {
                if let Some(ref mut gpu) = self.gpu {
                    gpu.config.width = new_size.width.max(1);
                    gpu.config.height = new_size.height.max(1);
                    gpu.surface.configure(&gpu.renderer.device, &gpu.config);
                }
            }
            WindowEvent::ModifiersChanged(m) => {
                self.modifiers = m.state();
            }
            WindowEvent::CursorMoved { position, .. } => {
                let px = [position.x as f32, position.y as f32];
                if self.middle_orbit_down {
                    let dx = px[0] - self.last_orbit_px[0];
                    let dy = px[1] - self.last_orbit_px[1];
                    self.cam_yaw = (self.cam_yaw - dx * 0.0028).clamp(-0.95, 0.95);
                    self.cam_pitch = (self.cam_pitch - dy * 0.0022).clamp(-0.60, 0.45);
                }
                self.last_orbit_px = px;
                self.cursor_px = px;
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let down = state == ElementState::Pressed;
                match button {
                    MouseButton::Left => self.left_down = down,
                    MouseButton::Right => {
                        if down {
                            self.right_just_clicked = true;
                        }
                    }
                    MouseButton::Middle => {
                        self.middle_orbit_down = down;
                        self.last_orbit_px = self.cursor_px;
                    }
                    MouseButton::Back => {
                        if down {
                            self.back_just_clicked = true;
                        }
                    }
                    _ => {}
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let dy = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(pos) => (pos.y as f32) / 36.0,
                };
                if self.modifiers.control_key() {
                    self.app.workspace_radius =
                        (self.app.workspace_radius - dy * 0.08).clamp(1.05, 3.20);
                } else if dy.abs() > 0.1 {
                    self.scroll_stick_y = dy.signum() * 0.85;
                }
            }
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => {
                let ctrl = self.modifiers.control_key();
                if ctrl {
                    if let Key::Character(ref s) = event.logical_key {
                        match s.to_ascii_lowercase().as_str() {
                            "c" => {
                                self.app.copy_selected_to_clipboard(ClipboardOp::Copy);
                                return;
                            }
                            "x" => {
                                self.app.copy_selected_to_clipboard(ClipboardOp::Cut);
                                return;
                            }
                            "v" => {
                                let _ = self.app.paste_clipboard_here();
                                return;
                            }
                            "z" => {
                                let _ = self.app.undo_last();
                                return;
                            }
                            "n" => {
                                let _ = self.app.create_folder_named("新建文件夹");
                                return;
                            }
                            "h" => {
                                self.app.show_hidden = !self.app.show_hidden;
                                self.app.reload_directory();
                                return;
                            }
                            "q" => {
                                event_loop.exit();
                                return;
                            }
                            _ => {}
                        }
                    }
                }
                match event.logical_key {
                    Key::Named(NamedKey::Escape) => self.app.cancel_modals(),
                    Key::Named(NamedKey::Enter) => self.app.commit_prompt(),
                    Key::Named(NamedKey::Backspace) => self.app.handle_backspace(),
                    Key::Named(NamedKey::Delete) => {
                        let _ = self.app.trash_or_restore_selected();
                    }
                    Key::Named(NamedKey::Tab) => {
                        let next = match self.app.view_mode {
                            ViewMode::Workbench => ViewMode::Treemap3D,
                            ViewMode::Treemap3D => ViewMode::Workbench,
                        };
                        self.app.set_view_mode(next);
                    }
                    Key::Named(NamedKey::F2) => {
                        if let Some(idx) = self.app.primary_selected {
                            self.app.prompt_mode = Some(crate::app::PromptMode::Rename);
                            self.app.prompt_buffer = self.app.all_entries[idx].name.clone();
                        }
                    }
                    Key::Named(NamedKey::PageDown) => self.app.step_page(1),
                    Key::Named(NamedKey::PageUp) => self.app.step_page(-1),
                    _ => {
                        if let Some(ref txt) = event.text {
                            self.app.handle_key_text(txt.as_str());
                        }
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                self.render_frame(event_loop);
            }
            _ => {}
        }
        let _ = &self.current_eye(1, 1);
    }
}
