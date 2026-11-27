// SPDX-License-Identifier: GPL-3.0-or-later
//! Entry point for `xr-files` — openKylin XR Spatial File Manager (Rust / wgpu / SIMD).

use anyhow::{bail, Result};
use std::{
    env, fs,
    path::PathBuf,
    thread,
    time::{Duration, Instant},
};
use xr_files::{
    app::{HandControllerInput, SpatialFilesApp, ViewMode},
    control::{self, ControlServer},
    renderer::WgpuSpatialRenderer,
    simd,
    spatial::{EyeView, Pose3D},
    text::GlyphAtlas,
    window_mode, xr_runtime,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RunMode {
    Auto,
    OpenXr,
    Window,
    Headless,
}

fn print_usage() {
    println!(
        "\
空间文件 (xr-files) {} — openKylin XR 原生三维空间文件管理器
SIMD 硬件加速后端: {}

用法:
  xr-files [选项] [初始目录或文件路径]
  xr-files ctl '<JSON控制命令>'

选项:
  --xr                  强制使用原生 OpenXR 双目立体空间会话
  --window              在 Wayland / X11 桌面窗口中运行交互式空间视图
  --headless            无头 Vulkan 离屏立体渲染模式（用于自动化验证与控制端测试）
  --treemap             启动时直接进入「3D 空间体积星图」模式
  --seconds <N>         运行指定秒数后正常退出（0 为持续交互）
  --capture-dir <DIR>   在无头模式下导出双目立体渲染截图 (left.png / right.png) 与 summary.json
  -h, --help            显示此帮助信息
  -V, --version         显示版本信息",
        env!("CARGO_PKG_VERSION"),
        simd::active_simd_backend()
    );
}

fn main() -> Result<()> {
    let mut args = env::args().skip(1).peekable();

    if args.peek().map(String::as_str) == Some("ctl") {
        args.next();
        let payload = args.collect::<Vec<_>>().join(" ");
        let resp = control::send_raw_command(&control::default_socket_path(), &payload)?;
        println!("{resp}");
        return Ok(());
    }

    let mut mode = RunMode::Auto;
    let mut initial_treemap = false;
    let mut initial_search: Option<String> = None;
    let mut max_seconds: Option<u64> = None;
    let mut capture_dir: Option<PathBuf> = None;
    let mut initial_path: Option<PathBuf> = None;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                print_usage();
                return Ok(());
            }
            "-V" | "--version" => {
                println!(
                    "xr-files {} ({})",
                    env!("CARGO_PKG_VERSION"),
                    simd::active_simd_backend()
                );
                return Ok(());
            }
            "--xr" => mode = RunMode::OpenXr,
            "--window" => mode = RunMode::Window,
            "--headless" => mode = RunMode::Headless,
            "--treemap" => initial_treemap = true,
            "--search" => {
                let raw = args
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("--search 需要查询词参数"))?;
                initial_search = Some(raw);
            }
            "--seconds" => {
                let raw = args
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("--seconds 需要整数秒参数"))?;
                max_seconds = Some(raw.parse()?);
            }
            "--capture-dir" => {
                let raw = args
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("--capture-dir 需要目录路径"))?;
                capture_dir = Some(PathBuf::from(raw));
            }
            s if s.starts_with('-') => bail!("未知选项: {s}"),
            path_str => {
                initial_path = Some(PathBuf::from(path_str));
            }
        }
    }

    // Forward target path to an already-running xr-files session if present
    if mode != RunMode::Headless {
        if let Some(ref target) = initial_path {
            if control::try_forward_open_to_existing(target) {
                return Ok(());
            }
        }
    }

    let mut app = SpatialFilesApp::new(initial_path);
    if initial_treemap {
        app.set_view_mode(ViewMode::Treemap3D);
    }
    if let Some(q) = initial_search {
        app.set_search_query(q);
    }

    let ctl = ControlServer::bind_default().ok();

    match mode {
        RunMode::Headless => run_headless(&mut app, ctl.as_ref(), max_seconds, capture_dir),
        RunMode::Window => window_mode::run_window(&mut app, ctl.as_ref()),
        RunMode::OpenXr => xr_runtime::run_openxr(&mut app, ctl.as_ref(), max_seconds),
        RunMode::Auto => {
            let prefer_xr = env::var_os("XR_RUNTIME_JSON").is_some()
                || env::var_os("XR_APPLICATION_ID").is_some()
                || xr_runtime::probe_openxr_available();
            if prefer_xr {
                match xr_runtime::run_openxr(&mut app, ctl.as_ref(), max_seconds) {
                    Ok(()) => Ok(()),
                    Err(xr_err) => {
                        if env::var_os("WAYLAND_DISPLAY").is_some()
                            || env::var_os("DISPLAY").is_some()
                        {
                            eprintln!(
                                "OpenXR 会话未就绪 ({xr_err})，自动回退至空间桌面窗口模式"
                            );
                            window_mode::run_window(&mut app, ctl.as_ref())
                        } else {
                            Err(xr_err)
                        }
                    }
                }
            } else {
                window_mode::run_window(&mut app, ctl.as_ref())
            }
        }
    }
}

fn run_headless(
    app: &mut SpatialFilesApp,
    control: Option<&ControlServer>,
    max_seconds: Option<u64>,
    capture_dir: Option<PathBuf>,
) -> Result<()> {
    let mut renderer = WgpuSpatialRenderer::new_headless()?;
    let mut glyphs = GlyphAtlas::new();

    // Realistic 64mm IPD binocular stereo eye pair + tracked right-hand ray pointing at center cards
    let eyes = [
        EyeView {
            position: [-0.032, 1.60, 0.0],
            orientation: [0.0, 0.0, 0.0, 1.0],
            fov: [-0.82, 0.78, 0.76, -0.76],
        },
        EyeView {
            position: [0.032, 1.60, 0.0],
            orientation: [0.0, 0.0, 0.0, 1.0],
            fov: [-0.78, 0.82, 0.76, -0.76],
        },
    ];
    let right_hand = HandControllerInput {
        active: true,
        grip_pose: Pose3D {
            position: [0.22, 1.34, -0.32],
            orientation: [0.0, 0.0, 0.0, 1.0],
        },
        aim_origin: [0.22, 1.35, -0.35],
        aim_dir: xr_files::spatial::vec3_normalize([-0.14, 0.12, -1.35]),
        ..Default::default()
    };
    let hands = [HandControllerInput::default(), right_hand];

    let width = 1280u32;
    let height = 960u32;
    let duration = Duration::from_secs(max_seconds.unwrap_or(0));
    let started = Instant::now();

    let last_stereo = loop {
        if let Some(ctl) = control {
            ctl.poll_and_apply(app);
        }
        let scene = app.step_and_build_scene(&eyes, &hands, &mut glyphs);
        let _ = app.take_haptics();
        renderer.sync_textures(
            &mut glyphs,
            &mut app.pending_gpu_thumbs,
            &mut app.pending_gpu_inspect_image,
        );
        let pair = renderer.render_stereo_offscreen_rgba(width, height, &eyes, &scene)?;
        app.submitted_frames += 1;

        let min_frames = if capture_dir.is_some() { 14 } else { 3 };
        if app.shutdown_requested() || (duration.is_zero() && app.submitted_frames >= min_frames) {
            break pair;
        }
        if !duration.is_zero() && started.elapsed() >= duration {
            break pair;
        }
        thread::sleep(Duration::from_millis(16));
    };

    if let Some(dir) = capture_dir {
        let [left_rgba, right_rgba] = last_stereo;
        fs::create_dir_all(&dir)?;
        image::save_buffer(
            dir.join("left.png"),
            &left_rgba,
            width,
            height,
            image::ColorType::Rgba8,
        )?;
        image::save_buffer(
            dir.join("right.png"),
            &right_rgba,
            width,
            height,
            image::ColorType::Rgba8,
        )?;
        let summary = serde_json::to_string_pretty(&app.status_snapshot())?;
        fs::write(dir.join("summary.json"), format!("{summary}\n"))?;
    }
    Ok(())
}
