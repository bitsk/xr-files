// SPDX-License-Identifier: GPL-3.0-or-later
//! End-to-end GPU stereo pixel readback & local control socket regression tests for `xr-files`.

use serde_json::Value;
use std::{
    env, fs,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use xr_files::{
    app::{HandControllerInput, SpatialFilesApp, ViewMode},
    control::{self, ControlServer},
    renderer::WgpuSpatialRenderer,
    spatial::{vec3_normalize, EyeView, Pose3D},
    text::GlyphAtlas,
};

#[test]
fn wgpu_stereo_workbench_and_treemap3d_render_distinct_parallax_pixels() {
    let tmp = env::temp_dir().join(format!(
        "xr-files-gpu-test-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir_all(tmp.join("空间文档")).unwrap();
    fs::write(tmp.join("readme.md"), "# openKylin XR Spatial Files\nSIMD + wgpu\n").unwrap();
    fs::write(
        tmp.join("crystal.obj"),
        "v -1 0 0\nv 1 0 0\nv 0 1.5 0\nv 0 0.5 1\nf 1 2 3\nf 1 2 4\n",
    )
    .unwrap();
    fs::write(tmp.join("movie.mp4"), vec![0x55u8; 64 * 1024]).unwrap();

    let mut app = SpatialFilesApp::new(Some(tmp.clone()));
    // Select the 3D OBJ file and wait briefly so background 3D hologram preview finishes
    if let Some((idx, _)) = app
        .all_entries
        .iter()
        .enumerate()
        .find(|(_, e)| e.name == "crystal.obj")
    {
        app.select_single(idx);
    }
    std::thread::sleep(Duration::from_millis(60));

    let mut renderer = WgpuSpatialRenderer::new_headless().expect("init headless Vulkan wgpu");
    let mut glyphs = GlyphAtlas::new();

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
    let hands = [
        HandControllerInput::default(),
        HandControllerInput {
            active: true,
            grip_pose: Pose3D {
                position: [0.22, 1.34, -0.32],
                orientation: [0.0, 0.0, 0.0, 1.0],
            },
            aim_origin: [0.22, 1.35, -0.35],
            aim_dir: vec3_normalize([-0.12, 0.10, -1.35]),
            ..Default::default()
        },
    ];

    for mode in [ViewMode::Workbench, ViewMode::Treemap3D] {
        app.set_view_mode(mode);
        let scene = app.step_and_build_scene(&eyes, &hands, &mut glyphs);
        assert!(!scene.quads.is_empty(), "quads must not be empty in {mode:?}");
        assert!(
            !scene.solid_vertices.is_empty(),
            "3D solid vertices must not be empty in {mode:?}"
        );
        renderer.sync_textures(
            &mut glyphs,
            &mut app.pending_gpu_thumbs,
            &mut app.pending_gpu_inspect_image,
        );
        let [left, right] = renderer
            .render_stereo_offscreen_rgba(480, 360, &eyes, &scene)
            .expect("stereo offscreen render");

        let bright_pixels = left
            .chunks_exact(4)
            .filter(|px| px[0] > 30 || px[1] > 45 || px[2] > 80)
            .count();
        assert!(
            bright_pixels > 4_000,
            "expected rich rendered spatial UI in {mode:?}, got only {bright_pixels} lit pixels"
        );

        let parallax_diffs = left
            .chunks_exact(4)
            .zip(right.chunks_exact(4))
            .filter(|(l, r)| l != r)
            .count();
        assert!(
            parallax_diffs > 2_000,
            "left and right stereo views must exhibit real 64mm IPD parallax in {mode:?} (diffs={parallax_diffs})"
        );
    }

    let _ = fs::remove_dir_all(&tmp);
}

#[test]
fn control_socket_executes_search_mkdir_view_and_undo() {
    let tmp = env::temp_dir().join(format!(
        "xr-files-ctl-test-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir_all(&tmp).unwrap();
    let sock = tmp.join("control.sock");
    let work = tmp.join("workspace");
    fs::create_dir_all(&work).unwrap();
    fs::write(work.join("alpha.rs"), "fn main() {}\n").unwrap();
    fs::write(work.join("beta.txt"), "hello\n").unwrap();

    let mut app = SpatialFilesApp::new(Some(work.clone()));
    let server = ControlServer::bind(&sock).unwrap();

    // Spawn client thread sending commands while main thread pumps `poll_and_apply`
    let sock_client = sock.clone();
    let client = std::thread::spawn(move || {
        let r1: Value = serde_json::from_str(
            &control::send_raw_command(&sock_client, r#"{"op":"mkdir","name":"三维资产"}"#).unwrap(),
        )
        .unwrap();
        assert_eq!(r1["ok"], true);
        assert_eq!(r1["state"]["undo_depth"], 1);

        let r2: Value = serde_json::from_str(
            &control::send_raw_command(&sock_client, r#"{"op":"search","query":"alpha"}"#).unwrap(),
        )
        .unwrap();
        assert_eq!(r2["ok"], true);
        assert_eq!(r2["state"]["visible_entries"], 1);

        let r3: Value = serde_json::from_str(
            &control::send_raw_command(&sock_client, r#"{"op":"set_view","view":"treemap3d"}"#)
                .unwrap(),
        )
        .unwrap();
        assert_eq!(r3["ok"], true);
        assert_eq!(r3["state"]["view_mode"], "treemap3d");

        let r4: Value = serde_json::from_str(
            &control::send_raw_command(&sock_client, r#"{"op":"undo"}"#).unwrap(),
        )
        .unwrap();
        assert_eq!(r4["ok"], true);
        assert_eq!(r4["state"]["undo_depth"], 0);

        let r5: Value = serde_json::from_str(
            &control::send_raw_command(&sock_client, r#"{"op":"quit"}"#).unwrap(),
        )
        .unwrap();
        assert_eq!(r5["ok"], true);
    });

    let started = std::time::Instant::now();
    while !app.shutdown_requested() && !client.is_finished() {
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "timed out waiting for control socket client"
        );
        server.poll_and_apply(&mut app);
        std::thread::sleep(Duration::from_millis(5));
    }
    server.poll_and_apply(&mut app);
    client.join().unwrap();
    assert!(!work.join("三维资产").exists(), "undo must remove created folder");

    drop(server);
    let _ = fs::remove_dir_all(&tmp);
}
