// SPDX-License-Identifier: GPL-3.0-or-later
//! Native OpenXR Stereo Session + `wgpu::hal::api::Vulkan` (`XR_KHR_vulkan_enable2`) Bridge.
//! Renders true binocular stereo views directly into OpenXR single-layer swapchains
//! using `WgpuSpatialRenderer`, supporting Monado, WiVRn, and SteamVR runtimes.

use crate::{
    app::SpatialFilesApp,
    control::ControlServer,
    renderer::WgpuSpatialRenderer,
    spatial::EyeView,
    text::GlyphAtlas,
    xr_input::XrControllers,
};
use anyhow::{anyhow, bail, ensure, Context, Result};
use ash::{vk, vk::Handle};
use openxr as xr;
use std::{env, ffi::CStr, fs, path::PathBuf, thread, time::Duration};

const STEREO: xr::ViewConfigurationType = xr::ViewConfigurationType::PRIMARY_STEREO;

pub fn probe_openxr_available() -> bool {
    // SAFETY: Dynamically loads `libopenxr_loader.so.1` to probe runtime availability.
    let Ok(entry) = (unsafe { xr::Entry::load(&()) }) else {
        return false;
    };
    let Ok(exts) = entry.enumerate_extensions() else {
        return false;
    };
    exts.khr_vulkan_enable2
}

pub fn run_openxr(
    app: &mut SpatialFilesApp,
    control: Option<&ControlServer>,
    max_seconds: Option<u64>,
) -> Result<()> {
    // SAFETY: Loads the system OpenXR loader shared library.
    let entry = unsafe { xr::Entry::load(&()).context("加载 libopenxr_loader 失败")? };
    let available = entry.enumerate_extensions()?;
    ensure!(
        available.khr_vulkan_enable2,
        "当前 OpenXR 运行时不支持 XR_KHR_vulkan_enable2"
    );

    let mut extensions = xr::ExtensionSet::default();
    extensions.khr_vulkan_enable2 = true;
    extensions.bd_controller_interaction = available.bd_controller_interaction;

    let xr_instance = entry.create_instance(
        &xr::ApplicationInfo {
            application_name: "openKylin XR Files",
            application_version: 1,
            engine_name: "wgpu-spatial-simd",
            engine_version: 1,
            api_version: xr::Version::new(1, 0, 0),
        },
        &extensions,
        &[],
        &(),
    )?;
    let props = xr_instance.properties()?;
    eprintln!(
        "XR Files OpenXR runtime: {} {}",
        props.runtime_name, props.runtime_version
    );

    let system = xr_instance
        .system(xr::FormFactor::HEAD_MOUNTED_DISPLAY)
        .context("未检测到 OpenXR 头显系统，请检查运行时与头显连接")?;

    let blend_modes = xr_instance.enumerate_environment_blend_modes(system, STEREO)?;
    // Prefer ALPHA_BLEND passthrough when available, falling back to OPAQUE / ADDITIVE
    let blend = [
        xr::EnvironmentBlendMode::ALPHA_BLEND,
        xr::EnvironmentBlendMode::OPAQUE,
        xr::EnvironmentBlendMode::ADDITIVE,
    ]
    .into_iter()
    .find(|m| blend_modes.contains(m))
    .context("OpenXR 运行时无可用混合模式")?;
    let clear_alpha = if blend == xr::EnvironmentBlendMode::ALPHA_BLEND {
        0.0
    } else {
        1.0
    };

    let view_configs = xr_instance.enumerate_view_configuration_views(system, STEREO)?;
    ensure!(view_configs.len() == 2, "PRIMARY_STEREO 必须返回两个视图");
    let width = view_configs[0]
        .recommended_image_rect_width
        .max(view_configs[1].recommended_image_rect_width)
        .min(view_configs[0].max_image_rect_width)
        .min(view_configs[1].max_image_rect_width)
        .max(64);
    let height = view_configs[0]
        .recommended_image_rect_height
        .max(view_configs[1].recommended_image_rect_height)
        .min(view_configs[0].max_image_rect_height)
        .min(view_configs[1].max_image_rect_height)
        .max(64);
    let rect = xr::Rect2Di {
        offset: xr::Offset2Di { x: 0, y: 0 },
        extent: xr::Extent2Di {
            width: width.try_into()?,
            height: height.try_into()?,
        },
    };

    let (wgpu_device, wgpu_queue, raw_vk_instance, raw_vk_phd, raw_vk_dev, queue_family) =
        create_wgpu_on_openxr_vulkan(&xr_instance, system)?;

    // SAFETY: Vulkan handles were created via `XR_KHR_vulkan_enable2` for this `system`.
    let (session, mut frame_waiter, mut frame_stream) = unsafe {
        xr_instance.create_session::<xr::Vulkan>(
            system,
            &xr::vulkan::SessionCreateInfo {
                instance: raw_vk_instance.as_raw() as _,
                physical_device: raw_vk_phd.as_raw() as _,
                device: raw_vk_dev.as_raw() as _,
                queue_family_index: queue_family,
                queue_index: 0,
            },
        )?
    };

    let base_space =
        session.create_reference_space(xr::ReferenceSpaceType::LOCAL, xr::Posef::IDENTITY)?;
    let view_space =
        session.create_reference_space(xr::ReferenceSpaceType::VIEW, xr::Posef::IDENTITY)?;
    let mut seat_space: Option<xr::Space> = None;

    let mut controllers = XrControllers::new(
        &xr_instance,
        &session,
        extensions.bd_controller_interaction,
    )?;

    let runtime_formats = session.enumerate_swapchain_formats()?;
    let (vk_format, wgpu_format) = [
        (
            vk::Format::R8G8B8A8_SRGB,
            wgpu::TextureFormat::Rgba8UnormSrgb,
        ),
        (
            vk::Format::B8G8R8A8_SRGB,
            wgpu::TextureFormat::Bgra8UnormSrgb,
        ),
        (vk::Format::R8G8B8A8_UNORM, wgpu::TextureFormat::Rgba8Unorm),
        (vk::Format::B8G8R8A8_UNORM, wgpu::TextureFormat::Bgra8Unorm),
    ]
    .into_iter()
    .find(|(vf, _)| runtime_formats.contains(&(vf.as_raw() as _)))
    .ok_or_else(|| anyhow!("OpenXR 运行时不提供兼容的 8-bit RGBA/BGRA 交换链格式"))?;

    // Separate single-layer swapchains per eye avoid cross-process array-layer assumptions on SM8250/Monado
    let mut chains = Vec::with_capacity(2);
    let mut swapchain_views: [Vec<(wgpu::Texture, wgpu::TextureView)>; 2] =
        [Vec::new(), Vec::new()];
    for eye_idx in 0..2 {
        let chain = session.create_swapchain(&xr::SwapchainCreateInfo {
            create_flags: xr::SwapchainCreateFlags::EMPTY,
            usage_flags: xr::SwapchainUsageFlags::COLOR_ATTACHMENT
                | xr::SwapchainUsageFlags::SAMPLED,
            format: vk_format.as_raw() as _,
            sample_count: 1,
            width,
            height,
            face_count: 1,
            array_size: 1,
            mip_count: 1,
        })?;
        for raw_img in chain.enumerate_images()? {
            let vk_img = vk::Image::from_raw(raw_img);
            let wrapped = wrap_openxr_swapchain_image(&wgpu_device, vk_img, width, height, wgpu_format);
            let view = wrapped.create_view(&wgpu::TextureViewDescriptor::default());
            swapchain_views[eye_idx].push((wrapped, view));
        }
        chains.push(chain);
    }

    let mut renderer = WgpuSpatialRenderer::new(wgpu_device, wgpu_queue, wgpu_format)?;
    let mut glyphs = GlyphAtlas::new();

    let recenter_file = env::var_os("XR_APPLICATION_RECENTER_FILE").map(PathBuf::from);
    let deadline = max_seconds
        .filter(|&s| s > 0)
        .map(|s| std::time::Instant::now() + Duration::from_secs(s));

    let mut events = xr::EventDataBuffer::new();
    let (mut running, mut focused, mut exiting) = (false, false, false);
    let mut recenter = true;
    let mut space_changed_at: Option<i64> = None;

    let loop_result: Result<()> = (|| {
        loop {
            while let Some(ev) = xr_instance.poll_event(&mut events)? {
                match ev {
                    xr::Event::SessionStateChanged(sc) => {
                        focused = sc.state() == xr::SessionState::FOCUSED;
                        if !focused {
                            controllers.reset();
                        }
                        match sc.state() {
                            xr::SessionState::READY => {
                                session.begin(STEREO)?;
                                running = true;
                            }
                            xr::SessionState::STOPPING => {
                                session.end()?;
                                running = false;
                                if exiting {
                                    return Ok(());
                                }
                            }
                            xr::SessionState::EXITING => return Ok(()),
                            xr::SessionState::LOSS_PENDING => bail!("OpenXR 会话丢失"),
                            _ => {}
                        }
                    }
                    xr::Event::InstanceLossPending(_) => bail!("OpenXR 运行时连接中断"),
                    xr::Event::ReferenceSpaceChangePending(rc)
                        if rc.reference_space_type() == xr::ReferenceSpaceType::LOCAL =>
                    {
                        space_changed_at = Some(rc.change_time().as_nanos());
                    }
                    _ => {}
                }
            }

            if let Some(ctl) = control {
                ctl.poll_and_apply(app);
            }
            if let Some(ref rf) = recenter_file {
                if rf.exists() {
                    let _ = fs::remove_file(rf);
                    recenter = true;
                }
            }
            recenter |= app.take_recenter_request();

            if deadline.is_some_and(|dl| std::time::Instant::now() >= dl) {
                app.request_shutdown();
            }

            if app.shutdown_requested() && !exiting {
                if !running {
                    return Ok(());
                }
                session.request_exit()?;
                exiting = true;
            }

            if !running {
                thread::sleep(Duration::from_millis(16));
                continue;
            }

            let frame_state = frame_waiter.wait()?;
            frame_stream.begin()?;
            let display_time = frame_state.predicted_display_time;

            if !frame_state.should_render || exiting {
                controllers.reset();
                frame_stream.end(display_time, blend, &[])?;
                continue;
            }

            if space_changed_at.is_some_and(|chg| display_time.as_nanos() >= chg) {
                space_changed_at = None;
                recenter = true;
                controllers.reset();
            }

            if recenter {
                if let Some(pose) = upright_seat_pose(view_space.locate(&base_space, display_time)?) {
                    seat_space =
                        Some(session.create_reference_space(xr::ReferenceSpaceType::LOCAL, pose)?);
                    recenter = false;
                    controllers.reset();
                }
            }

            let active_space = seat_space.as_ref().unwrap_or(&base_space);
            let (view_flags, xr_views) = session.locate_views(STEREO, display_time, active_space)?;
            if !view_flags.contains(
                xr::ViewStateFlags::ORIENTATION_VALID | xr::ViewStateFlags::POSITION_VALID,
            ) {
                controllers.reset();
                frame_stream.end(display_time, blend, &[])?;
                continue;
            }
            ensure!(xr_views.len() == 2, "双目视图数量异常");

            let hand_inputs = if focused {
                controllers.poll(&session, active_space, display_time)?
            } else {
                Default::default()
            };

            let eyes = [xr_view_to_eye(&xr_views[0]), xr_view_to_eye(&xr_views[1])];
            let scene = app.step_and_build_scene(&eyes, &hand_inputs, &mut glyphs);
            let haptics = app.take_haptics();
            if focused {
                controllers.apply_haptics(&session, haptics);
            }

            renderer.sync_textures(
                &mut glyphs,
                &mut app.pending_gpu_thumbs,
                &mut app.pending_gpu_inspect_image,
            );

            let left_idx = chains[0].acquire_image()? as usize;
            chains[0].wait_image(xr::Duration::INFINITE)?;
            let right_idx = match chains[1].acquire_image() {
                Ok(i) => i as usize,
                Err(e) => {
                    let _ = chains[0].release_image();
                    return Err(e.into());
                }
            };
            chains[1].wait_image(xr::Duration::INFINITE)?;

            renderer.render_views(
                &[
                    (&swapchain_views[0][left_idx].1, width, height, eyes[0]),
                    (&swapchain_views[1][right_idx].1, width, height, eyes[1]),
                ],
                &scene,
                clear_alpha,
            );

            chains[0].release_image()?;
            chains[1].release_image()?;

            let proj_views = [
                xr::CompositionLayerProjectionView::new()
                    .pose(xr_views[0].pose)
                    .fov(xr_views[0].fov)
                    .sub_image(
                        xr::SwapchainSubImage::new()
                            .swapchain(&chains[0])
                            .image_array_index(0)
                            .image_rect(rect),
                    ),
                xr::CompositionLayerProjectionView::new()
                    .pose(xr_views[1].pose)
                    .fov(xr_views[1].fov)
                    .sub_image(
                        xr::SwapchainSubImage::new()
                            .swapchain(&chains[1])
                            .image_array_index(0)
                            .image_rect(rect),
                    ),
            ];
            frame_stream.end(
                display_time,
                blend,
                &[&xr::CompositionLayerProjection::new()
                    .space(active_space)
                    .views(&proj_views)],
            )?;
            renderer.device.poll(wgpu::Maintain::Poll);
            app.submitted_frames += 1;
        }
    })();

    // Ensure GPU finishes before dropping wgpu views, swapchains, and OpenXR session
    renderer.device.poll(wgpu::Maintain::Wait);
    drop(swapchain_views);
    drop(chains);
    drop(controllers);
    drop(seat_space);
    drop(view_space);
    drop(base_space);
    drop(frame_stream);
    drop(frame_waiter);
    drop(session);
    drop(renderer);

    loop_result
}

fn create_wgpu_on_openxr_vulkan(
    xr_instance: &xr::Instance,
    system: xr::SystemId,
) -> Result<(
    wgpu::Device,
    wgpu::Queue,
    vk::Instance,
    vk::PhysicalDevice,
    vk::Device,
    u32,
)> {
    let reqs = xr_instance.graphics_requirements::<xr::Vulkan>(system)?;
    // SAFETY: Bridges OpenXR `XR_KHR_vulkan_enable2` instance & logical device into `wgpu::hal::vulkan`.
    unsafe {
        let vk_entry = ash::Entry::load().context("加载 libvulkan.so.1 失败")?;
        let loader_ver = vk_entry
            .try_enumerate_instance_version()?
            .unwrap_or(vk::API_VERSION_1_1);
        let runtime_max_api = vk::make_api_version(
            0,
            reqs.max_api_version_supported.major() as u32,
            reqs.max_api_version_supported.minor() as u32,
            0,
        );
        // wgpu uses promoted core entry points on modern devices. Request the
        // highest mutually supported version instead of the runtime's minimum.
        let target_api = loader_ver.min(runtime_max_api).min(vk::API_VERSION_1_3);
        let runtime_min_api = vk::make_api_version(
            0,
            reqs.min_api_version_supported.major() as u32,
            reqs.min_api_version_supported.minor() as u32,
            0,
        );
        ensure!(
            target_api >= runtime_min_api.max(vk::API_VERSION_1_1),
            "OpenXR 与 Vulkan 加载器没有兼容的 Vulkan 1.1+ 版本"
        );

        let flags = wgpu::InstanceFlags::ALLOW_UNDERLYING_NONCOMPLIANT_ADAPTER;
        let desired_inst_exts =
            <wgpu::hal::api::Vulkan as wgpu::hal::Api>::Instance::desired_extensions(
                &vk_entry, target_api, flags,
            )
            .unwrap_or_default();
        let inst_ext_ptrs: Vec<*const std::ffi::c_char> =
            desired_inst_exts.iter().map(|s| s.as_ptr()).collect();

        let app_info = vk::ApplicationInfo::default()
            .application_name(c"openKylin XR Files")
            .engine_name(c"wgpu-spatial-simd")
            .api_version(target_api);
        let inst_create = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&inst_ext_ptrs);

        let get_proc = std::mem::transmute::<
            vk::PFN_vkGetInstanceProcAddr,
            xr::sys::platform::VkGetInstanceProcAddr,
        >(vk_entry.static_fn().get_instance_proc_addr);

        let raw_inst_ptr = xr_instance
            .create_vulkan_instance(system, get_proc, &inst_create as *const _ as *const _)?
            .map_err(vk::Result::from_raw)?;
        let raw_vk_instance = vk::Instance::from_raw(raw_inst_ptr as _);
        let ash_instance = ash::Instance::load(vk_entry.static_fn(), raw_vk_instance);

        let hal_instance = <wgpu::hal::api::Vulkan as wgpu::hal::Api>::Instance::from_raw(
            vk_entry,
            ash_instance.clone(),
            target_api,
            0,
            None,
            desired_inst_exts,
            flags,
            false,
            None,
        )?;

        let raw_phd_ptr = xr_instance
            .vulkan_graphics_device(system, raw_vk_instance.as_raw() as _)?
            as u64;
        let raw_vk_phd = vk::PhysicalDevice::from_raw(raw_phd_ptr);

        let queue_family = ash_instance
            .get_physical_device_queue_family_properties(raw_vk_phd)
            .iter()
            .position(|p| p.queue_count > 0 && p.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .context("OpenXR Vulkan 物理设备无图形队列")? as u32;

        let exposed = hal_instance
            .expose_adapter(raw_vk_phd)
            .context("wgpu-hal 无法暴露 OpenXR 选择的 Vulkan 物理设备")?;

        let features = wgpu::Features::empty();
        let enabled_dev_exts = exposed.adapter.required_device_extensions(features);
        let dev_ext_ptrs: Vec<*const std::ffi::c_char> =
            enabled_dev_exts.iter().map(|s| s.as_ptr()).collect();
        let mut enabled_phd_features = exposed
            .adapter
            .physical_device_features(&enabled_dev_exts, features);

        let priorities = [1.0f32];
        let queue_infos = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&priorities)];
        let dev_create_base = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&dev_ext_ptrs);
        let dev_create = enabled_phd_features.add_to_device_create(dev_create_base);

        let raw_dev_ptr = xr_instance
            .create_vulkan_device(
                system,
                get_proc,
                raw_vk_phd.as_raw() as _,
                &dev_create as *const _ as *const _,
            )?
            .map_err(vk::Result::from_raw)?;
        let raw_vk_dev = vk::Device::from_raw(raw_dev_ptr as _);
        let ash_device = ash::Device::load(ash_instance.fp_v1_0(), raw_vk_dev);

        let gpu_props = ash_instance.get_physical_device_properties(raw_vk_phd);
        eprintln!(
            "XR Files Vulkan GPU: {}",
            CStr::from_ptr(gpu_props.device_name.as_ptr()).to_string_lossy()
        );

        let memory_hints = wgpu::MemoryHints::Performance;
        let open_device = exposed.adapter.device_from_raw(
            ash_device,
            None,
            &enabled_dev_exts,
            features,
            &memory_hints,
            queue_family,
            0,
        )?;

        let wgpu_instance = wgpu::Instance::from_hal::<wgpu::hal::api::Vulkan>(hal_instance);
        let wgpu_adapter = wgpu_instance.create_adapter_from_hal(exposed);
        let limits = wgpu_adapter.limits();
        let (wgpu_device, wgpu_queue) = wgpu_adapter.create_device_from_hal(
            open_device,
            &wgpu::DeviceDescriptor {
                label: Some("xr-files-openxr-wgpu-device"),
                required_features: features,
                required_limits: limits,
                memory_hints,
            },
            None,
        )?;

        Ok((
            wgpu_device,
            wgpu_queue,
            raw_vk_instance,
            raw_vk_phd,
            raw_vk_dev,
            queue_family,
        ))
    }
}

fn wrap_openxr_swapchain_image(
    device: &wgpu::Device,
    vk_image: vk::Image,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
) -> wgpu::Texture {
    let hal_desc = wgpu::hal::TextureDescriptor {
        label: Some("xr-files-openxr-swapchain-image"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::hal::TextureUses::COLOR_TARGET | wgpu::hal::TextureUses::RESOURCE,
        memory_flags: wgpu::hal::MemoryFlags::empty(),
        view_formats: vec![],
    };
    let wgpu_desc = wgpu::TextureDescriptor {
        label: Some("xr-files-openxr-swapchain-tex"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    };
    // SAFETY: `vk_image` belongs to an active OpenXR swapchain; the no-op DropCallback prevents
    // `wgpu` from calling `vkDestroyImage` on images owned by the OpenXR runtime.
    unsafe {
        let hal_tex = <wgpu::hal::api::Vulkan as wgpu::hal::Api>::Device::texture_from_raw(
            vk_image,
            &hal_desc,
            Some(Box::new(|| {})),
        );
        device.create_texture_from_hal::<wgpu::hal::api::Vulkan>(hal_tex, &wgpu_desc)
    }
}

fn upright_seat_pose(loc: xr::SpaceLocation) -> Option<xr::Posef> {
    if !loc.location_flags.contains(
        xr::SpaceLocationFlags::POSITION_VALID | xr::SpaceLocationFlags::ORIENTATION_VALID,
    ) {
        return None;
    }
    let p = loc.pose.position;
    let q = loc.pose.orientation;
    let fx = 2.0 * (q.x * q.z + q.w * q.y);
    let fz = 1.0 - 2.0 * (q.x * q.x + q.y * q.y);
    if ![fx, fz, p.x, p.y, p.z].iter().all(|v| v.is_finite()) || fx * fx + fz * fz < 1e-4 {
        return None;
    }
    let (sy, cy) = (fx.atan2(fz) * 0.5).sin_cos();
    Some(xr::Posef {
        orientation: xr::Quaternionf {
            x: 0.0,
            y: sy,
            z: 0.0,
            w: cy,
        },
        position: xr::Vector3f {
            x: p.x,
            y: 0.0, // Keep LOCAL floor/stage height offset natural while centering XZ + Yaw
            z: p.z,
        },
    })
}

fn xr_view_to_eye(view: &xr::View) -> EyeView {
    let p = view.pose.position;
    let q = view.pose.orientation;
    let f = view.fov;
    EyeView {
        position: [p.x, p.y, p.z],
        orientation: [q.x, q.y, q.z, q.w],
        fov: [f.angle_left, f.angle_right, f.angle_up, f.angle_down],
    }
}
