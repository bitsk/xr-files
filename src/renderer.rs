// SPDX-License-Identifier: GPL-3.0-or-later
//! Unified `wgpu` 3D Spatial Stereo Renderer for `xr-files` (UKUI XR "Qingkong" Edition).
//! Includes:
//! - 360° Qingkong equirectangular panorama background (`qingkong-360.png` + procedural fallback)
//! - Supersampled 2.5D UKUI icon layers (`0..8`) + live SIMD-mipmapped photo thumbnails (`8..64`)
//! - Daylight hemisphere-lit 3D solids & holograms + exterior-shadowed frosted silver glass

use crate::{
    app::{FrameScene, QuadInstance, SolidVertex},
    icons::{self, BUILTIN_ICON_COUNT},
    preview::MipmappedRgba,
    simd,
    spatial::{mat4_view_proj, quat_normalize, EyeView},
    text::{GlyphAtlas, ATLAS_SIZE},
};
use anyhow::{Context, Result};
use bytemuck::{Pod, Zeroable};
use std::{env, path::Path};
use wgpu::util::DeviceExt;

pub const THUMB_SIZE: u32 = 128;
pub const THUMB_LAYERS: u32 = 96;
pub const INSPECT_TEX_SIZE: u32 = 384;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 4],
    camera_quat: [f32; 4],
    fov_tan: [f32; 4],
}

pub struct WgpuSpatialRenderer {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub target_format: wgpu::TextureFormat,
    sky_pipeline: wgpu::RenderPipeline,
    solid_pipeline: wgpu::RenderPipeline,
    quad_pipeline: wgpu::RenderPipeline,
    camera_buffers: [wgpu::Buffer; 2],
    camera_bind_groups: [wgpu::BindGroup; 2],
    texture_bind_group: wgpu::BindGroup,
    glyph_texture: wgpu::Texture,
    thumb_array_texture: wgpu::Texture,
    inspect_texture: wgpu::Texture,
    depth_textures: [Option<(wgpu::TextureView, u32, u32)>; 2],
    wayland_offscreen: Option<(wgpu::Texture, wgpu::TextureView, wgpu::Buffer, u32, u32, u32)>,
    solid_vb: wgpu::Buffer,
    solid_vb_cap: usize,
    quad_ib: wgpu::Buffer,
    quad_ib_cap: usize,
}

impl WgpuSpatialRenderer {
    pub fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        target_format: wgpu::TextureFormat,
    ) -> Result<Self> {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("xr-files-qingkong-wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders.wgsl").into()),
        });

        let camera_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("xr-files-camera-bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let tex_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("xr-files-textures-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let sky_and_quad_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("xr-files-sky-quad-pl"),
            bind_group_layouts: &[&camera_bgl, &tex_bgl],
            push_constant_ranges: &[],
        });

        let solid_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("xr-files-solid-pl"),
            bind_group_layouts: &[&camera_bgl],
            push_constant_ranges: &[],
        });

        let premul_blend = wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING;

        let sky_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("xr-files-qingkong-sky-pipeline"),
            layout: Some(&sky_and_quad_pl),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_sky"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_sky"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });

        let solid_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("xr-files-solid-pipeline"),
            layout: Some(&solid_pl),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_solid"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<SolidVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![
                        0 => Float32x3,
                        1 => Float32x3,
                        2 => Float32x4,
                        3 => Float32,
                    ],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_solid"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(premul_blend),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });

        let quad_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("xr-files-quad-pipeline"),
            layout: Some(&sky_and_quad_pl),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_quad"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<QuadInstance>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &wgpu::vertex_attr_array![
                        0 => Float32x3,
                        1 => Float32x3,
                        2 => Float32x3,
                        3 => Float32x4,
                        4 => Float32x4,
                        5 => Float32x4,
                        6 => Float32x4,
                    ],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_quad"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(premul_blend),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });

        let glyph_mips = (ATLAS_SIZE.ilog2() + 1).max(1);
        let glyph_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("xr-files-glyph-atlas"),
            size: wgpu::Extent3d {
                width: ATLAS_SIZE,
                height: ATLAS_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: glyph_mips,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let thumb_mips = (THUMB_SIZE.ilog2() + 1).max(1);
        let thumb_array_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("xr-files-thumb-array"),
            size: wgpu::Extent3d {
                width: THUMB_SIZE,
                height: THUMB_SIZE,
                depth_or_array_layers: THUMB_LAYERS,
            },
            mip_level_count: thumb_mips,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // Preload supersampled 2.5D UKUI Qingkong icons into slots `0..8`
        for (slot, icon_mips) in icons::build_qingkong_icons() {
            if slot < BUILTIN_ICON_COUNT {
                upload_mip_chain_2d(&queue, &thumb_array_texture, slot, &icon_mips.mips);
            }
        }

        let inspect_mips = (INSPECT_TEX_SIZE.ilog2() + 1).max(1);
        let inspect_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("xr-files-inspect-tex"),
            size: wgpu::Extent3d {
                width: INSPECT_TEX_SIZE,
                height: INSPECT_TEX_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: inspect_mips,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let (sky_rgba, sky_w, sky_h) = load_or_synthesize_qingkong_sky();
        let sky_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("xr-files-qingkong-sky"),
            size: wgpu::Extent3d {
                width: sky_w,
                height: sky_h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &sky_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &sky_rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(sky_w * 4),
                rows_per_image: Some(sky_h),
            },
            wgpu::Extent3d {
                width: sky_w,
                height: sky_h,
                depth_or_array_layers: 1,
            },
        );

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("xr-files-mip-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            anisotropy_clamp: 4,
            ..Default::default()
        });

        let glyph_view = glyph_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let thumb_view = thumb_array_texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let inspect_view = inspect_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sky_view = sky_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("xr-files-tex-bg"),
            layout: &tex_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&glyph_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&thumb_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&inspect_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&sky_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        let default_cam = CameraUniform {
            view_proj: [[0.0; 4]; 4],
            camera_pos: [0.0, 1.6, 0.0, 1.0],
            camera_quat: [0.0, 0.0, 0.0, 1.0],
            fov_tan: [-0.8, 0.8, 0.75, -0.75],
        };
        let cam_buf0 = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("xr-files-cam-0"),
            contents: bytemuck::bytes_of(&default_cam),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let cam_buf1 = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("xr-files-cam-1"),
            contents: bytemuck::bytes_of(&default_cam),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let cam_bg0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("xr-files-cam-bg-0"),
            layout: &camera_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: cam_buf0.as_entire_binding(),
            }],
        });
        let cam_bg1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("xr-files-cam-bg-1"),
            layout: &camera_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: cam_buf1.as_entire_binding(),
            }],
        });

        let solid_cap = 16_384usize;
        let solid_vb = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("xr-files-solid-vb"),
            size: (solid_cap * std::mem::size_of::<SolidVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let quad_cap = 12_288usize;
        let quad_ib = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("xr-files-quad-ib"),
            size: (quad_cap * std::mem::size_of::<QuadInstance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Self {
            device,
            queue,
            target_format,
            sky_pipeline,
            solid_pipeline,
            quad_pipeline,
            camera_buffers: [cam_buf0, cam_buf1],
            camera_bind_groups: [cam_bg0, cam_bg1],
            texture_bind_group,
            glyph_texture,
            thumb_array_texture,
            inspect_texture,
            depth_textures: [None, None],
            wayland_offscreen: None,
            solid_vb,
            solid_vb_cap: solid_cap,
            quad_ib,
            quad_ib_cap: quad_cap,
        })
    }

    pub fn sync_textures(
        &mut self,
        glyphs: &mut GlyphAtlas,
        new_thumbs: &mut Vec<(u32, MipmappedRgba)>,
        new_inspect: &mut Option<MipmappedRgba>,
    ) {
        if glyphs.dirty {
            let mips = glyphs.build_mip_chain();
            upload_mip_chain_2d(&self.queue, &self.glyph_texture, 0, &mips);
        }
        for (slot, img) in new_thumbs.drain(..) {
            if slot >= BUILTIN_ICON_COUNT && slot < THUMB_LAYERS {
                let fitted = fit_cover_square_canvas(&img, THUMB_SIZE);
                upload_mip_chain_2d(&self.queue, &self.thumb_array_texture, slot, &fitted);
            }
        }
        if let Some(img) = new_inspect.take() {
            let fitted = fit_cover_square_canvas(&img, INSPECT_TEX_SIZE);
            upload_mip_chain_2d(&self.queue, &self.inspect_texture, 0, &fitted);
        }
    }

    pub fn upload_scene(&mut self, scene: &FrameScene) {
        if scene.solid_vertices.len() > self.solid_vb_cap {
            self.solid_vb_cap = scene.solid_vertices.len().next_power_of_two();
            self.solid_vb = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("xr-files-solid-vb"),
                size: (self.solid_vb_cap * std::mem::size_of::<SolidVertex>()) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        if !scene.solid_vertices.is_empty() {
            self.queue.write_buffer(
                &self.solid_vb,
                0,
                bytemuck::cast_slice(&scene.solid_vertices),
            );
        }

        if scene.quads.len() > self.quad_ib_cap {
            self.quad_ib_cap = scene.quads.len().next_power_of_two();
            self.quad_ib = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("xr-files-quad-ib"),
                size: (self.quad_ib_cap * std::mem::size_of::<QuadInstance>()) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        if !scene.quads.is_empty() {
            self.queue
                .write_buffer(&self.quad_ib, 0, bytemuck::cast_slice(&scene.quads));
        }
    }

    pub fn render_views(
        &mut self,
        targets: &[(&wgpu::TextureView, u32, u32, EyeView)],
        scene: &FrameScene,
        clear_alpha: f64,
    ) {
        self.upload_scene(scene);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("xr-files-stereo-encoder"),
            });

        for (eye_idx, (color_view, width, height, eye)) in targets.iter().take(2).enumerate() {
            self.ensure_depth_view(eye_idx, *width, *height);
            let depth_view = &self.depth_textures[eye_idx].as_ref().unwrap().0;

            let q = quat_normalize(eye.orientation);
            let cam = CameraUniform {
                view_proj: mat4_view_proj(eye, 0.04, 40.0),
                camera_pos: [eye.position[0], eye.position[1], eye.position[2], 1.0],
                camera_quat: q,
                fov_tan: [
                    eye.fov[0].tan(),
                    eye.fov[1].tan(),
                    eye.fov[2].tan(),
                    eye.fov[3].tan(),
                ],
            };
            self.queue
                .write_buffer(&self.camera_buffers[eye_idx], 0, bytemuck::bytes_of(&cam));

            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("xr-files-eye-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.72 * clear_alpha,
                            g: 0.84 * clear_alpha,
                            b: 0.98 * clear_alpha,
                            a: clear_alpha,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            // 0. Draw 360° Qingkong sky panorama in opaque sessions (keeps passthrough transparent when clear_alpha == 0)
            if clear_alpha > 0.5 {
                rpass.set_pipeline(&self.sky_pipeline);
                rpass.set_bind_group(0, &self.camera_bind_groups[eye_idx], &[]);
                rpass.set_bind_group(1, &self.texture_bind_group, &[]);
                rpass.draw(0..3, 0..1);
            }

            // 1. Draw daylight-lit 3D solids (3D treemap podium/blocks, 3D OBJ/GLB hologram, 6DoF beam)
            if !scene.solid_vertices.is_empty() {
                rpass.set_pipeline(&self.solid_pipeline);
                rpass.set_bind_group(0, &self.camera_bind_groups[eye_idx], &[]);
                rpass.set_vertex_buffer(0, self.solid_vb.slice(..));
                rpass.draw(0..scene.solid_vertices.len() as u32, 0..1);
            }

            // 2. Draw ordered Qingkong frosted glass, exterior shadows, 2.5D icons & crisp CJK text
            if !scene.quads.is_empty() {
                rpass.set_pipeline(&self.quad_pipeline);
                rpass.set_bind_group(0, &self.camera_bind_groups[eye_idx], &[]);
                rpass.set_bind_group(1, &self.texture_bind_group, &[]);
                rpass.set_vertex_buffer(0, self.quad_ib.slice(..));
                rpass.draw(0..6, 0..scene.quads.len() as u32);
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
    }

    fn ensure_depth_view(&mut self, slot: usize, width: u32, height: u32) {
        let needs_recreate = match &self.depth_textures[slot] {
            Some((_, w, h)) => *w != width || *h != height,
            None => true,
        };
        if needs_recreate {
            let tex = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("xr-files-depth"),
                size: wgpu::Extent3d {
                    width: width.max(1),
                    height: height.max(1),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
            self.depth_textures[slot] = Some((view, width, height));
        }
    }

    pub fn new_headless() -> Result<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            flags: wgpu::InstanceFlags::ALLOW_UNDERLYING_NONCOMPLIANT_ADAPTER,
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .context("未找到可用的 Vulkan GPU / Lavapipe 适配器")?;
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("xr-files-headless-device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))?;
        Self::new(device, queue, wgpu::TextureFormat::Rgba8UnormSrgb)
    }

    /// Renders a single transparent-background (`clear_alpha = 0.0`) frame on Vulkan GPU
    /// and writes Cairo/Wayland-compatible premultiplied sRGB `Argb8888` (`[B, G, R, A]` on little-endian)
    /// directly into `dst_argb8888`, reusing persistent GPU texture & staging buffers.
    pub fn render_offscreen_wayland_argb8888(
        &mut self,
        width: u32,
        height: u32,
        eye: EyeView,
        scene: &FrameScene,
        dst_argb8888: &mut [u8],
    ) -> Result<()> {
        let width = width.max(1);
        let height = height.max(1);
        anyhow::ensure!(
            dst_argb8888.len() >= (width as usize) * (height as usize) * 4,
            "dst_argb8888 buffer too small"
        );

        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let unpadded = width * 4;
        let padded_bpr = ((unpadded + align - 1) / align) * align;

        let needs_alloc = match &self.wayland_offscreen {
            Some((_, _, _, w, h, _)) => *w != width || *h != height,
            None => true,
        };
        if needs_alloc {
            let tex = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("xr-files-wayland-color"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.target_format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
            let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("xr-files-wayland-staging"),
                size: (padded_bpr as u64) * (height as u64),
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            self.wayland_offscreen = Some((tex, view, staging, width, height, padded_bpr));
        }

        // Borrow view via temporary raw split or render before copy
        self.upload_scene(scene);
        self.ensure_depth_view(0, width, height);

        let (tex, view, staging, _, _, bpr) = self.wayland_offscreen.as_ref().unwrap();
        let depth_view = &self.depth_textures[0].as_ref().unwrap().0;

        let q = quat_normalize(eye.orientation);
        let cam = CameraUniform {
            view_proj: mat4_view_proj(&eye, 0.04, 40.0),
            camera_pos: [eye.position[0], eye.position[1], eye.position[2], 1.0],
            camera_quat: q,
            fov_tan: [
                eye.fov[0].tan(),
                eye.fov[1].tan(),
                eye.fov[2].tan(),
                eye.fov[3].tan(),
            ],
        };
        self.queue
            .write_buffer(&self.camera_buffers[0], 0, bytemuck::bytes_of(&cam));

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("xr-files-wayland-enc"),
            });
        {
            let mut rpass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("xr-files-wayland-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            if !scene.solid_vertices.is_empty() {
                rpass.set_pipeline(&self.solid_pipeline);
                rpass.set_bind_group(0, &self.camera_bind_groups[0], &[]);
                rpass.set_vertex_buffer(0, self.solid_vb.slice(..));
                rpass.draw(0..scene.solid_vertices.len() as u32, 0..1);
            }
            if !scene.quads.is_empty() {
                rpass.set_pipeline(&self.quad_pipeline);
                rpass.set_bind_group(0, &self.camera_bind_groups[0], &[]);
                rpass.set_bind_group(1, &self.texture_bind_group, &[]);
                rpass.set_vertex_buffer(0, self.quad_ib.slice(..));
                rpass.draw(0..6, 0..scene.quads.len() as u32);
            }
        }
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(*bpr),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit(std::iter::once(enc.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()??;

        let mapped = slice.get_mapped_range();
        let lut = srgb_premul_alpha_lut();
        let row_bytes = (width as usize) * 4;
        for row in 0..height as usize {
            let src_off = row * (*bpr as usize);
            let dst_off = row * row_bytes;
            convert_hw_rgba_srgb_to_wayland_argb8888(
                &mapped[src_off..src_off + row_bytes],
                &mut dst_argb8888[dst_off..dst_off + row_bytes],
                lut,
            );
        }
        drop(mapped);
        staging.unmap();
        Ok(())
    }

    pub fn render_stereo_offscreen_rgba(
        &mut self,
        width: u32,
        height: u32,
        eyes: &[EyeView; 2],
        scene: &FrameScene,
    ) -> Result<[Vec<u8>; 2]> {
        let make_tex = |label| {
            self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.target_format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            })
        };
        let tex_l = make_tex("xr-files-offscreen-left");
        let tex_r = make_tex("xr-files-offscreen-right");
        let view_l = tex_l.create_view(&wgpu::TextureViewDescriptor::default());
        let view_r = tex_r.create_view(&wgpu::TextureViewDescriptor::default());

        self.render_views(
            &[
                (&view_l, width, height, eyes[0]),
                (&view_r, width, height, eyes[1]),
            ],
            scene,
            1.0,
        );

        Ok([
            readback_texture_rgba(&self.device, &self.queue, &tex_l, width, height)?,
            readback_texture_rgba(&self.device, &self.queue, &tex_r, width, height)?,
        ])
    }
}

fn load_or_synthesize_qingkong_sky() -> (Vec<u8>, u32, u32) {
    let mut candidates = Vec::new();
    if let Some(custom) = env::var_os("XR_FILES_SKY_PNG") {
        candidates.push(custom.to_string_lossy().into_owned());
    }
    candidates.push("/usr/share/xr/environments/qingkong/qingkong-360.png".into());
    candidates.push("../xr-environments/environments/qingkong/qingkong-360.png".into());
    candidates.push("xr-environments/environments/qingkong/qingkong-360.png".into());

    for path in candidates {
        if !Path::new(&path).is_file() {
            continue;
        }
        if let Ok(img) = image::open(&path) {
            let scaled = if img.width() > 2048 {
                img.resize_exact(2048, 1024, image::imageops::FilterType::Triangle)
            } else {
                img
            };
            let rgba = scaled.to_rgba8();
            let (w, h) = (rgba.width().max(1), rgba.height().max(1));
            return (rgba.into_raw(), w, h);
        }
    }
    let (w, h) = (1024, 512);
    (icons::synthesize_qingkong_panorama(w, h), w, h)
}

fn upload_mip_chain_2d(
    queue: &wgpu::Queue,
    tex: &wgpu::Texture,
    array_layer: u32,
    mips: &[(Vec<u8>, u32, u32)],
) {
    let max_mips = tex.mip_level_count();
    for (level, (bytes, w, h)) in mips.iter().enumerate() {
        if level as u32 >= max_mips {
            break;
        }
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: tex,
                mip_level: level as u32,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: array_layer,
                },
                aspect: wgpu::TextureAspect::All,
            },
            bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w * 4),
                rows_per_image: Some(*h),
            },
            wgpu::Extent3d {
                width: *w,
                height: *h,
                depth_or_array_layers: 1,
            },
        );
    }
}

/// Center-crops (`object-fit: cover`) an image onto an `edge × edge` premultiplied RGBA8 square
/// so photo thumbnails fill rounded spatial cards edge-to-edge without black bars.
fn fit_cover_square_canvas(src: &MipmappedRgba, edge: u32) -> Vec<(Vec<u8>, u32, u32)> {
    let (ref base, sw, sh) = src.mips[0];
    let mut canvas = vec![0u8; (edge * edge * 4) as usize];
    let crop_side = sw.min(sh).max(1);
    let sx_off = (sw - crop_side) / 2;
    let sy_off = (sh - crop_side) / 2;

    for y in 0..edge {
        let sy = sy_off + ((y as u64 * crop_side as u64) / edge as u64).min((crop_side - 1) as u64) as u32;
        for x in 0..edge {
            let sx = sx_off + ((x as u64 * crop_side as u64) / edge as u64).min((crop_side - 1) as u64) as u32;
            let s_off = ((sy * sw + sx) * 4) as usize;
            let d_off = ((y * edge + x) * 4) as usize;
            canvas[d_off..d_off + 4].copy_from_slice(&base[s_off..s_off + 4]);
        }
    }
    simd::generate_mip_chain_rgba(canvas, edge, edge)
}

fn readback_texture_rgba(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    tex: &wgpu::Texture,
    width: u32,
    height: u32,
) -> Result<Vec<u8>> {
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let unpadded = width * 4;
    let padded_bpr = ((unpadded + align - 1) / align) * align;
    let buf_size = (padded_bpr * height) as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("xr-files-readback"),
        size: buf_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("xr-files-readback-enc"),
    });
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bpr),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(std::iter::once(enc.finish()));
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv()??;
    let mapped = slice.get_mapped_range();
    let mut out = Vec::with_capacity((width * height * 4) as usize);
    for row in 0..height {
        let start = (row * padded_bpr) as usize;
        out.extend_from_slice(&mapped[start..start + unpadded as usize]);
    }
    drop(mapped);
    staging.unmap();
    Ok(out)
}

fn srgb_premul_alpha_lut() -> &'static [u16; 256] {
    static LUT: std::sync::OnceLock<[u16; 256]> = std::sync::OnceLock::new();
    LUT.get_or_init(|| {
        let mut table = [0u16; 256];
        table[255] = 256;
        for a in 1..255usize {
            let af = (a as f32) / 255.0;
            // Hardware `Rgba8UnormSrgb` stores `srgb(L * a) ≈ srgb(L) * a^(1/2.2)`.
            // Wayland/Cairo `Argb8888` (`xr-shell` `spatial.frag`) expects `srgb(L) * a`,
            // which equals `hw_byte * a^(1 - 1/2.2) = hw_byte * a^0.54545`.
            table[a] = (af.powf(0.54545456) * 256.0).round().clamp(0.0, 256.0) as u16;
        }
        table
    })
}

/// Converts GPU `Rgba8UnormSrgb` (linear-premultiplied) pixels into Wayland `wl_shm::Format::Argb8888`
/// (`[B, G, R, A]` sRGB-premultiplied on little-endian), matching `xr-shell`'s `spatial.frag`.
pub fn convert_hw_rgba_srgb_to_wayland_argb8888(
    src_rgba: &[u8],
    dst_bgra: &mut [u8],
    lut: &[u16; 256],
) {
    for (s, d) in src_rgba.chunks_exact(4).zip(dst_bgra.chunks_exact_mut(4)) {
        let a = s[3];
        if a == 255 {
            d[0] = s[2];
            d[1] = s[1];
            d[2] = s[0];
            d[3] = 255;
        } else if a == 0 {
            d[0] = 0;
            d[1] = 0;
            d[2] = 0;
            d[3] = 0;
        } else {
            let k = lut[a as usize];
            d[0] = (((s[2] as u16 * k + 128) >> 8) as u8).min(a);
            d[1] = (((s[1] as u16 * k + 128) >> 8) as u8).min(a);
            d[2] = (((s[0] as u16 * k + 128) >> 8) as u8).min(a);
            d[3] = a;
        }
    }
}
