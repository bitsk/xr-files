// SPDX-License-Identifier: GPL-3.0-or-later
// WGSL Spatial Shaders for openKylin XR Files (`xr-files`)
// - Premultiplied linear alpha composition
// - Per-eye camera_pos Fresnel & half-angle specular parallax (Constraint 3.1)
// - GPU analytical rounded-rectangle SDF for zero-CPU-redraw hover/press states (Constraint 5.1)
// - Lit 3D volumetric extrusions & hologram meshes

struct CameraUniform {
    view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    light_dir: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

// ============================================================================
// Pipeline 1: Lit 3D Solid & Hologram Mesh Pipeline
// ============================================================================

struct SolidVertexIn {
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) emissive: f32,
};

struct SolidVertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) emissive: f32,
};

@vertex
fn vs_solid(in: SolidVertexIn) -> SolidVertexOut {
    var out: SolidVertexOut;
    out.clip_pos = camera.view_proj * vec4<f32>(in.pos, 1.0);
    out.world_pos = in.pos;
    out.world_normal = in.normal;
    out.color = in.color;
    out.emissive = in.emissive;
    return out;
}

@fragment
fn fs_solid(in: SolidVertexOut) -> @location(0) vec4<f32> {
    let n = normalize(in.world_normal);
    let v = normalize(camera.camera_pos.xyz - in.world_pos);
    let l1 = normalize(camera.light_dir.xyz);
    let l2 = normalize(vec3<f32>(-0.55, 0.42, 0.72));

    let ndotl1 = max(dot(n, l1), 0.0);
    let ndotl2 = max(dot(n, l2), 0.0);
    let h = normalize(l1 + v);
    let spec = pow(max(dot(n, h), 0.0), 42.0) * 0.32;
    let fresnel = pow(1.0 - max(abs(dot(n, v)), 0.0), 3.0);

    let ambient = 0.28;
    let lit = in.color.rgb * (ambient + 0.56 * ndotl1 + 0.22 * ndotl2);
    let rim = mix(vec3<f32>(0.35, 0.72, 1.0), in.color.rgb, 0.4) * fresnel * (0.22 + 0.45 * in.emissive);
    let final_rgb = lit + vec3<f32>(spec) + rim + in.color.rgb * in.emissive * 0.55;
    let a = clamp(in.color.a, 0.0, 1.0);
    return vec4<f32>(final_rgb * a, a);
}

// ============================================================================
// Pipeline 2: Instanced Spatial SDF Glass / Card / Text / Thumbnail Quads
// ============================================================================

@group(1) @binding(0)
var glyph_tex: texture_2d<f32>;
@group(1) @binding(1)
var thumb_array_tex: texture_2d_array<f32>;
@group(1) @binding(2)
var inspect_tex: texture_2d<f32>;
@group(1) @binding(3)
var linear_sampler: sampler;

struct QuadInstanceIn {
    @builtin(vertex_index) vid: u32,
    @location(0) center: vec3<f32>,
    @location(1) right_vec: vec3<f32>,   // length = half_width_m
    @location(2) up_vec: vec3<f32>,      // length = half_height_m
    @location(3) uv_rect: vec4<f32>,     // [u0, v0, u1, v1]
    @location(4) fill_color: vec4<f32>,  // linear RGBA (straight alpha; shader premultiplies)
    @location(5) border_color: vec4<f32>,
    @location(6) params: vec4<f32>,      // [corner_radius_m, border_width_m, tex_kind, tex_layer]
};

struct QuadVertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) local_m: vec2<f32>,
    @location(3) half_extents_m: vec2<f32>,
    @location(4) uv: vec2<f32>,
    @location(5) fill_color: vec4<f32>,
    @location(6) border_color: vec4<f32>,
    @location(7) params: vec4<f32>,
};

fn quad_corner(vid: u32) -> vec2<f32> {
    switch vid {
        case 0u: { return vec2<f32>(-1.0, -1.0); }
        case 1u: { return vec2<f32>( 1.0, -1.0); }
        case 2u: { return vec2<f32>(-1.0,  1.0); }
        case 3u: { return vec2<f32>(-1.0,  1.0); }
        case 4u: { return vec2<f32>( 1.0, -1.0); }
        default: { return vec2<f32>( 1.0,  1.0); }
    }
}

@vertex
fn vs_quad(in: QuadInstanceIn) -> QuadVertexOut {
    let c = quad_corner(in.vid);
    let half_w = length(in.right_vec);
    let half_h = length(in.up_vec);
    let world = in.center + in.right_vec * c.x + in.up_vec * c.y;
    let n = normalize(cross(in.right_vec, in.up_vec));

    // UV: c.y = +1 is top (`v0`), c.y = -1 is bottom (`v1`)
    let u = mix(in.uv_rect.x, in.uv_rect.z, 0.5 * (c.x + 1.0));
    let v = mix(in.uv_rect.w, in.uv_rect.y, 0.5 * (c.y + 1.0));

    var out: QuadVertexOut;
    out.clip_pos = camera.view_proj * vec4<f32>(world, 1.0);
    out.world_pos = world;
    out.world_normal = n;
    out.local_m = vec2<f32>(c.x * half_w, c.y * half_h);
    out.half_extents_m = vec2<f32>(half_w, half_h);
    out.uv = vec2<f32>(u, v);
    out.fill_color = in.fill_color;
    out.border_color = in.border_color;
    out.params = in.params;
    return out;
}

fn sd_rounded_box(p: vec2<f32>, b: vec2<f32>, r: f32) -> f32 {
    let rr = min(r, min(b.x, b.y));
    let q = abs(p) - b + vec2<f32>(rr);
    return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - rr;
}

@fragment
fn fs_quad(in: QuadVertexOut) -> @location(0) vec4<f32> {
    let tex_kind = i32(round(in.params.z));
    let tex_layer = i32(round(in.params.w));

    // Always sample textures in uniform control flow for mipmap derivative accuracy
    let glyph_s = textureSample(glyph_tex, linear_sampler, in.uv);
    let thumb_s = textureSample(thumb_array_tex, linear_sampler, in.uv, max(tex_layer, 0));
    let insp_s  = textureSample(inspect_tex, linear_sampler, in.uv);

    if tex_kind == 1 {
        // Premultiplied white glyph coverage * fill_color
        let cov = glyph_s.a;
        if cov < 0.004 {
            discard;
        }
        let a = cov * in.fill_color.a;
        return vec4<f32>(in.fill_color.rgb * a, a);
    }

    let d = sd_rounded_box(in.local_m, in.half_extents_m, in.params.x);
    let aa = max(fwidth(d), 0.00045);
    let outer_cov = 1.0 - smoothstep(-aa, aa, d);
    if outer_cov < 0.003 {
        discard;
    }

    var base = in.fill_color;
    if tex_kind == 2 {
        // Premultiplied thumbnail array sample inside rounded SDF mask
        let ta = thumb_s.a * in.fill_color.a * outer_cov;
        let trgb = thumb_s.rgb * in.fill_color.rgb * in.fill_color.a * outer_cov;
        return vec4<f32>(trgb, ta);
    }
    if tex_kind == 3 {
        // High-res inspector image sample inside rounded SDF mask
        let ia = insp_s.a * in.fill_color.a * outer_cov;
        let irgb = insp_s.rgb * in.fill_color.rgb * in.fill_color.a * outer_cov;
        return vec4<f32>(irgb, ia);
    }

    // Spatial SDF Glass / Button / Card with single-layer crisp border + per-eye specular parallax
    let bw = in.params.y;
    var border_mix = 0.0;
    if bw > 0.0001 {
        border_mix = smoothstep(-bw - aa, -bw + aa, d);
    }

    let n = normalize(in.world_normal);
    let toward_eye = normalize(camera.camera_pos.xyz - in.world_pos);
    let l = normalize(camera.light_dir.xyz);
    let h = normalize(l + toward_eye);
    let spec = pow(max(dot(n, h), 0.0), 48.0) * 0.08;
    let fresnel = pow(1.0 - max(abs(dot(n, toward_eye)), 0.0), 3.5) * 0.06;

    let col = mix(base, in.border_color, border_mix);
    let out_a = clamp(col.a * outer_cov, 0.0, 1.0);
    let out_rgb = col.rgb * out_a + vec3<f32>(spec + fresnel) * outer_cov * step(0.15, col.a);
    return vec4<f32>(out_rgb, out_a);
}
