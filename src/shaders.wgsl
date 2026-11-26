// SPDX-License-Identifier: GPL-3.0-or-later
// WGSL Shaders for openKylin XR Files — UKUI XR "Qingkong" (`ukui.spatial.qingkong.light`)
// - Pipeline 0 (`vs_sky` / `fs_sky`): 360° Qingkong equirectangular panorama background
// - Pipeline 1 (`vs_solid` / `fs_solid`): Daylight hemisphere-lit 3D podium, pastel-ceramic blocks & 3D holograms
// - Pipeline 2 (`vs_quad` / `fs_quad`):
//   * `tex_kind == 4`: Exterior-only feathered drop shadow (zero internal darkening under glass!)
//   * `tex_kind == 0`: Frosted silver-mist glass (`#E8F0F8`) & warm white (`#F5F7FA`) with directional
//                      specular rim + per-eye Fresnel & half-angle specular parallax
//   * `tex_kind == 1`: Premultiplied dual-weight CJK/Latin typography
//   * `tex_kind == 2`: Mipmapped 2.5D UKUI icon / photo thumbnail inside rounded SDF mask
//   * `tex_kind == 3`: High-resolution Inspector image inside rounded SDF mask

struct CameraUniform {
    view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    camera_quat: vec4<f32>,
    fov_tan: vec4<f32>, // [tan_left, tan_right, tan_up, tan_down]
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

@group(1) @binding(0)
var glyph_tex: texture_2d<f32>;
@group(1) @binding(1)
var thumb_array_tex: texture_2d_array<f32>;
@group(1) @binding(2)
var inspect_tex: texture_2d<f32>;
@group(1) @binding(3)
var sky_tex: texture_2d<f32>;
@group(1) @binding(4)
var linear_sampler: sampler;

const PI: f32 = 3.141592653589793;

// ============================================================================
// Pipeline 0: 360° Qingkong Equirectangular Panorama Background
// ============================================================================

struct SkyVertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) screen_uv: vec2<f32>,
};

@vertex
fn vs_sky(@builtin(vertex_index) vid: u32) -> SkyVertexOut {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0)
    );
    let p = pos[vid];
    var out: SkyVertexOut;
    out.clip_pos = vec4<f32>(p, 1.0, 1.0);
    out.screen_uv = vec2<f32>(0.5 * (p.x + 1.0), 0.5 * (p.y + 1.0));
    return out;
}

@fragment
fn fs_sky(in: SkyVertexOut) -> @location(0) vec4<f32> {
    // Reconstruct world view ray from per-eye asymmetric FOV + orientation (matching xr-shell panorama.frag)
    let local_dir = normalize(vec3<f32>(
        mix(camera.fov_tan.x, camera.fov_tan.y, in.screen_uv.x),
        mix(camera.fov_tan.w, camera.fov_tan.z, in.screen_uv.y),
        -1.0
    ));
    let q = camera.camera_quat;
    let d = normalize(local_dir + 2.0 * cross(q.xyz, cross(q.xyz, local_dir) + q.w * local_dir));
    let u = fract(atan2(d.x, -d.z) / (2.0 * PI) + 0.5);
    let v = clamp(0.5 - asin(clamp(d.y, -1.0, 1.0)) / PI, 0.001, 0.999);
    let sky = textureSampleLevel(sky_tex, linear_sampler, vec2<f32>(u, v), 0.0);
    return vec4<f32>(sky.rgb, 1.0);
}

// ============================================================================
// Pipeline 1: Daylight Hemisphere-Lit 3D Solids & Holograms
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
    let sun = normalize(vec3<f32>(-0.42, 0.86, 0.65));
    let fill = normalize(vec3<f32>(0.55, 0.35, 0.75));

    // Qingkong daylight sky/ground hemisphere irradiance
    let hemi = 0.5 * n.y + 0.5;
    let env_light = mix(vec3<f32>(0.55, 0.64, 0.82), vec3<f32>(0.96, 0.98, 1.0), hemi);

    let ndotl = max(dot(n, sun), 0.0);
    let ndotf = max(dot(n, fill), 0.0);
    let h = normalize(sun + v);
    let spec = pow(max(dot(n, h), 0.0), 56.0) * 0.28;
    let fresnel = pow(1.0 - max(abs(dot(n, v)), 0.0), 3.5);

    let lit = in.color.rgb * (env_light * 0.54 + 0.44 * ndotl + 0.14 * ndotf);
    let rim = vec3<f32>(0.88, 0.95, 1.0) * fresnel * (0.16 + 0.30 * in.emissive);
    let final_rgb = lit + vec3<f32>(spec) + rim + in.color.rgb * in.emissive * 0.28;
    let a = clamp(in.color.a, 0.0, 1.0);
    return vec4<f32>(final_rgb * a, a);
}

// ============================================================================
// Pipeline 2: Qingkong Spatial Glass, Soft Exterior Shadow, Icons & Typography
// ============================================================================

struct QuadInstanceIn {
    @builtin(vertex_index) vid: u32,
    @location(0) center: vec3<f32>,
    @location(1) right_vec: vec3<f32>,
    @location(2) up_vec: vec3<f32>,
    @location(3) uv_rect: vec4<f32>,
    @location(4) fill_color: vec4<f32>,   // Linear RGBA
    @location(5) border_color: vec4<f32>, // Linear RGBA
    @location(6) params: vec4<f32>,       // [corner_radius_m, border_width_m, tex_kind, tex_layer]
};

struct QuadVertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) local_m: vec2<f32>,
    @location(3) half_extents_m: vec2<f32>,
    @location(4) uv: vec2<f32>,
    @location(5) norm_uv: vec2<f32>,
    @location(6) fill_color: vec4<f32>,
    @location(7) border_color: vec4<f32>,
    @location(8) params: vec4<f32>,
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

    let nuv = vec2<f32>(0.5 * (c.x + 1.0), 0.5 * (1.0 - c.y)); // (0,0) top-left, (1,1) bottom-right
    let u = mix(in.uv_rect.x, in.uv_rect.z, nuv.x);
    let v = mix(in.uv_rect.y, in.uv_rect.w, nuv.y);

    var out: QuadVertexOut;
    out.clip_pos = camera.view_proj * vec4<f32>(world, 1.0);
    out.world_pos = world;
    out.world_normal = n;
    out.local_m = vec2<f32>(c.x * half_w, c.y * half_h);
    out.half_extents_m = vec2<f32>(half_w, half_h);
    out.uv = vec2<f32>(u, v);
    out.norm_uv = nuv;
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

    let glyph_s = textureSample(glyph_tex, linear_sampler, in.uv);
    let thumb_s = textureSample(thumb_array_tex, linear_sampler, in.uv, max(tex_layer, 0));
    let insp_s  = textureSample(inspect_tex, linear_sampler, in.uv);

    // 1. Typography glyph (`tex_kind == 1`)
    if tex_kind == 1 {
        let cov = glyph_s.a;
        if cov < 0.004 {
            discard;
        }
        let a = cov * in.fill_color.a;
        return vec4<f32>(in.fill_color.rgb * a, a);
    }

    // 2. Exterior-only feathered drop shadow (`tex_kind == 4`, matching xr-shell material==4)
    if tex_kind == 4 {
        let spread = max(in.params.y, 0.008);
        let inner_half = max(in.half_extents_m - vec2<f32>(spread), vec2<f32>(0.002));
        let d_shd = sd_rounded_box(in.local_m, inner_half, in.params.x);
        // Zero alpha inside the panel so translucent frosted glass never gets a muddy gray film!
        let outside = smoothstep(-0.001, 0.004, d_shd);
        let falloff = exp(-pow(max(d_shd, 0.0) / (spread * 0.52), 2.0))
            * (1.0 - smoothstep(spread * 0.72, spread, d_shd));
        let a = in.fill_color.a * outside * falloff;
        if a < 0.001 {
            discard;
        }
        return vec4<f32>(in.fill_color.rgb * a, a);
    }

    let d = sd_rounded_box(in.local_m, in.half_extents_m, in.params.x);
    let aa = max(fwidth(d), 0.00038);
    let outer_cov = 1.0 - smoothstep(-aa, aa, d);
    if outer_cov < 0.003 {
        discard;
    }

    // 3. Mipmapped 2.5D UKUI icon or photo thumbnail (`tex_kind == 2`)
    if tex_kind == 2 {
        let a = thumb_s.a * in.fill_color.a * outer_cov;
        if a < 0.003 {
            discard;
        }
        return vec4<f32>(thumb_s.rgb * in.fill_color.rgb * in.fill_color.a * outer_cov, a);
    }

    // 4. High-res Inspector image (`tex_kind == 3`)
    if tex_kind == 3 {
        let a = insp_s.a * in.fill_color.a * outer_cov;
        if a < 0.003 {
            discard;
        }
        return vec4<f32>(insp_s.rgb * in.fill_color.rgb * in.fill_color.a * outer_cov, a);
    }

    // 5. Qingkong Frosted Glass / Content Card / Pill (`tex_kind == 0`)
    let n = normalize(in.world_normal);
    let toward_eye = normalize(camera.camera_pos.xyz - in.world_pos);
    let light = normalize(vec3<f32>(-0.5, 1.0, 1.0));
    let facing = clamp(dot(n, toward_eye), 0.0, 1.0);
    let fresnel = 0.03 + 0.97 * pow(1.0 - facing, 5.0);
    let spec = pow(max(dot(normal_safe(n), normalize(light + toward_eye)), 0.0), 64.0);

    // Subtle top-to-bottom frost luminance + diagonal soft sheen (like style-reference.png 01/04)
    let diag_sheen = exp(-pow((in.norm_uv.x * 0.65 + in.norm_uv.y * 0.75 - 0.42) * 3.2, 2.0)) * 0.045;
    var lit_rgb = in.fill_color.rgb + vec3<f32>(diag_sheen) + vec3<f32>(0.055 * spec + 0.025 * fresnel);

    let bw = in.params.y;
    var rim_alpha = 0.0;
    if bw > 0.00005 {
        let lit_edge = max(1.0 - in.norm_uv.y, (1.0 - in.norm_uv.x) * 0.65);
        let edge_ring = smoothstep(-bw - aa, -bw + aa, d);
        rim_alpha = edge_ring * in.border_color.a * mix(0.38, 1.0, lit_edge);
    }

    let base_a = in.fill_color.a;
    let base_pm = lit_rgb * base_a;
    let rim_pm = in.border_color.rgb * rim_alpha;

    // Porter-Duff `rim OVER base`, masked by anti-aliased rounded SDF
    let combined_pm = rim_pm + base_pm * (1.0 - rim_alpha);
    let combined_a = clamp(rim_alpha + base_a * (1.0 - rim_alpha), 0.0, 1.0);
    return vec4<f32>(combined_pm * outer_cov, combined_a * outer_cov);
}

fn normal_safe(v: vec3<f32>) -> vec3<f32> {
    return normalize(v);
}
