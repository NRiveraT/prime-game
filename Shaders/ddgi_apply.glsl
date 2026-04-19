#[compute]

#version 460
#extension GL_EXT_shader_image_load_store : enable

#define GRID_X          8
#define GRID_Y          8
#define GRID_Z          8
#define ATLAS_COLS      (GRID_X * GRID_Z)   // 64

// Probe tile resolutions (must match update shaders)
#define IRR_PROBE_SIDE  16
#define DEP_PROBE_SIDE  32

// Atlas dimensions per cascade
//#define IRR_ATLAS_W     (ATLAS_COLS * IRR_PROBE_SIDE)   // 1024
//#define IRR_ATLAS_H     (GRID_Y     * IRR_PROBE_SIDE)   // 128
//#define DEP_ATLAS_W     (ATLAS_COLS * DEP_PROBE_SIDE)   // 2048
//#define DEP_ATLAS_H     (GRID_Y     * DEP_PROBE_SIDE)   // 256

#define IRR_TILE_STRIDE (IRR_PROBE_SIDE + 2)
#define DEP_TILE_STRIDE (DEP_PROBE_SIDE + 2)
#define IRR_ATLAS_W (ATLAS_COLS * IRR_TILE_STRIDE) // 640
#define IRR_ATLAS_H (GRID_Y * IRR_TILE_STRIDE)     // 40
#define DEP_ATLAS_W (ATLAS_COLS * DEP_TILE_STRIDE) // 1152
#define DEP_ATLAS_H (GRID_Y * DEP_TILE_STRIDE)     // 72

#define PI              3.14159265358979

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, rgba16f) uniform image2D   color_buf;
layout(set = 0, binding = 1)          uniform sampler2D depth_buf;
layout(set = 0, binding = 2)          uniform sampler2D normal_roughness_buf;

// Cascade 0 — innermost, finest spacing
layout(set = 0, binding = 3)          uniform sampler2D irr_atlas_0;
layout(set = 0, binding = 4)          uniform sampler2D dep_atlas_0;

// Params SSBO: scalars (16 B) + four cascade grid descriptors (4 × 16 B) = 80 B
// grid_N.xyz = probe grid world origin   grid_N.w = probe_spacing for that cascade
layout(set = 0, binding = 5, std430) readonly buffer DDGIParams {
    float ddgi_strength;
    float energy_preservation;
    float normal_bias;
    float view_bias;
    vec4  grid_0;
    vec4  grid_1;
    vec4  grid_2;
    vec4  grid_3;
} params;

// Cascades 1–3 — progressively coarser spacing
layout(set = 0, binding = 6)          uniform sampler2D irr_atlas_1;
layout(set = 0, binding = 7)          uniform sampler2D dep_atlas_1;
layout(set = 0, binding = 8)          uniform sampler2D irr_atlas_2;
layout(set = 0, binding = 9)          uniform sampler2D dep_atlas_2;
layout(set = 0, binding = 10)         uniform sampler2D irr_atlas_3;
layout(set = 0, binding = 11)         uniform sampler2D dep_atlas_3;

// 112 bytes = 28 floats — camera matrices only.
// Cascade grid origins live in the params SSBO above.
layout(push_constant, std430) uniform PC {
    mat4  inv_proj;        // floats  0-15
    vec4  inv_view_r0;     // floats 16-19  basis.x + cam_origin.x
    vec4  inv_view_r1;     // floats 20-23  basis.y + cam_origin.y
    vec4  inv_view_r2;     // floats 24-27  basis.z + cam_origin.z
} pc;

// ── Octahedral encode: unit vec → [0,1]² ─────────────────────────────────────
vec2 oct_encode(vec3 n) {
    float l1 = abs(n.x) + abs(n.y) + abs(n.z);
    vec2  p  = n.xy / l1;
    if (n.z < 0.0)
        p = (1.0 - abs(p.yx)) * sign(p);
    return p * 0.5 + 0.5;
}

// ── Atlas UV helpers ──────────────────────────────────────────────────────────
vec2 irr_atlas_uv(uint probe_idx, vec3 dir) {
    vec2 oct = clamp(oct_encode(normalize(dir)), vec2(0.5 / float(IRR_PROBE_SIDE)), vec2(1.0 - 0.5 / float(IRR_PROBE_SIDE)));
    uint tile_col = probe_idx % uint(ATLAS_COLS);
    uint tile_row = probe_idx / uint(ATLAS_COLS);
    vec2 texel    = vec2(float(tile_col * IRR_TILE_STRIDE) + 1.0, float(tile_row * IRR_TILE_STRIDE) + 1.0) + oct * float(IRR_PROBE_SIDE);
    return texel / vec2(float(IRR_ATLAS_W), float(IRR_ATLAS_H));
}

vec2 dep_atlas_uv(uint probe_idx, vec3 dir) {
    vec2 oct = clamp(oct_encode(normalize(dir)), vec2(0.5 / float(DEP_PROBE_SIDE)), vec2(1.0 - 0.5 / float(DEP_PROBE_SIDE)));
    uint tile_col = probe_idx % uint(ATLAS_COLS);
    uint tile_row = probe_idx / uint(ATLAS_COLS);
    vec2 texel    = vec2(float(tile_col * DEP_TILE_STRIDE) + 1.0, float(tile_row * DEP_TILE_STRIDE) + 1.0) + oct * float(DEP_PROBE_SIDE);
    return texel / vec2(float(DEP_ATLAS_W), float(DEP_ATLAS_H));
}

// ── Chebyshev upper-bound visibility — cubed for sharp shadow contrast ────────
float chebyshev_vis(float mean, float mean2, float dist) {
    // Cold-start guard: atlas not yet warmed up (all zeros) → assume unoccluded.
    // Without this, variance=0 at cold-start makes everything fully occluded.
    if (mean < 0.01) return 1.0;
    if (dist <= mean) return 1.0;
    float variance = max(mean2 - mean * mean, 0.0001);
    float t        = dist - mean;
    float w = variance / (variance + t * t);
    // w³ with 0.05 floor — per RTXGI spec (Majercik 2019).
    // Floor prevents fully-occluded probes from contributing zero weight
    // (which causes dark blotches when all cage probes are occluded).
    float w3 = w * w * w;
//    return max(0.05, w3);
    return 1.0;
}

// ── Weight crush — smooth cubic rolloff below 0.2 ─────────────────────────────
float crush_weight(float w) {
    const float CRUSH = 0.2;
    if (w < CRUSH)
        w = w * w * w * (1.0 / (CRUSH * CRUSH));
    return w;
}

// ── Per-cascade blend weight: 1.0 inside the grid, fades over 1 probe cell ───
float cascade_blend(vec3 world_pos, vec3 origin, float spacing) {
    vec3 local     = (world_pos - origin) / spacing;
    vec3 face_dist = min(local, vec3(float(GRID_X - 1), float(GRID_Y - 1), float(GRID_Z - 1)) - local);
    return smoothstep(0.0, 1.0, min(min(face_dist.x, face_dist.y), face_dist.z));
}

void main() {
    ivec2 coord       = ivec2(gl_GlobalInvocationID.xy);
    ivec2 screen_size = imageSize(color_buf);
    if (coord.x >= screen_size.x || coord.y >= screen_size.y) return;

    vec2 screen_uv = (vec2(coord) + 0.5) / vec2(screen_size);

    float depth = texture(depth_buf, screen_uv).r;
    if (depth <= 0.0001) return;  // sky — no GI

    // ── Reconstruct world position ────────────────────────────────────────────
    vec2 ndc_uv    = screen_uv * 2.0 - 1.0;
    vec4 view_pos  = pc.inv_proj * vec4(ndc_uv, depth, 1.0);
    view_pos.xyz  /= view_pos.w;

    vec3 world_pos = vec3(
        pc.inv_view_r0.x * view_pos.x + pc.inv_view_r1.x * view_pos.y + pc.inv_view_r2.x * view_pos.z + pc.inv_view_r0.w,
        pc.inv_view_r0.y * view_pos.x + pc.inv_view_r1.y * view_pos.y + pc.inv_view_r2.y * view_pos.z + pc.inv_view_r1.w,
        pc.inv_view_r0.z * view_pos.x + pc.inv_view_r1.z * view_pos.y + pc.inv_view_r2.z * view_pos.z + pc.inv_view_r2.w
    );

    // ── World-space normal ────────────────────────────────────────────────────
    vec3 normal_vs = texture(normal_roughness_buf, screen_uv).xyz * 2.0 - 1.0;
    vec3 normal = normalize(vec3(
        pc.inv_view_r0.x * normal_vs.x + pc.inv_view_r1.x * normal_vs.y + pc.inv_view_r2.x * normal_vs.z,
        pc.inv_view_r0.y * normal_vs.x + pc.inv_view_r1.y * normal_vs.y + pc.inv_view_r2.y * normal_vs.z,
        pc.inv_view_r0.z * normal_vs.x + pc.inv_view_r1.z * normal_vs.y + pc.inv_view_r2.z * normal_vs.z
    ));

    vec3 cam_pos  = vec3(pc.inv_view_r0.w, pc.inv_view_r1.w, pc.inv_view_r2.w);
    // view_dir: surface→camera (matches RTXGI -camDir convention, negated)
    vec3 view_dir = normalize(cam_pos - world_pos);

    // ── Trilinear probe cage — inlined per cascade (Vulkan can't pass samplers) ─

    // ── Cascade 0 ─────────────────────────────────────────────────────────────
    vec3  irr_sum_0    = vec3(0.0);
    float weight_sum_0 = 0.0;
    {
        vec3  origin  = params.grid_0.xyz;
        float spacing = params.grid_0.w;
        vec3  pc0     = (world_pos - origin) / spacing;
        ivec3 base    = clamp(ivec3(floor(pc0)), ivec3(0), ivec3(GRID_X-2, GRID_Y-2, GRID_Z-2));
        vec3  alpha   = clamp(pc0 - vec3(base), vec3(0.0), vec3(1.0));

        for (int i = 0; i < 8; i++) {
            ivec3 off  = ivec3(i & 1, (i >> 1) & 1, (i >> 2) & 1);
            ivec3 cage = base + off;
            uint  pid  = uint(cage.x) + uint(cage.z) * uint(GRID_X)
                       + uint(cage.y) * uint(GRID_X * GRID_Z);

            vec3  t   = mix(vec3(1.0) - alpha, alpha, vec3(off));
            float w   = t.x * t.y * t.z;

            vec3  probe_pos    = origin + vec3(cage) * spacing;
            vec3  dir_to_probe = normalize(probe_pos - world_pos);
            float facing       = max(0.0001, (dot(dir_to_probe, normal) + 1.0) * 0.5);
            w *= facing * facing + 0.2;

            vec3  ptp      = world_pos - probe_pos + normal * params.normal_bias + view_dir * params.view_bias;
            float pdist    = length(ptp);
            vec2  dep_uv   = dep_atlas_uv(pid, normalize(ptp));
            vec2  dep      = textureLod(dep_atlas_0, dep_uv, 0.0).rg;
            // chebyshev_vis handles cold-start (mean < 0.01 → unoccluded)
            w *= chebyshev_vis(dep.x, dep.y, pdist);
            w  = crush_weight(max(0.000001, w));

            vec3 irr = textureLod(irr_atlas_0, irr_atlas_uv(pid, normal), 0.0).rgb;
            irr_sum_0    += sqrt(max(irr, vec3(0.0))) * w;
            weight_sum_0 += w;
        }
    }

    // ── Cascade 1 ─────────────────────────────────────────────────────────────
    vec3  irr_sum_1    = vec3(0.0);
    float weight_sum_1 = 0.0;
    {
        vec3  origin  = params.grid_1.xyz;
        float spacing = params.grid_1.w;
        vec3  pc1     = (world_pos - origin) / spacing;
        ivec3 base    = clamp(ivec3(floor(pc1)), ivec3(0), ivec3(GRID_X-2, GRID_Y-2, GRID_Z-2));
        vec3  alpha   = clamp(pc1 - vec3(base), vec3(0.0), vec3(1.0));

        for (int i = 0; i < 8; i++) {
            ivec3 off  = ivec3(i & 1, (i >> 1) & 1, (i >> 2) & 1);
            ivec3 cage = base + off;
            uint  pid  = uint(cage.x) + uint(cage.z) * uint(GRID_X)
                       + uint(cage.y) * uint(GRID_X * GRID_Z);

            vec3  t   = mix(vec3(1.0) - alpha, alpha, vec3(off));
            float w   = t.x * t.y * t.z;

            vec3  probe_pos    = origin + vec3(cage) * spacing;
            vec3  dir_to_probe = normalize(probe_pos - world_pos);
            float facing       = max(0.0001, (dot(dir_to_probe, normal) + 1.0) * 0.5);
            w *= facing * facing + 0.2;

            vec3  ptp      = world_pos - probe_pos + normal * params.normal_bias + view_dir * params.view_bias;
            float pdist    = length(ptp);
            vec2  dep_uv   = dep_atlas_uv(pid, normalize(ptp));
            vec2  dep      = textureLod(dep_atlas_1, dep_uv, 0.0).rg;
            // chebyshev_vis handles cold-start (mean < 0.01 → unoccluded)
            w *= chebyshev_vis(dep.x, dep.y, pdist);
            w  = crush_weight(max(0.000001, w));

            vec3 irr = textureLod(irr_atlas_1, irr_atlas_uv(pid, normal), 0.0).rgb;
            irr_sum_1    += sqrt(max(irr, vec3(0.0))) * w;
            weight_sum_1 += w;
        }
    }

    // ── Cascade 2 ─────────────────────────────────────────────────────────────
    vec3  irr_sum_2    = vec3(0.0);
    float weight_sum_2 = 0.0;
    {
        vec3  origin  = params.grid_2.xyz;
        float spacing = params.grid_2.w;
        vec3  pc2     = (world_pos - origin) / spacing;
        ivec3 base    = clamp(ivec3(floor(pc2)), ivec3(0), ivec3(GRID_X-2, GRID_Y-2, GRID_Z-2));
        vec3  alpha   = clamp(pc2 - vec3(base), vec3(0.0), vec3(1.0));

        for (int i = 0; i < 8; i++) {
            ivec3 off  = ivec3(i & 1, (i >> 1) & 1, (i >> 2) & 1);
            ivec3 cage = base + off;
            uint  pid  = uint(cage.x) + uint(cage.z) * uint(GRID_X)
                       + uint(cage.y) * uint(GRID_X * GRID_Z);

            vec3  t   = mix(vec3(1.0) - alpha, alpha, vec3(off));
            float w   = t.x * t.y * t.z;

            vec3  probe_pos    = origin + vec3(cage) * spacing;
            vec3  dir_to_probe = normalize(probe_pos - world_pos);
            float facing       = max(0.0001, (dot(dir_to_probe, normal) + 1.0) * 0.5);
            w *= facing * facing + 0.2;

            vec3  ptp      = world_pos - probe_pos + normal * params.normal_bias + view_dir * params.view_bias;
            float pdist    = length(ptp);
            vec2  dep_uv   = dep_atlas_uv(pid, normalize(ptp));
            vec2  dep      = textureLod(dep_atlas_2, dep_uv, 0.0).rg;
            // chebyshev_vis handles cold-start (mean < 0.01 → unoccluded)
            w *= chebyshev_vis(dep.x, dep.y, pdist);
            w  = crush_weight(max(0.000001, w));

            vec3 irr = textureLod(irr_atlas_2, irr_atlas_uv(pid, normal), 0.0).rgb;
            irr_sum_2    += sqrt(max(irr, vec3(0.0))) * w;
            weight_sum_2 += w;
        }
    }

    // ── Cascade 3 — outermost fallback ────────────────────────────────────────
    vec3  irr_sum_3    = vec3(0.0);
    float weight_sum_3 = 0.0;
    {
        vec3  origin  = params.grid_3.xyz;
        float spacing = params.grid_3.w;
        vec3  pc3     = (world_pos - origin) / spacing;
        ivec3 base    = clamp(ivec3(floor(pc3)), ivec3(0), ivec3(GRID_X-2, GRID_Y-2, GRID_Z-2));
        vec3  alpha   = clamp(pc3 - vec3(base), vec3(0.0), vec3(1.0));

        for (int i = 0; i < 8; i++) {
            ivec3 off  = ivec3(i & 1, (i >> 1) & 1, (i >> 2) & 1);
            ivec3 cage = base + off;
            uint  pid  = uint(cage.x) + uint(cage.z) * uint(GRID_X)
                       + uint(cage.y) * uint(GRID_X * GRID_Z);

            vec3  t   = mix(vec3(1.0) - alpha, alpha, vec3(off));
            float w   = t.x * t.y * t.z;

            vec3  probe_pos    = origin + vec3(cage) * spacing;
            vec3  dir_to_probe = normalize(probe_pos - world_pos);
            float facing       = max(0.0001, (dot(dir_to_probe, normal) + 1.0) * 0.5);
            w *= facing * facing + 0.2;

            vec3  ptp      = world_pos - probe_pos + normal * params.normal_bias + view_dir * params.view_bias;
            float pdist    = length(ptp);
            vec2  dep_uv   = dep_atlas_uv(pid, normalize(ptp));
            vec2  dep      = textureLod(dep_atlas_3, dep_uv, 0.0).rg;
            // chebyshev_vis handles cold-start (mean < 0.01 → unoccluded)
            w *= chebyshev_vis(dep.x, dep.y, pdist);
            w  = crush_weight(max(0.000001, w));

            vec3 irr = textureLod(irr_atlas_3, irr_atlas_uv(pid, normal), 0.0).rgb;
            irr_sum_3    += sqrt(max(irr, vec3(0.0))) * w;
            weight_sum_3 += w;
        }
    }

    // ── Resolve from sqrt-space ───────────────────────────────────────────────
    vec3 r0 = irr_sum_0 / max(weight_sum_0, 1e-4); r0 = r0 * r0;
    vec3 r1 = irr_sum_1 / max(weight_sum_1, 1e-4); r1 = r1 * r1;
    vec3 r2 = irr_sum_2 / max(weight_sum_2, 1e-4); r2 = r2 * r2;
    vec3 r3 = irr_sum_3 / max(weight_sum_3, 1e-4); r3 = r3 * r3;

    // NaN guards
    if (isnan(r0.x) || isnan(r0.y) || isnan(r0.z)) r0 = vec3(0.0);
    if (isnan(r1.x) || isnan(r1.y) || isnan(r1.z)) r1 = vec3(0.0);
    if (isnan(r2.x) || isnan(r2.y) || isnan(r2.z)) r2 = vec3(0.0);
    if (isnan(r3.x) || isnan(r3.y) || isnan(r3.z)) r3 = vec3(0.0);

    // ── Cascade blend: composite outermost → innermost ────────────────────────
    // Each blend weight is 1.0 inside that cascade's bounds, fades over 1 probe
    // cell at the boundary, 0.0 fully outside.  Cascade 3 is the unconditional
    // fallback; inner cascades progressively override it.
    float b2 = cascade_blend(world_pos, params.grid_2.xyz, params.grid_2.w);
    float b1 = cascade_blend(world_pos, params.grid_1.xyz, params.grid_1.w);
    float b0 = cascade_blend(world_pos, params.grid_0.xyz, params.grid_0.w);

    vec3 net_irr = r3;
    net_irr = mix(net_irr, r2, b2);
    net_irr = mix(net_irr, r1, b1);
    net_irr = mix(net_irr, r0, b0);

    // ── Scale and apply ───────────────────────────────────────────────────────
    // 0.5 * PI converts stored irradiance → outgoing radiance (Lambertian
    // hemisphere integral).  ddgi_strength absorbs surface albedo / PI.
    vec3 irradiance = net_irr * (0.5 * PI) * params.energy_preservation * params.ddgi_strength;

    vec4 old_color = imageLoad(color_buf, coord);
    imageStore(color_buf, coord, vec4(old_color.rgb + irradiance, old_color.a));
}
