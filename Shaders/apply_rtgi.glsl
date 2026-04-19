#[compute]
#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

struct Probe { vec4 rgb_meta; };

layout(set = 0, binding = 0, rgba16f) uniform image2D color_buffer;
layout(set = 0, binding = 1) uniform sampler2D depth_buffer;
layout(set = 0, binding = 2) uniform sampler2D normal_roughness_buffer;

// C0 — near-field probe grid (same as Phase A single-irradiance, 1m spacing default).
layout(set = 0, binding = 3, std430) readonly buffer C0Probes {
    Probe data[];
} c0;

layout(set = 0, binding = 4, std430) readonly buffer CameraBuf {
    mat4 inv_proj;
    vec4 inv_view_r0;
    vec4 inv_view_r1;
    vec4 inv_view_r2;
} camera;

// C1 — mid-field probe grid (same probe count as C0, spacing×N → N× larger volume).
layout(set = 0, binding = 5, std430) readonly buffer C1Probes {
    Probe data[];
} c1;

// C2 — far-field probe grid (spacing×N² → N²× larger volume than C0).
layout(set = 0, binding = 6, std430) readonly buffer C2Probes {
    Probe data[];
} c2;

// Phase B1 screen-space cache.
layout(set = 0, binding = 7, rgba16f) uniform image2D screen_cache;

// Push constant: 112 bytes.
//    0..15  vec4  c0_origin_spacing    (xyz=origin, w=spacing)
//   16..31  ivec4 c0_size_debug        (xyz=size, w=debug mode)
//   32..47  vec4  misc                 (x=gi_strength, y=cache_enable,
//                                        z=num_extra_cascades (0..2),
//                                        w=smooth_blend (0/1))
//   48..63  vec4  c1_origin_spacing
//   64..79  ivec4 c1_size_pad
//   80..95  vec4  c2_origin_spacing
//   96..111 ivec4 c2_size_pad
layout(push_constant, std430) uniform PC {
    vec4  c0_origin_spacing;
    ivec4 c0_size_debug;
    vec4  misc;
    vec4  c1_origin_spacing;
    ivec4 c1_size_pad;
    vec4  c2_origin_spacing;
    ivec4 c2_size_pad;
} pc;

const int DBG_OFF         = 0;
const int DBG_INDIRECT    = 1;
const int DBG_WORLD_POS   = 2;
const int DBG_GRID_CELL   = 3;
const int DBG_PROBE_LOCAL = 4;
const int DBG_CACHE_ONLY  = 5;
// Per-cascade debug modes — show just that cascade's contribution.
const int DBG_C0_ONLY     = 6;
const int DBG_C1_ONLY     = 7;
const int DBG_C2_ONLY     = 8;

uint flat_idx(ivec3 p, ivec3 size) {
    return uint(p.x) + uint(p.y) * uint(size.x) + uint(p.z) * uint(size.x * size.y);
}

vec3 view_normal_to_world(vec3 n_vs) {
    return normalize(vec3(
        camera.inv_view_r0.x * n_vs.x + camera.inv_view_r1.x * n_vs.y + camera.inv_view_r2.x * n_vs.z,
        camera.inv_view_r0.y * n_vs.x + camera.inv_view_r1.y * n_vs.y + camera.inv_view_r2.y * n_vs.z,
        camera.inv_view_r0.z * n_vs.x + camera.inv_view_r1.z * n_vs.y + camera.inv_view_r2.z * n_vs.z
    ));
}

// Normal-weighted trilinear sample of a probe cascade at world_pos. Returns
// irradiance; caller applies albedo.
vec3 sample_cascade(int cascade_id, vec3 world_pos, vec3 normal,
                    vec3 origin, float spacing, ivec3 gsize) {
    vec3 grid_pos = (world_pos - origin) / spacing;
    if (any(lessThan(grid_pos, vec3(0.0)))
     || any(greaterThanEqual(grid_pos, vec3(gsize) - 1.0))) {
        return vec3(0.0);
    }
    ivec3 base = ivec3(floor(grid_pos));
    vec3  frac = grid_pos - vec3(base);
    vec3  sum  = vec3(0.0);
    float wsum = 0.0;
    for (int dz = 0; dz <= 1; dz++)
    for (int dy = 0; dy <= 1; dy++)
    for (int dx = 0; dx <= 1; dx++) {
        ivec3 idx = clamp(base + ivec3(dx, dy, dz), ivec3(0), gsize - 1);
        float wb = (dx == 0 ? 1.0 - frac.x : frac.x)
                 * (dy == 0 ? 1.0 - frac.y : frac.y)
                 * (dz == 0 ? 1.0 - frac.z : frac.z);
        vec3 pw = origin + vec3(idx) * spacing;
        vec3 dp = normalize(pw - world_pos + normal * 0.001);
        float nw = max(dot(normal, dp), 0.0);
        nw = nw * nw;
        float w = wb * nw;
        uint f = flat_idx(idx, gsize);
        vec3 val = (cascade_id == 0) ? c0.data[f].rgb_meta.xyz
                  : (cascade_id == 1) ? c1.data[f].rgb_meta.xyz
                                      : c2.data[f].rgb_meta.xyz;
        sum  += val * w;
        wsum += w;
    }
    return (wsum > 1e-4) ? (sum / wsum) : vec3(0.0);
}

// Containment in [0, 1]:
//   1.0 = well inside volume (further than 1 cell from any edge)
//   0.0 = outside volume
// Smooth fade inside 1-cell band along the boundary for cascade blending.
float cascade_containment(vec3 world_pos, vec3 origin, float spacing, ivec3 gsize,
                          bool smooth_edge) {
    vec3 grid_pos = (world_pos - origin) / spacing;
    vec3 max_pos  = vec3(gsize) - 1.0;
    if (any(lessThan(grid_pos, vec3(0.0)))
     || any(greaterThan(grid_pos, max_pos))) {
        return 0.0;
    }
    if (!smooth_edge) return 1.0;
    vec3 dist_from_edge = min(grid_pos, max_pos - grid_pos);
    float min_dist = min(min(dist_from_edge.x, dist_from_edge.y), dist_from_edge.z);
    return smoothstep(0.0, 1.0, min_dist);
}

// Layered cascade merge. C0 near camera → C1 mid → C2 far.
// Each cascade's containment fades to zero past its edge; we use a
// sample-and-lerp chain so pixels outside the innermost cascade fall through
// to the next one. Order: C0 owns its volume; edges blend into C1; C1 edges
// blend into C2; outside C2 we return 0.
vec3 cascaded_gi(vec3 world_pos, vec3 normal,
                 int num_extra, bool smooth_blend) {
    vec3  c0_val = sample_cascade(0, world_pos, normal,
                                  pc.c0_origin_spacing.xyz, pc.c0_origin_spacing.w,
                                  pc.c0_size_debug.xyz);
    float c0_in  = cascade_containment(world_pos,
                                       pc.c0_origin_spacing.xyz, pc.c0_origin_spacing.w,
                                       pc.c0_size_debug.xyz, smooth_blend);
    if (num_extra == 0) return c0_val * c0_in;

    vec3  c1_val = sample_cascade(1, world_pos, normal,
                                  pc.c1_origin_spacing.xyz, pc.c1_origin_spacing.w,
                                  pc.c1_size_pad.xyz);
    if (num_extra == 1) {
        float c1_in = cascade_containment(world_pos,
                                          pc.c1_origin_spacing.xyz, pc.c1_origin_spacing.w,
                                          pc.c1_size_pad.xyz, smooth_blend);
        vec3 layered = mix(c1_val * c1_in, c0_val, c0_in);
        return layered;
    }

    vec3  c2_val = sample_cascade(2, world_pos, normal,
                                  pc.c2_origin_spacing.xyz, pc.c2_origin_spacing.w,
                                  pc.c2_size_pad.xyz);
    float c1_in = cascade_containment(world_pos,
                                      pc.c1_origin_spacing.xyz, pc.c1_origin_spacing.w,
                                      pc.c1_size_pad.xyz, smooth_blend);
    float c2_in = cascade_containment(world_pos,
                                      pc.c2_origin_spacing.xyz, pc.c2_origin_spacing.w,
                                      pc.c2_size_pad.xyz, smooth_blend);
    vec3 far     = mix(c2_val * c2_in, c1_val, c1_in);
    vec3 layered = mix(far,            c0_val, c0_in);
    return layered;
}

// Screen-cache bilateral gather.
vec3 screen_cache_gather(ivec2 full_coord, ivec2 full_size,
                         float ref_depth, vec3 ref_normal,
                         out bool cache_valid) {
    ivec2 cache_size = imageSize(screen_cache);
    vec2  cache_pos  = (vec2(full_coord) + 0.5) / vec2(full_size)
                     * vec2(cache_size) - 0.5;
    ivec2 cache_base = ivec2(floor(cache_pos));
    vec2  cache_frac = cache_pos - vec2(cache_base);

    vec3  sum    = vec3(0.0);
    float weight = 0.0;
    for (int dy = 0; dy <= 1; dy++)
    for (int dx = 0; dx <= 1; dx++) {
        ivec2 lc = clamp(cache_base + ivec2(dx, dy), ivec2(0), cache_size - 1);
        float wb = (dx == 0 ? 1.0 - cache_frac.x : cache_frac.x)
                 * (dy == 0 ? 1.0 - cache_frac.y : cache_frac.y);
        vec2  lc_uv    = (vec2(lc) + 0.5) / vec2(cache_size);
        ivec2 gb_coord = clamp(ivec2(lc_uv * vec2(full_size)),
                               ivec2(0), full_size - 1);
        float s_depth  = texelFetch(depth_buffer, gb_coord, 0).r;
        vec3  s_normal = view_normal_to_world(
            texelFetch(normal_roughness_buffer, gb_coord, 0).xyz * 2.0 - 1.0);
        if (s_depth <= 0.0001) continue;
        float wd = exp(-abs(ref_depth - s_depth) * 256.0);
        float wn = pow(max(dot(ref_normal, s_normal), 0.0), 8.0);
        float w  = wb * wd * wn;
        vec3  tap = imageLoad(screen_cache, lc).rgb;
        sum    += tap * w;
        weight += w;
    }
    cache_valid = weight > 1e-4;
    return cache_valid ? (sum / weight) : vec3(0.0);
}

void main() {
    ivec2 full_coord = ivec2(gl_GlobalInvocationID.xy);
    ivec2 full_size  = imageSize(color_buffer);
    if (full_coord.x >= full_size.x || full_coord.y >= full_size.y) return;

    float depth = texelFetch(depth_buffer, full_coord, 0).r;
    if (depth <= 0.0001) return;

    vec2 screen_uv = (vec2(full_coord) + 0.5) / vec2(full_size);
    vec2 ndc       = screen_uv * 2.0 - 1.0;
    vec4 view_pos  = camera.inv_proj * vec4(ndc, depth, 1.0);
    view_pos.xyz  /= view_pos.w;
    vec3 world_pos = vec3(
        camera.inv_view_r0.x * view_pos.x + camera.inv_view_r1.x * view_pos.y + camera.inv_view_r2.x * view_pos.z + camera.inv_view_r0.w,
        camera.inv_view_r0.y * view_pos.x + camera.inv_view_r1.y * view_pos.y + camera.inv_view_r2.y * view_pos.z + camera.inv_view_r1.w,
        camera.inv_view_r0.z * view_pos.x + camera.inv_view_r1.z * view_pos.y + camera.inv_view_r2.z * view_pos.z + camera.inv_view_r2.w
    );

    vec3 ref_normal_vs = texelFetch(normal_roughness_buffer, full_coord, 0).xyz * 2.0 - 1.0;
    vec3 ref_normal    = view_normal_to_world(ref_normal_vs);

    vec3  origin        = pc.c0_origin_spacing.xyz;
    float spacing       = pc.c0_origin_spacing.w;
    ivec3 gsize         = pc.c0_size_debug.xyz;
    int   debug         = pc.c0_size_debug.w;
    float gi_strength   = pc.misc.x;
    bool  cache_enabled = pc.misc.y > 0.5;
    int   num_extra     = int(pc.misc.z + 0.5);
    bool  smooth_blend  = pc.misc.w > 0.5;

    // ── Debug modes that don't need indirect ─────────────────────────────────
    vec3  grid_pos = (world_pos - origin) / spacing;
    bool  in_grid  = all(greaterThanEqual(grid_pos, vec3(0.0)))
                  && all(lessThan(grid_pos, vec3(gsize) - 1.0));

    if (debug == DBG_WORLD_POS) {
        vec3 tinted = fract(world_pos * 0.5) * 0.8 + 0.2;
        imageStore(color_buffer, full_coord, vec4(tinted, 1.0));
        return;
    }
    if (debug == DBG_GRID_CELL) {
        if (!in_grid) return;
        ivec3 cell = ivec3(floor(grid_pos));
        float r = float(cell.x & 3) / 3.0;
        float g = float(cell.y & 3) / 3.0;
        float b = float(cell.z & 3) / 3.0;
        imageStore(color_buffer, full_coord, vec4(r, g, b, 1.0));
        return;
    }
    if (debug == DBG_PROBE_LOCAL) {
        if (!in_grid) return;
        ivec3 base = ivec3(floor(grid_pos));
        vec3  frac = grid_pos - vec3(base);
        imageStore(color_buffer, full_coord, vec4(frac, 1.0));
        return;
    }

    // Per-cascade debug — show just that cascade's trilinear result.
    if (debug == DBG_C0_ONLY) {
        vec3 v = sample_cascade(0, world_pos, ref_normal,
                                pc.c0_origin_spacing.xyz, pc.c0_origin_spacing.w,
                                pc.c0_size_debug.xyz);
        imageStore(color_buffer, full_coord, vec4(v, 1.0));
        return;
    }
    if (debug == DBG_C1_ONLY) {
        vec3 v = sample_cascade(1, world_pos, ref_normal,
                                pc.c1_origin_spacing.xyz, pc.c1_origin_spacing.w,
                                pc.c1_size_pad.xyz);
        imageStore(color_buffer, full_coord, vec4(v, 1.0));
        return;
    }
    if (debug == DBG_C2_ONLY) {
        vec3 v = sample_cascade(2, world_pos, ref_normal,
                                pc.c2_origin_spacing.xyz, pc.c2_origin_spacing.w,
                                pc.c2_size_pad.xyz);
        imageStore(color_buffer, full_coord, vec4(v, 1.0));
        return;
    }

    // ── Indirect resolve ─────────────────────────────────────────────────────
    vec3 probe_indirect = cascaded_gi(world_pos, ref_normal, num_extra, smooth_blend);
    vec3 indirect = probe_indirect;

    if (cache_enabled) {
        bool cache_valid = false;
        vec3 cache_indirect = screen_cache_gather(full_coord, full_size,
                                                  depth, ref_normal, cache_valid);
        indirect = cache_valid ? cache_indirect : probe_indirect;
        if (debug == DBG_CACHE_ONLY) {
            imageStore(color_buffer, full_coord, vec4(cache_indirect, 1.0));
            return;
        }
    }

    if (debug == DBG_INDIRECT) {
        imageStore(color_buffer, full_coord, vec4(indirect, 1.0));
        return;
    }

    vec4 color  = imageLoad(color_buffer, full_coord);
    vec3 result = color.rgb + color.rgb * indirect * gi_strength;
    imageStore(color_buffer, full_coord, vec4(result, color.a));
}
