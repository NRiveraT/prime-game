#[compute]

#version 460
#extension GL_EXT_shader_image_load_store : enable

#define GRID_X          8
#define GRID_Y          8
#define GRID_Z          8
#define RAYS_PER_PROBE  128
#define TOTAL_PROBES    (GRID_X * GRID_Y * GRID_Z)   // 512
#define DEP_PROBE_SIDE  32
#define DEP_ATLAS_COLS  (GRID_X * GRID_Z)            // 64

// One workgroup = one probe depth tile (16×16 threads).
// Larger tile than irradiance gives higher-resolution Chebyshev visibility.
layout(local_size_x = DEP_PROBE_SIDE, local_size_y = DEP_PROBE_SIDE, local_size_z = 1) in;

// Ray buffer: [2i+0] = radiance.xyz + hit_dist,  [2i+1] = ray_dir.xyz + pad
layout(set = 0, binding = 0, std430) readonly buffer RayBuffer { vec4 data[]; } ray_buf;
layout(set = 0, binding = 1, rg16f) uniform image2D depth_atlas;  // .r = mean, .g = mean²

layout(push_constant, std430) uniform PC {
    float hysteresis;
    float depth_sharpness;  // default 50.0 — sharpens shadow boundaries
    float _pad0;
    float _pad1;
} pc;

// ── Octahedral decode: f ∈ [-1,1]² → unit sphere ─────────────────────────────
vec3 oct_decode(vec2 f) {
    vec3 n = vec3(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
    float t = clamp(-n.z, 0.0, 1.0);
    n.x += (n.x >= 0.0) ? -t : t;
    n.y += (n.y >= 0.0) ? -t : t;
    return normalize(n);
}
#define DEP_TILE_STRIDE (DEP_PROBE_SIDE + 2)
void main() {
    uint probe_idx = gl_WorkGroupID.x;
    if (probe_idx >= uint(TOTAL_PROBES)) return;

    uint local_x = gl_LocalInvocationID.x;
    uint local_y = gl_LocalInvocationID.y;

    // Octahedral direction for this depth texel
    vec2 oct_f   = ((vec2(local_x, local_y) + 0.5) / float(DEP_PROBE_SIDE)) * 2.0 - 1.0;
    vec3 oct_dir = oct_decode(oct_f);

    // Accumulate depth moments weighted by pow(cosine, depth_sharpness).
    // High depth_sharpness (50) concentrates depth data near the exact ray
    // direction, giving sharp Chebyshev shadow boundaries.
    float dep_sum  = 0.0;
    float dep_sum2 = 0.0;
    float dep_w    = 0.0;

    uint base = probe_idx * uint(RAYS_PER_PROBE);
    for (uint ri = 0u; ri < uint(RAYS_PER_PROBE); ri++) {
        uint buf_idx  = base + ri;
        float hit_dist = ray_buf.data[buf_idx * 2u].w;
        if (hit_dist < 0.0) continue;
        
        vec3  ray_dir  = ray_buf.data[buf_idx * 2u + 1u].xyz;

        float cosine = max(dot(ray_dir, oct_dir), 0.0);
        // depth_sharpness-powered weight — same as Majercik 2019 / hybrid-rendering
        float weight = pow(cosine, pc.depth_sharpness);

        dep_sum  += hit_dist            * weight;
        dep_sum2 += hit_dist * hit_dist * weight;
        dep_w    += weight;
    }

    vec2 new_dep = (dep_w > 1e-6)
        ? vec2(dep_sum / dep_w, dep_sum2 / dep_w)
        : vec2(0.0);

    // Atlas texel for this probe tile
    uint  tile_col    = probe_idx % uint(DEP_ATLAS_COLS);
    uint  tile_row    = probe_idx / uint(DEP_ATLAS_COLS);
    ivec2 atlas_coord = ivec2(int(tile_col) * DEP_TILE_STRIDE + int(local_x) + 1, int(tile_row) * DEP_TILE_STRIDE + int(local_y) + 1);

    vec2 old_dep = imageLoad(depth_atlas, atlas_coord).rg;
    imageStore(depth_atlas, atlas_coord, vec4(mix(new_dep, old_dep, pc.hysteresis), 0.0, 0.0));
}
