#[compute]

#version 460
#extension GL_EXT_shader_image_load_store : enable

// ── Grid / atlas constants ────────────────────────────────────────────────────
#define GRID_X         8
#define GRID_Y         8
#define GRID_Z         8
#define RAYS_PER_PROBE 128
#define TOTAL_PROBES   (GRID_X * GRID_Y * GRID_Z)  // 256
#define PROBE_RES      8
#define ATLAS_COLS     (GRID_X * GRID_Z)            // 64 tiles wide

// One workgroup = one probe (8×8 threads = one atlas tile).
layout(local_size_x = PROBE_RES, local_size_y = PROBE_RES, local_size_z = 1) in;

// Ray buffer written by ddgi_update raygen:
//   [2i+0]: radiance.xyz + hit_distance
//   [2i+1]: ray_direction.xyz + pad
layout(set = 0, binding = 0, std430) readonly buffer RayBuffer {
    vec4 data[];
} ray_buf;

layout(set = 0, binding = 1, rgba16f) uniform image2D irr_atlas;
layout(set = 0, binding = 2, rg16f)   uniform image2D depth_atlas;

// 16 bytes
layout(push_constant, std430) uniform PC {
    float hysteresis;  // fraction of old atlas value to keep (e.g. 0.97)
    float _pad0;
    float _pad1;
    float _pad2;
} pc;

// ── Octahedral decode: f ∈ [-1,1]² → unit sphere ─────────────────────────────
vec3 oct_decode(vec2 f) {
    vec3 n = vec3(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
    float t = clamp(-n.z, 0.0, 1.0);
    n.x += (n.x >= 0.0) ? -t : t;
    n.y += (n.y >= 0.0) ? -t : t;
    return normalize(n);
}

void main() {
    uint probe_idx = gl_WorkGroupID.x;
    if (probe_idx >= uint(TOTAL_PROBES)) return;

    uint local_x = gl_LocalInvocationID.x;
    uint local_y = gl_LocalInvocationID.y;

    // Decode octahedral direction for this atlas texel
    vec2 oct_uv  = (vec2(local_x, local_y) + 0.5) / float(PROBE_RES);
    vec3 oct_dir = oct_decode(oct_uv * 2.0 - 1.0);

    // Accumulate over all rays for this probe
    vec3  irr_sum  = vec3(0.0);
    float irr_w    = 0.0;
    float dep_sum  = 0.0;
    float dep_sum2 = 0.0;
    float dep_w    = 0.0;

    uint base = probe_idx * uint(RAYS_PER_PROBE);
    for (uint ri = 0u; ri < uint(RAYS_PER_PROBE); ri++) {
        uint buf_idx  = base + ri;
        vec4 r0 = ray_buf.data[buf_idx * 2u];       // radiance + hit_dist
        vec4 r1 = ray_buf.data[buf_idx * 2u + 1u];  // ray_dir  + pad

        vec3  radiance = r0.xyz;
        float hit_dist = r0.w;
        vec3  ray_dir  = r1.xyz;

        float cosine  = max(dot(ray_dir, oct_dir), 0.0);

        // Irradiance: cosine-weighted
        irr_sum += radiance * cosine;
        irr_w   += cosine;

        // Depth: squared-cosine weighting for Chebyshev sharpness
        float cosine2 = cosine * cosine;
        dep_sum   += hit_dist            * cosine2;
        dep_sum2  += hit_dist * hit_dist * cosine2;
        dep_w     += cosine2;
    }

    // Atlas texel coordinates for this probe tile
    uint  tile_col    = probe_idx % uint(ATLAS_COLS);
    uint  tile_row    = probe_idx / uint(ATLAS_COLS);
    ivec2 atlas_coord = ivec2(
        int(tile_col) * PROBE_RES + int(local_x),
        int(tile_row) * PROBE_RES + int(local_y)
    );

    // ── Irradiance atlas ──────────────────────────────────────────────────────
    vec3 new_irr = (irr_w > 1e-6) ? irr_sum / irr_w : vec3(0.0);
    vec3 old_irr = imageLoad(irr_atlas, atlas_coord).rgb;
    imageStore(irr_atlas, atlas_coord, vec4(mix(new_irr, old_irr, pc.hysteresis), 1.0));

    // ── Depth atlas (mean + mean²) ────────────────────────────────────────────
    vec2 new_dep = vec2(
        (dep_w > 1e-6) ? dep_sum  / dep_w : 0.0,
        (dep_w > 1e-6) ? dep_sum2 / dep_w : 0.0
    );
    vec2 old_dep = imageLoad(depth_atlas, atlas_coord).rg;
    imageStore(depth_atlas, atlas_coord, vec4(mix(new_dep, old_dep, pc.hysteresis), 0.0, 0.0));
}
