#[compute]

#version 460
#extension GL_EXT_shader_image_load_store : enable

#define GRID_X          8
#define GRID_Y          8
#define GRID_Z          8
#define RAYS_PER_PROBE  128
#define TOTAL_PROBES    (GRID_X * GRID_Y * GRID_Z)   // 512
#define IRR_PROBE_SIDE  16
#define IRR_ATLAS_COLS  (GRID_X * GRID_Z)            // 64

// One workgroup = one probe irradiance tile (8×8 threads).
layout(local_size_x = IRR_PROBE_SIDE, local_size_y = IRR_PROBE_SIDE, local_size_z = 1) in;

// Ray buffer: [2i+0] = radiance.xyz + hit_dist,  [2i+1] = ray_dir.xyz + pad
layout(set = 0, binding = 0, std430) readonly buffer RayBuffer { vec4 data[]; } ray_buf;
layout(set = 0, binding = 1, rgba16f) uniform image2D irr_atlas;

layout(push_constant, std430) uniform PC {
    float hysteresis;
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

    // Octahedral direction for this texel (centre of texel, [-1,1] range)
    vec2 oct_f   = ((vec2(local_x, local_y) + 0.5) / float(IRR_PROBE_SIDE)) * 2.0 - 1.0;
    vec3 oct_dir = oct_decode(oct_f);

    // Accumulate cosine-weighted radiance over all probe rays
    // energy_conservation = 0.95 (matches hybrid-rendering reference)
    const float ENERGY_CONSERVATION = 0.95;
    vec3  irr_sum = vec3(0.0);
    float irr_w   = 0.0;

    uint base = probe_idx * uint(RAYS_PER_PROBE);
    for (uint ri = 0u; ri < uint(RAYS_PER_PROBE); ri++) {
        uint buf_idx = base + ri;
        vec3  radiance = ray_buf.data[buf_idx * 2u].xyz * ENERGY_CONSERVATION;
        vec3  ray_dir  = ray_buf.data[buf_idx * 2u + 1u].xyz;

        float cosine = max(dot(ray_dir, oct_dir), 0.0);
        irr_sum += radiance * cosine;
        irr_w   += cosine;
    }

    vec3 new_irr = (irr_w > 1e-6) ? irr_sum / irr_w : vec3(0.0);

    // Atlas texel for this probe tile
    uint  tile_col    = probe_idx % uint(IRR_ATLAS_COLS);
    uint  tile_row    = probe_idx / uint(IRR_ATLAS_COLS);
    ivec2 atlas_coord = ivec2(
        int(tile_col) * IRR_PROBE_SIDE + int(local_x),
        int(tile_row) * IRR_PROBE_SIDE + int(local_y)
    );

    // Temporal blend in sqrt-space (perceptual, reduces contrast flicker)
    // sqrt(irr) blending then square out, same as hybrid-rendering reference
    vec3 old_irr = imageLoad(irr_atlas, atlas_coord).rgb;
    vec3 blended = mix(sqrt(max(new_irr, vec3(0.0))), sqrt(max(old_irr, vec3(0.0))), pc.hysteresis);
    imageStore(irr_atlas, atlas_coord, vec4(blended * blended, 1.0));
}
