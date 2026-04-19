#[compute]
#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, rgba16f) uniform image2D color_buffer;
layout(set = 0, binding = 1, rgba16f) uniform image2D ao_mask;
layout(set = 0, binding = 2) uniform sampler2D normal_roughness_buffer;
layout(set = 0, binding = 3) uniform sampler2D depth_buffer;

layout(push_constant, std430) uniform PC {
    float ao_strength;
    float pad;
    vec2  rt_size;   // dispatch size of the RT raygen pass
} pc;

// Plain 2×2 bilinear upsample of the (à-trous-denoised) AO mask onto the
// full-res colour buffer. Denoising has already happened in the à-trous
// passes, so this stage only has to reconstruct full-res values.
void main() {
    ivec2 full_coord = ivec2(gl_GlobalInvocationID.xy);
    ivec2 full_size  = imageSize(color_buffer);
    if (full_coord.x >= full_size.x || full_coord.y >= full_size.y) return;

    // Sky pixel — never modify the colour buffer. Without this gate the
    // 2×2 bilinear near reflective-object silhouettes would pull partially
    // occluded taps into sky pixels and show up as glow-amplified halos.
    float depth = texelFetch(depth_buffer, full_coord, 0).r;
    if (depth <= 0.0001) return;

    vec2  low_pos  = (vec2(full_coord) + 0.5) / vec2(full_size) * pc.rt_size - 0.5;
    ivec2 low_base = ivec2(floor(low_pos));
    vec2  frac     = low_pos - vec2(low_base);

    float ao = 0.0;
    for (int dy = 0; dy <= 1; dy++) {
        for (int dx = 0; dx <= 1; dx++) {
            ivec2 lc = clamp(low_base + ivec2(dx, dy), ivec2(0), ivec2(pc.rt_size) - 1);
            float wb = (dx == 0 ? 1.0 - frac.x : frac.x)
                     * (dy == 0 ? 1.0 - frac.y : frac.y);
            ao += imageLoad(ao_mask, lc).a * wb;
        }
    }

    vec4 color = imageLoad(color_buffer, full_coord);
    // ao=1 → no darkening; ao=0 → full ao_strength darkening.
    imageStore(color_buffer, full_coord,
               vec4(color.rgb * mix(1.0 - pc.ao_strength, 1.0, ao), color.a));
}
