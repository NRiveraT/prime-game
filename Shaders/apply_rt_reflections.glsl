#[compute]
#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, rgba16f) uniform image2D color_buffer;
layout(set = 0, binding = 1, rgba16f) uniform image2D reflection_mask;

layout(push_constant, std430) uniform PC {
    float reflection_strength;
} pc;

void main() {
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size  = imageSize(color_buffer);
    if (coord.x >= size.x || coord.y >= size.y) return;

    vec4 refl = imageLoad(reflection_mask, coord);
    // refl.a == 0.0 for sky pixels — no geometry, skip blending.
    // Geometry pixels have a = 1.0 + fresnel * spec_vis, always > 1.0.
    if (refl.a < 0.5) return;

    // Unpack Fresnel * spec_vis weight from the offset alpha channel.
    float weight = (refl.a - 1.0) * pc.reflection_strength;

    // Additive blend: preserves existing diffuse, adds only the specular
    // contribution bounded by Fresnel (max ~4% at normal incidence for dielectrics).
    vec4 color = imageLoad(color_buffer, coord);
    imageStore(color_buffer, coord, vec4(color.rgb + refl.rgb * weight, color.a));
    // Debug: visualise reflection_mask directly
//    imageStore(color_buffer, coord, vec4(refl.rgb, color.a));
}
