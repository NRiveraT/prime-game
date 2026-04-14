#[compute]
#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, rgba16f) uniform image2D color_buffer;
layout(set = 0, binding = 1, rgba16f) uniform image2D ao_mask;

layout(push_constant, std430) uniform PC {
    float ao_strength;
} pc;

void main() {
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size  = imageSize(color_buffer);
    if (coord.x >= size.x || coord.y >= size.y) return;

    float ao    = imageLoad(ao_mask, coord).a;  // 1=fully open, 0=fully occluded
    vec4  color = imageLoad(color_buffer, coord);
    // ao=1 → no darkening (multiply by 1.0); ao=0 → full darkening (multiply by 1-strength)
    imageStore(color_buffer, coord, vec4(color.rgb * mix(1.0 - pc.ao_strength, 1.0, ao), color.a));
    // Debug: visualise ao_mask directly
//    imageStore(color_buffer, coord, vec4(imageLoad(ao_mask, coord).rgb, color.a));
}
