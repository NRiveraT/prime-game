#[compute]
#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, rgba16f) uniform image2D color_buffer;
layout(set = 0, binding = 1, rgba16f)    uniform image2D shadow_mask;

void main() {
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size  = imageSize(color_buffer);
    if (coord.x >= size.x || coord.y >= size.y) return;

    vec4 shadow = imageLoad(shadow_mask, coord);
    vec4  color  = imageLoad(color_buffer, coord);
    
    imageStore(color_buffer, coord, vec4(color.rgb * (1.0 - shadow.a * 0.8f), color.a));
//    imageStore(color_buffer, coord, vec4(shadow));
}