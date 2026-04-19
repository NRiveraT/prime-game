#[compute]
#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

// Ping-pong: pass 1 reads the raw mask into _blur, pass 2 reads _blur back into mask.
layout(set = 0, binding = 0, rgba16f) uniform image2D ao_input;
layout(set = 0, binding = 1, rgba16f) uniform image2D ao_output;
layout(set = 0, binding = 2) uniform sampler2D normal_roughness_buffer;
layout(set = 0, binding = 3) uniform sampler2D depth_buffer;

layout(push_constant, std430) uniform PC {
    ivec2 direction;  // (1,0) horizontal pass, (0,1) vertical pass
    ivec2 full_size;  // full-res G-buffer size (for bilateral reads)
} pc;

// 7-tap gaussian (σ ≈ 1.0).
const float WEIGHTS[7] = float[7](0.006, 0.061, 0.242, 0.383, 0.242, 0.061, 0.006);

// Separable bilateral blur on the low-res AO mask. Runs before the apply pass,
// so ao_samples can drop to 1 or 2 per raygen pixel and still produce clean
// output — spatial neighbours denoise each other, and the bilateral keeps
// the blur from crossing silhouettes.
void main() {
    ivec2 coord   = ivec2(gl_GlobalInvocationID.xy);
    ivec2 rt_size = imageSize(ao_output);
    if (coord.x >= rt_size.x || coord.y >= rt_size.y) return;

    // Full-res G-buffer coord for THIS low-res pixel (must match the mapping
    // the raygen used when it generated ao_input).
    vec2  uv       = (vec2(coord) + 0.5) / vec2(rt_size);
    ivec2 gb_coord = clamp(ivec2(uv * vec2(pc.full_size)),
                           ivec2(0), pc.full_size - 1);

    float ref_depth = texelFetch(depth_buffer, gb_coord, 0).r;
    // Sky pixel — keep fully open, skip the blur.
    if (ref_depth <= 0.0001) {
        imageStore(ao_output, coord, vec4(1.0));
        return;
    }
    vec3 ref_normal = texelFetch(normal_roughness_buffer, gb_coord, 0).xyz * 2.0 - 1.0;

    float ao_sum     = 0.0;
    float weight_sum = 0.0;

    for (int i = -3; i <= 3; i++) {
        ivec2 sc = coord + pc.direction * i;
        sc = clamp(sc, ivec2(0), rt_size - 1);

        vec2  sc_uv    = (vec2(sc) + 0.5) / vec2(rt_size);
        ivec2 sgb      = clamp(ivec2(sc_uv * vec2(pc.full_size)),
                               ivec2(0), pc.full_size - 1);
        float s_depth  = texelFetch(depth_buffer, sgb, 0).r;
        vec3  s_normal = texelFetch(normal_roughness_buffer, sgb, 0).xyz * 2.0 - 1.0;

        // Bilateral rejection: silhouettes get near-zero contribution.
        float wd = exp(-abs(ref_depth - s_depth) * 256.0);
        float wn = max(dot(ref_normal, s_normal), 0.0);
        wn = wn * wn;

        float w  = WEIGHTS[i + 3] * wd * wn;
        float ao = imageLoad(ao_input, sc).a;
        ao_sum     += ao * w;
        weight_sum += w;
    }

    // If every neighbour rejected (isolated silhouette pixel), keep the raw AO.
    float ao = weight_sum > 1e-4
        ? (ao_sum / weight_sum)
        : imageLoad(ao_input, coord).a;

    imageStore(ao_output, coord, vec4(vec3(ao), ao));
}
