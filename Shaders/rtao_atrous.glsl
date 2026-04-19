#[compute]
#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, rgba16f) uniform image2D ao_input;
layout(set = 0, binding = 1, rgba16f) uniform image2D ao_output;
layout(set = 0, binding = 2) uniform sampler2D normal_roughness_buffer;
layout(set = 0, binding = 3) uniform sampler2D depth_buffer;

layout(push_constant, std430) uniform PC {
    int   stride;      // 1, 2, 4 — grows each pass (à-trous)
    int   pad;
    ivec2 full_size;   // full-res G-buffer dimensions (for bilateral lookups)
} pc;

// B3 spline kernel — standard à-trous wavelet weights.
const float KERNEL[5] = float[5](0.0625, 0.25, 0.375, 0.25, 0.0625);

// One pass of à-trous wavelet denoising on the RTAO mask. Three dispatches
// with strides 1, 2, 4 cover a 17×17 effective window at just 25 taps per
// pass — far cheaper than a direct wide bilateral and much better at
// integrating away the stochastic noise that low-sample AO produces.
//
// Edge stopping via depth + normal keeps the widening kernel from bleeding
// occlusion across silhouettes.
void main() {
    ivec2 coord    = ivec2(gl_GlobalInvocationID.xy);
    ivec2 low_size = imageSize(ao_input);
    if (coord.x >= low_size.x || coord.y >= low_size.y) return;

    // Map low-res pixel → full-res G-buffer lookup (matches the raygen mapping).
    vec2  uv       = (vec2(coord) + 0.5) / vec2(low_size);
    ivec2 gb_coord = clamp(ivec2(uv * vec2(pc.full_size)),
                           ivec2(0), pc.full_size - 1);

    float ref_depth = texelFetch(depth_buffer, gb_coord, 0).r;
    // Sky — write fully open and skip the filter. Matches the RT raygen
    // behaviour so ping-pong state stays consistent.
    if (ref_depth <= 0.0001) {
        imageStore(ao_output, coord, vec4(1.0));
        return;
    }
    vec3 ref_normal = texelFetch(normal_roughness_buffer, gb_coord, 0).xyz * 2.0 - 1.0;

    float ao_sum     = 0.0;
    float weight_sum = 0.0;

    for (int dy = -2; dy <= 2; dy++) {
        for (int dx = -2; dx <= 2; dx++) {
            ivec2 sc = clamp(coord + ivec2(dx, dy) * pc.stride,
                             ivec2(0), low_size - 1);

            vec2  suv = (vec2(sc) + 0.5) / vec2(low_size);
            ivec2 sgb = clamp(ivec2(suv * vec2(pc.full_size)),
                              ivec2(0), pc.full_size - 1);
            float s_depth  = texelFetch(depth_buffer, sgb, 0).r;
            vec3  s_normal = texelFetch(normal_roughness_buffer, sgb, 0).xyz * 2.0 - 1.0;

            // Edge stopping — same formulas as reflection à-trous.
            float wd = exp(-abs(ref_depth - s_depth) * 256.0);
            float wn = pow(max(dot(ref_normal, s_normal), 0.0), 8.0);
            float wk = KERNEL[dx + 2] * KERNEL[dy + 2];

            float s_ao = imageLoad(ao_input, sc).a;
            float w    = wk * wd * wn;
            ao_sum     += s_ao * w;
            weight_sum += w;
        }
    }

    float result = (weight_sum > 1e-4)
        ? (ao_sum / weight_sum)
        : imageLoad(ao_input, coord).a;

    imageStore(ao_output, coord, vec4(vec3(result), result));
}
