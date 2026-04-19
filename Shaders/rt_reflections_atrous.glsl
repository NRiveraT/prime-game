#[compute]
#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, rgba16f) uniform image2D reflection_input;
layout(set = 0, binding = 1, rgba16f) uniform image2D reflection_output;
layout(set = 0, binding = 2) uniform sampler2D normal_roughness_buffer;
layout(set = 0, binding = 3) uniform sampler2D depth_buffer;

layout(push_constant, std430) uniform PC {
    int   stride;      // 1, 2, 4, ... grows each pass (à-trous)
    int   pad;
    ivec2 full_size;   // full-res G-buffer dimensions (for bilateral lookups)
} pc;

// B3 spline kernel — the standard à-trous wavelet weights.
//   Outer product over x and y gives a 5×5 kernel normalised to 1.
const float KERNEL[5] = float[5](0.0625, 0.25, 0.375, 0.25, 0.0625);

// One pass of à-trous wavelet filtering. Run with strides 1, 2, 4 (three
// dispatches) to cover a 17×17 effective window in only 25 taps per pass,
// vs 289 taps a bare 17×17 bilateral would need.
//
// Edge stopping via depth + normal similarity — identical in spirit to what
// bilateral was doing, but spread across multiple cheap passes. The widening
// stride is what turns this into a hole-filler: pass 0 smooths neighbours,
// pass 1 reaches past small gaps, pass 2 reaches across sub-low-res holes
// that single-pass bilateral can't see past.
void main() {
    ivec2 coord    = ivec2(gl_GlobalInvocationID.xy);
    ivec2 low_size = imageSize(reflection_input);
    if (coord.x >= low_size.x || coord.y >= low_size.y) return;

    vec4 center = imageLoad(reflection_input, coord);
    // Pass invalid (alpha < 0.5) pixels through unchanged so ping-pong state
    // stays consistent — we never *fill* a rough/sky pixel, only smooth
    // within regions that already have valid reflections.
    if (center.a < 0.5) {
        imageStore(reflection_output, coord, center);
        return;
    }

    vec2  uv         = (vec2(coord) + 0.5) / vec2(low_size);
    ivec2 gb_coord   = clamp(ivec2(uv * vec2(pc.full_size)),
                             ivec2(0), pc.full_size - 1);
    float ref_depth  = texelFetch(depth_buffer, gb_coord, 0).r;
    vec4  ref_nr     = texelFetch(normal_roughness_buffer, gb_coord, 0);
    vec3  ref_normal = ref_nr.xyz * 2.0 - 1.0;
    float roughness  = ref_nr.w;

    // ── Mirror gate ───────────────────────────────────────────────────────────
    // Near-mirror pixels are deterministic reflections — one ray gives the
    // exact answer. Running à-trous on them averages neighbours whose
    // reflection directions differ sharply (smooth curvature on a chrome ball,
    // silhouettes of reflected detail, etc.) and produces banding / blocky
    // patches. Pass those pixels through without filtering.
    //
    // Cutoff raised to 0.15 — materials near but not exactly 0 roughness
    // (common in practice: 0.05–0.1 for polished metal) still need full
    // pass-through. The ramp reaches full filter at 0.4 roughness so the
    // transition into "glossy but needs denoise" territory is gradual.
    float filter_strength = smoothstep(0.15, 0.4, roughness);
    if (filter_strength < 0.001) {
        imageStore(reflection_output, coord, center);
        return;
    }

    vec4  accum      = vec4(0.0);
    float weight_sum = 0.0;

    for (int dy = -2; dy <= 2; dy++) {
        for (int dx = -2; dx <= 2; dx++) {
            ivec2 sc = clamp(coord + ivec2(dx, dy) * pc.stride,
                             ivec2(0), low_size - 1);

            vec2  suv      = (vec2(sc) + 0.5) / vec2(low_size);
            ivec2 sgb      = clamp(ivec2(suv * vec2(pc.full_size)),
                                   ivec2(0), pc.full_size - 1);
            float s_depth  = texelFetch(depth_buffer, sgb, 0).r;
            vec3  s_normal = texelFetch(normal_roughness_buffer, sgb, 0).xyz * 2.0 - 1.0;

            // Edge stopping. Depth falloff is aggressive (silhouettes reject
            // cleanly), normal is raised to a power to sharpen transitions.
            float wd = exp(-abs(ref_depth - s_depth) * 256.0);
            float wn = pow(max(dot(ref_normal, s_normal), 0.0), 8.0);
            float wk = KERNEL[dx + 2] * KERNEL[dy + 2];

            vec4 s = imageLoad(reflection_input, sc);
            if (s.a >= 0.5) {
                float w = wk * wd * wn;
                accum      += s * w;
                weight_sum += w;
            }
        }
    }

    vec4 filtered = (weight_sum > 1e-4) ? (accum / weight_sum) : center;
    // Ramp between unfiltered center (mirror-like) and filtered neighbourhood
    // (rougher) based on local roughness — prevents hard transitions between
    // filtered and unfiltered regions in the mask.
    vec4 result = mix(center, filtered, filter_strength);
    imageStore(reflection_output, coord, result);
}
