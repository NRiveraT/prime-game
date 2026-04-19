#[compute]
#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, rgba16f) uniform image2D color_buffer;
layout(set = 0, binding = 1, rgba16f) uniform image2D reflection_mask;
layout(set = 0, binding = 2) uniform sampler2D normal_roughness_buffer;
layout(set = 0, binding = 3) uniform sampler2D depth_buffer;

layout(push_constant, std430) uniform PC {
    float reflection_strength;
    float pad;
    vec2  rt_size;   // dispatch size of the RT raygen pass (may be < color_buffer size)
} pc;

// Upsamples the low-res reflection mask to full-res output.
//
// Two modes based on reference pixel roughness:
//
//   MIRROR  (roughness < 0.15) — nearest-best-match tap.
//     Mirror pixels carry unique deterministic reflections. Bilinear blending
//     would average adjacent reflection DIRECTIONS across a curved surface
//     and produce banding / ghost edges. Instead: evaluate all 4 candidate
//     low-res taps' source G-buffer against the full-res reference pixel,
//     pick the best match, use its reflection. Blocky at low res but never
//     smears distinct reflections together.
//
//   GLOSSY/ROUGH (roughness ≥ 0.15) — 2×2 bilinear with bilateral rejection.
//     Smooth interpolation where it's safe (reflections are already blurry).
//     Cross-silhouette taps still get rejected by full-res G-buffer bilateral.
//
// Both modes gate on depth/roughness at the reference pixel, so sky and
// non-reflective surfaces never receive reflection data.
void main() {
    ivec2 full_coord = ivec2(gl_GlobalInvocationID.xy);
    ivec2 full_size  = imageSize(color_buffer);
    if (full_coord.x >= full_size.x || full_coord.y >= full_size.y) return;

    float ref_depth = texelFetch(depth_buffer, full_coord, 0).r;
    if (ref_depth <= 0.0001) return;

    vec4  ref_nr    = texelFetch(normal_roughness_buffer, full_coord, 0);
    float roughness = ref_nr.w;
    if (roughness >= 0.5) return;
    vec3 ref_normal = ref_nr.xyz * 2.0 - 1.0;

    vec2  low_pos  = (vec2(full_coord) + 0.5) / vec2(full_size) * pc.rt_size - 0.5;
    ivec2 low_base = ivec2(floor(low_pos));

    vec4 refl = vec4(0.0);

    if (roughness < 0.15) {
        // ── MIRROR: pick the best-matching low-res tap ────────────────────────
        ivec2 best_tap = low_base;
        float best_w   = -1.0;
        for (int dy = 0; dy <= 1; dy++) {
            for (int dx = 0; dx <= 1; dx++) {
                ivec2 lc = clamp(low_base + ivec2(dx, dy), ivec2(0), ivec2(pc.rt_size) - 1);

                // The full-res pixel where the raygen actually sampled this tap.
                vec2  lc_uv    = (vec2(lc) + 0.5) / pc.rt_size;
                ivec2 gb_coord = clamp(ivec2(lc_uv * vec2(full_size)),
                                       ivec2(0), full_size - 1);
                float s_depth  = texelFetch(depth_buffer, gb_coord, 0).r;
                vec3  s_normal = texelFetch(normal_roughness_buffer, gb_coord, 0).xyz * 2.0 - 1.0;

                float wd = exp(-abs(ref_depth - s_depth) * 256.0);
                float wn = pow(max(dot(ref_normal, s_normal), 0.0), 8.0);
                float w  = wd * wn;

                // Prefer valid taps (alpha ≥ 0.5) over invalid ones.
                vec4 candidate = imageLoad(reflection_mask, lc);
                if (candidate.a < 0.5) w *= 0.01;

                if (w > best_w) {
                    best_w   = w;
                    best_tap = lc;
                    refl     = candidate;
                }
            }
        }
        // No valid nearest tap — leave colour alone.
        if (refl.a < 0.5) return;
    } else {
        // ── GLOSSY/ROUGH: 2×2 bilinear with bilateral rejection ───────────────
        vec2  frac       = low_pos - vec2(low_base);
        vec4  accum      = vec4(0.0);
        float weight_sum = 0.0;
        for (int dy = 0; dy <= 1; dy++) {
            for (int dx = 0; dx <= 1; dx++) {
                ivec2 lc = clamp(low_base + ivec2(dx, dy), ivec2(0), ivec2(pc.rt_size) - 1);
                float wb = (dx == 0 ? 1.0 - frac.x : frac.x)
                         * (dy == 0 ? 1.0 - frac.y : frac.y);

                vec2  lc_uv    = (vec2(lc) + 0.5) / pc.rt_size;
                ivec2 gb_coord = clamp(ivec2(lc_uv * vec2(full_size)),
                                       ivec2(0), full_size - 1);
                float s_depth  = texelFetch(depth_buffer, gb_coord, 0).r;
                vec3  s_normal = texelFetch(normal_roughness_buffer, gb_coord, 0).xyz * 2.0 - 1.0;

                float wd = exp(-abs(ref_depth - s_depth) * 256.0);
                float wn = pow(max(dot(ref_normal, s_normal), 0.0), 8.0);

                vec4 tap = imageLoad(reflection_mask, lc);
                if (tap.a >= 0.5) {
                    float w = wb * wd * wn;
                    accum      += tap * w;
                    weight_sum += w;
                }
            }
        }
        if (weight_sum < 1e-4) return;
        refl = accum / weight_sum;
        if (refl.a < 0.5) return;
    }

    // Fresnel × spec_vis encoded as (1 + weight). Smooth fade at the cutoff
    // so the 0.5 roughness boundary doesn't show a hard seam.
    float weight     = refl.a - 1.0;
    float rough_fade = 1.0 - smoothstep(0.3, 0.5, roughness);
    weight *= rough_fade;

    vec4 color = imageLoad(color_buffer, full_coord);
    imageStore(color_buffer, full_coord,
               vec4(mix(color.rgb, refl.rgb, clamp(weight, 0.0, 1.0)), color.a));
}
