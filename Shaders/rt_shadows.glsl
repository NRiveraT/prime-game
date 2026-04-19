#[raygen]

#version 460
#extension GL_EXT_ray_tracing : enable

layout(location = 0) rayPayloadEXT float shadow_hit;

layout(set = 0, binding = 0) uniform accelerationStructureEXT tlas;
layout(set = 0, binding = 1, rgba16f) uniform image2D shadow_mask;
layout(set = 0, binding = 2) uniform sampler2D depth_buffer;
layout(set = 0, binding = 3) uniform sampler2D normal_roughness_buffer;

layout(push_constant, std430) uniform PC {
    mat4  inv_proj;
    vec4  inv_view_r0;
    vec4  inv_view_r1;
    vec4  inv_view_r2;
    vec2  screen_size;
    float shadow_bias;
    float max_dist;
} pc;

layout(set = 0, binding = 4, std430) readonly buffer LightBlock {
    vec4 direction_energy; // xyz = toward-light dir, w = energy
    vec4 color_sha;        // xyz = color, w = sin_half_angle (from angular_distance)
    vec4 frame_data;       // x = frame_index, y = shadow_samples, zw = unused
} light_block;

struct RTLight {
    vec4 pos_range;    // xyz = world position, w = range
    vec4 color_energy; // xyz = color, w = energy
    vec4 dir_size;     // xyz = spot forward (-basis.z), w = light_size (disk radius)
    vec4 cone_type;    // x = cos_outer, y = cos_inner, z = type (0=omni, 1=spot), w = unused
};
layout(set = 0, binding = 5, std430) readonly buffer LocalLightsBlock {
    uint    count;
    uint    pad0, pad1, pad2;
    RTLight lights[32];
} local_lights_buf;

// ── LCG RNG ───────────────────────────────────────────────────────────────────
uint init_rand(uint val0, uint val1) {
    uint v0 = val0, v1 = val1, s0 = 0u;
    for (uint n = 0u; n < 16u; n++) {
        s0 += 0x9e3779b9u;
        v0 += ((v1 << 4u) + 0xa341316cu) ^ (v1 + s0) ^ ((v1 >> 5u) + 0xc8013ea4u);
        v1 += ((v0 << 4u) + 0xad90777du) ^ (v0 + s0) ^ ((v0 >> 5u) + 0x7e95761eu);
    }
    return v0;
}
float next_rand(inout uint s) {
    s = 1664525u * s + 1013904223u;
    return float(s & 0x00FFFFFFu) / float(0x01000000u);
}

// ── Helpers ───────────────────────────────────────────────────────────────────
vec3 perp(vec3 v) {
    vec3 a = abs(v);
    if (a.x <= a.y && a.x <= a.z) return vec3(0.0, -v.z,  v.y);
    if (a.y <= a.z)                return vec3(-v.z, 0.0,  v.x);
    return vec3(-v.y, v.x, 0.0);
}

// Uniform sample over spherical cap (sin_half_angle = sin of cone half-angle).
vec3 sample_cone(vec3 dir, float sin_half_angle, inout uint rng) {
    float r1 = next_rand(rng);
    float r2 = next_rand(rng);
    float phi       = 6.28318530718 * r1;
    float cos_max   = sqrt(max(0.0, 1.0 - sin_half_angle * sin_half_angle));
    float cos_theta = 1.0 - r2 * (1.0 - cos_max);
    float sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
    vec3 t = normalize(perp(dir));
    vec3 b = cross(dir, t);
    return sin_theta * cos(phi) * t + sin_theta * sin(phi) * b + cos_theta * dir;
}

void main() {
    ivec2 coord     = ivec2(gl_LaunchIDEXT.xy);
    vec2  screen_uv = (vec2(coord) + 0.5) / pc.screen_size;
    vec2  ndc_uv    = screen_uv * 2.0 - 1.0;

    float depth = texture(depth_buffer, screen_uv).r;
    if (depth <= 0.0001) {
        imageStore(shadow_mask, coord, vec4(0.0, 0.0, 1.0, 0.0));
        return;
    }

    vec4 view_pos = pc.inv_proj * vec4(ndc_uv, depth, 1.0);
    view_pos.xyz /= view_pos.w;
    vec3 world_pos = vec3(
        pc.inv_view_r0.x * view_pos.x + pc.inv_view_r1.x * view_pos.y + pc.inv_view_r2.x * view_pos.z + pc.inv_view_r0.w,
        pc.inv_view_r0.y * view_pos.x + pc.inv_view_r1.y * view_pos.y + pc.inv_view_r2.y * view_pos.z + pc.inv_view_r1.w,
        pc.inv_view_r0.z * view_pos.x + pc.inv_view_r1.z * view_pos.y + pc.inv_view_r2.z * view_pos.z + pc.inv_view_r2.w
    );

    vec3 normal_vs = texture(normal_roughness_buffer, screen_uv).xyz * 2.0 - 1.0;
    vec3 normal = normalize(vec3(
        pc.inv_view_r0.x * normal_vs.x + pc.inv_view_r1.x * normal_vs.y + pc.inv_view_r2.x * normal_vs.z,
        pc.inv_view_r0.y * normal_vs.x + pc.inv_view_r1.y * normal_vs.y + pc.inv_view_r2.y * normal_vs.z,
        pc.inv_view_r0.z * normal_vs.x + pc.inv_view_r1.z * normal_vs.y + pc.inv_view_r2.z * normal_vs.z
    ));

    vec3 shadow_origin = world_pos + normal * pc.shadow_bias;

    uint frame_idx = uint(light_block.frame_data.x);
    uint n_samples = max(uint(light_block.frame_data.y), 1u);
    uint rng       = init_rand(uint(coord.x) + uint(coord.y) * uint(pc.screen_size.x), frame_idx);

    // ── Directional shadow ────────────────────────────────────────────────────
    vec3  light_dir = light_block.direction_energy.xyz;
    float NdotL     = clamp(dot(normal, light_dir), 0.0, 1.0);
    float sin_ha    = light_block.color_sha.w;

    float dir_shadow = 1.0;
    float terminator = 1.0;
    if (NdotL > 0.001) {
        float accum = 0.0;
        for (uint s = 0u; s < n_samples; s++) {
            vec3 jdir = (sin_ha > 0.001) ? sample_cone(light_dir, sin_ha, rng) : light_dir;
            shadow_hit = 0.0;
            traceRayEXT(tlas, gl_RayFlagsTerminateOnFirstHitEXT,
                        0xFF, 0, 0, 0, shadow_origin, 0.001, jdir, pc.max_dist, 0);
            accum += shadow_hit > 0.5 ? 1.0 : 0.0;
        }
        dir_shadow = accum / float(n_samples);
        terminator = 1.0 - smoothstep(0.0, 0.2, NdotL);
    }
    float dir_alpha = max(dir_shadow, terminator);

    // ── Local light shadows ───────────────────────────────────────────────────
    // Shadow cast radius can exceed the illumination range so shadows fade out
    // smoothly past the attenuation edge rather than cutting off hard.
    float shadow_range_mult = max(light_block.frame_data.z, 1.0);
    float local_alpha = 0.0;
    for (uint li = 0u; li < local_lights_buf.count && li < 32u; li++) {
        RTLight lgt          = local_lights_buf.lights[li];
        vec3  to_light       = lgt.pos_range.xyz - world_pos;
        float dist           = length(to_light);
        float range          = lgt.pos_range.w;
        float shadow_range   = range * shadow_range_mult;
        if (dist >= shadow_range || dist < 0.001) continue;

        vec3  ldir   = to_light / dist;
        float lndotl = max(dot(normal, ldir), 0.0);
        if (lndotl < 0.001) continue;

        float spot_att = 1.0;
        if (lgt.cone_type.z > 0.5) {
            float cos_a = dot(-ldir, lgt.dir_size.xyz);
            spot_att = clamp((cos_a - lgt.cone_type.x) / max(lgt.cone_type.y - lgt.cone_type.x, 0.001), 0.0, 1.0);
            if (spot_att <= 0.0) continue;
        }

        // Separate shadow falloff from illumination: illumination cuts off at `range`,
        // shadow smoothly fades to zero at `shadow_range`. Quadratic for smoother edge.
        float t_shadow = 1.0 - clamp(dist / shadow_range, 0.0, 1.0);
        float shadow_weight = t_shadow * t_shadow;

        // Skip lights whose combined influence is too small — avoids shadow noise
        // on surfaces the light barely reaches.
        if (shadow_weight * lndotl * spot_att < 0.02) continue;

        float lsz    = lgt.dir_size.w;
        float laccum = 0.0;
        for (uint s = 0u; s < n_samples; s++) {
            vec3 jldir = ldir;
            if (lsz > 0.001) {
                vec3 lt = normalize(perp(ldir));
                vec3 lb = cross(ldir, lt);
                float r1 = next_rand(rng);
                float r2 = next_rand(rng);
                float ang = 6.28318530718 * r1;
                float rad = sqrt(r2) * lsz;
                jldir = normalize((lgt.pos_range.xyz + (cos(ang) * lt + sin(ang) * lb) * rad) - world_pos);
                if (dot(normal, jldir) < 0.001) jldir = ldir;
            }
            shadow_hit = 0.0;
            traceRayEXT(tlas, gl_RayFlagsTerminateOnFirstHitEXT,
                        0xFF, 0, 0, 0, shadow_origin, 0.001, jldir, dist + lsz + 0.1, 0);
            laccum += shadow_hit > 0.5 ? 1.0 : 0.0;
        }
        // Weight shadow by the smooth range falloff so the boundary fades out
        // rather than cutting off hard at attenuation edge.
        local_alpha = max(local_alpha, (laccum / float(n_samples)) * shadow_weight * spot_att);
    }

    float alpha = max(dir_alpha, local_alpha);
    imageStore(shadow_mask, coord, vec4(dir_shadow, local_alpha, 0.0, alpha));
}

#[miss]

#version 460
#extension GL_EXT_ray_tracing : enable

layout(location = 0) rayPayloadInEXT float shadow_hit;

void main() {
    shadow_hit = 0.0;
}

#[closest_hit]

#version 460
#extension GL_EXT_ray_tracing : enable

layout(location = 0) rayPayloadInEXT float shadow_hit;

void main() {
    shadow_hit = 1.0;
}
