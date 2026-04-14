#[raygen]

#version 460
#extension GL_EXT_ray_tracing : enable

layout(location = 0) rayPayloadEXT float shadow_hit;

layout(set = 0, binding = 0) uniform accelerationStructureEXT tlas;
layout(set = 0, binding = 1, rgba16f) uniform image2D shadow_mask;
layout(set = 0, binding = 2) uniform sampler2D depth_buffer;
layout(set = 0, binding = 3) uniform sampler2D normal_roughness_buffer;

layout(push_constant, std430) uniform PC {
    mat4  inv_proj;// floats  0-15  (64 bytes)
    vec4  inv_view_r0;// floats 16-19  basis.x + origin.x
    vec4  inv_view_r1;// floats 20-23  basis.y + origin.y
    vec4  inv_view_r2;// floats 24-27  basis.z + origin.z
    vec2  screen_size;// floats 28-29
    float shadow_bias;// float  30
    float max_dist;// float  31
} pc;

struct LightData {
    vec4 direction_energy;// xyz = toward-light dir, w = energy
    vec4 color;
};

layout(set = 0, binding = 4, std430) readonly buffer LightBlock {
    LightData light;
} light_block;

void main() {
    ivec2 coord     = ivec2(gl_LaunchIDEXT.xy);
    vec2  screen_uv = (vec2(coord) + 0.5) / pc.screen_size;
    vec2  ndc_uv    = screen_uv * 2.0 - 1.0;

    // Sky — no geometry, no shadow
    float depth = texture(depth_buffer, screen_uv).r;
    if (depth <= 0.0001) {
        imageStore(shadow_mask, coord, vec4(0.0, 0.0, 1.0, 0.0)); // blue=sky, a=0
        return;
    }

    // Depth → view space → world space
    vec4 view_pos = pc.inv_proj * vec4(ndc_uv, depth, 1.0);
    view_pos.xyz /= view_pos.w;

    vec3 world_pos = vec3(
        pc.inv_view_r0.x * view_pos.x + pc.inv_view_r1.x * view_pos.y + pc.inv_view_r2.x * view_pos.z + pc.inv_view_r0.w,
        pc.inv_view_r0.y * view_pos.x + pc.inv_view_r1.y * view_pos.y + pc.inv_view_r2.y * view_pos.z + pc.inv_view_r1.w,
        pc.inv_view_r0.z * view_pos.x + pc.inv_view_r1.z * view_pos.y + pc.inv_view_r2.z * view_pos.z + pc.inv_view_r2.w
    );

    // Normal: view space → world space
    vec3 normal_vs = texture(normal_roughness_buffer, screen_uv).xyz * 2.0 - 1.0;
    vec3 normal = normalize(vec3(
        pc.inv_view_r0.x * normal_vs.x + pc.inv_view_r1.x * normal_vs.y + pc.inv_view_r2.x * normal_vs.z,
        pc.inv_view_r0.y * normal_vs.x + pc.inv_view_r1.y * normal_vs.y + pc.inv_view_r2.y * normal_vs.z,
        pc.inv_view_r0.z * normal_vs.x + pc.inv_view_r1.z * normal_vs.y + pc.inv_view_r2.z * normal_vs.z
    ));

    vec3  light_dir = light_block.light.direction_energy.xyz; // toward light
    // Clamp to [0,1]: back-facing surfaces map to 0 (fully away from light).
    float NdotL = clamp(dot(normal, light_dir), 0.0, 1.0);

    // Surfaces at or past the terminator: fully dark, no ray needed.
    if (NdotL <= 0.001) {
        imageStore(shadow_mask, coord, vec4(0.0, 1.0, 0.0, 1.0)); // green=terminator, a=1 (dark)
        return;
    }

    vec3 shadow_origin = world_pos + normal * pc.shadow_bias;

    shadow_hit = 0.0;
    traceRayEXT(
        tlas,
        gl_RayFlagsTerminateOnFirstHitEXT,
        0xFF, 0, 0, 0,
        shadow_origin, 0.001,
        light_dir, pc.max_dist,
        0
    );

    float shadow = shadow_hit > 0.5 ? 1.0 : 0.0;

    // Smooth NdotL ramp over the first ~11° from the terminator (NdotL 0 → 0.2).
    // This blends the hard-dark terminator region into the ray-traced shadow without
    // a visible seam on curved surfaces. Outside that band the ray result is authoritative.
    float terminator_blend = 1.0 - smoothstep(0.0, 0.2, NdotL);
    float alpha = max(shadow, terminator_blend);

    if (shadow > 0.5) {
        imageStore(shadow_mask, coord, vec4(1.0, 0.0, 0.0, alpha)); // red=occluded
    } else {
        imageStore(shadow_mask, coord, vec4(1.0, 1.0, 1.0, alpha)); // white=lit (alpha=0 away from terminator)
    }
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
