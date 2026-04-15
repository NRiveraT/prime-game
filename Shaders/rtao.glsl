#[raygen]

#version 460
#extension GL_EXT_ray_tracing : enable

layout(location = 0) rayPayloadEXT float ao_hit;

layout(set = 0, binding = 0) uniform accelerationStructureEXT tlas;
layout(set = 0, binding = 1, rgba16f) uniform image2D ao_mask;
layout(set = 0, binding = 2) uniform sampler2D depth_buffer;
layout(set = 0, binding = 3) uniform sampler2D normal_roughness_buffer;

layout(push_constant, std430) uniform PC {
    mat4  inv_proj;       // floats  0-15  (64 bytes)
    vec4  inv_view_r0;    // floats 16-19  basis.x + origin.x
    vec4  inv_view_r1;    // floats 20-23  basis.y + origin.y
    vec4  inv_view_r2;    // floats 24-27  basis.z + origin.z
    vec2  screen_size;    // floats 28-29
    float ao_radius;      // float  30
    uint  frame_samples;  // uint   31 — high 16 bits = ao_samples, low 16 bits = frame_index
} pc;

// ── LCG RNG (Wyman DXR Tutorial 5) ────────────────────────────────────────────
// Seed mixer — call once per pixel with two uncorrelated values.
uint init_rand(uint val0, uint val1) {
    uint v0 = val0, v1 = val1, s0 = 0u;
    for (uint n = 0u; n < 16u; n++) {
        s0 += 0x9e3779b9u;
        v0 += ((v1 << 4u) + 0xa341316cu) ^ (v1 + s0) ^ ((v1 >> 5u) + 0xc8013ea4u);
        v1 += ((v0 << 4u) + 0xad90777du) ^ (v0 + s0) ^ ((v0 >> 5u) + 0x7e95761eu);
    }
    return v0;
}

// Advance the state and return a uniform float in [0, 1).
float next_rand(inout uint s) {
    s = 1664525u * s + 1013904223u;
    return float(s & 0x00FFFFFFu) / float(0x01000000u);
}

// ── Tangent-frame helpers ──────────────────────────────────────────────────────
vec3 perp(vec3 v) {
    vec3 a = abs(v);
    if (a.x <= a.y && a.x <= a.z) return vec3(0.0, -v.z,  v.y);
    if (a.y <= a.z)                return vec3(-v.z, 0.0,  v.x);
    return vec3(-v.y, v.x, 0.0);
}

mat3 tbn_from_normal(vec3 n) {
    vec3 t = normalize(perp(n));
    vec3 b = cross(n, t);
    return mat3(t, b, n);
}

// Cosine-weighted hemisphere sample (Shirley's concentric-disk mapping).
// r1, r2 in [0,1). Returns a direction in the +Z hemisphere (tangent space).
vec3 cosine_hemisphere(float r1, float r2) {
    float phi       = 6.28318530718 * r1;   // 2π * r1
    float sin_theta = sqrt(r2);
    float cos_theta = sqrt(1.0 - r2);
    return vec3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
}

void main() {
    ivec2 coord     = ivec2(gl_LaunchIDEXT.xy);
    vec2  screen_uv = (vec2(coord) + 0.5) / pc.screen_size;
    vec2  ndc_uv    = screen_uv * 2.0 - 1.0;

    float depth = texture(depth_buffer, screen_uv).r;
    if (depth <= 0.0001) {
        imageStore(ao_mask, coord, vec4(1.0)); // sky pixel: fully open (ao=1)
        return;
    }

    // Unproject depth → view space → world space
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

    mat3 tbn    = tbn_from_normal(normal);
    vec3 origin = world_pos + normal * 0.005; // small bias off surface

    // Unpack samples and frame index
    uint n_samples   = max((pc.frame_samples >> 16u) & 0xFFFFu, 1u);
    uint frame_index = pc.frame_samples & 0xFFFFu;

    // Seed RNG uniquely per-pixel per-frame
    uint pixel_idx = uint(coord.x) + uint(coord.y) * uint(pc.screen_size.x);
    uint rng       = init_rand(pixel_idx, frame_index);

    float ao_accum = 0.0;

    for (uint i = 0u; i < n_samples; i++) {
        float r1      = next_rand(rng);
        float r2      = next_rand(rng);
        vec3  ray_dir = normalize(tbn * cosine_hemisphere(r1, r2));

        // Default payload = 0.0 (occluded). Miss shader sets 1.0 (unoccluded).
        // SKIP_CLOSEST_HIT_SHADER means a geometry hit leaves payload at 0.0 cheaply.
        ao_hit = 0.0;
        traceRayEXT(
            tlas,
            gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT,
            0xFF, 0, 0, 0,
            origin, 0.001,
            ray_dir, pc.ao_radius,
            0
        );
        ao_accum += ao_hit; // 1.0 = ray escaped (open), 0.0 = ray hit geometry (occluded)
    }

    // ao = fraction of unoccluded rays: 1.0 = fully open, 0.0 = fully occluded
    float ao = ao_accum / float(max(n_samples, 1u));
    // RGB = grayscale visualisation of openness, A = ao value used by apply pass
    imageStore(ao_mask, coord, vec4(vec3(ao), ao));
}

#[miss]

#version 460
#extension GL_EXT_ray_tracing : enable

layout(location = 0) rayPayloadInEXT float ao_hit;

void main() {
    ao_hit = 1.0; // ray reached max distance — surface is unoccluded
}

#[closest_hit]

#version 460
#extension GL_EXT_ray_tracing : enable

layout(location = 0) rayPayloadInEXT float ao_hit;

void main() {
    // Unreachable: gl_RayFlagsSkipClosestHitShaderEXT is always set.
    // Payload stays 0.0 (occluded) from the raygen initialisation.
    ao_hit = 0.0;
}
