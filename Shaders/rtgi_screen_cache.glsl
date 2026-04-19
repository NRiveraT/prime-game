#[raygen]

#version 460
#extension GL_EXT_ray_tracing : enable

// Same payload layout as rt_reflections + rtgi_probe_update so we can share
// the chit/miss bodies across all GI-related ray passes.
struct ReflPayload { vec4 ra; vec4 rb; vec4 rc; vec4 rd; };
layout(location = 0) rayPayloadEXT ReflPayload rpl;

struct Probe { vec4 rgb_meta; };

layout(set = 0, binding = 0) uniform accelerationStructureEXT tlas;
layout(set = 0, binding = 1, rgba16f) uniform image2D screen_cache;
layout(set = 0, binding = 2) uniform sampler2D depth_buffer;
layout(set = 0, binding = 3) uniform sampler2D normal_roughness_buffer;

layout(set = 0, binding = 11, std430) readonly buffer LightBlock {
    vec4 direction_energy;
    vec4 color_sky_energy;
    vec4 ambient_pad;
    vec4 params;       // x = indirect_intensity (direct-bounce boost)
} light;

layout(set = 0, binding = 12, std430) readonly buffer ProbesBuf {
    Probe data[];
} probes;

// Camera matrices for world-position/normal reconstruction.
layout(set = 0, binding = 14, std430) readonly buffer CameraBuf {
    mat4 inv_proj;
    vec4 inv_view_r0;
    vec4 inv_view_r1;
    vec4 inv_view_r2;
} camera;

layout(push_constant, std430) uniform PC {
    vec3  grid_origin;
    float grid_spacing;
    ivec3 grid_size;
    uint  frame_idx;
    float max_ray_dist;
    float cache_ema_alpha;  // 0 = fresh sample every frame, 1 = keep old forever
    float _pad0;
    float _pad1;
} pc;

// ── RNG, helpers ──────────────────────────────────────────────────────────────
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

vec3 perp(vec3 v) {
    vec3 a = abs(v);
    if (a.x <= a.y && a.x <= a.z) return vec3(0.0, -v.z,  v.y);
    if (a.y <= a.z)                return vec3(-v.z, 0.0,  v.x);
    return vec3(-v.y, v.x, 0.0);
}

// Cosine-weighted hemisphere sample — +Z is surface normal.
vec3 cosine_hemisphere(float r1, float r2) {
    float phi       = 6.28318530718 * r1;
    float sin_theta = sqrt(r2);
    float cos_theta = sqrt(1.0 - r2);
    return vec3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
}

// Normal-weighted trilinear sample of the probe grid at world_pos. Used to
// approximate 2nd-bounce indirect at the primary ray's hit point.
vec3 sample_probes(vec3 world_pos, vec3 normal) {
    vec3 grid_pos = (world_pos - pc.grid_origin) / pc.grid_spacing;
    if (any(lessThan(grid_pos, vec3(0.0)))
     || any(greaterThanEqual(grid_pos, vec3(pc.grid_size) - 1.0))) {
        return vec3(0.0);
    }
    ivec3 base = ivec3(floor(grid_pos));
    vec3  frac = grid_pos - vec3(base);

    vec3  sum  = vec3(0.0);
    float wsum = 0.0;
    for (int dz = 0; dz <= 1; dz++) {
        for (int dy = 0; dy <= 1; dy++) {
            for (int dx = 0; dx <= 1; dx++) {
                ivec3 idx = clamp(base + ivec3(dx, dy, dz), ivec3(0), pc.grid_size - 1);
                float wb = (dx == 0 ? 1.0 - frac.x : frac.x)
                         * (dy == 0 ? 1.0 - frac.y : frac.y)
                         * (dz == 0 ? 1.0 - frac.z : frac.z);
                vec3 probe_world = pc.grid_origin + vec3(idx) * pc.grid_spacing;
                vec3 dir_to_probe = normalize(probe_world - world_pos + normal * 0.001);
                float nw = max(dot(normal, dir_to_probe), 0.0);
                nw = nw * nw;
                float w  = wb * nw;
                uint flat_i = uint(idx.x) + uint(idx.y) * uint(pc.grid_size.x)
                            + uint(idx.z) * uint(pc.grid_size.x * pc.grid_size.y);
                sum  += probes.data[flat_i].rgb_meta.xyz * w;
                wsum += w;
            }
        }
    }
    return (wsum > 1e-4) ? (sum / wsum) : vec3(0.0);
}

// Shadow-only probe — chit skipped via ray flags, returns 1 if miss, 0 if hit.
float shadow_probe(vec3 origin, vec3 dir, float max_t) {
    rpl.ra = vec4(-1.0);
    traceRayEXT(tlas,
        gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT | gl_RayFlagsSkipClosestHitShaderEXT,
        0xFF, 0, 0, 0, origin, 0.001, dir, max_t, 0);
    return (rpl.ra.x > -0.5) ? 1.0 : 0.0;
}

void main() {
    // Quarter-res dispatch coord → full-res G-buffer coord at the cache pixel's centre.
    ivec2 coord       = ivec2(gl_LaunchIDEXT.xy);
    ivec2 cache_size  = ivec2(gl_LaunchSizeEXT.xy);
    ivec2 full_size   = textureSize(depth_buffer, 0);
    vec2  cache_uv    = (vec2(coord) + 0.5) / vec2(cache_size);
    ivec2 gb_coord    = clamp(ivec2(cache_uv * vec2(full_size)),
                              ivec2(0), full_size - 1);

    float depth = texelFetch(depth_buffer, gb_coord, 0).r;
    // Sky pixel — clear cache entry so apply-pass fallback handles it.
    if (depth <= 0.0001) {
        imageStore(screen_cache, coord, vec4(0.0));
        return;
    }

    // Reconstruct view-space position → world-space position.
    vec2 ndc      = cache_uv * 2.0 - 1.0;
    vec4 view_pos = camera.inv_proj * vec4(ndc, depth, 1.0);
    view_pos.xyz /= view_pos.w;
    vec3 world_pos = vec3(
        camera.inv_view_r0.x * view_pos.x + camera.inv_view_r1.x * view_pos.y + camera.inv_view_r2.x * view_pos.z + camera.inv_view_r0.w,
        camera.inv_view_r0.y * view_pos.x + camera.inv_view_r1.y * view_pos.y + camera.inv_view_r2.y * view_pos.z + camera.inv_view_r1.w,
        camera.inv_view_r0.z * view_pos.x + camera.inv_view_r1.z * view_pos.y + camera.inv_view_r2.z * view_pos.z + camera.inv_view_r2.w
    );

    // View-space normal → world-space normal.
    vec3 nr_vs = texelFetch(normal_roughness_buffer, gb_coord, 0).xyz * 2.0 - 1.0;
    vec3 normal_ws = normalize(vec3(
        camera.inv_view_r0.x * nr_vs.x + camera.inv_view_r1.x * nr_vs.y + camera.inv_view_r2.x * nr_vs.z,
        camera.inv_view_r0.y * nr_vs.x + camera.inv_view_r1.y * nr_vs.y + camera.inv_view_r2.y * nr_vs.z,
        camera.inv_view_r0.z * nr_vs.x + camera.inv_view_r1.z * nr_vs.y + camera.inv_view_r2.z * nr_vs.z
    ));

    // Cosine-weighted hemisphere ray oriented around the surface normal.
    uint rng = init_rand(uint(coord.x) + uint(coord.y) * uint(cache_size.x),
                         pc.frame_idx + 1u);
    vec3 tangent   = normalize(perp(normal_ws));
    vec3 bitangent = cross(normal_ws, tangent);
    mat3 tbn = mat3(tangent, bitangent, normal_ws);

    vec3 hemi_local = cosine_hemisphere(next_rand(rng), next_rand(rng));
    vec3 ray_dir    = normalize(tbn * hemi_local);

    vec3 ray_origin = world_pos + normal_ws * 0.02;

    // Fire primary ray. Chit returns albedo + hit pos + NdotL + hit normal.
    rpl.ra = vec4(-1.0);
    traceRayEXT(tlas, gl_RayFlagsNoneEXT, 0xFF, 0, 0, 0,
                ray_origin, 0.001, ray_dir, pc.max_ray_dist, 0);

    vec3 radiance;
    if (rpl.ra.w < 0.5) {
        // Sky miss — sky colour is already exposure-scaled in the miss shader.
        radiance = rpl.ra.xyz;
    } else {
        vec3  albedo       = rpl.ra.xyz;
        vec3  hit_pos      = rpl.rb.xyz;
        float NdotL        = rpl.rb.w;
        vec3  hit_normal   = rpl.rc.xyz;
        vec3  hit_emission = rpl.rd.xyz;
        vec3  lit_origin   = hit_pos + hit_normal * 0.02;

        // Direct lighting at the hit surface.
        vec3  light_dir    = light.direction_energy.xyz;
        float light_energy = light.direction_energy.w;
        vec3  light_color  = light.color_sky_energy.rgb;
        float intensity    = light.params.x;

        float shadow_vis = 0.0;
        if (NdotL > 0.001) {
            shadow_vis = shadow_probe(lit_origin, light_dir, 1000.0) * NdotL;
        }
        vec3 direct = albedo * light_energy * light_color * shadow_vis * intensity;

        // 2nd-bounce indirect from the probe grid at the hit position.
        vec3 probe_indirect = sample_probes(hit_pos, hit_normal);
        vec3 indirect_at_hit = albedo * probe_indirect;

        radiance = direct + indirect_at_hit + hit_emission;
    }

    // Temporal stabilisation via plain EMA (no reprojection).
    if (pc.cache_ema_alpha > 0.001 && pc.cache_ema_alpha < 0.999) {
        vec3 prev = imageLoad(screen_cache, coord).rgb;
        radiance = mix(radiance, prev, pc.cache_ema_alpha);
    }

    imageStore(screen_cache, coord, vec4(radiance, 1.0));
}

#[miss]

#version 460
#extension GL_EXT_ray_tracing : enable

struct ReflPayload { vec4 ra; vec4 rb; vec4 rc; vec4 rd; };
layout(location = 0) rayPayloadInEXT ReflPayload rpl;

layout(set = 0, binding = 9) uniform sampler2D sky_tex;
layout(set = 0, binding = 11, std430) readonly buffer LightBlock {
    vec4 direction_energy;
    vec4 color_sky_energy;
    vec4 ambient_pad;
} light;

#define PI 3.14159265358979

void main() {
    vec3 dir = normalize(gl_WorldRayDirectionEXT);
    float u = atan(dir.x, -dir.z) / (2.0 * PI) + 0.5;
    float v = clamp(acos(clamp(-dir.y, -1.0, 1.0)) / PI, 0.001, 0.999);
    float sky_energy = light.color_sky_energy.w;
    rpl.ra = vec4(texture(sky_tex, vec2(u, v)).rgb * sky_energy, 0.0);
}

#[closest_hit]

#version 460
#extension GL_EXT_ray_tracing          : enable
#extension GL_EXT_nonuniform_qualifier : enable

struct ReflPayload { vec4 ra; vec4 rb; vec4 rc; vec4 rd; };
layout(location = 0) rayPayloadInEXT ReflPayload rpl;

hitAttributeEXT vec2 bary;

layout(set = 0, binding = 4, std430) readonly buffer MatInstHdrs {
    uvec2 data[];
} mat_inst;

struct SurfaceMat {
    uvec4 range;
    vec4  albedo;
    vec4  props;
    vec4  uv_offset;
    vec4  emission;
};
layout(set = 0, binding = 5, std430) readonly buffer MatSurfs {
    SurfaceMat data[];
} mat_surfs;

layout(set = 0, binding = 6, std430) readonly buffer GeomInst {
    uvec2 data[];
} geom_inst;

layout(set = 0, binding = 7, std430) readonly buffer UVBuffer {
    vec2 data[];
} uv_buf;

#define MAX_TEXTURES 128
layout(set = 0, binding = 8) uniform sampler2D albedo_textures[MAX_TEXTURES];

layout(set = 0, binding = 10, std430) readonly buffer GeomVerts {
    float data[];
} geom_verts;

layout(set = 0, binding = 11, std430) readonly buffer LightBlock {
    vec4 direction_energy;
    vec4 color_sky_energy;
    vec4 ambient_pad;
    vec4 params;
} light;

void main() {
    uint surf_off   = mat_inst.data[uint(gl_InstanceID)].x;
    uint surf_count = mat_inst.data[uint(gl_InstanceID)].y;

    SurfaceMat mat = mat_surfs.data[surf_off];
    for (uint i = 1u; i < surf_count; i++) {
        SurfaceMat candidate = mat_surfs.data[surf_off + i];
        if (candidate.range.x > uint(gl_PrimitiveID)) break;
        mat = candidate;
    }
    if (uint(gl_PrimitiveID) > mat.range.y && surf_count > 0u)
        mat = mat_surfs.data[surf_off];

    uint base_v = geom_inst.data[uint(gl_InstanceID)].x;
    uint vi0    = base_v + uint(gl_PrimitiveID) * 3u;
    float b1 = bary.x, b2 = bary.y, b0 = 1.0 - b1 - b2;
    vec2 uv = b0 * uv_buf.data[vi0]
            + b1 * uv_buf.data[vi0 + 1u]
            + b2 * uv_buf.data[vi0 + 2u];
    uv = uv * mat.props.zw + mat.uv_offset.xy;

    uint tex_idx   = mat.range.z;
    vec3 tex_color = textureLod(albedo_textures[nonuniformEXT(tex_idx)], uv, 0.0).rgb;
    vec3 albedo    = mat.albedo.rgb * tex_color;

    uint vp0 = vi0 * 3u;
    vec3 p0 = vec3(geom_verts.data[vp0],      geom_verts.data[vp0 + 1u], geom_verts.data[vp0 + 2u]);
    uint vp1 = (vi0 + 1u) * 3u;
    vec3 p1 = vec3(geom_verts.data[vp1],      geom_verts.data[vp1 + 1u], geom_verts.data[vp1 + 2u]);
    uint vp2 = (vi0 + 2u) * 3u;
    vec3 p2 = vec3(geom_verts.data[vp2],      geom_verts.data[vp2 + 1u], geom_verts.data[vp2 + 2u]);

    vec3 obj_normal = normalize(cross(p1 - p0, p2 - p0));
    mat3 m = mat3(gl_ObjectToWorldEXT);
    vec3 normal_ws = normalize(transpose(inverse(m)) * obj_normal);
    if (dot(gl_WorldRayDirectionEXT, normal_ws) > 0.0) normal_ws = -normal_ws;

    float NdotL = max(dot(normal_ws, light.direction_energy.xyz), 0.0);
    vec3  hit_ws = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;

    rpl.ra = vec4(albedo,    1.0 + mat.props.x);
    rpl.rb = vec4(hit_ws,    NdotL);
    rpl.rc = vec4(normal_ws, mat.props.y);
    rpl.rd = vec4(mat.emission.rgb, 0.0);
}
