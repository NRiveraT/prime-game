#[raygen]

#version 460
#extension GL_EXT_ray_tracing : enable

// ── Payload ─────────────────────────────────────────────────────────────────
// ra: xyz = albedo (geom) | sky color (miss) | probe sentinel (-1)
//     w   = 1.0 geometry hit / 0.0 sky miss
// rb: xyz = hit world position,  w = NdotL  (geometry hit only)
// rc: xyz = world hit normal,    w = unused
struct DDGIPayload { vec4 ra; vec4 rb; vec4 rc; };
layout(location = 0) rayPayloadEXT DDGIPayload rpl;

// ── Grid constants ───────────────────────────────────────────────────────────
#define GRID_X         8
#define GRID_Y         4
#define GRID_Z         8
#define RAYS_PER_PROBE 64
#define TOTAL_PROBES   (GRID_X * GRID_Y * GRID_Z)

// ── Bindings ─────────────────────────────────────────────────────────────────
layout(set = 0, binding = 0) uniform accelerationStructureEXT tlas;

// Ray buffer: index = probe_idx * RAYS_PER_PROBE + ray_idx, two vec4s per ray:
//   [2i+0]: radiance.xyz + hit_distance
//   [2i+1]: ray_direction.xyz + pad
layout(set = 0, binding = 1, std430) writeonly buffer RayBuffer {
    vec4 data[];
} ray_buf;

// Light + environment (48 bytes — same layout as RTReflectionEffect)
layout(set = 0, binding = 3, std430) readonly buffer LightBlock {
    vec4 direction_energy;   // xyz = dir to light, w = light_energy
    vec4 color_sky_energy;   // xyz = light_color,  w = background_energy_multiplier
    vec4 ambient_pad;        // xyz = ambient_light_color * ambient_light_energy
} light;

layout(push_constant, std430) uniform PC {
    vec3  grid_origin;    // world position of probe (0,0,0)
    float probe_spacing;  // metres between adjacent probes
    float frame_index;    // cast to uint for Fibonacci rotation
    float max_ray_dist;   // max probe ray distance
    vec2  _pad;
} pc; // 32 bytes

// ── Fibonacci sphere direction ────────────────────────────────────────────────
// Evenly distributes RAYS_PER_PROBE directions over the unit sphere.
// rotation varies per frame so samples shift across temporal frames.
const float GOLDEN_ANGLE = 2.399963;

vec3 fibonacci_dir(uint idx, float rotation) {
    float phi       = GOLDEN_ANGLE * float(idx) + rotation;
    float cos_theta = 1.0 - (2.0 * float(idx) + 1.0) / float(RAYS_PER_PROBE);
    float sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
    return normalize(vec3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta));
}

// ── Occlusion probe ───────────────────────────────────────────────────────────
// Returns 1.0 if unoccluded (miss), 0.0 if occluded. Reuses rpl sentinel trick.
float occlusion_probe(vec3 origin, vec3 dir, float max_t) {
    rpl.ra = vec4(-1.0);
    traceRayEXT(tlas,
        gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT | gl_RayFlagsSkipClosestHitShaderEXT,
        0xFF, 0, 0, 0, origin, 0.001, dir, max_t, 0);
    return (rpl.ra.x > -0.5) ? 1.0 : 0.0;
}

void main() {
    uint ray_idx   = gl_LaunchIDEXT.x;
    uint probe_idx = gl_LaunchIDEXT.y;
    if (probe_idx >= uint(TOTAL_PROBES) || ray_idx >= uint(RAYS_PER_PROBE)) return;

    // Probe world position from grid index
    uint px = probe_idx % uint(GRID_X);
    uint pz = (probe_idx / uint(GRID_X)) % uint(GRID_Z);
    uint py = probe_idx / uint(GRID_X * GRID_Z);
    vec3 probe_pos = pc.grid_origin + vec3(float(px), float(py), float(pz)) * pc.probe_spacing;

    // Fibonacci sphere ray direction with per-frame rotation
    float rotation = pc.frame_index * GOLDEN_ANGLE;
    vec3  ray_dir  = fibonacci_dir(ray_idx, rotation);

    // Trace primary ray
    traceRayEXT(tlas, gl_RayFlagsNoneEXT,
                0xFF, 0, 0, 0, probe_pos, 0.01, ray_dir, pc.max_ray_dist, 0);

    vec3  radiance;
    float hit_dist;

    if (rpl.ra.w < 0.5) {
        // Sky miss — radiance = sky colour * sky_energy (already done in miss shader)
        radiance = rpl.ra.xyz;
        hit_dist = pc.max_ray_dist;
    } else {
        // Geometry hit — compute single-bounce direct + ambient
        vec3  albedo     = rpl.ra.xyz;
        vec3  hit_pos    = rpl.rb.xyz;
        float NdotL      = rpl.rb.w;
        vec3  hit_normal = rpl.rc.xyz;
        vec3  lit_origin = hit_pos + hit_normal * 0.02;

        vec3  light_dir    = light.direction_energy.xyz;
        float light_energy = light.direction_energy.w;
        vec3  light_color  = light.color_sky_energy.rgb;
        vec3  godot_ambient = light.ambient_pad.xyz;

        float shadow_vis = 0.0;
        if (NdotL > 0.001) {
            shadow_vis = occlusion_probe(lit_origin, light_dir, 1e4) * NdotL;
        }
        radiance = albedo * (light_energy * light_color * shadow_vis + godot_ambient);
        hit_dist = length(hit_pos - probe_pos);
    }

    // Write ray result to flat buffer (2 vec4s per ray)
    uint buf_idx = probe_idx * uint(RAYS_PER_PROBE) + ray_idx;
    ray_buf.data[buf_idx * 2u + 0u] = vec4(radiance, hit_dist);
    ray_buf.data[buf_idx * 2u + 1u] = vec4(ray_dir,  0.0);
}

#[miss]

#version 460
#extension GL_EXT_ray_tracing : enable

struct DDGIPayload { vec4 ra; vec4 rb; vec4 rc; };
layout(location = 0) rayPayloadInEXT DDGIPayload rpl;

layout(set = 0, binding = 2) uniform sampler2D sky_tex;

layout(set = 0, binding = 3, std430) readonly buffer LightBlock {
    vec4 direction_energy;
    vec4 color_sky_energy;  // w = background_energy_multiplier
    vec4 ambient_pad;
} light;

#define PI 3.14159265358979

void main() {
    vec3  dir = normalize(gl_WorldRayDirectionEXT);
    float u   = atan(dir.x, -dir.z) / (2.0 * PI) + 0.5;
    float v   = clamp(acos(clamp(-dir.y, -1.0, 1.0)) / PI, 0.001, 0.999);
    // ra.w = 0.0 marks sky (never the -1 sentinel), scaled by sky energy.
    rpl.ra = vec4(texture(sky_tex, vec2(u, v)).rgb * light.color_sky_energy.w, 0.0);
}

#[closest_hit]

#version 460
#extension GL_EXT_ray_tracing          : enable
#extension GL_EXT_nonuniform_qualifier : enable

struct DDGIPayload { vec4 ra; vec4 rb; vec4 rc; };
layout(location = 0) rayPayloadInEXT DDGIPayload rpl;

hitAttributeEXT vec2 bary;

layout(set = 0, binding = 3, std430) readonly buffer LightBlock {
    vec4 direction_energy;
    vec4 color_sky_energy;
    vec4 ambient_pad;
} light;

layout(set = 0, binding = 4, std430) readonly buffer MatInstHdrs {
    uvec2 data[];
} mat_inst;

struct SurfaceMat {
    uvec4 range;   // x=start_prim, y=end_prim, z=tex_idx, w=pad
    vec4  albedo;
    vec4  props;   // x=metallic, y=roughness
};
layout(set = 0, binding = 5, std430) readonly buffer MatSurfs {
    SurfaceMat data[];
} mat_surfs;

layout(set = 0, binding = 6, std430) readonly buffer GeomInst {
    uvec2 data[];  // x=base_vertex, y=vertex_count
} geom_inst;

layout(set = 0, binding = 7, std430) readonly buffer UVBuffer {
    vec2 data[];
} uv_buf;

#define MAX_TEXTURES 128
layout(set = 0, binding = 8) uniform sampler2D albedo_textures[MAX_TEXTURES];

layout(set = 0, binding = 9, std430) readonly buffer GeomVerts {
    float data[];
} geom_verts;

void main() {
    // ── Surface material ──────────────────────────────────────────────────
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

    // ── UV → albedo ──────────────────────────────────────────────────────
    uint base_v = geom_inst.data[uint(gl_InstanceID)].x;
    uint vi0    = base_v + uint(gl_PrimitiveID) * 3u;
    float b1 = bary.x, b2 = bary.y, b0 = 1.0 - b1 - b2;
    vec2 uv = b0 * uv_buf.data[vi0]
            + b1 * uv_buf.data[vi0 + 1u]
            + b2 * uv_buf.data[vi0 + 2u];

    uint tex_idx  = mat.range.z;
    vec3 albedo   = mat.albedo.rgb
                  * texture(albedo_textures[nonuniformEXT(tex_idx)], uv).rgb;

    // ── Geometric normal → world space ───────────────────────────────────
    uint vp0 = vi0 * 3u;
    vec3 p0 = vec3(geom_verts.data[vp0],        geom_verts.data[vp0 + 1u], geom_verts.data[vp0 + 2u]);
    uint vp1 = (vi0 + 1u) * 3u;
    vec3 p1 = vec3(geom_verts.data[vp1],        geom_verts.data[vp1 + 1u], geom_verts.data[vp1 + 2u]);
    uint vp2 = (vi0 + 2u) * 3u;
    vec3 p2 = vec3(geom_verts.data[vp2],        geom_verts.data[vp2 + 1u], geom_verts.data[vp2 + 2u]);

    vec3 obj_normal = normalize(cross(p1 - p0, p2 - p0));
    mat3 m = mat3(gl_ObjectToWorldEXT);
    vec3 normal_ws = normalize(transpose(inverse(m)) * obj_normal);
    if (dot(gl_WorldRayDirectionEXT, normal_ws) > 0.0) normal_ws = -normal_ws;

    float NdotL  = max(dot(normal_ws, light.direction_energy.xyz), 0.0);
    vec3  hit_ws = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;

    rpl.ra = vec4(albedo,    1.0);
    rpl.rb = vec4(hit_ws,    NdotL);
    rpl.rc = vec4(normal_ws, 0.0);
}
