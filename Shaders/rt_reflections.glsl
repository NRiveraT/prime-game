#[raygen]

#version 460
#extension GL_EXT_ray_tracing : enable

// ── Payload ───────────────────────────────────────────────────────────────────
// ra: xyz = albedo (geom hit) | sky color (miss) | probe sentinel (-1)
//     w   = 1.0 for geometry hit, 0.0 for sky miss
// rb: xyz = world hit position,  w = NdotL  (geometry hit only)
// rc: xyz = world hit normal,    w = unused (geometry hit only)
struct ReflPayload { vec4 ra; vec4 rb; vec4 rc; };
layout(location = 0) rayPayloadEXT ReflPayload rpl;

layout(set = 0, binding = 0)  uniform accelerationStructureEXT tlas;
layout(set = 0, binding = 1, rgba16f) uniform image2D reflection_mask;
layout(set = 0, binding = 2)  uniform sampler2D depth_buffer;
layout(set = 0, binding = 3)  uniform sampler2D normal_roughness_buffer;

// Light data (same layout as RTShadowEffect): direction.xyz + energy, color.rgb + pad.
layout(set = 0, binding = 11, std430) readonly buffer LightBlock {
    vec4 direction_energy;
    vec4 color_pad;
} light;

layout(push_constant, std430) uniform PC {
    mat4  inv_proj;           // floats  0-15
    vec4  inv_view_r0;        // floats 16-19  basis.x + origin.x
    vec4  inv_view_r1;        // floats 20-23  basis.y + origin.y
    vec4  inv_view_r2;        // floats 24-27  basis.z + origin.z
    vec2  screen_size;        // floats 28-29
    float reflection_max_dist;// float  30
    uint  frame_samples;      // uint   31 — high 16 = sample_count, low 16 = frame_index
} pc;

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

// ── Tangent frame ─────────────────────────────────────────────────────────────
vec3 perp(vec3 v) {
    vec3 a = abs(v);
    if (a.x <= a.y && a.x <= a.z) return vec3(0.0, -v.z,  v.y);
    if (a.y <= a.z)                return vec3(-v.z, 0.0,  v.x);
    return vec3(-v.y, v.x, 0.0);
}

mat3 tbn_from_normal(vec3 n) {
    vec3 t = normalize(perp(n));
    return mat3(t, cross(n, t), n);
}

// GGX microfacet distribution sample (tangent space, +Z hemisphere).
vec3 ggx_sample(float r1, float r2, float alpha) {
    float phi       = 6.28318530718 * r1;
    float cos_theta = sqrt((1.0 - r2) / max(1.0 + (alpha * alpha - 1.0) * r2, 1e-6));
    float sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
    return vec3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
}

// ── Secondary-ray probe ───────────────────────────────────────────────────────
// Traces an occlusion-only ray. Returns 1.0 if unoccluded (ray missed all geometry),
// 0.0 if occluded (hit geometry, chit skipped). The miss shader overwrites rpl.ra.x
// with sky color (>= 0); a geometry hit leaves it at the -1 sentinel.
float probe(vec3 origin, vec3 dir, float max_t) {
    rpl.ra = vec4(-1.0);
    traceRayEXT(tlas,
        gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT | gl_RayFlagsSkipClosestHitShaderEXT,
        0xFF, 0, 0, 0, origin, 0.001, dir, max_t, 0);
    return (rpl.ra.x > -0.5) ? 1.0 : 0.0;
}

void main() {
    ivec2 coord     = ivec2(gl_LaunchIDEXT.xy);
    vec2  screen_uv = (vec2(coord) + 0.5) / pc.screen_size;
    vec2  ndc_uv    = screen_uv * 2.0 - 1.0;

    float depth = texture(depth_buffer, screen_uv).r;
    if (depth <= 0.0001) {
        imageStore(reflection_mask, coord, vec4(0.0));
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

    // Normal + roughness from G-buffer
    vec4  nr_sample = texture(normal_roughness_buffer, screen_uv);
    vec3  normal_vs = nr_sample.xyz * 2.0 - 1.0;
    vec3  normal    = normalize(vec3(
        pc.inv_view_r0.x * normal_vs.x + pc.inv_view_r1.x * normal_vs.y + pc.inv_view_r2.x * normal_vs.z,
        pc.inv_view_r0.y * normal_vs.x + pc.inv_view_r1.y * normal_vs.y + pc.inv_view_r2.y * normal_vs.z,
        pc.inv_view_r0.z * normal_vs.x + pc.inv_view_r1.z * normal_vs.y + pc.inv_view_r2.z * normal_vs.z
    ));
    float roughness = nr_sample.w;

    if (roughness >= 0.75) {
        imageStore(reflection_mask, coord, vec4(0.0));
        return;
    }

    vec3 cam_origin = vec3(pc.inv_view_r0.w, pc.inv_view_r1.w, pc.inv_view_r2.w);
    vec3 view_dir   = normalize(world_pos - cam_origin);
    vec3 refl_dir   = reflect(view_dir, normal);
    vec3 origin     = world_pos + normal * 0.02;

    float alpha = roughness * roughness;

    mat3 tbn       = tbn_from_normal(refl_dir);
    uint n_samples = max((pc.frame_samples >> 16u) & 0xFFFFu, 1u);
    uint frame_idx = pc.frame_samples & 0xFFFFu;
    uint rng       = init_rand(uint(coord.x) + uint(coord.y) * uint(pc.screen_size.x), frame_idx);

    vec3  light_dir    = light.direction_energy.xyz;
    float light_energy = light.direction_energy.w;
    vec3  light_color  = light.color_pad.rgb;

    // 4-ray tetrahedral AO hemisphere (tangent space, Y = along normal).
    const vec3 AO_TS[4] = vec3[4](
        vec3( 0.0,    1.0,   0.0  ),
        vec3( 0.866,  0.5,   0.0  ),
        vec3(-0.433,  0.5,   0.75 ),
        vec3(-0.433,  0.5,  -0.75 )
    );

    vec3 accum = vec3(0.0);
    for (uint i = 0u; i < n_samples; i++) {
        vec3 ray_dir;
        if (alpha < 0.001) {
            ray_dir = refl_dir;
        } else {
            ray_dir = normalize(tbn * ggx_sample(next_rand(rng), next_rand(rng), alpha));
            if (dot(ray_dir, normal) <= 0.0) ray_dir = refl_dir;
        }

        // Primary reflection ray — chit writes geometry data into payload.
        traceRayEXT(tlas, gl_RayFlagsNoneEXT,
                    0xFF, 0, 0, 0, origin, 0.001, ray_dir, pc.reflection_max_dist, 0);

        vec3 sample_color;
        if (rpl.ra.w < 0.5) {
            // Sky miss — already lit by definition.
            sample_color = rpl.ra.xyz;
        } else {
            // Geometry hit — apply direct + ambient lighting at the reflected surface.
            vec3  albedo     = rpl.ra.xyz;
            vec3  hit_pos    = rpl.rb.xyz;
            float NdotL      = rpl.rb.w;
            vec3  hit_normal = rpl.rc.xyz;
            vec3  lit_origin = hit_pos + hit_normal * 0.02;

            // Shadow ray toward the directional light.
            float shadow_vis = 0.0;
            if (NdotL > 0.001) {
                shadow_vis = probe(lit_origin, light_dir, 1000.0) * NdotL;
            }

            // 4-ray AO sampling at the reflected hit point.
            vec3  ht = normalize(perp(hit_normal));
            vec3  hb = cross(hit_normal, ht);
            float ao_vis = 0.0;
            for (int ai = 0; ai < 4; ai++) {
                vec3 ao_ws = normalize(ht * AO_TS[ai].x + hit_normal * AO_TS[ai].y + hb * AO_TS[ai].z);
                ao_vis += probe(lit_origin, ao_ws, 1.0);
            }
            ao_vis *= 0.25;

            sample_color = albedo * (light_energy * light_color * shadow_vis
                                     + 0.05 * ao_vis);
        }

        accum += sample_color;
    }

    vec3 result = accum / float(n_samples);

    float spec_vis = (1.0 - alpha) * (1.0 - alpha);
    imageStore(reflection_mask, coord, vec4(result, 1.0 + spec_vis));
}

#[miss]

#version 460
#extension GL_EXT_ray_tracing : enable

struct ReflPayload { vec4 ra; vec4 rb; vec4 rc; };
layout(location = 0) rayPayloadInEXT ReflPayload rpl;

layout(set = 0, binding = 9) uniform sampler2D sky_tex;

#define PI 3.14159265358979

void main() {
    vec3 dir = normalize(gl_WorldRayDirectionEXT);
    float u = atan(dir.x, -dir.z) / (2.0 * PI) + 0.5;
    float v = acos(clamp(-dir.y, -1.0, 1.0)) / PI;
    // ra.w = 0.0 marks a sky miss. ra.xyz = sky color (always >= 0, never the -1 sentinel).
    rpl.ra = vec4(texture(sky_tex, vec2(u, v)).rgb, 0.0);
}

#[closest_hit]

#version 460
#extension GL_EXT_ray_tracing          : enable
#extension GL_EXT_nonuniform_qualifier : enable

struct ReflPayload { vec4 ra; vec4 rb; vec4 rc; };
layout(location = 0) rayPayloadInEXT ReflPayload rpl;

hitAttributeEXT vec2 bary;

// ── Bindings ──────────────────────────────────────────────────────────────────

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

// Vertex positions: flat xyz float triples, same ordering as uv_buf.
layout(set = 0, binding = 10, std430) readonly buffer GeomVerts {
    float data[];
} geom_verts;

// Light data: direction.xyz + energy, color.rgb + pad.
layout(set = 0, binding = 11, std430) readonly buffer LightBlock {
    vec4 direction_energy;
    vec4 color_pad;
} light;

// ── Main ──────────────────────────────────────────────────────────────────────

void main() {
    // ── Surface material ──────────────────────────────────────────────────
    uint surf_off   = mat_inst.data[uint(gl_InstanceID)].x;
    uint surf_count = mat_inst.data[uint(gl_InstanceID)].y;

    SurfaceMat mat = mat_surfs.data[surf_off];
    for (uint i = 1u; i < surf_count; i++) {
        if (mat_surfs.data[surf_off + i].range.x > uint(gl_PrimitiveID)) break;
        mat = mat_surfs.data[surf_off + i];
    }

    // ── UV interpolation → albedo sample ─────────────────────────────────
    uint base_v = geom_inst.data[uint(gl_InstanceID)].x;
    uint vi0    = base_v + uint(gl_PrimitiveID) * 3u;
    float b1 = bary.x, b2 = bary.y, b0 = 1.0 - b1 - b2;
    vec2 uv = b0 * uv_buf.data[vi0]
            + b1 * uv_buf.data[vi0 + 1u]
            + b2 * uv_buf.data[vi0 + 2u];

    uint tex_idx  = mat.range.z;
    vec3 tex_color = texture(albedo_textures[nonuniformEXT(tex_idx)], uv).rgb;
    vec3 albedo    = mat.albedo.rgb * tex_color;

    // ── Geometric normal (object space → world space) ─────────────────────
    // Vertex positions are flat xyz floats: vertex N is at data[N*3..N*3+2].
    uint vp0 = vi0 * 3u;
    vec3 p0 = vec3(geom_verts.data[vp0],      geom_verts.data[vp0 + 1u], geom_verts.data[vp0 + 2u]);
    uint vp1 = (vi0 + 1u) * 3u;
    vec3 p1 = vec3(geom_verts.data[vp1],      geom_verts.data[vp1 + 1u], geom_verts.data[vp1 + 2u]);
    uint vp2 = (vi0 + 2u) * 3u;
    vec3 p2 = vec3(geom_verts.data[vp2],      geom_verts.data[vp2 + 1u], geom_verts.data[vp2 + 2u]);

    vec3 obj_normal = normalize(cross(p1 - p0, p2 - p0));
    // mat3(gl_ObjectToWorldEXT) is the 3×3 rotation+scale part of the instance transform.
    // For uniform scaling this correctly transforms normals; non-uniform scale is rare in game meshes.
    vec3 normal_ws = normalize(mat3(gl_ObjectToWorldEXT) * obj_normal);

    // ── NdotL ─────────────────────────────────────────────────────────────
    float NdotL = max(dot(normal_ws, light.direction_energy.xyz), 0.0);

    // ── Hit world position ─────────────────────────────────────────────────
    vec3 hit_ws = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;

    // ── Write payload — raygen handles shadow/AO, then combines ───────────
    rpl.ra = vec4(albedo,    1.0);   // w=1 = geometry hit
    rpl.rb = vec4(hit_ws,    NdotL);
    rpl.rc = vec4(normal_ws, 0.0);
}
