#[raygen]

#version 460
#extension GL_EXT_ray_tracing : enable

// Payload reused from rt_reflections — same chit/miss pattern.
//   ra: xyz = albedo (geom hit) | sky colour (miss) | probe sentinel (-1)
//       w   = 1 + metallic (hit, ≥1) | 0 (sky miss) | -1 (shadow probe)
//   rb: xyz = world hit position, w = NdotL
//   rc: xyz = world hit normal, w = roughness
struct ReflPayload { vec4 ra; vec4 rb; vec4 rc; vec4 rd; };
layout(location = 0) rayPayloadEXT ReflPayload rpl;

// Probe entry: vec3 radiance + float meta (age/confidence placeholder).
// Padded to 16 bytes by std430.
struct Probe {
    vec4 rgb_meta;
};

layout(set = 0, binding = 0) uniform accelerationStructureEXT tlas;
layout(set = 0, binding = 1, std430) buffer ProbesBuf {
    Probe data[];
} probes;

layout(set = 0, binding = 11, std430) readonly buffer LightBlock {
    vec4 direction_energy; // xyz = toward light, w = exposure-corrected energy
    vec4 color_sky_energy; // xyz = colour, w = sky energy (already × exposure_norm)
    vec4 ambient_pad;      // xyz = ambient * energy * exposure_norm
    vec4 params;           // x = indirect_intensity (direct-bounce boost)
} light;

layout(push_constant, std430) uniform PC {
    vec3  grid_origin;
    float grid_spacing;
    ivec3 grid_size;
    uint  rotation_offset;  // which slice of probes this frame (0..rotation_slices-1)
    uint  rotation_slices;  // how many frames to spread a full grid update across
    float ema_alpha;
    float max_ray_dist;
    uint  frame_idx;        // rotates the random-sample set each frame
} pc;

// Per-probe ray count. With temporal EMA integration over rotation cycles,
// effective sample count is NUM_RAYS × frames_since_start, so 8 rays × 25 frames
// of convergence ≈ 200 samples averaged — plenty for smooth indirect.
#define NUM_RAYS 8

// ── Simple hash-based RNG (Wyman DXR Tutorial 5 pattern) ──────────────────────
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

// Uniform sampling over the full sphere — probes care about radiance from all
// directions, not just one hemisphere like a surface would.
vec3 sample_sphere(inout uint rng) {
    float u1 = next_rand(rng);
    float u2 = next_rand(rng);
    float phi       = 6.28318530718 * u1;
    float cos_theta = 2.0 * u2 - 1.0;
    float sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
    return vec3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
}

// Shadow probe — same helper pattern as rt_reflections.
float shadow_probe(vec3 origin, vec3 dir, float max_t) {
    rpl.ra = vec4(-1.0);
    traceRayEXT(tlas,
        gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT | gl_RayFlagsSkipClosestHitShaderEXT,
        0xFF, 0, 0, 0, origin, 0.001, dir, max_t, 0);
    return (rpl.ra.x > -0.5) ? 1.0 : 0.0;
}

void main() {
    // Slice-based rotation: each frame only a fraction of probes updates.
    uint local_idx   = gl_LaunchIDEXT.x;
    uint total       = uint(pc.grid_size.x * pc.grid_size.y * pc.grid_size.z);
    uint slice_size  = (total + pc.rotation_slices - 1u) / pc.rotation_slices;
    uint global_idx  = pc.rotation_offset * slice_size + local_idx;
    if (global_idx >= total) return;

    // Unflatten → 3D grid coord.
    uint gx = uint(pc.grid_size.x);
    uint gy = uint(pc.grid_size.y);
    ivec3 gc;
    gc.x = int(global_idx % gx);
    gc.y = int((global_idx / gx) % gy);
    gc.z = int(global_idx / (gx * gy));

    vec3 world_pos = pc.grid_origin + vec3(gc) * pc.grid_spacing;

    vec3  light_dir       = light.direction_energy.xyz;
    float light_energy    = light.direction_energy.w;
    vec3  light_color     = light.color_sky_energy.rgb;
    vec3  ambient         = light.ambient_pad.xyz;
    // Converts Lambertian reflected radiance at the hit point into an
    // irradiance-like magnitude that a downstream surface actually receives.
    // Applied only to DIRECT bounces so coloured light bleed stands out
    // relative to the neutral ambient floor.
    float direct_boost    = light.params.x;

    // Close-hit distance threshold. Rays that hit geometry closer than this
    // probably originated inside solid geometry — those samples would return
    // backface noise. Ignore them and, if too many rays are "close hits",
    // abandon the whole probe update (it's embedded, previous value is less
    // wrong than fresh garbage).
    const float CLOSE_HIT = 0.25;
    const int   EMBEDDED_THRESHOLD = NUM_RAYS * 2 / 3;  // 2/3 embedded → skip

    // Seed RNG from probe index + frame so sample directions differ between
    // probes (kills axis-alignment) and between frames (temporal integration
    // via the EMA delivers a bias-free average over many frames).
    uint rng = init_rand(global_idx + 1u, pc.frame_idx + 1u);

    vec3 accum = vec3(0.0);
    int  valid_rays = 0;
    int  close_hits = 0;

    for (int i = 0; i < NUM_RAYS; i++) {
        vec3 ray_dir = sample_sphere(rng);

        rpl.ra = vec4(-1.0);
        traceRayEXT(tlas, gl_RayFlagsNoneEXT,
                    0xFF, 0, 0, 0, world_pos, 0.01, ray_dir, pc.max_ray_dist, 0);

        if (rpl.ra.w < 0.5) {
            // Sky miss — safe, use full sky radiance.
            accum += rpl.ra.xyz;
            valid_rays++;
        } else {
            vec3  hit_pos  = rpl.rb.xyz;
            float hit_dist = distance(world_pos, hit_pos);
            if (hit_dist < CLOSE_HIT) {
                // Embedded-likely hit; skip this ray entirely.
                close_hits++;
                continue;
            }
            vec3  albedo       = rpl.ra.xyz;
            float NdotL        = rpl.rb.w;
            vec3  hit_normal   = rpl.rc.xyz;
            vec3  hit_emission = rpl.rd.xyz;
            vec3  lit_origin   = hit_pos + hit_normal * 0.02;

            float shadow_vis = 0.0;
            if (NdotL > 0.001) {
                shadow_vis = shadow_probe(lit_origin, light_dir, 1000.0) * NdotL;
            }
            // Direct-bounce radiance boosted by irradiance-conversion factor.
            // Ambient stays at physical scale — boosting it would dilute
            // colour bleed by raising the neutral floor. Emission adds raw
            // radiance from the hit surface (already light-independent).
            vec3 direct      = albedo * light_energy * light_color * shadow_vis * direct_boost;
            vec3 hit_ambient = albedo * ambient;
            accum += direct + hit_ambient + hit_emission;
            valid_rays++;
        }
    }

    // If most of the probe's rays were embedded hits, don't update — the
    // previous (or zero) value is less wrong than averaging a couple of
    // valid rays with a pile of rejected backface garbage.
    if (close_hits >= EMBEDDED_THRESHOLD || valid_rays == 0) {
        return;
    }

    accum /= float(valid_rays);

    // EMA blend with previous probe value.
    vec3 prev = probes.data[global_idx].rgb_meta.xyz;
    vec3 blended = mix(prev, accum, pc.ema_alpha);
    probes.data[global_idx].rgb_meta = vec4(blended, 0.0);
}

#[miss]

#version 460
#extension GL_EXT_ray_tracing : enable

struct ReflPayload { vec4 ra; vec4 rb; vec4 rc; vec4 rd; };
layout(location = 0) rayPayloadInEXT ReflPayload rpl;

layout(set = 0, binding = 9) uniform sampler2D sky_tex;
layout(set = 0, binding = 11, std430) readonly buffer LightBlock {
    vec4 direction_energy;
    vec4 color_sky_energy; // w = sky energy × exposure_norm
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
