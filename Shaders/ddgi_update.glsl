#[raygen]

#version 460
#extension GL_EXT_ray_tracing : enable

// ── Payload ──────────────────────────────────────────────────────────────────
// ra: xyz = albedo (geom) | sky colour (miss)   w = 1.0 geom / 0.0 sky
// rb: xyz = hit world position                  w = NdotL
// rc: xyz = world hit normal                    w = unused
struct DDGIPayload { vec4 ra; vec4 rb; vec4 rc; };
layout(location = 0) rayPayloadEXT DDGIPayload rpl;

// ── Grid constants ────────────────────────────────────────────────────────────
#define GRID_X          8
#define GRID_Y          8
#define GRID_Z          8
#define RAYS_PER_PROBE  128
#define TOTAL_PROBES    (GRID_X * GRID_Y * GRID_Z)

// ── Irradiance atlas constants (must match apply + update shaders) ────────────
#define ATLAS_COLS      (GRID_X * GRID_Z)             // 64
#define IRR_PROBE_SIDE  16
#define IRR_ATLAS_W     (ATLAS_COLS * IRR_PROBE_SIDE) // 1024
#define IRR_ATLAS_H     (GRID_Y * IRR_PROBE_SIDE)     // 128

#define PI 3.14159265358979

// ── Bindings ──────────────────────────────────────────────────────────────────
layout(set = 0, binding = 0) uniform accelerationStructureEXT tlas;

// Ray buffer: index = probe_idx * RAYS_PER_PROBE + ray_idx, two vec4s per ray:
//   [2i+0]: radiance.xyz + hit_distance
//   [2i+1]: ray_direction.xyz + pad
layout(set = 0, binding = 1, std430) writeonly buffer RayBuffer {
    vec4 data[];
} ray_buf;

// Sky texture (for miss shader — also bound here so raygen can share light block)
layout(set = 0, binding = 2) uniform sampler2D sky_tex;

layout(set = 0, binding = 3, std430) readonly buffer LightBlock {
    vec4 direction_energy;   // xyz = dir to light, w = light_energy
    vec4 color_sky_energy;   // xyz = light_color,  w = sky_energy_multiplier
    vec4 ambient_pad;        // xyz = ambient light colour × energy
} light;

layout(set = 0, binding = 4, std430) readonly buffer MatInstHdrs { uvec2 data[]; } mat_inst;

struct SurfaceMat {
    uvec4 range;
    vec4  albedo;
    vec4  props;
    vec4  uv_offset;
    vec4  emission;
};
layout(set = 0, binding = 5, std430) readonly buffer MatSurfs   { SurfaceMat data[]; } mat_surfs;
layout(set = 0, binding = 6, std430) readonly buffer GeomInst   { uvec2 data[]; } geom_inst;
layout(set = 0, binding = 7, std430) readonly buffer UVBuffer   { vec2  data[]; } uv_buf;

#define MAX_TEXTURES 128
layout(set = 0, binding = 8) uniform sampler2D albedo_textures[MAX_TEXTURES];

layout(set = 0, binding = 9,  std430) readonly buffer GeomVerts { float data[]; } geom_verts;

// C0 irradiance atlas from the PREVIOUS frame — used for multi-bounce indirect.
// Because the RT pass runs before the irr_update pass, this is last frame's atlas.
layout(set = 0, binding = 10) uniform sampler2D irr_atlas_indirect;

// ── Push constant (48 bytes) ──────────────────────────────────────────────────
// First 32 bytes: this cascade's grid (unchanged from single-cascade layout).
// Last 16 bytes: C0 irradiance grid for multi-bounce indirect lookup.
layout(push_constant, std430) uniform PC {
    vec3  grid_origin;      // this cascade's grid origin
    float probe_spacing;
    float frame_index;
    float max_ray_dist;
    vec2  _pad;             // keep 32-byte first block intact
    vec3  irr_origin;       // C0 irradiance grid origin (for indirect bounce)
    float irr_spacing;      // C0 probe spacing
} pc;

// ── Fibonacci sphere direction ─────────────────────────────────────────────────
const float GOLDEN_ANGLE = 2.399963;

vec3 fibonacci_dir(uint idx, float rotation) {
    float phi       = GOLDEN_ANGLE * float(idx) + rotation;
    float cos_theta = 1.0 - (2.0 * float(idx) + 1.0) / float(RAYS_PER_PROBE);
    float sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
    return normalize(vec3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta));
}

// ── Octahedral encode (unit vec → [0,1]²) ────────────────────────────────────
vec2 oct_encode(vec3 n) {
    float l1 = abs(n.x) + abs(n.y) + abs(n.z);
    vec2  p  = n.xy / l1;
    if (n.z < 0.0) p = (1.0 - abs(p.yx)) * sign(p);
    return p * 0.5 + 0.5;
}

#define IRR_TILE_STRIDE (IRR_PROBE_SIDE + 2)

// ── Irradiance atlas UV for a probe tile ─────────────────────────────────────
vec2 irr_atlas_uv(uint probe_idx, vec3 dir) {
    vec2 oct = clamp(oct_encode(normalize(dir)),
                     vec2(0.5 / float(IRR_PROBE_SIDE)),
                     vec2(1.0 - 0.5 / float(IRR_PROBE_SIDE)));
    uint tile_col = probe_idx % uint(ATLAS_COLS);
    uint tile_row = probe_idx / uint(ATLAS_COLS);
    vec2 texel    = vec2(float(tile_col * IRR_PROBE_SIDE), float(tile_row * IRR_PROBE_SIDE))
                  + oct * float(IRR_PROBE_SIDE);
    return texel / vec2(float(IRR_ATLAS_W), float(IRR_ATLAS_H));
}

// ── Sample C0 irradiance at a world-space hit position / normal ───────────────
// Trilinear cage interpolation over the C0 probe grid.  No Chebyshev here —
// the probes already encode indirect lighting; visibility is handled by how
// they were updated.  Used to provide a second bounce of GI in probe rays.
vec3 sample_indirect_irradiance(vec3 hit_pos, vec3 hit_normal) {
    vec3  probe_coords = (hit_pos - pc.irr_origin) / pc.irr_spacing;
    ivec3 base = clamp(ivec3(floor(probe_coords)),
                       ivec3(0),
                       ivec3(GRID_X - 2, GRID_Y - 2, GRID_Z - 2));
    vec3  alpha = clamp(probe_coords - vec3(base), vec3(0.0), vec3(1.0));

    vec3  irr_sum    = vec3(0.0);
    float weight_sum = 0.0;

    for (int i = 0; i < 8; i++) {
        ivec3 off  = ivec3(i & 1, (i >> 1) & 1, (i >> 2) & 1);
        ivec3 cage = base + off;
        uint  pid  = uint(cage.x) + uint(cage.z) * uint(GRID_X)
                   + uint(cage.y) * uint(GRID_X * GRID_Z);

        vec3  t  = mix(vec3(1.0) - alpha, alpha, vec3(off));
        float tw = t.x * t.y * t.z;

        // Smooth backface weight so back-facing probes contribute less
        vec3  probe_pos    = pc.irr_origin + vec3(cage) * pc.irr_spacing;
        vec3  dir_to_probe = normalize(probe_pos - hit_pos);
        float facing       = max(0.0001, (dot(dir_to_probe, hit_normal) + 1.0) * 0.5);
        tw *= facing * facing + 0.2;

        irr_sum    += textureLod(irr_atlas_indirect, irr_atlas_uv(pid, hit_normal), 0.0).rgb * tw;
        weight_sum += tw;
    }

    return irr_sum / max(weight_sum, 1e-4);
}

// ── Occlusion probe (shadow ray) ──────────────────────────────────────────────
// gl_RayFlagsSkipClosestHitShaderEXT prevents chit running; rpl.ra stays at
// the -1 sentinel.  Miss writes non-negative sky RGB → marks unoccluded.
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

    // Fibonacci direction with per-frame rotation
    float rotation = pc.frame_index * GOLDEN_ANGLE;
    vec3  ray_dir  = fibonacci_dir(ray_idx, rotation);

    // Primary probe ray
    traceRayEXT(tlas, gl_RayFlagsNoneEXT,
        0xFF, 0, 0, 0, probe_pos, 0.01, ray_dir, pc.max_ray_dist, 0);

    vec3  radiance;
    float hit_dist;

    if (rpl.ra.w < 0.5) {
        // ── Sky miss ─────────────────────────────────────────────────────────
        // Sky colour is already stored in ra.xyz by the miss shader.
        radiance = rpl.ra.xyz;
        hit_dist = -1.0;
    } else {
        // ── Geometry hit ──────────────────────────────────────────────────────
        vec3  albedo     = rpl.ra.xyz;
        vec3  hit_pos    = rpl.rb.xyz;
        float NdotL      = rpl.rb.w;
        vec3  hit_normal = rpl.rc.xyz;
        vec3  lit_origin = hit_pos + hit_normal * 0.02;

        vec3  light_dir    = light.direction_energy.xyz;
        float light_energy = light.direction_energy.w;
        vec3  light_color  = light.color_sky_energy.rgb;

        // Direct lighting
        float shadow_vis = 0.0;
        if (NdotL > 0.001) {
            shadow_vis = occlusion_probe(lit_origin, light_dir, pc.max_ray_dist) * NdotL;
        }
        vec3 direct = light_energy * light_color * shadow_vis;

        // Indirect bounce from C0 irradiance atlas (previous frame).
        // ENERGY_CONSERVATION factor prevents infinite energy accumulation.
        const float ENERGY_CONSERVATION = 0.95;
        vec3 indirect = sample_indirect_irradiance(hit_pos, hit_normal);

        radiance = albedo * (direct + ENERGY_CONSERVATION * indirect);
        hit_dist = length(hit_pos - probe_pos);
    }

    uint buf_idx = probe_idx * uint(RAYS_PER_PROBE) + ray_idx;
    ray_buf.data[buf_idx * 2u + 0u] = vec4(radiance, hit_dist);
    ray_buf.data[buf_idx * 2u + 1u] = vec4(ray_dir, 0.0);
}

#[miss]

#version 460
#extension GL_EXT_ray_tracing : enable

struct DDGIPayload { vec4 ra; vec4 rb; vec4 rc; };
layout(location = 0) rayPayloadInEXT DDGIPayload rpl;

layout(set = 0, binding = 2) uniform sampler2D sky_tex;

layout(set = 0, binding = 3, std430) readonly buffer LightBlock {
    vec4 direction_energy;
    vec4 color_sky_energy;   // w = sky_energy_multiplier
    vec4 ambient_pad;
} light;

#define PI 3.14159265358979

void main() {
    vec3  dir = normalize(gl_WorldRayDirectionEXT);
    float u   = atan(dir.x, -dir.z) / (2.0 * PI) + 0.5;
    float v   = clamp(acos(clamp(-dir.y, -1.0, 1.0)) / PI, 0.001, 0.999);
    // Multiply by sky_energy_multiplier so probes correctly capture sky brightness.
    // ra.w = 0.0 marks sky miss (distinct from the -1.0 occlusion probe sentinel).
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

layout(set = 0, binding = 4, std430) readonly buffer MatInstHdrs { uvec2 data[]; } mat_inst;

struct SurfaceMat {
    uvec4 range;       // x=start_prim, y=end_prim, z=tex_idx, w=pad
    vec4  albedo;
    vec4  props;       // x=metallic, y=roughness, z=uv_scale_x, w=uv_scale_y
    vec4  uv_offset;   // x=uv_offset_x, y=uv_offset_y, zw=pad
};
layout(set = 0, binding = 5, std430) readonly buffer MatSurfs { SurfaceMat data[]; } mat_surfs;
layout(set = 0, binding = 6, std430) readonly buffer GeomInst { uvec2 data[]; } geom_inst;
layout(set = 0, binding = 7, std430) readonly buffer UVBuffer { vec2  data[]; } uv_buf;

#define MAX_TEXTURES 128
layout(set = 0, binding = 8) uniform sampler2D albedo_textures[MAX_TEXTURES];

layout(set = 0, binding = 9, std430) readonly buffer GeomVerts { float data[]; } geom_verts;

void main() {
    // ── Surface material ──────────────────────────────────────────────────────
    uint surf_off   = mat_inst.data[uint(gl_InstanceID)].x;
    uint surf_count = mat_inst.data[uint(gl_InstanceID)].y;

    SurfaceMat mat = mat_surfs.data[surf_off];
    for (uint i = 1u; i < surf_count; i++) {
        SurfaceMat candidate = mat_surfs.data[surf_off + i];
        if (uint(gl_PrimitiveID) >= candidate.range.x && uint(gl_PrimitiveID) <= candidate.range.y) {
            mat = candidate;
            break;
        }
    }
    if (uint(gl_PrimitiveID) > mat.range.y && surf_count > 0u)
        mat = mat_surfs.data[surf_off];

    // ── UV → albedo ───────────────────────────────────────────────────────────
    uint base_v = geom_inst.data[uint(gl_InstanceID)].x;
    uint vi0    = base_v + uint(gl_PrimitiveID) * 3u;
    float b1 = bary.x, b2 = bary.y, b0 = 1.0 - b1 - b2;
    vec2 uv = b0 * uv_buf.data[vi0]
            + b1 * uv_buf.data[vi0 + 1u]
            + b2 * uv_buf.data[vi0 + 2u];

    uv = uv * mat.props.zw + mat.uv_offset.xy;

    uint tex_idx = mat.range.z;
    vec3 albedo  = mat.albedo.rgb
                 * textureLod(albedo_textures[nonuniformEXT(tex_idx)], uv, 0.0).rgb;

    // ── Geometric normal → world space ────────────────────────────────────────
    uint vp0 = vi0 * 3u;
    vec3 p0 = vec3(geom_verts.data[vp0], geom_verts.data[vp0 + 1u], geom_verts.data[vp0 + 2u]);
    uint vp1 = (vi0 + 1u) * 3u;
    vec3 p1 = vec3(geom_verts.data[vp1], geom_verts.data[vp1 + 1u], geom_verts.data[vp1 + 2u]);
    uint vp2 = (vi0 + 2u) * 3u;
    vec3 p2 = vec3(geom_verts.data[vp2], geom_verts.data[vp2 + 1u], geom_verts.data[vp2 + 2u]);

    vec3 obj_normal = normalize(cross(p1 - p0, p2 - p0));
    mat3 m          = mat3(gl_ObjectToWorldEXT);
    vec3 normal_ws  = normalize(transpose(inverse(m)) * obj_normal);
    if (dot(gl_WorldRayDirectionEXT, normal_ws) > 0.0) normal_ws = -normal_ws;

    float NdotL = max(dot(normal_ws, light.direction_energy.xyz), 0.0);
    vec3  hit_ws = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;

    rpl.ra = vec4(albedo, 1.0);
    rpl.rb = vec4(hit_ws, NdotL);
    rpl.rc = vec4(normal_ws, 0.0);
}
