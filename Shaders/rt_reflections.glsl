#[raygen]

#version 460
#extension GL_EXT_ray_tracing : enable

// ── Payload ───────────────────────────────────────────────────────────────────
// ra: xyz = albedo (geom hit) | sky color (miss) | probe sentinel (-1)
//     w   = 1.0 for geometry hit, 0.0 for sky miss
// rb: xyz = world hit position,  w = NdotL  (geometry hit only)
// rc: xyz = world hit normal,    w = unused (geometry hit only)
struct ReflPayload { vec4 ra; vec4 rb; vec4 rc; vec4 rd; };
layout(location = 0) rayPayloadEXT ReflPayload rpl;

layout(set = 0, binding = 0)  uniform accelerationStructureEXT tlas;
layout(set = 0, binding = 1, rgba16f) uniform image2D reflection_mask;
layout(set = 0, binding = 2)  uniform sampler2D depth_buffer;
layout(set = 0, binding = 3)  uniform sampler2D normal_roughness_buffer;
// Light + environment data SSBO (64 bytes):
//   direction_energy:  xyz = direction to light,  w = 1.0 (normalized)
//   color_sky_energy:  xyz = light_color,          w = background_energy_multiplier
//   ambient_pad:       xyz = ambient_light_color * ambient_light_energy,
//                      w   = dir_light_sin_half_angle (sin of directional light angular radius)
//   params:            x = unused, y = tonemap_exposure, z = saturation, w = contrast
layout(set = 0, binding = 11, std430) readonly buffer LightBlock {
    vec4 direction_energy;
    vec4 color_sky_energy;
    vec4 ambient_pad;
    vec4 params;
} light;

// Local lights (OmniLight3D / SpotLight3D) — up to 32 entries.
struct RTLight {
    vec4 pos_range;    // xyz = world position, w = range
    vec4 color_energy; // xyz = color, w = energy
    vec4 dir_size;     // xyz = spot forward (-basis.z), w = light_size (disk radius)
    vec4 cone_type;    // x = cos_outer, y = cos_inner, z = type (0=omni, 1=spot), w = unused
};
layout(set = 0, binding = 13, std430) readonly buffer LocalLightsBlock {
    uint    count;
    uint    pad0, pad1, pad2;
    RTLight lights[32];
} local_lights_buf;

// ── GI probe grid (C0 cascade) — sourced from RTGIEffect ──────────────────────
// Lets reflection hits pick up indirect lighting (green-wall bleed onto
// orange-bench visible in a mirror, etc.) instead of just direct + flat
// ambient. Zero buffer / disabled grid → sample returns 0 and reflections
// fall back to their existing direct+ambient model.
struct Probe { vec4 rgb_meta; };
layout(set = 0, binding = 12, std430) readonly buffer ProbesBuf {
    Probe data[];
} gi_probes;

layout(set = 0, binding = 14, std430) readonly buffer GiGridBlock {
    vec4  origin_spacing;   // xyz=origin, w=spacing
    ivec4 size_enable;      // xyz=size, w=enable (0/1)
} gi_grid;

// Normal-weighted trilinear probe sample, matches RTGI apply shader.
vec3 sample_gi_probes(vec3 world_pos, vec3 normal) {
    if (gi_grid.size_enable.w == 0) return vec3(0.0);
    vec3  origin  = gi_grid.origin_spacing.xyz;
    float spacing = gi_grid.origin_spacing.w;
    ivec3 gsize   = gi_grid.size_enable.xyz;
    vec3 grid_pos = (world_pos - origin) / spacing;
    if (any(lessThan(grid_pos, vec3(0.0)))
     || any(greaterThanEqual(grid_pos, vec3(gsize) - 1.0))) {
        return vec3(0.0);
    }
    ivec3 base = ivec3(floor(grid_pos));
    vec3  frac = grid_pos - vec3(base);
    vec3  sum  = vec3(0.0);
    float wsum = 0.0;
    for (int dz = 0; dz <= 1; dz++)
    for (int dy = 0; dy <= 1; dy++)
    for (int dx = 0; dx <= 1; dx++) {
        ivec3 idx = clamp(base + ivec3(dx, dy, dz), ivec3(0), gsize - 1);
        float wb = (dx == 0 ? 1.0 - frac.x : frac.x)
                 * (dy == 0 ? 1.0 - frac.y : frac.y)
                 * (dz == 0 ? 1.0 - frac.z : frac.z);
        vec3 pw = origin + vec3(idx) * spacing;
        vec3 dp = normalize(pw - world_pos + normal * 0.001);
        float nw = max(dot(normal, dp), 0.0);
        nw = nw * nw;
        float w = wb * nw;
        uint f = uint(idx.x) + uint(idx.y) * uint(gsize.x)
               + uint(idx.z) * uint(gsize.x * gsize.y);
        sum  += gi_probes.data[f].rgb_meta.xyz * w;
        wsum += w;
    }
    return (wsum > 1e-4) ? (sum / wsum) : vec3(0.0);
}

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

// Cosine-weighted hemisphere sample (matches RTAO sampling pattern).
vec3 cosine_hemisphere(float r1, float r2) {
    float phi       = 6.28318530718 * r1;
    float sin_theta = sqrt(r2);
    float cos_theta = sqrt(1.0 - r2);
    return vec3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
}

// GGX microfacet distribution sample (tangent space, +Z hemisphere).
vec3 ggx_sample(float r1, float r2, float alpha) {
    float phi       = 6.28318530718 * r1;
    float cos_theta = sqrt((1.0 - r2) / max(1.0 + (alpha * alpha - 1.0) * r2, 1e-6));
    float sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
    return vec3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
}

// Uniform sample over spherical cap defined by sin_half_angle (sin of cone half-angle).
// Used for soft-shadow jitter on directional and local lights.
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

vec3 agx(vec3 val) {
    const mat3 agx_mat = mat3(
    0.842479062283038, 0.0423282422610123, 0.0429164076159921,
    0.077812740921479, 0.74190508323946, 0.0766464566191395,
    0.0797127593165214, 0.215752320399317, 0.880435150240172
    );

    // Epsilon to avoid log(0)
    val = max(val, 1e-6);
    val = agx_mat * val;

    // Log2 space encoding
    val = clamp(log2(val), -10.0, 2.0);
    val = (val + 10.0) / 12.0;

    // Sigmoid curve fitting
    vec3 val2 = val * val;
    vec3 val4 = val2 * val2;
    return 15.5 * val4 * val2 - 40.14 * val4 * val + 31.96 * val4 - 6.868 * val2 * val + 0.4298 * val2 + 0.1191 * val;
}


void main() {
    // Dispatch runs at `gl_LaunchSizeEXT` resolution (≤ G-buffer size). When
    // downsampled the mask is small; `pc.screen_size` remains the full-res
    // G-buffer size so texelFetch reads line up with the actual G-buffer pixels.
    ivec2 coord     = ivec2(gl_LaunchIDEXT.xy);
    vec2  rt_size   = vec2(gl_LaunchSizeEXT.xy);
    vec2  screen_uv = (vec2(coord) + 0.5) / rt_size;
    ivec2 full_size = ivec2(pc.screen_size);
    // Full-res G-buffer coord at the centre of this low-res pixel's region.
    ivec2 gb_coord  = clamp(ivec2(screen_uv * pc.screen_size),
                            ivec2(0), full_size - 1);

    float depth     = texelFetch(depth_buffer, gb_coord, 0).r;
    vec4  nr_sample = texelFetch(normal_roughness_buffer, gb_coord, 0);
    float roughness = nr_sample.w;

    // ── Thin-feature recovery at low res ───────────────────────────────────────
    // If the centre of the low-res pixel lands on sky or a rough surface, scan
    // the full-res region this low-res pixel covers and snap to the lowest-
    // roughness non-sky pixel. Catches thin mirrors (wires, small caps) whose
    // footprint is smaller than a low-res texel.
    vec2 scale = pc.screen_size / rt_size;
    if (scale.x > 1.5 && (depth <= 0.0001 || roughness >= 0.5)) {
        ivec2 region_base = ivec2(vec2(coord) * scale);
        ivec2 region_end  = min(ivec2(vec2(coord + 1) * scale), full_size);
        for (int y = region_base.y; y < region_end.y; y++) {
            for (int x = region_base.x; x < region_end.x; x++) {
                ivec2 sc = ivec2(x, y);
                float d  = texelFetch(depth_buffer, sc, 0).r;
                if (d <= 0.0001) continue;
                vec4  nr = texelFetch(normal_roughness_buffer, sc, 0);
                if (nr.w < roughness) {
                    gb_coord  = sc;
                    depth     = d;
                    nr_sample = nr;
                    roughness = nr.w;
                    if (roughness < 0.05) break;
                }
            }
            if (roughness < 0.05) break;
        }
    }

    if (depth <= 0.0001 || roughness >= 0.5) {
        imageStore(reflection_mask, coord, vec4(0.0));
        return;
    }

    // Use gb_coord's actual NDC position (may differ from the low-res centre
    // after the thin-feature search) so the ray originates at the selected
    // pixel's true world-space location.
    vec2 gb_uv = (vec2(gb_coord) + 0.5) / pc.screen_size;
    vec2 ndc_uv = gb_uv * 2.0 - 1.0;

    // Depth → view space → world space
    vec4 view_pos = pc.inv_proj * vec4(ndc_uv, depth, 1.0);
    view_pos.xyz /= view_pos.w;

    vec3 world_pos = vec3(
        pc.inv_view_r0.x * view_pos.x + pc.inv_view_r1.x * view_pos.y + pc.inv_view_r2.x * view_pos.z + pc.inv_view_r0.w,
        pc.inv_view_r0.y * view_pos.x + pc.inv_view_r1.y * view_pos.y + pc.inv_view_r2.y * view_pos.z + pc.inv_view_r1.w,
        pc.inv_view_r0.z * view_pos.x + pc.inv_view_r1.z * view_pos.y + pc.inv_view_r2.z * view_pos.z + pc.inv_view_r2.w
    );

    // Normal (view → world).
    vec3 normal_vs = nr_sample.xyz * 2.0 - 1.0;
    vec3 normal    = normalize(vec3(
        pc.inv_view_r0.x * normal_vs.x + pc.inv_view_r1.x * normal_vs.y + pc.inv_view_r2.x * normal_vs.z,
        pc.inv_view_r0.y * normal_vs.x + pc.inv_view_r1.y * normal_vs.y + pc.inv_view_r2.y * normal_vs.z,
        pc.inv_view_r0.z * normal_vs.x + pc.inv_view_r1.z * normal_vs.y + pc.inv_view_r2.z * normal_vs.z
    ));

    vec3 cam_origin = vec3(pc.inv_view_r0.w, pc.inv_view_r1.w, pc.inv_view_r2.w);
    vec3 view_dir   = normalize(world_pos - cam_origin);
    vec3 refl_dir   = reflect(view_dir, normal);
    vec3 origin     = world_pos + normal * 0.02;

    float alpha = roughness * roughness;

    mat3 tbn       = tbn_from_normal(refl_dir);
    uint n_samples = max((pc.frame_samples >> 16u) & 0xFFFFu, 1u);
    uint frame_idx = pc.frame_samples & 0xFFFFu;
    uint rng       = init_rand(uint(coord.x) + uint(coord.y) * uint(rt_size.x), frame_idx);

    vec3  light_dir    = light.direction_energy.xyz;
    float light_energy = light.direction_energy.w;
    light_energy = light_energy / (light_energy + 1.0);
    vec3  light_color  = light.color_sky_energy.rgb;

    // Separate geometry hits from sky misses.
    // Geometry is at finite distance — it physically occludes the sky (infinite).
    // Averaging them together dilutes geometry reflections with sky color.
    // If any samples hit geometry, use only the geometry average.
    // Fall back to sky average only when every sample missed.
    vec3 geom_accum = vec3(0.0);
    vec3 sky_accum  = vec3(0.0);
    uint n_hits = 0u;
    uint n_sky  = 0u;

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

        if (rpl.ra.w < 0.5) {
            // Sky miss — accumulate separately.
            sky_accum += rpl.ra.xyz;
            n_sky++;
        } else {
            // Geometry hit.
            vec3  albedo       = rpl.ra.xyz;
            float hit_metallic = rpl.ra.w - 1.0;          // unpacked from offset encoding
            vec3  hit_pos      = rpl.rb.xyz;
            float NdotL        = rpl.rb.w;
            vec3  hit_normal   = rpl.rc.xyz;
            vec3  hit_emission = rpl.rd.xyz;              // cached — probe() shadow rays reuse payload
            vec3  lit_origin   = hit_pos + hit_normal * 0.02;

            // ── Compute lighting (raw HDR, no tonemap) ──────────────────────────
            // Screen-space reprojection disabled — every geometry hit uses the
            // manually-computed lighting path below. Values are in the same HDR
            // range Godot's renderer produces (exposure_normalization applied to
            // each light's packed energy on the CPU side).
            vec3 sample_color;
            {
                // Directional light — HARD shadow probe.
                // Cone jitter (angular_distance penumbra) is only applied in the
                // main-scene shadow pass (rt_shadows.glsl) where we can afford many
                // samples and Godot's TAA smooths the result. Reflections typically
                // run one sample per pixel and the reflection mask isn't temporally
                // filtered, so jittering turned shadows into per-pixel coin-flips —
                // breaking both shadow definition and average direct brightness.
                float shadow_vis = 0.0;
                if (NdotL > 0.001) {
                    shadow_vis = probe(lit_origin, light_dir, 1000.0) * NdotL;
                }

                mat3  ao_tbn = tbn_from_normal(hit_normal);
                float ao_vis = 0.0;
                for (int ai = 0; ai < 2; ai++) {
                    vec3 ao_dir = normalize(ao_tbn * cosine_hemisphere(next_rand(rng), next_rand(rng)));
                    ao_vis += probe(lit_origin, ao_dir, 1.0);
                }
//                ao_vis *= 0.5;

                vec3  F0_hit        = mix(vec3(0.04), albedo, hit_metallic);
                float diffuse_scale = 1.0 - hit_metallic;
                vec3 direct  = (albedo * diffuse_scale + F0_hit) * light_energy * light_color * shadow_vis;
                // Ambient must be scaled by the environment's ambient_light (color × energy,
                // already exposure-normalized on the CPU). Without this, ambient is at full
                // albedo brightness (~1.0) while direct and sky are in exposed HDR range —
                // reflections looked ~2× brighter than the surfaces they reflect (chrome ball
                // turning uniformly white instead of mirroring the scene's shaded walls).
                vec3 ambient = albedo * diffuse_scale * light.ambient_pad.xyz * ao_vis;
                // Sample RTGI probe cascade at the reflection hit so mirrors
                // pick up indirect colour bleed instead of only direct +
                // flat ambient. When the RTGI effect isn't present, gi_grid
                // size_enable.w = 0 and the sample returns zero.
                vec3 gi_hit  = sample_gi_probes(hit_pos, hit_normal);
                vec3 gi_bounce = albedo * diffuse_scale * gi_hit;
                sample_color = direct + ambient + gi_bounce + hit_emission;

                // Local lights (OmniLight3D / SpotLight3D) with penumbra.
                for (uint li = 0u; li < local_lights_buf.count && li < 32u; li++) {
                    RTLight lgt    = local_lights_buf.lights[li];
                    vec3  to_light = lgt.pos_range.xyz - hit_pos;
                    float dist     = length(to_light);
                    float range    = lgt.pos_range.w;
                    if (dist >= range || dist < 0.001) continue;
                    vec3  ldir  = to_light / dist;
                    float lndotl = max(dot(hit_normal, ldir), 0.0);
                    if (lndotl < 0.001) continue;

                    // Godot 4 omni/spot attenuation:
                    //   att = ((1 - (dist/range)^4)^2) * dist^(-decay)
                    // Matches RendererSceneRenderRD's get_omni_attenuation() so our
                    // output lands on the same HDR range as the opaque scene under
                    // either physical or practical light units.
                    float decay = lgt.cone_type.w;
                    float nd    = dist / range;
                    nd *= nd; nd *= nd;
                    float cutoff = max(1.0 - nd, 0.0);
                    cutoff *= cutoff;
                    float att = cutoff * pow(max(dist, 0.0001), -decay);

                    // Spot cone attenuation.
                    float spot_att = 1.0;
                    if (lgt.cone_type.z > 0.5) {
                        float cos_angle = dot(-ldir, lgt.dir_size.xyz);
                        float cos_outer = lgt.cone_type.x;
                        float cos_inner = lgt.cone_type.y;
                        spot_att = clamp((cos_angle - cos_outer) / max(cos_inner - cos_outer, 0.001), 0.0, 1.0);
                        if (spot_att <= 0.0) continue;
                    }

                    // Hard shadow probe toward the light center. Disk-sampled
                    // penumbras belong in the scene shadow pass — see directional
                    // shadow note above.
                    float lshadow = probe(lit_origin, ldir, dist + 0.1) * lndotl;
                    vec3 lF0  = mix(vec3(0.04), albedo, hit_metallic);
                    float ldiff = 1.0 - hit_metallic;
                    sample_color += (albedo * ldiff + lF0)
                                  * lgt.color_energy.rgb * lgt.color_energy.w
                                  * att * spot_att * lshadow;
                }
            }

//            sample_color *= 1.0;
//            
            vec3 ldr = agx(sample_color);
            
            geom_accum += sample_color;
            n_hits++;
        }
    }

    // Geometry forecloses sky along those ray directions — use geometry average
    // if any hits exist, otherwise fall back to sky.
    vec3 result;
    if (n_hits > 0u) {
        result = geom_accum / float(n_hits);
    } else {
        result = (n_sky > 0u) ? sky_accum / float(n_sky) : vec3(0.0);
    }

    // Saturation: mix toward luminance (params.z = 1.0 neutral, 0.0 greyscale, >1 boosted).
    float luma = dot(result, vec3(0.2126, 0.7152, 0.0722));
    result = mix(vec3(luma), result, light.params.z);

    // Contrast: power curve centred at linear midgrey (params.w = 1.0 neutral).
    result = pow(max(result, vec3(0.0)), vec3(light.params.w));

    // F0 for the primary surface.
    // Metallic is not available in the G-buffer at this stage, so roughness is
    // used as a proxy: smooth surfaces (roughness→0) are treated as metal-like
    // (F0=1.0), rough surfaces fall back to dielectric (F0=0.04).
    float F0            = mix(0.04, 1.0, clamp(1.0 - roughness * 2.0, 0.0, 1.0));
    float VdotN         = max(dot(-view_dir, normal), 0.0);
    float fresnel       = mix(F0, 1.0, pow(1.0 - VdotN, 5.0));
    float spec_vis      = pow(1.0 - roughness, 4.0);
    float refl_weight   = fresnel * spec_vis;
    // Hard cutoff: if final weight is below 1% the reflection can't be visible
    // anyway. Writing alpha=0 guarantees the apply pass skips the pixel rather
    // than blending `mix(color, refl, tiny)` with subpixel fractional weight.
    if (refl_weight < 0.01) {
        imageStore(reflection_mask, coord, vec4(0.0));
        return;
    }
    imageStore(reflection_mask, coord, vec4(result, 1.0 + refl_weight));
}

#[miss]

#version 460
#extension GL_EXT_ray_tracing : enable

struct ReflPayload { vec4 ra; vec4 rb; vec4 rc; vec4 rd; };
layout(location = 0) rayPayloadInEXT ReflPayload rpl;

layout(set = 0, binding = 9) uniform sampler2D sky_tex;

layout(set = 0, binding = 11, std430) readonly buffer LightBlock {
    vec4 direction_energy;
    vec4 color_sky_energy;  // w = background_energy_multiplier
    vec4 ambient_pad;
} light;

#define PI 3.14159265358979

void main() {
    vec3 dir = normalize(gl_WorldRayDirectionEXT);
    float u = atan(dir.x, -dir.z) / (2.0 * PI) + 0.5;
    float v = clamp(acos(clamp(-dir.y, -1.0, 1.0)) / PI, 0.001, 0.999);
    // Multiply raw sky texture by the WorldEnvironment background_energy_multiplier.
    // When sky energy is 0, this returns black — matching what Godot renders as the background.
    // ra.w = 0.0 marks a sky miss (never the -1 probe sentinel, regardless of sky_energy).
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

// ── Bindings ──────────────────────────────────────────────────────────────────

layout(set = 0, binding = 4, std430) readonly buffer MatInstHdrs {
    uvec2 data[];
} mat_inst;

struct SurfaceMat {
    uvec4 range;       // x=start_prim, y=end_prim, z=tex_idx, w=pad
    vec4  albedo;
    vec4  props;       // x=metallic, y=roughness, z=uv_scale_x, w=uv_scale_y
    vec4  uv_offset;   // x=uv_offset_x, y=uv_offset_y, zw=pad
    vec4  emission;    // xyz = emission_color × emission_energy, w=pad
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

layout(set = 0, binding = 11, std430) readonly buffer LightBlock {
    vec4 direction_energy;
    vec4 color_sky_energy;
    vec4 ambient_pad;
    vec4 params;           // x = reflection_exposure (pre-tonemap scale)
} light;

// ── Main ──────────────────────────────────────────────────────────────────────

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
    // Guard: if prim is beyond mat.range.y, fall back to first surface.
    if (uint(gl_PrimitiveID) > mat.range.y && surf_count > 0u)
        mat = mat_surfs.data[surf_off];

    // ── UV interpolation → albedo sample ─────────────────────────────────
    uint base_v = geom_inst.data[uint(gl_InstanceID)].x;
    uint vi0    = base_v + uint(gl_PrimitiveID) * 3u;
    float b1 = bary.x, b2 = bary.y, b0 = 1.0 - b1 - b2;
    vec2 uv = b0 * uv_buf.data[vi0]
            + b1 * uv_buf.data[vi0 + 1u]
            + b2 * uv_buf.data[vi0 + 2u];

    // Apply material UV transform (tiling / offset).
    uv = uv * mat.props.zw + mat.uv_offset.xy;

    uint tex_idx  = mat.range.z;
    // textureLod required in RT stages — implicit LOD not available outside fragment.
    vec3 tex_color = textureLod(albedo_textures[nonuniformEXT(tex_idx)], uv, 0.0).rgb;
    vec3 albedo    = (mat.albedo.rgb) * tex_color;

    // ── Geometric normal (object space → world space) ─────────────────────
    // Vertex positions are flat xyz floats: vertex N is at data[N*3..N*3+2].
    uint vp0 = vi0 * 3u;
    vec3 p0 = vec3(geom_verts.data[vp0],      geom_verts.data[vp0 + 1u], geom_verts.data[vp0 + 2u]);
    uint vp1 = (vi0 + 1u) * 3u;
    vec3 p1 = vec3(geom_verts.data[vp1],      geom_verts.data[vp1 + 1u], geom_verts.data[vp1 + 2u]);
    uint vp2 = (vi0 + 2u) * 3u;
    vec3 p2 = vec3(geom_verts.data[vp2],      geom_verts.data[vp2 + 1u], geom_verts.data[vp2 + 2u]);

    vec3 obj_normal = normalize(cross(p1 - p0, p2 - p0));
    // Inverse-transpose correctly transforms normals under non-uniform scale.
    mat3 m = mat3(gl_ObjectToWorldEXT);
    vec3 normal_ws = normalize(transpose(inverse(m)) * obj_normal);
    // Ensure the normal faces toward the incoming ray (handles back-face / inverted winding).
    if (dot(gl_WorldRayDirectionEXT, normal_ws) > 0.0) normal_ws = -normal_ws;

    // ── NdotL ─────────────────────────────────────────────────────────────
    float NdotL = max(dot(normal_ws, light.direction_energy.xyz), 0.0);

    // ── Hit world position ─────────────────────────────────────────────────
    vec3 hit_ws = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;

    // ── Write payload — raygen handles shadow/AO, then combines ───────────
    // ra.w = 1 + metallic  (≥1 → geometry hit; sky miss = 0.0 → still < 0.5)
    // rc.w = roughness of the hit surface
    rpl.ra = vec4(albedo,    1.0 + mat.props.x);
    rpl.rb = vec4(hit_ws,    NdotL);
    rpl.rc = vec4(normal_ws, mat.props.y);
    rpl.rd = vec4(mat.emission.rgb, 0.0);
}
