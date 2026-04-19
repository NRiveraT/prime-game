#[compute]

#version 460
#extension GL_EXT_shader_image_load_store : enable

// One invocation per probe across all cascades.
// Dispatched as: NUM_CASCADES * TOTAL_PROBES threads (2048 total).
// Each thread projects its probe to screen space and scatter-writes a disc.
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, rgba16f) uniform image2D color_buf;

// Same params SSBO as apply pass — contains all 4 cascade grid descriptors.
layout(set = 0, binding = 1, std430) readonly buffer DDGIParams {
    float ddgi_strength;
    float energy_preservation;
    float normal_bias;
    float _pad;
    vec4  grid_0;
    vec4  grid_1;
    vec4  grid_2;
    vec4  grid_3;
} params;

// Irradiance atlases — sample probe colour from the stored irradiance.
layout(set = 0, binding = 2) uniform sampler2D irr_atlas_0;
layout(set = 0, binding = 3) uniform sampler2D irr_atlas_1;
layout(set = 0, binding = 4) uniform sampler2D irr_atlas_2;
layout(set = 0, binding = 5) uniform sampler2D irr_atlas_3;

// Depth buffer for occlusion culling — hides probes behind surfaces.
layout(set = 0, binding = 6) uniform sampler2D depth_buf;

// Push constant: forward projection matrix + camera inv-view rows + screen info.
// Layout identical to apply pass except inv_proj → forward proj.
// 128 bytes = 32 floats.
layout(push_constant, std430) uniform PC {
    mat4  proj;           // forward projection (view → clip)  floats 0-15
    vec4  inv_view_r0;    // basis.x + cam_origin.x            floats 16-19
    vec4  inv_view_r1;    // basis.y + cam_origin.y            floats 20-23
    vec4  inv_view_r2;    // basis.z + cam_origin.z            floats 24-27
    vec2  screen_size;    //                                   floats 28-29
    float disc_radius;    //                                   float  30
    float _pad2;          //                                   float  31
} pc;

#define GRID_X       8
#define GRID_Y       8
#define GRID_Z       8
#define ATLAS_COLS   (GRID_X * GRID_Z)  // 64
#define IRR_PROBE_SIDE 16
#define IRR_ATLAS_W  (ATLAS_COLS * IRR_PROBE_SIDE)  // 1024
#define IRR_ATLAS_H  (GRID_Y * IRR_PROBE_SIDE)      // 128
#define TOTAL_PROBES (GRID_X * GRID_Y * GRID_Z)     // 512
#define NUM_CASCADES 4

// ── Octahedral encode ─────────────────────────────────────────────────────────
vec2 oct_encode(vec3 n) {
    float l1 = abs(n.x) + abs(n.y) + abs(n.z);
    vec2  p  = n.xy / l1;
    if (n.z < 0.0) p = (1.0 - abs(p.yx)) * sign(p);
    return p * 0.5 + 0.5;
}

vec2 irr_atlas_uv(uint pid, vec3 dir) {
    vec2 oct = clamp(oct_encode(normalize(dir)),
                     vec2(0.5 / float(IRR_PROBE_SIDE)),
                     vec2(1.0 - 0.5 / float(IRR_PROBE_SIDE)));
    uint tile_col = pid % uint(ATLAS_COLS);
    uint tile_row = pid / uint(ATLAS_COLS);
    vec2 texel = vec2(float(tile_col * IRR_PROBE_SIDE), float(tile_row * IRR_PROBE_SIDE))
               + oct * float(IRR_PROBE_SIDE);
    return texel / vec2(float(IRR_ATLAS_W), float(IRR_ATLAS_H));
}

void main() {
    uint global_idx  = gl_GlobalInvocationID.x;
    uint cascade_idx = global_idx / uint(TOTAL_PROBES);
    uint probe_idx   = global_idx % uint(TOTAL_PROBES);

    if (cascade_idx >= uint(NUM_CASCADES)) return;

    // ── Cascade grid params ───────────────────────────────────────────────────
    vec3  origin;
    float spacing;
    if      (cascade_idx == 0u) { origin = params.grid_0.xyz; spacing = params.grid_0.w; }
    else if (cascade_idx == 1u) { origin = params.grid_1.xyz; spacing = params.grid_1.w; }
    else if (cascade_idx == 2u) { origin = params.grid_2.xyz; spacing = params.grid_2.w; }
    else                        { origin = params.grid_3.xyz; spacing = params.grid_3.w; }

    // ── Probe world position ──────────────────────────────────────────────────
    uint px = probe_idx % uint(GRID_X);
    uint pz = (probe_idx / uint(GRID_X)) % uint(GRID_Z);
    uint py = probe_idx / uint(GRID_X * GRID_Z);
    vec3 probe_world = origin + vec3(float(px), float(py), float(pz)) * spacing;

    // ── World → view → clip projection ───────────────────────────────────────
    vec3 cam_origin = vec3(pc.inv_view_r0.w, pc.inv_view_r1.w, pc.inv_view_r2.w);
    vec3 v = probe_world - cam_origin;
    vec3 view_space = vec3(
        dot(pc.inv_view_r0.xyz, v),
        dot(pc.inv_view_r1.xyz, v),
        dot(pc.inv_view_r2.xyz, v)
    );

    vec4 clip = pc.proj * vec4(view_space, 1.0);
    if (clip.w <= 0.001) return;  // behind camera

    vec3 ndc = clip.xyz / clip.w;
    if (any(greaterThan(abs(ndc.xy), vec2(1.05)))) return;  // off screen

    vec2 screen_pos = (ndc.xy * 0.5 + 0.5) * pc.screen_size;

    // ── Depth occlusion — hide probes behind geometry ─────────────────────────
    vec2 screen_uv  = screen_pos / pc.screen_size;
    float scene_depth = texture(depth_buf, screen_uv).r;
    // Convert probe clip-depth to same space as depth buffer (0..1)
    float probe_depth = ndc.z * 0.5 + 0.5;
    // Use a small bias so a probe exactly on a surface still shows
    if (scene_depth < probe_depth - 0.005) return;

    // ── Sample irradiance as probe colour (average of 5 octahedral samples) ───
    // We can't easily average the whole tile, so we sample 5 canonical directions.
    const vec3 SAMPLE_DIRS[5] = vec3[5](
        vec3( 0.0,  1.0,  0.0),
        vec3( 0.0, -1.0,  0.0),
        vec3( 1.0,  0.0,  0.0),
        vec3(-1.0,  0.0,  0.0),
        vec3( 0.0,  0.0,  1.0)
    );
    vec3 irr_color = vec3(0.0);
    for (int d = 0; d < 5; d++) {
        vec2 uv;
        if      (cascade_idx == 0u) uv = irr_atlas_uv(probe_idx, SAMPLE_DIRS[d]);
        else if (cascade_idx == 1u) uv = irr_atlas_uv(probe_idx, SAMPLE_DIRS[d]);
        else if (cascade_idx == 2u) uv = irr_atlas_uv(probe_idx, SAMPLE_DIRS[d]);
        else                        uv = irr_atlas_uv(probe_idx, SAMPLE_DIRS[d]);
        // (sampler selection can't use a variable index in Vulkan GLSL — sampled below)
        irr_color += vec3(0.0);  // placeholder, overwritten below
    }
    // Sample atlas for this cascade (Vulkan requires static sampler indexing)
    vec3 c0 = vec3(0.0), c1 = vec3(0.0), c2 = vec3(0.0), c3 = vec3(0.0);
    for (int d = 0; d < 5; d++) {
        vec2 uv = irr_atlas_uv(probe_idx, SAMPLE_DIRS[d]);
        c0 += textureLod(irr_atlas_0, uv, 0.0).rgb;
        c1 += textureLod(irr_atlas_1, uv, 0.0).rgb;
        c2 += textureLod(irr_atlas_2, uv, 0.0).rgb;
        c3 += textureLod(irr_atlas_3, uv, 0.0).rgb;
    }
    if      (cascade_idx == 0u) irr_color = c0 / 5.0;
    else if (cascade_idx == 1u) irr_color = c1 / 5.0;
    else if (cascade_idx == 2u) irr_color = c2 / 5.0;
    else                        irr_color = c3 / 5.0;

    // ── Cascade tint ring (outline color — distinct per cascade) ─────────────
    // C0=red  C1=yellow  C2=green  C3=blue
    const vec3 CASCADE_RING[4] = vec3[4](
        vec3(1.0, 0.15, 0.15),
        vec3(1.0, 0.75, 0.10),
        vec3(0.15, 1.0, 0.15),
        vec3(0.15, 0.50, 1.0)
    );
    vec3 ring_color = CASCADE_RING[cascade_idx];

    // ── Scatter-write disc ────────────────────────────────────────────────────
    float radius = pc.disc_radius;
    int   ir     = int(ceil(radius));
    ivec2 sz     = imageSize(color_buf);

    for (int dy = -ir; dy <= ir; dy++) {
        for (int dx = -ir; dx <= ir; dx++) {
            float dist = sqrt(float(dx * dx + dy * dy));
            if (dist > radius) continue;

            ivec2 pixel = ivec2(screen_pos) + ivec2(dx, dy);
            if (pixel.x < 0 || pixel.y < 0 || pixel.x >= sz.x || pixel.y >= sz.y) continue;

            // Ring at the outer edge, irradiance colour in the interior
            float ring  = smoothstep(radius, radius - 1.5, dist);
            vec3  color = mix(ring_color, max(irr_color, vec3(0.05)), ring);

            imageStore(color_buf, pixel, vec4(color, 1.0));
        }
    }
}
