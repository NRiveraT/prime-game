@tool
class_name RTGIEffect
extends CompositorEffect

# Cascaded probe grid GI.
#   C0 — near-field single-irradiance probe grid (was Phase A).
#   C1 — optional mid-field cascade, same probe count, larger spacing.
#   C2 — optional far-field cascade, same probe count, even larger spacing.
# Apply samples C0 first; pixels outside C0 fall through to C1, then C2.
# Smooth transition at each cascade boundary when smooth_cascade_blend = true.
#
# Phase B1 screen-space cache remains available as an optional per-surface
# quality layer (bilateral gather over cache taps with probe-grid fallback).

# ── Fixed layout constants ───────────────────────────────────────────────────
const PROBE_STRIDE_BYTES: int = 16   # vec4 per probe (rgb + meta pad)
const ROTATION_SLICES:    int = 6

# Singleton access — RTReflectionEffect reads the C0 probe buf from here so
# reflected surfaces can sample GI at their hit points.
static var instance: RTGIEffect = null

# ── Exports: probe grid dimensions (shared across all cascades) ──────────────
@export var c0_grid_x: int = 32
@export var c0_grid_y: int = 16
@export var c0_grid_z: int = 32

@export var follow_camera:        bool    = true
@export var grid_origin_offset:   Vector3 = Vector3(-16.0, -8.0, -16.0)
@export var grid_spacing:         float   = 1.0
@export var gi_strength:          float   = 1.0
@export var ema_alpha:            float   = 0.10
@export var max_ray_dist:         float   = 100.0
@export var indirect_intensity:   float   = 3.14159

enum DebugMode { OFF, INDIRECT, WORLD_POS, GRID_CELL, PROBE_LOCAL, CACHE_ONLY,
				 C0_ONLY, C1_ONLY, C2_ONLY }
@export var debug_mode: DebugMode = DebugMode.OFF

# ── Exports: Phase B1 screen cache ──────────────────────────────────────────
@export var enable_screen_cache: bool  = true
@export var cache_scale:         int   = 4           # 2, 4, 8 supported.
@export var cache_ema:           float = 0.80

# ── Exports: probe cascades ─────────────────────────────────────────────────
# 0 = C0 only. 1 = +C1. 2 = +C1+C2.
@export_range(0, 2) var num_extra_cascades:   int   = 1
# Each cascade's spacing = c0 spacing × ratio^level. With ratio=4:
#   C1 spacing = 4m → 128m × 64m × 128m volume (at 32×16×32 probes)
#   C2 spacing = 16m → 512m × 256m × 512m volume
@export var cascade_spacing_ratio: float = 4.0
# Smooth blend across cascade boundaries (costs ~1 extra sample per pixel near
# boundary). Discrete mode produces a visible seam at the C0 edge but is
# cheaper and sometimes preferred for perf A/B.
@export var smooth_cascade_blend:  bool  = true

# ── State ────────────────────────────────────────────────────────────────────
var _rd: RenderingDevice

var _current_c0_origin: Vector3 = Vector3.ZERO
var _current_c1_origin: Vector3 = Vector3.ZERO
var _current_c2_origin: Vector3 = Vector3.ZERO

var _allocated_probe_count: int = 0  # tracks buffer allocation vs. export dims

# Phase A probe update RT pipeline (shared across all cascades).
var _rt_shader:   RID
var _rt_pipeline: RID

# Phase B1 screen cache pipeline.
var _cache_shader:      RID
var _cache_pipeline:    RID
var _screen_cache_tex:  RID
var _last_cache_size:   Vector2i = Vector2i(0, 0)
var _fallback_cache_tex: RID

# Apply compute pipeline.
var _apply_shader:   RID
var _apply_pipeline: RID

# Cascade probe SSBOs (single-irradiance, 16 bytes per probe).
var _probes_buf:    RID  # C0 (near-field) — also exposed via singleton to reflections
var _c1_probes_buf: RID
var _c2_probes_buf: RID
var _camera_buf:    RID
var _light_buf:     RID
var _sampler:       RID
var _repeat_sampler: RID
var _rotation_offset: int = 0

func _init() -> void:
	effect_callback_type = EFFECT_CALLBACK_TYPE_POST_TRANSPARENT
	access_resolved_color  = true
	access_resolved_depth  = true
	needs_normal_roughness = true

	instance = self

	_rd = RenderingServer.get_rendering_device()
	if not _rd.has_feature(RenderingDevice.SUPPORTS_RAYTRACING_PIPELINE):
		push_error("[RTGI] Hardware raytracing not supported.")
		return

	_build_rt_pipeline()
	_build_cache_pipeline()
	_build_apply_pipeline()
	_build_persistent_resources()

func _build_rt_pipeline() -> void:
	var full_src := FileAccess.get_file_as_string("res://Shaders/rtgi_probe_update.glsl")
	if full_src.is_empty():
		push_error("[RTGI] Could not load rtgi_probe_update.glsl")
		return
	var shader_source := RDShaderSource.new()
	shader_source.source_raygen      = _extract_section(full_src, "#[raygen]",      "#[miss]")
	shader_source.source_miss        = _extract_section(full_src, "#[miss]",        "#[closest_hit]")
	shader_source.source_closest_hit = _extract_section(full_src, "#[closest_hit]", "")
	var spirv := _rd.shader_compile_spirv_from_source(shader_source)
	var err_rg := spirv.get_stage_compile_error(RenderingDevice.SHADER_STAGE_RAYGEN)
	var err_ms := spirv.get_stage_compile_error(RenderingDevice.SHADER_STAGE_MISS)
	var err_ch := spirv.get_stage_compile_error(RenderingDevice.SHADER_STAGE_CLOSEST_HIT)
	if not err_rg.is_empty(): push_error("[RTGI] raygen: ", err_rg)
	if not err_ms.is_empty(): push_error("[RTGI] miss: ",   err_ms)
	if not err_ch.is_empty(): push_error("[RTGI] chit: ",   err_ch)
	_rt_shader   = _rd.shader_create_from_spirv(spirv)
	_rt_pipeline = _rd.raytracing_pipeline_create(_rt_shader)
	print("[RTGI] RT pipeline valid: ", _rt_pipeline.is_valid())

func _build_cache_pipeline() -> void:
	var full_src := FileAccess.get_file_as_string("res://Shaders/rtgi_screen_cache.glsl")
	if full_src.is_empty():
		push_error("[RTGI] Could not load rtgi_screen_cache.glsl")
		return
	var shader_source := RDShaderSource.new()
	shader_source.source_raygen      = _extract_section(full_src, "#[raygen]",      "#[miss]")
	shader_source.source_miss        = _extract_section(full_src, "#[miss]",        "#[closest_hit]")
	shader_source.source_closest_hit = _extract_section(full_src, "#[closest_hit]", "")
	var spirv := _rd.shader_compile_spirv_from_source(shader_source)
	var err_rg := spirv.get_stage_compile_error(RenderingDevice.SHADER_STAGE_RAYGEN)
	var err_ms := spirv.get_stage_compile_error(RenderingDevice.SHADER_STAGE_MISS)
	var err_ch := spirv.get_stage_compile_error(RenderingDevice.SHADER_STAGE_CLOSEST_HIT)
	if not err_rg.is_empty(): push_error("[RTGI] cache raygen: ", err_rg)
	if not err_ms.is_empty(): push_error("[RTGI] cache miss: ",   err_ms)
	if not err_ch.is_empty(): push_error("[RTGI] cache chit: ",   err_ch)
	_cache_shader   = _rd.shader_create_from_spirv(spirv)
	_cache_pipeline = _rd.raytracing_pipeline_create(_cache_shader)
	print("[RTGI] Cache pipeline valid: ", _cache_pipeline.is_valid())

func _ensure_cache_tex(size: Vector2i) -> void:
	if _screen_cache_tex.is_valid() and _last_cache_size == size:
		return
	if _screen_cache_tex.is_valid():
		_rd.free_rid(_screen_cache_tex)
	var fmt := RDTextureFormat.new()
	fmt.format     = RenderingDevice.DATA_FORMAT_R16G16B16A16_SFLOAT
	fmt.width      = size.x
	fmt.height     = size.y
	fmt.usage_bits = (
		RenderingDevice.TEXTURE_USAGE_STORAGE_BIT |
		RenderingDevice.TEXTURE_USAGE_SAMPLING_BIT |
		RenderingDevice.TEXTURE_USAGE_CAN_COPY_FROM_BIT
	)
	_screen_cache_tex = _rd.texture_create(fmt, RDTextureView.new())
	_last_cache_size  = size

func _build_apply_pipeline() -> void:
	var src := FileAccess.get_file_as_string("res://Shaders/apply_rtgi.glsl")
	if src.is_empty():
		push_error("[RTGI] Could not load apply_rtgi.glsl")
		return
	var shader_source := RDShaderSource.new()
	shader_source.source_compute = _extract_section(src, "#[compute]", "")
	var spirv := _rd.shader_compile_spirv_from_source(shader_source)
	var err := spirv.get_stage_compile_error(RenderingDevice.SHADER_STAGE_COMPUTE)
	if not err.is_empty():
		push_error("[RTGI] apply error: ", err)
		return
	_apply_shader   = _rd.shader_create_from_spirv(spirv)
	_apply_pipeline = _rd.compute_pipeline_create(_apply_shader)
	print("[RTGI] Apply pipeline valid: ", _apply_pipeline.is_valid())

func _build_persistent_resources() -> void:
	var ss := RDSamplerState.new()
	ss.min_filter = RenderingDevice.SAMPLER_FILTER_LINEAR
	ss.mag_filter = RenderingDevice.SAMPLER_FILTER_LINEAR
	_sampler = _rd.sampler_create(ss)

	var ss_rep := RDSamplerState.new()
	ss_rep.min_filter = RenderingDevice.SAMPLER_FILTER_LINEAR
	ss_rep.mag_filter = RenderingDevice.SAMPLER_FILTER_LINEAR
	ss_rep.repeat_u   = RenderingDevice.SAMPLER_REPEAT_MODE_REPEAT
	ss_rep.repeat_v   = RenderingDevice.SAMPLER_REPEAT_MODE_REPEAT
	_repeat_sampler = _rd.sampler_create(ss_rep)

	_camera_buf = _rd.storage_buffer_create(128)

	# 1×1 zero fallback for the screen cache binding — apply always binds the
	# texture at binding 7; when cache is off/bypassed we point at this.
	var fmt := RDTextureFormat.new()
	fmt.format     = RenderingDevice.DATA_FORMAT_R16G16B16A16_SFLOAT
	fmt.width      = 1
	fmt.height     = 1
	fmt.usage_bits = (
		RenderingDevice.TEXTURE_USAGE_STORAGE_BIT |
		RenderingDevice.TEXTURE_USAGE_SAMPLING_BIT
	)
	var zeros := PackedByteArray()
	zeros.resize(8)
	zeros.fill(0)
	_fallback_cache_tex = _rd.texture_create(fmt, RDTextureView.new(), [zeros])

func _ensure_probe_buffers() -> void:
	var probe_count: int = c0_grid_x * c0_grid_y * c0_grid_z
	if probe_count == _allocated_probe_count:
		return
	if _probes_buf.is_valid():    _rd.free_rid(_probes_buf)
	if _c1_probes_buf.is_valid(): _rd.free_rid(_c1_probes_buf)
	if _c2_probes_buf.is_valid(): _rd.free_rid(_c2_probes_buf)
	var byte_count: int = probe_count * PROBE_STRIDE_BYTES
	var zero := PackedByteArray()
	zero.resize(byte_count)
	zero.fill(0)
	_probes_buf    = _rd.storage_buffer_create(byte_count, zero)
	_c1_probes_buf = _rd.storage_buffer_create(byte_count, zero)
	_c2_probes_buf = _rd.storage_buffer_create(byte_count, zero)
	_allocated_probe_count = probe_count

func _render_callback(p_callback_type: int, p_render_data: RenderData) -> void:
	var sm := RTScene.instance
	if sm == null:
		return
	sm.tick()
	if not sm.is_ready:
		return
	if not _rt_pipeline.is_valid() or not _apply_pipeline.is_valid():
		return
	if not sm.mat_inst_ssbo.is_valid() or not sm.mat_surf_ssbo.is_valid():
		return
	if not sm.uv_ssbo.is_valid() or sm.texture_rids.is_empty():
		return
	if not sm.sky_texture_rid.is_valid():
		return
	if not sm.geom_vertex_ssbo.is_valid():
		return

	_rd = RenderingServer.get_rendering_device()

	var scene_buffers := p_render_data.get_render_scene_buffers() as RenderSceneBuffersRD
	var scene_data    := p_render_data.get_render_scene_data()    as RenderSceneDataRD
	if scene_buffers == null or scene_data == null:
		return

	var size := scene_buffers.get_internal_size()

	_ensure_probe_buffers()

	# Per-cascade camera-snapped origins. Each cascade snaps to its own
	# spacing so its cells stay aligned as camera moves.
	var c1_spacing: float = grid_spacing * cascade_spacing_ratio
	var c2_spacing: float = grid_spacing * cascade_spacing_ratio * cascade_spacing_ratio

	if follow_camera:
		var cam_pos: Vector3 = scene_data.get_cam_transform().origin
		_current_c0_origin = _snap_origin(cam_pos, grid_spacing)
		_current_c1_origin = _snap_origin(cam_pos, c1_spacing)
		_current_c2_origin = _snap_origin(cam_pos, c2_spacing)
	else:
		_current_c0_origin = grid_origin_offset
		_current_c1_origin = grid_origin_offset
		_current_c2_origin = grid_origin_offset

	_update_light_buffer(sm)
	var inv_proj: Projection  = scene_data.get_cam_projection().inverse()
	var inv_view: Transform3D = scene_data.get_cam_transform()
	_update_camera_buffer(inv_proj, inv_view)

	var depth_tex:  RID = scene_buffers.get_depth_texture()
	var normal_tex: RID = scene_buffers.get_texture("forward_clustered", "normal_roughness")
	var color_tex:  RID = scene_buffers.get_color_texture()

	# ── Cascade probe updates (C0 always; C1 and C2 conditional) ─────────────
	_dispatch_probe_cascade(sm, _probes_buf, _current_c0_origin, grid_spacing)
	if num_extra_cascades >= 1:
		_dispatch_probe_cascade(sm, _c1_probes_buf, _current_c1_origin, c1_spacing)
	if num_extra_cascades >= 2:
		_dispatch_probe_cascade(sm, _c2_probes_buf, _current_c2_origin, c2_spacing)

	_rotation_offset = (_rotation_offset + 1) % ROTATION_SLICES

	# ── Phase B1 screen cache (independent toggle) ───────────────────────────
	if enable_screen_cache and _cache_pipeline.is_valid():
		var cache_size := Vector2i(
			max((size.x + cache_scale - 1) / cache_scale, 1),
			max((size.y + cache_scale - 1) / cache_scale, 1)
		)
		_ensure_cache_tex(cache_size)
		_dispatch_screen_cache(sm, depth_tex, normal_tex, cache_size)

	# ── Apply ────────────────────────────────────────────────────────────────
	_dispatch_apply(color_tex, depth_tex, normal_tex, size,
					c1_spacing, c2_spacing)

static func _snap_origin(cam_pos: Vector3, spacing: float) -> Vector3:
	# Centre the grid on the camera, snapped to integer multiples of spacing.
	var snapped := Vector3(
		floor(cam_pos.x / spacing) * spacing,
		floor(cam_pos.y / spacing) * spacing,
		floor(cam_pos.z / spacing) * spacing
	)
	# Half-extent shift so grid is centred; caller passes its own dims.
	return snapped  # half-extent is applied in _dispatch_probe_cascade per cascade

func _dispatch_probe_cascade(sm: RTScene, probe_buf: RID,
							 snapped: Vector3, spacing: float) -> void:
	# Centre the grid on the camera (or static offset if !follow_camera).
	var origin: Vector3 = snapped
	if follow_camera:
		var half_extent := Vector3(c0_grid_x, c0_grid_y, c0_grid_z) * spacing * 0.5
		origin = snapped - half_extent

	var u_tlas := RDUniform.new()
	u_tlas.uniform_type = RenderingDevice.UNIFORM_TYPE_ACCELERATION_STRUCTURE
	u_tlas.binding = 0
	u_tlas.add_id(sm.tlas)

	var u_probes := RDUniform.new()
	u_probes.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u_probes.binding = 1
	u_probes.add_id(probe_buf)

	var u_mat_inst := RDUniform.new()
	u_mat_inst.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u_mat_inst.binding = 4
	u_mat_inst.add_id(sm.mat_inst_ssbo)

	var u_mat_surf := RDUniform.new()
	u_mat_surf.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u_mat_surf.binding = 5
	u_mat_surf.add_id(sm.mat_surf_ssbo)

	var u_geom_inst := RDUniform.new()
	u_geom_inst.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u_geom_inst.binding = 6
	u_geom_inst.add_id(sm.geom_instance_ssbo)

	var u_uvs := RDUniform.new()
	u_uvs.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u_uvs.binding = 7
	u_uvs.add_id(sm.uv_ssbo)

	var u_textures := RDUniform.new()
	u_textures.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u_textures.binding = 8
	for tex_rid in sm.texture_rids:
		u_textures.add_id(_repeat_sampler)
		u_textures.add_id(tex_rid)

	var u_sky := RDUniform.new()
	u_sky.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u_sky.binding = 9
	u_sky.add_id(_sampler)
	u_sky.add_id(sm.sky_texture_rid)

	var u_geom_verts := RDUniform.new()
	u_geom_verts.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u_geom_verts.binding = 10
	u_geom_verts.add_id(sm.geom_vertex_ssbo)

	var u_light := RDUniform.new()
	u_light.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u_light.binding = 11
	u_light.add_id(_light_buf)

	var rt_set := _rd.uniform_set_create(
		[u_tlas, u_probes, u_mat_inst, u_mat_surf, u_geom_inst, u_uvs,
		 u_textures, u_sky, u_geom_verts, u_light],
		_rt_shader, 0)

	var push_bytes := PackedByteArray()
	push_bytes.resize(48)
	push_bytes.encode_float( 0, origin.x)
	push_bytes.encode_float( 4, origin.y)
	push_bytes.encode_float( 8, origin.z)
	push_bytes.encode_float(12, spacing)
	push_bytes.encode_s32(16, c0_grid_x)
	push_bytes.encode_s32(20, c0_grid_y)
	push_bytes.encode_s32(24, c0_grid_z)
	push_bytes.encode_u32(28, _rotation_offset)
	push_bytes.encode_u32(32, ROTATION_SLICES)
	push_bytes.encode_float(36, ema_alpha)
	push_bytes.encode_float(40, max_ray_dist)
	push_bytes.encode_u32(44, Engine.get_frames_drawn() & 0xFFFFFFFF)

	var total_probes: int = c0_grid_x * c0_grid_y * c0_grid_z
	var slice_size: int = (total_probes + ROTATION_SLICES - 1) / ROTATION_SLICES

	var rt_list := _rd.raytracing_list_begin()
	_rd.raytracing_list_bind_raytracing_pipeline(rt_list, _rt_pipeline)
	_rd.raytracing_list_bind_uniform_set(rt_list, rt_set, 0)
	_rd.raytracing_list_set_push_constant(rt_list, push_bytes, push_bytes.size())
	_rd.raytracing_list_trace_rays(rt_list, slice_size, 1)
	_rd.raytracing_list_end()
	_rd.free_rid(rt_set)

# ── Phase B1 screen-space cache dispatch ─────────────────────────────────────
func _dispatch_screen_cache(sm: RTScene, depth_tex: RID, normal_tex: RID,
							cache_size: Vector2i) -> void:
	# Sample C0 origin+spacing for the cache's probe-grid lookup inside its chit
	# (multi-bounce indirect at hit).
	var c0_origin: Vector3 = _current_c0_origin
	if follow_camera:
		var half_extent := Vector3(c0_grid_x, c0_grid_y, c0_grid_z) * grid_spacing * 0.5
		c0_origin = _current_c0_origin - half_extent

	var u_tlas := RDUniform.new()
	u_tlas.uniform_type = RenderingDevice.UNIFORM_TYPE_ACCELERATION_STRUCTURE
	u_tlas.binding = 0
	u_tlas.add_id(sm.tlas)

	var u_cache := RDUniform.new()
	u_cache.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u_cache.binding = 1
	u_cache.add_id(_screen_cache_tex)

	var u_depth := RDUniform.new()
	u_depth.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u_depth.binding = 2
	u_depth.add_id(_sampler)
	u_depth.add_id(depth_tex)

	var u_normal := RDUniform.new()
	u_normal.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u_normal.binding = 3
	u_normal.add_id(_sampler)
	u_normal.add_id(normal_tex)

	var u_mat_inst := RDUniform.new()
	u_mat_inst.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u_mat_inst.binding = 4
	u_mat_inst.add_id(sm.mat_inst_ssbo)

	var u_mat_surf := RDUniform.new()
	u_mat_surf.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u_mat_surf.binding = 5
	u_mat_surf.add_id(sm.mat_surf_ssbo)

	var u_geom_inst := RDUniform.new()
	u_geom_inst.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u_geom_inst.binding = 6
	u_geom_inst.add_id(sm.geom_instance_ssbo)

	var u_uvs := RDUniform.new()
	u_uvs.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u_uvs.binding = 7
	u_uvs.add_id(sm.uv_ssbo)

	var u_textures := RDUniform.new()
	u_textures.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u_textures.binding = 8
	for tex_rid in sm.texture_rids:
		u_textures.add_id(_repeat_sampler)
		u_textures.add_id(tex_rid)

	var u_sky := RDUniform.new()
	u_sky.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u_sky.binding = 9
	u_sky.add_id(_sampler)
	u_sky.add_id(sm.sky_texture_rid)

	var u_geom_verts := RDUniform.new()
	u_geom_verts.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u_geom_verts.binding = 10
	u_geom_verts.add_id(sm.geom_vertex_ssbo)

	var u_light := RDUniform.new()
	u_light.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u_light.binding = 11
	u_light.add_id(_light_buf)

	var u_probes := RDUniform.new()
	u_probes.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u_probes.binding = 12
	u_probes.add_id(_probes_buf)

	var u_camera := RDUniform.new()
	u_camera.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u_camera.binding = 14
	u_camera.add_id(_camera_buf)

	var rt_set := _rd.uniform_set_create(
		[u_tlas, u_cache, u_depth, u_normal, u_mat_inst, u_mat_surf,
		 u_geom_inst, u_uvs, u_textures, u_sky, u_geom_verts, u_light,
		 u_probes, u_camera],
		_cache_shader, 0)

	var pb := PackedByteArray()
	pb.resize(48)
	pb.encode_float( 0, c0_origin.x)
	pb.encode_float( 4, c0_origin.y)
	pb.encode_float( 8, c0_origin.z)
	pb.encode_float(12, grid_spacing)
	pb.encode_s32(16, c0_grid_x)
	pb.encode_s32(20, c0_grid_y)
	pb.encode_s32(24, c0_grid_z)
	pb.encode_u32(28, Engine.get_frames_drawn() & 0xFFFFFFFF)
	pb.encode_float(32, max_ray_dist)
	pb.encode_float(36, clamp(cache_ema, 0.0, 0.99))

	var rt_list := _rd.raytracing_list_begin()
	_rd.raytracing_list_bind_raytracing_pipeline(rt_list, _cache_pipeline)
	_rd.raytracing_list_bind_uniform_set(rt_list, rt_set, 0)
	_rd.raytracing_list_set_push_constant(rt_list, pb, pb.size())
	_rd.raytracing_list_trace_rays(rt_list, cache_size.x, cache_size.y)
	_rd.raytracing_list_end()
	_rd.free_rid(rt_set)

# ── Apply pass ───────────────────────────────────────────────────────────────
func _dispatch_apply(color_tex: RID, depth_tex: RID, normal_tex: RID,
					 size: Vector2i,
					 c1_spacing: float, c2_spacing: float) -> void:
	var u_color := RDUniform.new()
	u_color.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u_color.binding = 0
	u_color.add_id(color_tex)

	var u_depth := RDUniform.new()
	u_depth.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u_depth.binding = 1
	u_depth.add_id(_sampler)
	u_depth.add_id(depth_tex)

	var u_normal := RDUniform.new()
	u_normal.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u_normal.binding = 2
	u_normal.add_id(_sampler)
	u_normal.add_id(normal_tex)

	var u_c0 := RDUniform.new()
	u_c0.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u_c0.binding = 3
	u_c0.add_id(_probes_buf)

	var u_camera := RDUniform.new()
	u_camera.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u_camera.binding = 4
	u_camera.add_id(_camera_buf)

	var u_c1 := RDUniform.new()
	u_c1.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u_c1.binding = 5
	u_c1.add_id(_c1_probes_buf)

	var u_c2 := RDUniform.new()
	u_c2.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u_c2.binding = 6
	u_c2.add_id(_c2_probes_buf)

	var cache_active: bool = enable_screen_cache and _screen_cache_tex.is_valid()
	var cache_read: RID = _screen_cache_tex if cache_active else _fallback_cache_tex
	var u_cache := RDUniform.new()
	u_cache.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u_cache.binding = 7
	u_cache.add_id(cache_read)

	var apply_set := _rd.uniform_set_create(
		[u_color, u_depth, u_normal, u_c0, u_camera, u_c1, u_c2, u_cache],
		_apply_shader, 0)

	# Cascade origins: apply shader expects world-space origin (already
	# half-extent shifted). Phase A dispatch does that shift internally —
	# here we apply it to the snapped-to-spacing origins.
	var half_c0 := Vector3(c0_grid_x, c0_grid_y, c0_grid_z) * grid_spacing * 0.5
	var half_c1 := Vector3(c0_grid_x, c0_grid_y, c0_grid_z) * c1_spacing  * 0.5
	var half_c2 := Vector3(c0_grid_x, c0_grid_y, c0_grid_z) * c2_spacing  * 0.5
	var c0_origin: Vector3 = _current_c0_origin - half_c0 if follow_camera else _current_c0_origin
	var c1_origin: Vector3 = _current_c1_origin - half_c1 if follow_camera else _current_c1_origin
	var c2_origin: Vector3 = _current_c2_origin - half_c2 if follow_camera else _current_c2_origin

	# Push constant: 112 bytes.
	var ap := PackedByteArray()
	ap.resize(112)
	ap.encode_float( 0, c0_origin.x)
	ap.encode_float( 4, c0_origin.y)
	ap.encode_float( 8, c0_origin.z)
	ap.encode_float(12, grid_spacing)
	ap.encode_s32(16, c0_grid_x)
	ap.encode_s32(20, c0_grid_y)
	ap.encode_s32(24, c0_grid_z)
	ap.encode_s32(28, int(debug_mode))
	ap.encode_float(32, gi_strength)
	ap.encode_float(36, 1.0 if cache_active else 0.0)
	ap.encode_float(40, float(num_extra_cascades))
	ap.encode_float(44, 1.0 if smooth_cascade_blend else 0.0)
	ap.encode_float(48, c1_origin.x)
	ap.encode_float(52, c1_origin.y)
	ap.encode_float(56, c1_origin.z)
	ap.encode_float(60, c1_spacing)
	ap.encode_s32(64, c0_grid_x)
	ap.encode_s32(68, c0_grid_y)
	ap.encode_s32(72, c0_grid_z)
	# 76..79 pad
	ap.encode_float(80, c2_origin.x)
	ap.encode_float(84, c2_origin.y)
	ap.encode_float(88, c2_origin.z)
	ap.encode_float(92, c2_spacing)
	ap.encode_s32(96,  c0_grid_x)
	ap.encode_s32(100, c0_grid_y)
	ap.encode_s32(104, c0_grid_z)
	# 108..111 pad

	var apply_list := _rd.compute_list_begin()
	_rd.compute_list_bind_compute_pipeline(apply_list, _apply_pipeline)
	_rd.compute_list_bind_uniform_set(apply_list, apply_set, 0)
	_rd.compute_list_set_push_constant(apply_list, ap, ap.size())
	_rd.compute_list_dispatch(apply_list,
		int(ceil(size.x / 8.0)),
		int(ceil(size.y / 8.0)),
		1)
	_rd.compute_list_end()
	_rd.free_rid(apply_set)

# ── Singleton accessors (used by RTReflectionEffect for GI sampling) ────────
func get_c0_probe_buf() -> RID:
	return _probes_buf
func get_c0_grid_origin() -> Vector3:
	if follow_camera:
		var half_extent := Vector3(c0_grid_x, c0_grid_y, c0_grid_z) * grid_spacing * 0.5
		return _current_c0_origin - half_extent
	return _current_c0_origin
func get_c0_grid_spacing() -> float:
	return grid_spacing
func get_c0_grid_size() -> Vector3i:
	return Vector3i(c0_grid_x, c0_grid_y, c0_grid_z)

# ── Light buffer ─────────────────────────────────────────────────────────────
func _update_light_buffer(sm: RTScene) -> void:
	if not _light_buf.is_valid():
		_light_buf = _rd.storage_buffer_create(64)

	var physical_units: bool = ProjectSettings.get_setting(
		"rendering/lights_and_shadows/use_physical_light_units", false)
	var exposure_norm: float = sm.exposure_normalization

	var dl : DirectionalLight3D = sm.directional_light
	var light_dir   := Vector3(0.0, 1.0, 0.0)
	var light_color := Vector3(1.0, 1.0, 1.0)
	var dir_energy  := 1.0
	if dl != null:
		light_dir   = dl.global_transform.basis.z.normalized()
		light_color = Vector3(dl.light_color.r, dl.light_color.g, dl.light_color.b)
		dir_energy  = _compute_light_final_energy(dl, physical_units, exposure_norm)

	var amb := sm.ambient_light * exposure_norm
	var sky_energy := sm.sky_energy_multiplier * exposure_norm
	var data := PackedFloat32Array([
		light_dir.x, light_dir.y, light_dir.z, dir_energy,
		light_color.x, light_color.y, light_color.z, sky_energy,
		amb.x, amb.y, amb.z, 0.0,
		indirect_intensity, 0.0, 0.0, 0.0
	])
	_rd.buffer_update(_light_buf, 0, 64, data.to_byte_array())

func _update_camera_buffer(inv_proj: Projection, inv_view: Transform3D) -> void:
	var b := inv_view.basis
	var o := inv_view.origin
	var cam_data := PackedFloat32Array()
	var c0: Vector4 = inv_proj[0]; cam_data.append(c0.x); cam_data.append(c0.y); cam_data.append(c0.z); cam_data.append(c0.w)
	var c1: Vector4 = inv_proj[1]; cam_data.append(c1.x); cam_data.append(c1.y); cam_data.append(c1.z); cam_data.append(c1.w)
	var c2: Vector4 = inv_proj[2]; cam_data.append(c2.x); cam_data.append(c2.y); cam_data.append(c2.z); cam_data.append(c2.w)
	var c3: Vector4 = inv_proj[3]; cam_data.append(c3.x); cam_data.append(c3.y); cam_data.append(c3.z); cam_data.append(c3.w)
	cam_data.append(b.x.x); cam_data.append(b.x.y); cam_data.append(b.x.z); cam_data.append(o.x)
	cam_data.append(b.y.x); cam_data.append(b.y.y); cam_data.append(b.y.z); cam_data.append(o.y)
	cam_data.append(b.z.x); cam_data.append(b.z.y); cam_data.append(b.z.z); cam_data.append(o.z)
	_rd.buffer_update(_camera_buf, 0, 112, cam_data.to_byte_array())

static func _compute_light_final_energy(lgt: Light3D, physical_units: bool, exposure_norm: float) -> float:
	if physical_units:
		if lgt is DirectionalLight3D:
			var lux: float = lgt.light_intensity_lux
			if lux > 0.0:
				return lux * lgt.light_energy * exposure_norm
	return lgt.light_energy * exposure_norm

# ── Boilerplate ──────────────────────────────────────────────────────────────
func _extract_section(src: String, start_tag: String, end_tag: String) -> String:
	var start_idx := src.find(start_tag)
	if start_idx == -1:
		return ""
	start_idx = src.find("\n", start_idx) + 1
	if end_tag.is_empty():
		return src.substr(start_idx).strip_edges()
	var end_idx := src.find(end_tag, start_idx)
	if end_idx == -1:
		return src.substr(start_idx).strip_edges()
	return src.substr(start_idx, end_idx - start_idx).strip_edges()

func _notification(what: int) -> void:
	if what == NOTIFICATION_PREDELETE:
		if instance == self:
			instance = null
		if _probes_buf.is_valid():    _rd.free_rid(_probes_buf)
		if _c1_probes_buf.is_valid(): _rd.free_rid(_c1_probes_buf)
		if _c2_probes_buf.is_valid(): _rd.free_rid(_c2_probes_buf)
		if _camera_buf.is_valid():    _rd.free_rid(_camera_buf)
		if _light_buf.is_valid():     _rd.free_rid(_light_buf)
		if _sampler.is_valid():       _rd.free_rid(_sampler)
		if _repeat_sampler.is_valid():_rd.free_rid(_repeat_sampler)
		if _rt_pipeline.is_valid():   _rd.free_rid(_rt_pipeline)
		if _rt_shader.is_valid():     _rd.free_rid(_rt_shader)
		if _cache_pipeline.is_valid():_rd.free_rid(_cache_pipeline)
		if _cache_shader.is_valid():  _rd.free_rid(_cache_shader)
		if _screen_cache_tex.is_valid():  _rd.free_rid(_screen_cache_tex)
		if _fallback_cache_tex.is_valid():_rd.free_rid(_fallback_cache_tex)
		if _apply_pipeline.is_valid():_rd.free_rid(_apply_pipeline)
		if _apply_shader.is_valid():  _rd.free_rid(_apply_shader)
