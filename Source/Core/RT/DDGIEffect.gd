@tool
class_name DDGIEffect
extends CompositorEffect

# Four cascades with doubling probe spacing (1×, 2×, 4×, 8×):
#   C0: probe_spacing × 1  (fine near-camera detail)
#   C1: probe_spacing × 2
#   C2: probe_spacing × 4
#   C3: probe_spacing × 8  (wide world coverage)
#
# With probe_spacing=2.0 and 8×8×8 grid (7 cells per axis):
#   C0: 14m   C1: 28m   C2: 56m   C3: 112m
@export var probe_spacing:       float = 2.0
@export var max_ray_dist:        float = 50.0
@export var hysteresis:          float = 0.97   # fraction of old atlas to keep
@export var depth_sharpness:     float = 50.0   # Chebyshev pow() exponent
@export var energy_preservation: float = 0.85   # indirect light attenuation
@export var normal_bias:         float = 0.25   # world-units normal bias
@export var view_bias:           float = 0.25   # world-units view-direction bias (RTXGI surface bias)
# Absorbs (surface_albedo / PI) — tune per scene:
#   0.3 = mid-grey    0.1 = bright outdoor    0.5+ = dark interior
@export var ddgi_strength:       float = 0.3

# ── Debug ─────────────────────────────────────────────────────────────────────
@export var debug_show_probes:  bool  = false
@export var debug_probe_radius: float = 6.0

var _rd: RenderingDevice

# ── RT pipeline ───────────────────────────────────────────────────────────────
var _rt_shader:    RID
var _rt_pipeline:  RID

# ── Atlas update pipelines ────────────────────────────────────────────────────
var _irr_update_shader:   RID
var _irr_update_pipeline: RID
var _dep_update_shader:   RID
var _dep_update_pipeline: RID

# ── Apply pipeline ────────────────────────────────────────────────────────────
var _apply_shader:   RID
var _apply_pipeline: RID

# ── Debug pipeline ────────────────────────────────────────────────────────────
var _debug_shader:   RID
var _debug_pipeline: RID

# ── Per-cascade GPU resources (use untyped Array to avoid GDScript resize bugs)
var _ray_bufs:    Array = []   # SSBO: TOTAL_PROBES × RAYS_PER_PROBE × 2 × vec4
var _irr_atlases: Array = []   # rgba16f  IRR_ATLAS_W × IRR_ATLAS_H
var _dep_atlases: Array = []   # rg16f    DEP_ATLAS_W × DEP_ATLAS_H

# ── Shared GPU resources ───────────────────────────────────────────────────────
var _light_buf:       RID   # 48 B (dir+energy, color+sky_mult, ambient)
var _ddgi_params_buf: RID   # 80 B: 4 scalar floats + 4 × vec4 cascade descriptors
var _sampler:         RID
var _repeat_sampler:  RID

# ── Grid / atlas constants — must match ALL shader #defines ───────────────────
const GRID_X         := 8
const GRID_Y         := 8
const GRID_Z         := 8
const RAYS_PER_PROBE := 128
const TOTAL_PROBES   := GRID_X * GRID_Y * GRID_Z   # 512
const ATLAS_COLS     := GRID_X * GRID_Z             # 64

const IRR_PROBE_SIDE := 16
const IRR_TILE_STRIDE := IRR_PROBE_SIDE + 2            # 18
const IRR_ATLAS_W    := ATLAS_COLS * IRR_TILE_STRIDE   # 1152
const IRR_ATLAS_H    := GRID_Y * IRR_TILE_STRIDE       # 144

const DEP_PROBE_SIDE := 32
const DEP_TILE_STRIDE := DEP_PROBE_SIDE + 2            # 18   ← should be 34
const DEP_ATLAS_W    := ATLAS_COLS * DEP_TILE_STRIDE   # 1152 ← should be 2176
const DEP_ATLAS_H    := GRID_Y * DEP_TILE_STRIDE       # 72   ← should be 272

const NUM_CASCADES   := 4

# ── Probe snap state — reset hysteresis when grid origin moves a whole cell ───
var _last_origins:  Array = [Vector3(INF,INF,INF), Vector3(INF,INF,INF), Vector3(INF,INF,INF), Vector3(INF,INF,INF)]
var _fresh_frames:  Array = [0, 0, 0, 0]   # remaining frames of fast-convergence

# ─────────────────────────────────────────────────────────────────────────────

func _init() -> void:
	effect_callback_type = CompositorEffect.EFFECT_CALLBACK_TYPE_POST_TRANSPARENT
	access_resolved_color  = true
	access_resolved_depth  = true
	needs_normal_roughness = true

	_rd = RenderingServer.get_rendering_device()
	if not _rd.has_feature(RenderingDevice.SUPPORTS_RAYTRACING_PIPELINE):
		push_error("[DDGI] Hardware raytracing not supported.")
		return

	_build_rt_pipeline()
	_build_irr_update_pipeline()
	_build_dep_update_pipeline()
	_build_apply_pipeline()
	_build_debug_pipeline()
	_build_persistent_resources()

# ── Pipeline builders ─────────────────────────────────────────────────────────

func _build_rt_pipeline() -> void:
	var src := FileAccess.get_file_as_string("res://Shaders/ddgi_update.glsl")
	if src.is_empty(): push_error("[DDGI] Could not load ddgi_update.glsl"); return

	var ss := RDShaderSource.new()
	ss.source_raygen      = _extract_section(src, "#[raygen]",      "#[miss]")
	ss.source_miss        = _extract_section(src, "#[miss]",        "#[closest_hit]")
	ss.source_closest_hit = _extract_section(src, "#[closest_hit]", "")

	var spirv := _rd.shader_compile_spirv_from_source(ss)
	for pair in [[RenderingDevice.SHADER_STAGE_RAYGEN, "raygen"],
				 [RenderingDevice.SHADER_STAGE_MISS,   "miss"],
				 [RenderingDevice.SHADER_STAGE_CLOSEST_HIT, "chit"]]:
		var err := spirv.get_stage_compile_error(pair[0])
		if not err.is_empty(): push_error("[DDGI] ", pair[1], ": ", err)

	_rt_shader   = _rd.shader_create_from_spirv(spirv)
	_rt_pipeline = _rd.raytracing_pipeline_create(_rt_shader)
	print("[DDGI] RT pipeline valid: ", _rt_pipeline.is_valid())

func _build_irr_update_pipeline() -> void:
	var src := FileAccess.get_file_as_string("res://Shaders/ddgi_irr_update.glsl")
	if src.is_empty(): push_error("[DDGI] Could not load ddgi_irr_update.glsl"); return

	var ss := RDShaderSource.new()
	ss.source_compute = _extract_section(src, "#[compute]", "")
	var spirv := _rd.shader_compile_spirv_from_source(ss)
	var err := spirv.get_stage_compile_error(RenderingDevice.SHADER_STAGE_COMPUTE)
	if not err.is_empty():
		push_error("[DDGI] irr_update: ", err)
		return

	_irr_update_shader   = _rd.shader_create_from_spirv(spirv)
	_irr_update_pipeline = _rd.compute_pipeline_create(_irr_update_shader)
	print("[DDGI] Irr-update pipeline valid: ", _irr_update_pipeline.is_valid())

func _build_dep_update_pipeline() -> void:
	var src := FileAccess.get_file_as_string("res://Shaders/ddgi_dep_update.glsl")
	if src.is_empty(): push_error("[DDGI] Could not load ddgi_dep_update.glsl"); return

	var ss := RDShaderSource.new()
	ss.source_compute = _extract_section(src, "#[compute]", "")
	var spirv := _rd.shader_compile_spirv_from_source(ss)
	var err := spirv.get_stage_compile_error(RenderingDevice.SHADER_STAGE_COMPUTE)
	if not err.is_empty(): push_error("[DDGI] dep_update: ", err); return

	_dep_update_shader   = _rd.shader_create_from_spirv(spirv)
	_dep_update_pipeline = _rd.compute_pipeline_create(_dep_update_shader)
	print("[DDGI] Dep-update pipeline valid: ", _dep_update_pipeline.is_valid())

func _build_apply_pipeline() -> void:
	var src := FileAccess.get_file_as_string("res://Shaders/ddgi_apply.glsl")
	if src.is_empty(): push_error("[DDGI] Could not load ddgi_apply.glsl"); return

	var ss := RDShaderSource.new()
	ss.source_compute = _extract_section(src, "#[compute]", "")
	var spirv := _rd.shader_compile_spirv_from_source(ss)
	var err := spirv.get_stage_compile_error(RenderingDevice.SHADER_STAGE_COMPUTE)
	if not err.is_empty(): push_error("[DDGI] apply: ", err); return

	_apply_shader   = _rd.shader_create_from_spirv(spirv)
	_apply_pipeline = _rd.compute_pipeline_create(_apply_shader)
	print("[DDGI] Apply pipeline valid: ", _apply_pipeline.is_valid())

func _build_debug_pipeline() -> void:
	var src := FileAccess.get_file_as_string("res://Shaders/ddgi_debug.glsl")
	if src.is_empty():
		push_warning("[DDGI] ddgi_debug.glsl not found — probe visualization disabled.")
		return

	var ss := RDShaderSource.new()
	ss.source_compute = _extract_section(src, "#[compute]", "")
	var spirv := _rd.shader_compile_spirv_from_source(ss)
	var err := spirv.get_stage_compile_error(RenderingDevice.SHADER_STAGE_COMPUTE)
	if not err.is_empty(): push_error("[DDGI] debug: ", err); return

	_debug_shader   = _rd.shader_create_from_spirv(spirv)
	_debug_pipeline = _rd.compute_pipeline_create(_debug_shader)
	print("[DDGI] Debug pipeline valid: ", _debug_pipeline.is_valid())

func _build_persistent_resources() -> void:
	var ss_state := RDSamplerState.new()
	ss_state.min_filter = RenderingDevice.SAMPLER_FILTER_LINEAR
	ss_state.mag_filter = RenderingDevice.SAMPLER_FILTER_LINEAR
	_sampler = _rd.sampler_create(ss_state)

	var ss_rep := RDSamplerState.new()
	ss_rep.min_filter = RenderingDevice.SAMPLER_FILTER_LINEAR
	ss_rep.mag_filter = RenderingDevice.SAMPLER_FILTER_LINEAR
	ss_rep.repeat_u   = RenderingDevice.SAMPLER_REPEAT_MODE_REPEAT
	ss_rep.repeat_v   = RenderingDevice.SAMPLER_REPEAT_MODE_REPEAT
	_repeat_sampler = _rd.sampler_create(ss_rep)

	# Use append() — typed Array resize() is unreliable for RID in GDScript 4
	_ray_bufs.clear()
	_irr_atlases.clear()
	_dep_atlases.clear()

	for c in NUM_CASCADES:
		# Ray buffer: 512 probes × 128 rays × 2 vec4s × 16 bytes = 2 097 152 B
		_ray_bufs.append(_rd.storage_buffer_create(TOTAL_PROBES * RAYS_PER_PROBE * 2 * 16))

		# Irradiance atlas: rgba16f  1024 × 128
		var fmt_irr := RDTextureFormat.new()
		fmt_irr.format     = RenderingDevice.DATA_FORMAT_R16G16B16A16_SFLOAT
		fmt_irr.width      = IRR_ATLAS_W
		fmt_irr.height     = IRR_ATLAS_H
		fmt_irr.usage_bits = (RenderingDevice.TEXTURE_USAGE_STORAGE_BIT |
							  RenderingDevice.TEXTURE_USAGE_SAMPLING_BIT |
							  RenderingDevice.TEXTURE_USAGE_CAN_COPY_FROM_BIT)
		_irr_atlases.append(_rd.texture_create(fmt_irr, RDTextureView.new()))

		# Depth atlas: rg16f  2048 × 256
		var fmt_dep := RDTextureFormat.new()
		fmt_dep.format     = RenderingDevice.DATA_FORMAT_R16G16_SFLOAT
		fmt_dep.width      = DEP_ATLAS_W
		fmt_dep.height     = DEP_ATLAS_H
		fmt_dep.usage_bits = (RenderingDevice.TEXTURE_USAGE_STORAGE_BIT |
							  RenderingDevice.TEXTURE_USAGE_SAMPLING_BIT |
							  RenderingDevice.TEXTURE_USAGE_CAN_COPY_FROM_BIT)
		_dep_atlases.append(_rd.texture_create(fmt_dep, RDTextureView.new()))

	# Light SSBO: 48 B
	_light_buf = _rd.storage_buffer_create(48)
	# Params SSBO: 80 B — 4 scalars (16 B) + 4 × vec4 cascade descriptors (64 B)
	_ddgi_params_buf = _rd.storage_buffer_create(80)

# ── Render callback ───────────────────────────────────────────────────────────

func _render_callback(_p_callback_type: int, p_render_data: RenderData) -> void:
	var sm := RTScene.instance
	if sm == null: return

	sm.tick()
	if not sm.is_ready: return

	# Bail if any pipeline or resource is not ready
	if not (_rt_pipeline.is_valid() and _irr_update_pipeline.is_valid()
			and _dep_update_pipeline.is_valid() and _apply_pipeline.is_valid()):
		return
	if _ray_bufs.size() < NUM_CASCADES or _irr_atlases.size() < NUM_CASCADES: return
	if not sm.mat_inst_ssbo.is_valid() or not sm.mat_surf_ssbo.is_valid(): return
	if not sm.uv_ssbo.is_valid() or sm.texture_rids.is_empty(): return
	if not sm.sky_texture_rid.is_valid(): return
	if not sm.geom_vertex_ssbo.is_valid(): return

	_rd = RenderingServer.get_rendering_device()

	# ── Camera ────────────────────────────────────────────────────────────────
	var scene_buffers := p_render_data.get_render_scene_buffers() as RenderSceneBuffersRD
	var scene_data    := p_render_data.get_render_scene_data()    as RenderSceneDataRD
	if scene_buffers == null or scene_data == null: return

	var size     := scene_buffers.get_internal_size()
	var inv_proj : Projection  = scene_data.get_cam_projection().inverse()
	var proj     : Projection  = scene_data.get_cam_projection()
	var inv_view : Transform3D = scene_data.get_cam_transform()
	var b := inv_view.basis
	var o := inv_view.origin

	# Cascade spacings: 1×, 2×, 4×, 8× probe_spacing
	var spacings: Array = []
	var origins:  Array = []
	for c in NUM_CASCADES:
		var sp := probe_spacing * pow(2.0, float(c))
		spacings.append(sp)
		var ori := _compute_grid_origin(o, sp)
		origins.append(ori)

		# Detect grid snap: if origin moved a full probe cell, reset hysteresis
		var last: Vector3 = _last_origins[c]
		if last.x >= 1e38 or ori.distance_to(last) > sp * 0.5:
			_fresh_frames[c] = 5   # fast-converge for 5 frames after snap
		_last_origins[c] = ori

	# ── Light buffer ──────────────────────────────────────────────────────────
	var dl    := sm.directional_light
	var l_dir := Vector3(0, 1, 0)
	var l_nrg := 1.0
	var l_col := Vector3(1, 1, 1)
	if dl != null:
		l_dir = dl.global_transform.basis.z.normalized()
		l_nrg = dl.light_energy
		l_col = Vector3(dl.light_color.r, dl.light_color.g, dl.light_color.b)
	var amb := sm.ambient_light
	_rd.buffer_update(_light_buf, 0, 48, PackedFloat32Array([
		l_dir.x, l_dir.y, l_dir.z, l_nrg,
		l_col.x, l_col.y, l_col.z, sm.sky_energy_multiplier,
		amb.x, amb.y, amb.z, 0.0
	]).to_byte_array())

	# ── Params SSBO: scalars + all 4 cascade grid descriptors ────────────────
	_rd.buffer_update(_ddgi_params_buf, 0, 80, PackedFloat32Array([
		ddgi_strength, energy_preservation, normal_bias, view_bias,
		origins[0].x, origins[0].y, origins[0].z, spacings[0],
		origins[1].x, origins[1].y, origins[1].z, spacings[1],
		origins[2].x, origins[2].y, origins[2].z, spacings[2],
		origins[3].x, origins[3].y, origins[3].z, spacings[3],
	]).to_byte_array())

	# ── Per-cascade: RT + irr_update + dep_update ─────────────────────────────
	for c in NUM_CASCADES:
		var c_origin:  Vector3 = origins[c]
		var c_spacing: float   = spacings[c]

		# Fast-converge on probe snap: drop hysteresis to 0 for a few frames
		var eff_hysteresis := 0.0 if _fresh_frames[c] > 0 else hysteresis
		if _fresh_frames[c] > 0:
			_fresh_frames[c] -= 1

		# ── Pass 1: RT — probe ray cast ───────────────────────────────────────
		# Push constant: 48 bytes = 12 floats.
		# First 32 B: this cascade's grid (same as old layout).
		# Last 16 B: C0 irradiance grid for multi-bounce indirect.
		var rt_push := PackedFloat32Array([
			c_origin.x, c_origin.y, c_origin.z, c_spacing,
			float(Engine.get_frames_drawn()), max_ray_dist, 0.0, 0.0,
			origins[0].x, origins[0].y, origins[0].z, spacings[0],
		])
		assert(rt_push.size() == 12)  # 48 bytes

		var u_tlas  := _make_uniform(RenderingDevice.UNIFORM_TYPE_ACCELERATION_STRUCTURE, 0, [sm.tlas])
		var u_ray_w := _make_uniform(RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER,         1, [_ray_bufs[c]])
		var u_sky   := _make_uniform(RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE,   2, [_sampler, sm.sky_texture_rid])
		var u_lrt   := _make_uniform(RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER,         3, [_light_buf])
		var u_mi    := _make_uniform(RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER,         4, [sm.mat_inst_ssbo])
		var u_ms    := _make_uniform(RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER,         5, [sm.mat_surf_ssbo])
		var u_gi    := _make_uniform(RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER,         6, [sm.geom_instance_ssbo])
		var u_uv    := _make_uniform(RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER,         7, [sm.uv_ssbo])
		var u_gv    := _make_uniform(RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER,         9, [sm.geom_vertex_ssbo])
		# C0 irradiance atlas — provides previous-frame indirect for multi-bounce
		var u_irr_indirect := _make_uniform(RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 10, [_sampler, _irr_atlases[0]])

		var u_tex := RDUniform.new()
		u_tex.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
		u_tex.binding = 8
		for tex in sm.texture_rids:
			u_tex.add_id(_repeat_sampler); u_tex.add_id(tex)

		var rt_set := _rd.uniform_set_create(
			[u_tlas, u_ray_w, u_sky, u_lrt, u_mi, u_ms, u_gi, u_uv, u_tex, u_gv, u_irr_indirect],
			_rt_shader, 0)

		var rt_list := _rd.raytracing_list_begin()
		_rd.raytracing_list_bind_raytracing_pipeline(rt_list, _rt_pipeline)
		_rd.raytracing_list_bind_uniform_set(rt_list, rt_set, 0)
		_rd.raytracing_list_set_push_constant(rt_list, rt_push.to_byte_array(), 48)
		_rd.raytracing_list_trace_rays(rt_list, RAYS_PER_PROBE, TOTAL_PROBES)
		_rd.raytracing_list_end()
		_rd.free_rid(rt_set)

		# ── Pass 2a: irradiance atlas update ──────────────────────────────────
		var irr_push := PackedFloat32Array([eff_hysteresis, 0.0, 0.0, 0.0])
		var irr_set  := _rd.uniform_set_create([
			_make_uniform(RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER, 0, [_ray_bufs[c]]),
			_make_uniform(RenderingDevice.UNIFORM_TYPE_IMAGE,          1, [_irr_atlases[c]]),
		], _irr_update_shader, 0)

		var irr_list := _rd.compute_list_begin()
		_rd.compute_list_bind_compute_pipeline(irr_list, _irr_update_pipeline)
		_rd.compute_list_bind_uniform_set(irr_list, irr_set, 0)
		_rd.compute_list_set_push_constant(irr_list, irr_push.to_byte_array(), 16)
		_rd.compute_list_dispatch(irr_list, TOTAL_PROBES, 1, 1)
		_rd.compute_list_end()
		_rd.free_rid(irr_set)

		# ── Pass 2b: depth atlas update ───────────────────────────────────────
		var dep_push := PackedFloat32Array([eff_hysteresis, depth_sharpness, 0.0, 0.0])
		var dep_set  := _rd.uniform_set_create([
			_make_uniform(RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER, 0, [_ray_bufs[c]]),
			_make_uniform(RenderingDevice.UNIFORM_TYPE_IMAGE,          1, [_dep_atlases[c]]),
		], _dep_update_shader, 0)

		var dep_list := _rd.compute_list_begin()
		_rd.compute_list_bind_compute_pipeline(dep_list, _dep_update_pipeline)
		_rd.compute_list_bind_uniform_set(dep_list, dep_set, 0)
		_rd.compute_list_set_push_constant(dep_list, dep_push.to_byte_array(), 16)
		_rd.compute_list_dispatch(dep_list, TOTAL_PROBES, 1, 1)
		_rd.compute_list_end()
		_rd.free_rid(dep_set)

	# ── Pass 3: apply all cascades to the colour buffer ───────────────────────
	# Push constant: 28 floats = 112 bytes (inv_proj + inv_view rows).
	var apply_push := PackedFloat32Array()
	for i in range(4):
		var col: Vector4 = inv_proj[i]
		apply_push.append(col.x); apply_push.append(col.y)
		apply_push.append(col.z); apply_push.append(col.w)
	apply_push.append(b.x.x); apply_push.append(b.x.y); apply_push.append(b.x.z); apply_push.append(o.x)
	apply_push.append(b.y.x); apply_push.append(b.y.y); apply_push.append(b.y.z); apply_push.append(o.y)
	apply_push.append(b.z.x); apply_push.append(b.z.y); apply_push.append(b.z.z); apply_push.append(o.z)
	assert(apply_push.size() == 28)  # 112 bytes

	var color_tex  : RID = scene_buffers.get_color_texture()
	var depth_tex  : RID = scene_buffers.get_depth_texture()
	var normal_tex : RID = scene_buffers.get_texture("forward_clustered", "normal_roughness")

	var apply_set := _rd.uniform_set_create([
		_make_uniform(RenderingDevice.UNIFORM_TYPE_IMAGE,                0, [color_tex]),
		_make_uniform(RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 1, [_sampler, depth_tex]),
		_make_uniform(RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 2, [_sampler, normal_tex]),
		_make_uniform(RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 3, [_sampler, _irr_atlases[0]]),
		_make_uniform(RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 4, [_sampler, _dep_atlases[0]]),
		_make_uniform(RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER,       5, [_ddgi_params_buf]),
		_make_uniform(RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 6, [_sampler, _irr_atlases[1]]),
		_make_uniform(RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 7, [_sampler, _dep_atlases[1]]),
		_make_uniform(RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 8, [_sampler, _irr_atlases[2]]),
		_make_uniform(RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 9, [_sampler, _dep_atlases[2]]),
		_make_uniform(RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 10, [_sampler, _irr_atlases[3]]),
		_make_uniform(RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 11, [_sampler, _dep_atlases[3]]),
	], _apply_shader, 0)

	var apply_list := _rd.compute_list_begin()
	_rd.compute_list_bind_compute_pipeline(apply_list, _apply_pipeline)
	_rd.compute_list_bind_uniform_set(apply_list, apply_set, 0)
	_rd.compute_list_set_push_constant(apply_list, apply_push.to_byte_array(), 112)
	_rd.compute_list_dispatch(apply_list, int(ceil(size.x / 8.0)), int(ceil(size.y / 8.0)), 1)
	_rd.compute_list_end()
	_rd.free_rid(apply_set)

	# ── Pass 4 (optional): debug probe visualisation ──────────────────────────
	if debug_show_probes and _debug_pipeline.is_valid():
		# Push constant: forward proj (16f) + inv_view rows (12f) + screen+radius (4f)
		# = 32 floats = 128 bytes.
		var dbg_push := PackedFloat32Array()
		for i in range(4):
			var col: Vector4 = proj[i]
			dbg_push.append(col.x); dbg_push.append(col.y)
			dbg_push.append(col.z); dbg_push.append(col.w)
		dbg_push.append(b.x.x); dbg_push.append(b.x.y); dbg_push.append(b.x.z); dbg_push.append(o.x)
		dbg_push.append(b.y.x); dbg_push.append(b.y.y); dbg_push.append(b.y.z); dbg_push.append(o.y)
		dbg_push.append(b.z.x); dbg_push.append(b.z.y); dbg_push.append(b.z.z); dbg_push.append(o.z)
		dbg_push.append(float(size.x)); dbg_push.append(float(size.y))
		dbg_push.append(debug_probe_radius); dbg_push.append(0.0)
		assert(dbg_push.size() == 32)  # 128 bytes

		var dbg_set := _rd.uniform_set_create([
			_make_uniform(RenderingDevice.UNIFORM_TYPE_IMAGE,                0, [color_tex]),
			_make_uniform(RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER,       1, [_ddgi_params_buf]),
			_make_uniform(RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 2, [_sampler, _irr_atlases[0]]),
			_make_uniform(RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 3, [_sampler, _irr_atlases[1]]),
			_make_uniform(RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 4, [_sampler, _irr_atlases[2]]),
			_make_uniform(RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 5, [_sampler, _irr_atlases[3]]),
			_make_uniform(RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 6, [_sampler, depth_tex]),
		], _debug_shader, 0)

		var dbg_list := _rd.compute_list_begin()
		_rd.compute_list_bind_compute_pipeline(dbg_list, _debug_pipeline)
		_rd.compute_list_bind_uniform_set(dbg_list, dbg_set, 0)
		_rd.compute_list_set_push_constant(dbg_list, dbg_push.to_byte_array(), 128)
		_rd.compute_list_dispatch(dbg_list, NUM_CASCADES * TOTAL_PROBES, 1, 1)
		_rd.compute_list_end()
		_rd.free_rid(dbg_set)

# ── Helpers ───────────────────────────────────────────────────────────────────

func _make_uniform(type: int, binding: int, rids: Array) -> RDUniform:
	var u := RDUniform.new()
	u.uniform_type = type
	u.binding      = binding
	for r in rids:
		u.add_id(r)
	return u

# Centre the probe grid on cam_pos, snapped to probe_spacing so the atlas
# only goes stale on whole-probe-cell camera moves.
func _compute_grid_origin(cam_pos: Vector3, spacing: float) -> Vector3:
	var half_ext := Vector3(
		float(GRID_X - 1) * spacing * 0.5,
		float(GRID_Y - 1) * spacing * 0.5,
		float(GRID_Z - 1) * spacing * 0.5
	)
	return Vector3(
		snappedf(cam_pos.x, spacing),
		snappedf(cam_pos.y, spacing),
		snappedf(cam_pos.z, spacing)
	) - half_ext

func _extract_section(src: String, start_tag: String, end_tag: String) -> String:
	var si := src.find(start_tag)
	if si == -1: return ""
	si = src.find("\n", si) + 1
	if end_tag.is_empty(): return src.substr(si).strip_edges()
	var ei := src.find(end_tag, si)
	if ei == -1: return src.substr(si).strip_edges()
	return src.substr(si, ei - si).strip_edges()

func _notification(what: int) -> void:
	if what == NOTIFICATION_PREDELETE:
		for rid in _ray_bufs + _irr_atlases + _dep_atlases:
			if (rid as RID).is_valid(): _rd.free_rid(rid)
		for rid in [_light_buf, _ddgi_params_buf, _sampler, _repeat_sampler,
					_rt_pipeline, _rt_shader,
					_irr_update_pipeline, _irr_update_shader,
					_dep_update_pipeline, _dep_update_shader,
					_apply_pipeline, _apply_shader,
					_debug_pipeline, _debug_shader]:
			if rid.is_valid(): _rd.free_rid(rid)
