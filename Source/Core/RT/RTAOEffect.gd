@tool
class_name RTAOEffect
extends CompositorEffect

enum AOFilter { NONE, ATROUS }

@export var ao_radius:   float = 1.5
@export var ao_samples:  int   = 4     # stub supports max 4; increase when hemisphere sampling is upgraded
@export var ao_strength: float = 0.6
# NONE  — skip all filtering, apply uses the raw raygen mask directly.
# ATROUS — 3-pass à-trous wavelet denoise between raygen and apply.
@export var ao_filter: AOFilter = AOFilter.ATROUS
const AO_RESOLUTION_SCALE: int = 1

var _rd: RenderingDevice

# RT pipeline
var _rt_shader:   RID
var _rt_pipeline: RID

# À-trous wavelet denoiser — runs 3 ping-pong passes between RT and apply
# with increasing stride (1, 2, 4) to cover a 17×17 effective kernel cheaply.
# Lets ao_samples stay low (1–2) without visible noise.
var _atrous_shader:   RID
var _atrous_pipeline: RID

# Compute apply pipeline
var _comp_shader:   RID
var _comp_pipeline: RID

# Persistent resources
var _ao_mask_tex:      RID  # raygen output, ping-pong partner A
var _ao_mask_ping_tex: RID  # ping-pong partner B for à-trous passes
var _sampler:          RID
var _last_size:        Vector2i = Vector2i(0, 0)

func _init() -> void:
	effect_callback_type = EFFECT_CALLBACK_TYPE_POST_TRANSPARENT
	access_resolved_color  = true
	access_resolved_depth  = true
	needs_normal_roughness = true

	_rd = RenderingServer.get_rendering_device()

	if not _rd.has_feature(RenderingDevice.SUPPORTS_RAYTRACING_PIPELINE):
		push_error("[RTAO] Hardware raytracing not supported.")
		return

	_build_rt_pipeline()
	_build_atrous_pipeline()
	_build_compute_pipeline()
	_build_persistent_resources()

func _build_rt_pipeline() -> void:
	var full_src := FileAccess.get_file_as_string("res://Shaders/rtao.glsl")
	if full_src.is_empty():
		push_error("[RTAO] Could not load rtao.glsl")
		return

	var shader_source := RDShaderSource.new()
	shader_source.source_raygen      = _extract_section(full_src, "#[raygen]",      "#[miss]")
	shader_source.source_miss        = _extract_section(full_src, "#[miss]",        "#[closest_hit]")
	shader_source.source_closest_hit = _extract_section(full_src, "#[closest_hit]", "")

	var spirv := _rd.shader_compile_spirv_from_source(shader_source)
	var err_rg := spirv.get_stage_compile_error(RenderingDevice.SHADER_STAGE_RAYGEN)
	var err_ms := spirv.get_stage_compile_error(RenderingDevice.SHADER_STAGE_MISS)
	var err_ch := spirv.get_stage_compile_error(RenderingDevice.SHADER_STAGE_CLOSEST_HIT)
	if not err_rg.is_empty(): push_error("[RTAO] raygen: ", err_rg)
	if not err_ms.is_empty(): push_error("[RTAO] miss: ",   err_ms)
	if not err_ch.is_empty(): push_error("[RTAO] chit: ",   err_ch)

	_rt_shader   = _rd.shader_create_from_spirv(spirv)
	_rt_pipeline = _rd.raytracing_pipeline_create(_rt_shader)
	print("[RTAO] RT pipeline valid: ", _rt_pipeline.is_valid())

func _build_atrous_pipeline() -> void:
	var src := FileAccess.get_file_as_string("res://Shaders/rtao_atrous.glsl")
	if src.is_empty():
		push_error("[RTAO] Could not load rtao_atrous.glsl")
		return

	var shader_source := RDShaderSource.new()
	shader_source.source_compute = _extract_section(src, "#[compute]", "")

	var spirv := _rd.shader_compile_spirv_from_source(shader_source)
	var err := spirv.get_stage_compile_error(RenderingDevice.SHADER_STAGE_COMPUTE)
	if not err.is_empty():
		push_error("[RTAO] atrous error: ", err)
		return

	_atrous_shader   = _rd.shader_create_from_spirv(spirv)
	_atrous_pipeline = _rd.compute_pipeline_create(_atrous_shader)
	print("[RTAO] Atrous pipeline valid: ", _atrous_pipeline.is_valid())

func _build_compute_pipeline() -> void:
	var src := FileAccess.get_file_as_string("res://Shaders/apply_rtao.glsl")
	if src.is_empty():
		push_error("[RTAO] Could not load apply_rtao.glsl")
		return

	var shader_source := RDShaderSource.new()
	shader_source.source_compute = _extract_section(src, "#[compute]", "")

	var spirv := _rd.shader_compile_spirv_from_source(shader_source)
	var err := spirv.get_stage_compile_error(RenderingDevice.SHADER_STAGE_COMPUTE)
	if not err.is_empty():
		push_error("[RTAO] compute error: ", err)
		return

	_comp_shader   = _rd.shader_create_from_spirv(spirv)
	_comp_pipeline = _rd.compute_pipeline_create(_comp_shader)
	print("[RTAO] Compute pipeline valid: ", _comp_pipeline.is_valid())

func _build_persistent_resources() -> void:
	var ss := RDSamplerState.new()
	ss.min_filter = RenderingDevice.SAMPLER_FILTER_LINEAR
	ss.mag_filter = RenderingDevice.SAMPLER_FILTER_LINEAR
	_sampler = _rd.sampler_create(ss)

func _ensure_ao_mask(size: Vector2i) -> void:
	if _ao_mask_tex.is_valid() and _last_size == size:
		return
	if _ao_mask_tex.is_valid():      _rd.free_rid(_ao_mask_tex)
	if _ao_mask_ping_tex.is_valid(): _rd.free_rid(_ao_mask_ping_tex)

	var fmt := RDTextureFormat.new()
	fmt.format     = RenderingDevice.DATA_FORMAT_R16G16B16A16_SFLOAT
	fmt.width      = size.x
	fmt.height     = size.y
	fmt.usage_bits = (
		RenderingDevice.TEXTURE_USAGE_STORAGE_BIT |
		RenderingDevice.TEXTURE_USAGE_SAMPLING_BIT |
		RenderingDevice.TEXTURE_USAGE_CAN_COPY_FROM_BIT
	)
	_ao_mask_tex      = _rd.texture_create(fmt, RDTextureView.new())
	_ao_mask_ping_tex = _rd.texture_create(fmt, RDTextureView.new())
	_last_size = size

func _render_callback(p_callback_type: int, p_render_data: RenderData) -> void:
	var sm := RTScene.instance
	if sm == null:
		return

	# RTShadowEffect also calls sm.tick() each frame — RTScene guards against
	# double-advancing the state machine, so both calls are safe.
	sm.tick()

	if not sm.is_ready:
		return
	if not _rt_pipeline.is_valid() or not _comp_pipeline.is_valid():
		return

	_rd = RenderingServer.get_rendering_device()

	var scene_buffers := p_render_data.get_render_scene_buffers() as RenderSceneBuffersRD
	var scene_data    := p_render_data.get_render_scene_data()    as RenderSceneDataRD
	if scene_buffers == null or scene_data == null:
		return

	var size := scene_buffers.get_internal_size()
	var rt_size := Vector2i(
		(size.x + AO_RESOLUTION_SCALE - 1) / AO_RESOLUTION_SCALE,
		(size.y + AO_RESOLUTION_SCALE - 1) / AO_RESOLUTION_SCALE
	)
	_ensure_ao_mask(rt_size)

	# Build push constant — 128 bytes (32 floats), same layout as shadows minus light
	var proj: Projection  = scene_data.get_cam_projection()
	var inv_proj: Projection  = proj.inverse()
	var inv_view: Transform3D = scene_data.get_cam_transform()
	var b := inv_view.basis
	var o := inv_view.origin

	var push := PackedFloat32Array()
	var c0: Vector4 = inv_proj[0]; push.append(c0.x); push.append(c0.y); push.append(c0.z); push.append(c0.w)
	var c1: Vector4 = inv_proj[1]; push.append(c1.x); push.append(c1.y); push.append(c1.z); push.append(c1.w)
	var c2: Vector4 = inv_proj[2]; push.append(c2.x); push.append(c2.y); push.append(c2.z); push.append(c2.w)
	var c3: Vector4 = inv_proj[3]; push.append(c3.x); push.append(c3.y); push.append(c3.z); push.append(c3.w)
	push.append(b.x.x); push.append(b.x.y); push.append(b.x.z); push.append(o.x)
	push.append(b.y.x); push.append(b.y.y); push.append(b.y.z); push.append(o.y)
	push.append(b.z.x); push.append(b.z.y); push.append(b.z.z); push.append(o.z)
	push.append(float(size.x))
	push.append(float(size.y))
	push.append(ao_radius)
	# Slot 31 is a uint: high 16 bits = ao_samples, low 16 bits = frame_index.
	# Build 31 floats (124 bytes) then append the raw uint bytes to reach 128.
	var push_bytes := push.to_byte_array()
	var frame_uint: int = (ao_samples << 16) | (Engine.get_frames_drawn() & 0xFFFF)
	var uint_bytes := PackedByteArray()
	uint_bytes.resize(4)
	uint_bytes.encode_u32(0, frame_uint)
	push_bytes.append_array(uint_bytes)
	assert(push.size() == 31 and push_bytes.size() == 128)

	var depth_tex:  RID = scene_buffers.get_depth_texture()
	var normal_tex: RID = scene_buffers.get_texture("forward_clustered", "normal_roughness")

	# ── RT pass: write AO mask ────────────────────────────────────────────────
	var u_tlas := RDUniform.new()
	u_tlas.uniform_type = RenderingDevice.UNIFORM_TYPE_ACCELERATION_STRUCTURE
	u_tlas.binding = 0
	u_tlas.add_id(sm.tlas)

	var u_mask := RDUniform.new()
	u_mask.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u_mask.binding = 1
	u_mask.add_id(_ao_mask_tex)

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

	var rt_set := _rd.uniform_set_create(
		[u_tlas, u_mask, u_depth, u_normal], _rt_shader, 0
	)

	var rt_list := _rd.raytracing_list_begin()
	_rd.raytracing_list_bind_raytracing_pipeline(rt_list, _rt_pipeline)
	_rd.raytracing_list_bind_uniform_set(rt_list, rt_set, 0)
	_rd.raytracing_list_set_push_constant(rt_list, push_bytes, push_bytes.size())
	# Dispatch at downsampled resolution; raygen reads full-res G-buffer.
	_rd.raytracing_list_trace_rays(rt_list, rt_size.x, rt_size.y)
	_rd.raytracing_list_end()
	_rd.free_rid(rt_set)

	# ── Filtering (ao_filter export controls the path) ────────────────────────
	# NONE  — raw raygen mask goes straight to apply (noisy but exact).
	# ATROUS — 3-pass à-trous wavelet denoise, ping-pong between mask textures.
	var final_mask := _ao_mask_tex
	if ao_filter == AOFilter.ATROUS and _atrous_pipeline.is_valid():
		var a := _ao_mask_tex
		var c := _ao_mask_ping_tex
		for stride in [1, 2, 4]:
			_run_atrous_pass(a, c, depth_tex, normal_tex, stride, rt_size, size)
			var tmp := a
			a = c
			c = tmp
		final_mask = a

	# ── Compute pass: apply AO to color buffer ────────────────────────────────
	var color_tex: RID = scene_buffers.get_color_texture()

	var u_color := RDUniform.new()
	u_color.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u_color.binding = 0
	u_color.add_id(color_tex)

	var u_mask_read := RDUniform.new()
	u_mask_read.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u_mask_read.binding = 1
	u_mask_read.add_id(final_mask)

	# Bindings 2 & 3: G-buffer normal + depth for the bilateral upsample.
	var u_apply_normal := RDUniform.new()
	u_apply_normal.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u_apply_normal.binding = 2
	u_apply_normal.add_id(_sampler)
	u_apply_normal.add_id(normal_tex)

	var u_apply_depth := RDUniform.new()
	u_apply_depth.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u_apply_depth.binding = 3
	u_apply_depth.add_id(_sampler)
	u_apply_depth.add_id(depth_tex)

	var comp_set := _rd.uniform_set_create(
		[u_color, u_mask_read, u_apply_normal, u_apply_depth], _comp_shader, 0
	)

	# apply_rtao PC: strength, pad, rt_size.x, rt_size.y (16 bytes).
	var comp_push := PackedFloat32Array([
		ao_strength, 0.0,
		float(rt_size.x), float(rt_size.y)
	])
	var comp_push_bytes := comp_push.to_byte_array()

	var comp_list := _rd.compute_list_begin()
	_rd.compute_list_bind_compute_pipeline(comp_list, _comp_pipeline)
	_rd.compute_list_bind_uniform_set(comp_list, comp_set, 0)
	_rd.compute_list_set_push_constant(comp_list, comp_push_bytes, comp_push_bytes.size())
	_rd.compute_list_dispatch(comp_list,
		int(ceil(size.x / 8.0)),
		int(ceil(size.y / 8.0)),
		1
	)
	_rd.compute_list_end()
	_rd.free_rid(comp_set)

func _run_atrous_pass(input_tex: RID, output_tex: RID,
					  depth_tex: RID, normal_tex: RID,
					  stride: int, rt_size: Vector2i, full_size: Vector2i) -> void:
	var u_in := RDUniform.new()
	u_in.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u_in.binding = 0
	u_in.add_id(input_tex)

	var u_out := RDUniform.new()
	u_out.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u_out.binding = 1
	u_out.add_id(output_tex)

	var u_n := RDUniform.new()
	u_n.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u_n.binding = 2
	u_n.add_id(_sampler)
	u_n.add_id(normal_tex)

	var u_d := RDUniform.new()
	u_d.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u_d.binding = 3
	u_d.add_id(_sampler)
	u_d.add_id(depth_tex)

	var set := _rd.uniform_set_create([u_in, u_out, u_n, u_d], _atrous_shader, 0)

	# PC: int stride, int pad, ivec2 full_size — 16 bytes.
	var push := PackedInt32Array([stride, 0, full_size.x, full_size.y])
	var push_bytes := push.to_byte_array()

	var list := _rd.compute_list_begin()
	_rd.compute_list_bind_compute_pipeline(list, _atrous_pipeline)
	_rd.compute_list_bind_uniform_set(list, set, 0)
	_rd.compute_list_set_push_constant(list, push_bytes, push_bytes.size())
	_rd.compute_list_dispatch(list,
		int(ceil(rt_size.x / 8.0)),
		int(ceil(rt_size.y / 8.0)),
		1
	)
	_rd.compute_list_end()
	_rd.free_rid(set)

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
		if _ao_mask_tex.is_valid():      _rd.free_rid(_ao_mask_tex)
		if _ao_mask_ping_tex.is_valid(): _rd.free_rid(_ao_mask_ping_tex)
		if _sampler.is_valid():          _rd.free_rid(_sampler)
		if _rt_pipeline.is_valid():      _rd.free_rid(_rt_pipeline)
		if _rt_shader.is_valid():        _rd.free_rid(_rt_shader)
		if _atrous_pipeline.is_valid():  _rd.free_rid(_atrous_pipeline)
		if _atrous_shader.is_valid():    _rd.free_rid(_atrous_shader)
		if _comp_pipeline.is_valid():    _rd.free_rid(_comp_pipeline)
		if _comp_shader.is_valid():      _rd.free_rid(_comp_shader)
