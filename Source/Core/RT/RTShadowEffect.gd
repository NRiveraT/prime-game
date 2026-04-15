@tool
class_name RTShadowEffect
extends CompositorEffect

@export var shadow_bias: float    = 0.02
@export var max_shadow_dist: float = 200.0
@export var ambient_floor: float  = 0.05

var _rd: RenderingDevice

# RT pipeline
var _rt_shader:   RID
var _rt_pipeline: RID

# Compute blit pipeline
var _comp_shader:   RID
var _comp_pipeline: RID

# Persistent resources
var _shadow_mask_tex: RID
var _light_buf:       RID
var _sampler:         RID
var _last_size:       Vector2i = Vector2i(0, 0)

func _init() -> void:
	effect_callback_type = EFFECT_CALLBACK_TYPE_POST_TRANSPARENT
	access_resolved_color  = true
	access_resolved_depth  = true
	needs_normal_roughness = true
	
	_rd = RenderingServer.get_rendering_device()

	if not _rd.has_feature(RenderingDevice.SUPPORTS_RAYTRACING_PIPELINE):
		push_error("[RTShadows] Hardware raytracing not supported.")
		return

	_build_rt_pipeline()
	_build_compute_pipeline()
	_build_persistent_resources()

func _build_rt_pipeline() -> void:
	var full_src := FileAccess.get_file_as_string("res://Shaders/rt_shadows.glsl")
	if full_src.is_empty():
		push_error("[RTShadows] Could not load rt_shadows.glsl")
		return

	var shader_source := RDShaderSource.new()
	shader_source.source_raygen      = _extract_section(full_src, "#[raygen]",      "#[miss]")
	shader_source.source_miss        = _extract_section(full_src, "#[miss]",        "#[closest_hit]")
	shader_source.source_closest_hit = _extract_section(full_src, "#[closest_hit]", "")

	var spirv := _rd.shader_compile_spirv_from_source(shader_source)
	var err_rg := spirv.get_stage_compile_error(RenderingDevice.SHADER_STAGE_RAYGEN)
	var err_ms := spirv.get_stage_compile_error(RenderingDevice.SHADER_STAGE_MISS)
	var err_ch := spirv.get_stage_compile_error(RenderingDevice.SHADER_STAGE_CLOSEST_HIT)
	if not err_rg.is_empty(): push_error("[RTShadows] raygen: ", err_rg)
	if not err_ms.is_empty(): push_error("[RTShadows] miss: ",   err_ms)
	if not err_ch.is_empty(): push_error("[RTShadows] chit: ",   err_ch)

	_rt_shader   = _rd.shader_create_from_spirv(spirv)
	_rt_pipeline = _rd.raytracing_pipeline_create(_rt_shader)
	print("[RTShadows] RT pipeline valid: ", _rt_pipeline.is_valid())

func _build_compute_pipeline() -> void:
	var src := FileAccess.get_file_as_string("res://Shaders/apply_rt_shadows.glsl")
	if src.is_empty():
		push_error("[RTShadows] Could not load apply_shadows.glsl")
		return

	var compute_src := _extract_section(src, "#[compute]", "")

	var shader_source := RDShaderSource.new()
	shader_source.source_compute = compute_src

	var spirv := _rd.shader_compile_spirv_from_source(shader_source)
	var err := spirv.get_stage_compile_error(RenderingDevice.SHADER_STAGE_COMPUTE)
	if not err.is_empty():
		push_error("[RTShadows] compute error: ", err)
		return

	_comp_shader   = _rd.shader_create_from_spirv(spirv)
	_comp_pipeline = _rd.compute_pipeline_create(_comp_shader)
	print("[RTShadows] Compute pipeline valid: ", _comp_pipeline.is_valid())

func _build_persistent_resources() -> void:
	# Default linear sampler for depth + normal reads
	var ss := RDSamplerState.new()
	ss.min_filter = RenderingDevice.SAMPLER_FILTER_LINEAR
	ss.mag_filter = RenderingDevice.SAMPLER_FILTER_LINEAR
	_sampler = _rd.sampler_create(ss)

	# Light data SSBO — 2x vec4 = 32 bytes
	_light_buf = _rd.storage_buffer_create(32)

func _ensure_shadow_mask(size: Vector2i) -> void:
	if _shadow_mask_tex.is_valid() and _last_size == size:
		return
	if _shadow_mask_tex.is_valid():
		_rd.free_rid(_shadow_mask_tex)

	var fmt := RDTextureFormat.new()
	fmt.format     = RenderingDevice.DATA_FORMAT_R16G16B16A16_SFLOAT  
	fmt.width      = size.x
	fmt.height     = size.y
	fmt.usage_bits = (
		RenderingDevice.TEXTURE_USAGE_STORAGE_BIT |
		RenderingDevice.TEXTURE_USAGE_SAMPLING_BIT |
		RenderingDevice.TEXTURE_USAGE_CAN_COPY_FROM_BIT
	)
	_shadow_mask_tex = _rd.texture_create(fmt, RDTextureView.new())
	_last_size = size

func _render_callback(p_callback_type: int, p_render_data: RenderData) -> void:
	var sm := RTScene.instance
	if sm == null:
		return
	
	sm.tick()
	
	if not sm.is_ready: 

		return  # not safe to trace yet
	
	if not _rt_pipeline.is_valid() or not _comp_pipeline.is_valid() or not sm.is_ready:
		return
	
	_rd = RenderingServer.get_rendering_device()

	var scene_buffers := p_render_data.get_render_scene_buffers() as RenderSceneBuffersRD
	var scene_data    := p_render_data.get_render_scene_data()    as RenderSceneDataRD
	if scene_buffers == null or scene_data == null:
		return

	var size := scene_buffers.get_internal_size()
	_ensure_shadow_mask(size)

	# Update light buffer
	var dl := sm.directional_light
	var light_dir    := Vector3(0.0, 1.0, 0.0)
	var light_energy := 1.0
	var light_color  := Vector3(1.0, 1.0, 1.0)
	if dl != null:
		light_dir    = dl.global_transform.basis.z.normalized()
		light_energy = dl.light_energy
		light_color  = Vector3(dl.light_color.r, dl.light_color.g, dl.light_color.b)

	var light_data := PackedFloat32Array([
	light_dir.x, light_dir.y, light_dir.z, light_energy,
	light_color.x, light_color.y, light_color.z, 0.0
	])
	_rd.buffer_update(_light_buf, 0, 32, light_data.to_byte_array())

	# Build push constant — exactly 128 bytes (32 floats)
	var proj: Projection = scene_data.get_cam_projection()
	var inv_proj: Projection  = proj.inverse()
	var inv_view: Transform3D = scene_data.get_cam_transform()
	var b := inv_view.basis
	var o := inv_view.origin

	var push := PackedFloat32Array()

	# inv_proj column 0
	var c0: Vector4 = inv_proj[0]
	push.append(c0.x); push.append(c0.y); push.append(c0.z); push.append(c0.w)
	# inv_proj column 1
	var c1: Vector4 = inv_proj[1]
	push.append(c1.x); push.append(c1.y); push.append(c1.z); push.append(c1.w)
	# inv_proj column 2
	var c2: Vector4 = inv_proj[2]
	push.append(c2.x); push.append(c2.y); push.append(c2.z); push.append(c2.w)
	# inv_proj column 3
	var c3: Vector4 = inv_proj[3]
	push.append(c3.x); push.append(c3.y); push.append(c3.z); push.append(c3.w)

	# inv_view row 0 (basis.x + origin.x)
	push.append(b.x.x); push.append(b.x.y); push.append(b.x.z); push.append(o.x)
	# inv_view row 1 (basis.y + origin.y)
	push.append(b.y.x); push.append(b.y.y); push.append(b.y.z); push.append(o.y)
	# inv_view row 2 (basis.z + origin.z)
	push.append(b.z.x); push.append(b.z.y); push.append(b.z.z); push.append(o.z)

	# screen size, bias, max dist
	push.append(float(size.x))
	push.append(float(size.y))
	push.append(shadow_bias)
	push.append(max_shadow_dist)

	var push_bytes := push.to_byte_array()
	assert(push.size() == 32 and push_bytes.size() == 128)

	# In _render_callback, right before the RT dispatch
	var cam_xform: Transform3D = scene_data.get_cam_transform()
	#print("[RT] camera world pos: ", cam_xform.origin)
	#print("[RT] light_dir: ", light_dir)

	# Get G-buffer textures
	var depth_tex:  RID = scene_buffers.get_depth_texture()
	var normal_tex: RID = scene_buffers.get_texture("forward_clustered", "normal_roughness")

	# ── RT pass: write shadow mask ────────────────────────────────────────────
	var u_tlas := RDUniform.new()
	u_tlas.uniform_type = RenderingDevice.UNIFORM_TYPE_ACCELERATION_STRUCTURE
	u_tlas.binding = 0
	u_tlas.add_id(sm.tlas)

	var u_mask := RDUniform.new()
	u_mask.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u_mask.binding = 1
	u_mask.add_id(_shadow_mask_tex)

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

	var u_light := RDUniform.new()
	u_light.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u_light.binding = 4
	u_light.add_id(_light_buf)

	var rt_set := _rd.uniform_set_create(
	[u_tlas, u_mask, u_depth, u_normal, u_light], _rt_shader, 0
	)

	var rt_list := _rd.raytracing_list_begin()
	_rd.raytracing_list_bind_raytracing_pipeline(rt_list, _rt_pipeline)
	_rd.raytracing_list_bind_uniform_set(rt_list, rt_set, 0)
	_rd.raytracing_list_set_push_constant(rt_list, push_bytes, push_bytes.size())
	_rd.raytracing_list_trace_rays(rt_list, size.x, size.y)
	_rd.raytracing_list_end()
	
	_rd.free_rid(rt_set)

	# ── Compute pass: apply shadow mask to color buffer ───────────────────────
	var color_tex: RID = scene_buffers.get_color_texture()

	var u_color := RDUniform.new()
	u_color.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u_color.binding = 0
	u_color.add_id(color_tex)

	var u_mask_read := RDUniform.new()
	u_mask_read.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u_mask_read.binding = 1
	u_mask_read.add_id(_shadow_mask_tex)

	var comp_set := _rd.uniform_set_create(
	[u_color, u_mask_read], _comp_shader, 0
	)

	var comp_list := _rd.compute_list_begin()
	_rd.compute_list_bind_compute_pipeline(comp_list, _comp_pipeline)
	_rd.compute_list_bind_uniform_set(comp_list, comp_set, 0)
	_rd.compute_list_dispatch(comp_list,
	int(ceil(size.x / 8.0)),
	int(ceil(size.y / 8.0)),
	1
	)
	_rd.compute_list_end()
	_rd.free_rid(comp_set)

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
		if _shadow_mask_tex.is_valid(): _rd.free_rid(_shadow_mask_tex)
		if _light_buf.is_valid():       _rd.free_rid(_light_buf)
		if _sampler.is_valid():         _rd.free_rid(_sampler)
		if _rt_pipeline.is_valid():     _rd.free_rid(_rt_pipeline)
		if _rt_shader.is_valid():       _rd.free_rid(_rt_shader)
		if _comp_pipeline.is_valid():   _rd.free_rid(_comp_pipeline)
		if _comp_shader.is_valid():     _rd.free_rid(_comp_shader)
