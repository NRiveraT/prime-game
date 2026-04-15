@tool
class_name RTReflectionEffect
extends CompositorEffect

@export var reflection_max_dist: float = 20.0
@export var reflection_samples:  int   = 1
@export var reflection_strength: float = 0.8

var _rd: RenderingDevice

# RT pipeline
var _rt_shader:   RID
var _rt_pipeline: RID

# Compute apply pipeline
var _comp_shader:   RID
var _comp_pipeline: RID

# Persistent resources
var _reflection_mask_tex: RID
var _light_buf:           RID  # 32-byte SSBO: vec4(dir.xyz, energy) + vec4(color.rgb, 0)
var _sampler:             RID
var _last_size:           Vector2i = Vector2i(0, 0)

func _init() -> void:
	effect_callback_type = CompositorEffect.EFFECT_CALLBACK_TYPE_POST_TRANSPARENT
	access_resolved_color  = true
	access_resolved_depth  = true
	needs_normal_roughness = true

	_rd = RenderingServer.get_rendering_device()

	if not _rd.has_feature(RenderingDevice.SUPPORTS_RAYTRACING_PIPELINE):
		push_error("[RTRefl] Hardware raytracing not supported.")
		return

	_build_rt_pipeline()
	_build_compute_pipeline()
	_build_persistent_resources()

func _build_rt_pipeline() -> void:
	var full_src := FileAccess.get_file_as_string("res://Shaders/rt_reflections.glsl")
	if full_src.is_empty():
		push_error("[RTRefl] Could not load rt_reflections.glsl")
		return

	var shader_source := RDShaderSource.new()
	shader_source.source_raygen      = _extract_section(full_src, "#[raygen]",      "#[miss]")
	shader_source.source_miss        = _extract_section(full_src, "#[miss]",        "#[closest_hit]")
	shader_source.source_closest_hit = _extract_section(full_src, "#[closest_hit]", "")

	var spirv := _rd.shader_compile_spirv_from_source(shader_source)
	var err_rg := spirv.get_stage_compile_error(RenderingDevice.SHADER_STAGE_RAYGEN)
	var err_ms := spirv.get_stage_compile_error(RenderingDevice.SHADER_STAGE_MISS)
	var err_ch := spirv.get_stage_compile_error(RenderingDevice.SHADER_STAGE_CLOSEST_HIT)
	if not err_rg.is_empty(): push_error("[RTRefl] raygen: ", err_rg)
	if not err_ms.is_empty(): push_error("[RTRefl] miss: ",   err_ms)
	if not err_ch.is_empty(): push_error("[RTRefl] chit: ",   err_ch)

	_rt_shader   = _rd.shader_create_from_spirv(spirv)
	_rt_pipeline = _rd.raytracing_pipeline_create(_rt_shader)
	print("[RTRefl] RT pipeline valid: ", _rt_pipeline.is_valid())

func _build_compute_pipeline() -> void:
	var src := FileAccess.get_file_as_string("res://Shaders/apply_rt_reflections.glsl")
	if src.is_empty():
		push_error("[RTRefl] Could not load apply_rt_reflections.glsl")
		return

	var shader_source := RDShaderSource.new()
	shader_source.source_compute = _extract_section(src, "#[compute]", "")

	var spirv := _rd.shader_compile_spirv_from_source(shader_source)
	var err := spirv.get_stage_compile_error(RenderingDevice.SHADER_STAGE_COMPUTE)
	if not err.is_empty():
		push_error("[RTRefl] compute error: ", err)
		return

	_comp_shader   = _rd.shader_create_from_spirv(spirv)
	_comp_pipeline = _rd.compute_pipeline_create(_comp_shader)
	print("[RTRefl] Compute pipeline valid: ", _comp_pipeline.is_valid())

func _build_persistent_resources() -> void:
	var ss := RDSamplerState.new()
	ss.min_filter = RenderingDevice.SAMPLER_FILTER_LINEAR
	ss.mag_filter = RenderingDevice.SAMPLER_FILTER_LINEAR
	_sampler = _rd.sampler_create(ss)

	# Light data SSBO: vec4(dir.xyz, energy) + vec4(color.rgb, 0) = 32 bytes.
	_light_buf = _rd.storage_buffer_create(32)

func _ensure_reflection_mask(size: Vector2i) -> void:
	if _reflection_mask_tex.is_valid() and _last_size == size:
		return
	if _reflection_mask_tex.is_valid():
		_rd.free_rid(_reflection_mask_tex)

	var fmt := RDTextureFormat.new()
	fmt.format     = RenderingDevice.DATA_FORMAT_R16G16B16A16_SFLOAT
	fmt.width      = size.x
	fmt.height     = size.y
	fmt.usage_bits = (
		RenderingDevice.TEXTURE_USAGE_STORAGE_BIT |
		RenderingDevice.TEXTURE_USAGE_SAMPLING_BIT |
		RenderingDevice.TEXTURE_USAGE_CAN_COPY_FROM_BIT
	)
	_reflection_mask_tex = _rd.texture_create(fmt, RDTextureView.new())
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
	if not sm.mat_inst_ssbo.is_valid() or not sm.mat_surf_ssbo.is_valid():
		return
	if not sm.uv_ssbo.is_valid() or sm.texture_rids.is_empty():
		return
	if not sm.sky_texture_rid.is_valid():
		return
	if not sm.geom_vertex_ssbo.is_valid():
		return

	_rd = RenderingServer.get_rendering_device()

	# Update light buffer from scene's directional light.
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

	var scene_buffers := p_render_data.get_render_scene_buffers() as RenderSceneBuffersRD
	var scene_data    := p_render_data.get_render_scene_data()    as RenderSceneDataRD
	if scene_buffers == null or scene_data == null:
		return

	var size := scene_buffers.get_internal_size()
	_ensure_reflection_mask(size)

	# Build push constant — 128 bytes (31 floats + 1 uint)
	var proj: Projection   = scene_data.get_cam_projection()
	var inv_proj: Projection   = proj.inverse()
	var inv_view: Transform3D  = scene_data.get_cam_transform()
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
	push.append(reflection_max_dist)
	# Slot 31: packed uint — high 16 bits = sample count, low 16 bits = frame index
	var push_bytes := push.to_byte_array()
	var frame_uint: int = (reflection_samples << 16) | (Engine.get_frames_drawn() & 0xFFFF)
	var uint_bytes := PackedByteArray()
	uint_bytes.resize(4)
	uint_bytes.encode_u32(0, frame_uint)
	push_bytes.append_array(uint_bytes)
	assert(push.size() == 31 and push_bytes.size() == 128)

	var depth_tex:  RID = scene_buffers.get_depth_texture()
	var normal_tex: RID = scene_buffers.get_texture("forward_clustered", "normal_roughness")

	# ── RT pass: write reflection mask ────────────────────────────────────────
	var u_tlas := RDUniform.new()
	u_tlas.uniform_type = RenderingDevice.UNIFORM_TYPE_ACCELERATION_STRUCTURE
	u_tlas.binding = 0
	u_tlas.add_id(sm.tlas)

	var u_mask := RDUniform.new()
	u_mask.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u_mask.binding = 1
	u_mask.add_id(_reflection_mask_tex)

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

	# Bindless texture array: MAX_TEXTURES sampler+texture pairs at binding 8.
	var u_textures := RDUniform.new()
	u_textures.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u_textures.binding = 8
	for tex_rid in sm.texture_rids:
		u_textures.add_id(_sampler)
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
		[u_tlas, u_mask, u_depth, u_normal, u_mat_inst, u_mat_surf,
		 u_geom_inst, u_uvs, u_textures, u_sky, u_geom_verts, u_light],
		_rt_shader, 0
	)

	var rt_list := _rd.raytracing_list_begin()
	_rd.raytracing_list_bind_raytracing_pipeline(rt_list, _rt_pipeline)
	_rd.raytracing_list_bind_uniform_set(rt_list, rt_set, 0)
	_rd.raytracing_list_set_push_constant(rt_list, push_bytes, push_bytes.size())
	_rd.raytracing_list_trace_rays(rt_list, size.x, size.y)
	_rd.raytracing_list_end()
	_rd.free_rid(rt_set)

	# ── Compute pass: blend reflection onto color buffer ──────────────────────
	var color_tex: RID = scene_buffers.get_color_texture()

	var u_color := RDUniform.new()
	u_color.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u_color.binding = 0
	u_color.add_id(color_tex)

	var u_mask_read := RDUniform.new()
	u_mask_read.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u_mask_read.binding = 1
	u_mask_read.add_id(_reflection_mask_tex)

	var comp_set := _rd.uniform_set_create(
		[u_color, u_mask_read], _comp_shader, 0
	)

	var comp_push := PackedFloat32Array([reflection_strength])
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
		if _reflection_mask_tex.is_valid(): _rd.free_rid(_reflection_mask_tex)
		if _light_buf.is_valid():           _rd.free_rid(_light_buf)
		if _sampler.is_valid():             _rd.free_rid(_sampler)
		if _rt_pipeline.is_valid():         _rd.free_rid(_rt_pipeline)
		if _rt_shader.is_valid():           _rd.free_rid(_rt_shader)
		if _comp_pipeline.is_valid():       _rd.free_rid(_comp_pipeline)
		if _comp_shader.is_valid():         _rd.free_rid(_comp_shader)
