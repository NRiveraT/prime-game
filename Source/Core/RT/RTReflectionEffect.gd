@tool
class_name RTReflectionEffect
extends CompositorEffect

enum ResolutionScale { FULL = 1, HALF = 2, QUARTER = 4, EIGHTH = 8 }

@export var reflection_max_dist:   float = 20.0
@export var reflection_samples:    int   = 1
@export var reflection_strength:   float = 0.8
@export var reflection_saturation: float = 1.0
@export var reflection_contrast:   float = 1.0
# Downsample factor for the RT pass. Smaller = cheaper; bilateral upsample in
# the apply pass reconstructs full-res output. Mirror-roughness surfaces look
# best at FULL; rough-enough reflections stay indistinguishable at QUARTER.
@export var reflection_scale:      ResolutionScale = ResolutionScale.FULL

var _rd: RenderingDevice

# RT pipeline
var _rt_shader:   RID
var _rt_pipeline: RID

# À-trous wavelet denoiser — runs 3 ping-pong passes between RT and apply
# with increasing stride (1, 2, 4) to cover a 17×17 effective kernel cheaply.
var _atrous_shader:   RID
var _atrous_pipeline: RID

# Compute apply pipeline
var _comp_shader:   RID
var _comp_pipeline: RID

# Persistent resources
var _reflection_mask_tex:      RID
var _reflection_mask_ping_tex: RID  # ping-pong partner for à-trous passes
var _light_buf:                RID
var _local_lights_buf:         RID
# GI probe grid (C0 cascade) bindings — sourced from RTGIEffect.instance when
# available. Fallback buffers keep the bindings valid when RTGI isn't in the
# scene, in which case gi_grid.size_enable.w = 0 and the sample returns zero.
var _gi_grid_buf:              RID  # 32 bytes: vec4 origin_spacing + ivec4 size_enable
var _gi_probes_fallback:       RID  # 16-byte zero buffer for when no RTGI probes are live
var _sampler:                  RID
var _repeat_sampler:           RID
var _last_size:                Vector2i = Vector2i(0, 0)

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
	_build_atrous_pipeline()
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

func _build_atrous_pipeline() -> void:
	var src := FileAccess.get_file_as_string("res://Shaders/rt_reflections_atrous.glsl")
	if src.is_empty():
		push_error("[RTRefl] Could not load rt_reflections_atrous.glsl")
		return

	var shader_source := RDShaderSource.new()
	shader_source.source_compute = _extract_section(src, "#[compute]", "")

	var spirv := _rd.shader_compile_spirv_from_source(shader_source)
	var err := spirv.get_stage_compile_error(RenderingDevice.SHADER_STAGE_COMPUTE)
	if not err.is_empty():
		push_error("[RTRefl] atrous error: ", err)
		return

	_atrous_shader   = _rd.shader_create_from_spirv(spirv)
	_atrous_pipeline = _rd.compute_pipeline_create(_atrous_shader)
	print("[RTRefl] Atrous pipeline valid: ", _atrous_pipeline.is_valid())

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

	# Separate repeat sampler for albedo_textures[] — tiled materials need UV wrap.
	var ss_rep := RDSamplerState.new()
	ss_rep.min_filter = RenderingDevice.SAMPLER_FILTER_LINEAR
	ss_rep.mag_filter = RenderingDevice.SAMPLER_FILTER_LINEAR
	ss_rep.repeat_u   = RenderingDevice.SAMPLER_REPEAT_MODE_REPEAT
	ss_rep.repeat_v   = RenderingDevice.SAMPLER_REPEAT_MODE_REPEAT
	_repeat_sampler = _rd.sampler_create(ss_rep)

	# Light data SSBO: 64 bytes
	#   vec4(dir.xyz, 1.0)                  direction_energy
	#   vec4(color.rgb, sky_energy_mult)    color_sky_energy
	#   vec4(ambient.rgb, sin_half_angle)   ambient_pad
	#   vec4(0, tonemap_exp, sat, contrast) params
	_light_buf = _rd.storage_buffer_create(64)

	# Local lights SSBO: 16-byte header + 32 * 64-byte RTLight entries = 2064 bytes
	_local_lights_buf = _rd.storage_buffer_create(2064)

	# GI grid block (32 bytes). Updated per frame from RTGIEffect.instance.
	# When RTGI is absent or disabled, size_enable.w = 0 → shader skips the
	# sample and reflections use their existing direct+ambient path.
	_gi_grid_buf = _rd.storage_buffer_create(32)
	# Fallback probes buffer — 16 bytes (single vec4 zero) so the probe
	# binding remains valid when RTGI isn't in the compositor stack.
	var fb := PackedByteArray()
	fb.resize(16)
	fb.fill(0)
	_gi_probes_fallback = _rd.storage_buffer_create(16, fb)

func _ensure_reflection_mask(size: Vector2i) -> void:
	if _reflection_mask_tex.is_valid() and _last_size == size:
		return
	if _reflection_mask_tex.is_valid():      _rd.free_rid(_reflection_mask_tex)
	if _reflection_mask_ping_tex.is_valid(): _rd.free_rid(_reflection_mask_ping_tex)

	var fmt := RDTextureFormat.new()
	fmt.format     = RenderingDevice.DATA_FORMAT_R16G16B16A16_SFLOAT
	fmt.width      = size.x
	fmt.height     = size.y
	fmt.usage_bits = (
		RenderingDevice.TEXTURE_USAGE_STORAGE_BIT |
		RenderingDevice.TEXTURE_USAGE_SAMPLING_BIT |
		RenderingDevice.TEXTURE_USAGE_CAN_COPY_FROM_BIT
	)
	_reflection_mask_tex      = _rd.texture_create(fmt, RDTextureView.new())
	_reflection_mask_ping_tex = _rd.texture_create(fmt, RDTextureView.new())
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

	var physical_units: bool = ProjectSettings.get_setting(
		"rendering/lights_and_shadows/use_physical_light_units", false)
	var exposure_norm: float = sm.exposure_normalization

	# Update light buffer from scene's directional light.
	var dl : DirectionalLight3D = sm.directional_light
	var light_dir    := Vector3(0.0, 1.0, 0.0)
	var light_color  := Vector3(1.0, 1.0, 1.0)
	var dir_sin_ha   := 0.0
	var dir_energy   := 1.0
	if dl != null:
		light_dir   = dl.global_transform.basis.z.normalized()
		light_color = Vector3(dl.light_color.r, dl.light_color.g, dl.light_color.b)
		dir_sin_ha  = sin(deg_to_rad(dl.light_angular_distance))
		dir_energy  = _compute_light_final_energy(dl, physical_units, exposure_norm)
	# Ambient / sky energy / direct light all share one HDR range — the one Godot
	# outputs after applying exposure_normalization. Scale each contribution the
	# same way on the CPU so reflection output matches scene output pixel-for-pixel.
	var amb := sm.ambient_light * exposure_norm
	var sky_energy_exposed := sm.sky_energy_multiplier * exposure_norm
	var light_data := PackedFloat32Array([
		light_dir.x, light_dir.y, light_dir.z, dir_energy,
		light_color.x, light_color.y, light_color.z, sky_energy_exposed,
		amb.x, amb.y, amb.z, dir_sin_ha,
		0.0, sm.tonemap_exposure, reflection_saturation, reflection_contrast
	])
	_rd.buffer_update(_light_buf, 0, 64, light_data.to_byte_array())

	# Upload local lights (built fresh each frame for dynamic lights).
	var ll_bytes := _build_local_lights_bytes(sm.local_lights, physical_units, exposure_norm)
	_rd.buffer_update(_local_lights_buf, 0, ll_bytes.size(), ll_bytes)

	var scene_buffers := p_render_data.get_render_scene_buffers() as RenderSceneBuffersRD
	var scene_data    := p_render_data.get_render_scene_data()    as RenderSceneDataRD
	if scene_buffers == null or scene_data == null:
		return

	var size := scene_buffers.get_internal_size()
	var scale : int = max(int(reflection_scale), 1)
	# Ceil-divide so downsampling never loses the last edge pixels.
	var rt_size := Vector2i((size.x + scale - 1) / scale, (size.y + scale - 1) / scale)
	_ensure_reflection_mask(rt_size)

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
	var color_tex:  RID = scene_buffers.get_color_texture()

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

	var u_local_lights := RDUniform.new()
	u_local_lights.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u_local_lights.binding = 13
	u_local_lights.add_id(_local_lights_buf)

	# GI probe binding: fetch live probe buffer + grid params from RTGIEffect
	# singleton if present. When absent, fall back to zero buffers and let the
	# shader's size_enable.w = 0 check short-circuit the sample.
	var gi_probe_buf: RID = _gi_probes_fallback
	var gi_enable: int = 0
	var gi_origin: Vector3 = Vector3.ZERO
	var gi_spacing: float = 1.0
	var gi_size: Vector3i = Vector3i(1, 1, 1)
	var rtgi: RTGIEffect = RTGIEffect.instance
	if rtgi != null and rtgi.get_c0_probe_buf().is_valid():
		gi_probe_buf = rtgi.get_c0_probe_buf()
		gi_enable   = 1
		gi_origin   = rtgi.get_c0_grid_origin()
		gi_spacing  = rtgi.get_c0_grid_spacing()
		gi_size     = rtgi.get_c0_grid_size()
	var gi_bytes := PackedByteArray()
	gi_bytes.resize(32)
	gi_bytes.encode_float( 0, gi_origin.x)
	gi_bytes.encode_float( 4, gi_origin.y)
	gi_bytes.encode_float( 8, gi_origin.z)
	gi_bytes.encode_float(12, gi_spacing)
	gi_bytes.encode_s32(16, gi_size.x)
	gi_bytes.encode_s32(20, gi_size.y)
	gi_bytes.encode_s32(24, gi_size.z)
	gi_bytes.encode_s32(28, gi_enable)
	_rd.buffer_update(_gi_grid_buf, 0, 32, gi_bytes)

	var u_gi_probes := RDUniform.new()
	u_gi_probes.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u_gi_probes.binding = 12
	u_gi_probes.add_id(gi_probe_buf)

	var u_gi_grid := RDUniform.new()
	u_gi_grid.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u_gi_grid.binding = 14
	u_gi_grid.add_id(_gi_grid_buf)

	var rt_set := _rd.uniform_set_create(
		[u_tlas, u_mask, u_depth, u_normal, u_mat_inst, u_mat_surf,
		 u_geom_inst, u_uvs, u_textures, u_sky, u_geom_verts, u_light,
		 u_local_lights, u_gi_probes, u_gi_grid],
		_rt_shader, 0
	)

	var rt_list := _rd.raytracing_list_begin()
	_rd.raytracing_list_bind_raytracing_pipeline(rt_list, _rt_pipeline)
	_rd.raytracing_list_bind_uniform_set(rt_list, rt_set, 0)
	_rd.raytracing_list_set_push_constant(rt_list, push_bytes, push_bytes.size())
	_rd.raytracing_list_trace_rays(rt_list, rt_size.x, rt_size.y)
	_rd.raytracing_list_end()
	_rd.free_rid(rt_set)

	# ── À-trous denoise: 3 ping-pong passes at strides 1, 2, 4 ────────────────
	# After 3 passes the final result lives in whichever texture the last pass
	# wrote to — track it and bind that to the apply pass.
	var final_mask := _reflection_mask_tex
	if _atrous_pipeline.is_valid():
		var a := _reflection_mask_tex
		var c := _reflection_mask_ping_tex
		for stride in [1, 2, 4]:
			_run_atrous_pass(a, c, depth_tex, normal_tex, stride, rt_size, size)
			var tmp := a
			a = c
			c = tmp
		final_mask = a  # last-written texture after post-swap

	# ── Compute pass: blend reflection onto color buffer ──────────────────────
	var u_color := RDUniform.new()
	u_color.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u_color.binding = 0
	u_color.add_id(color_tex)

	var u_mask_read := RDUniform.new()
	u_mask_read.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u_mask_read.binding = 1
	u_mask_read.add_id(final_mask)

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

	# apply_rt_reflections PC: strength, pad, rt_size.x, rt_size.y (16 bytes).
	var comp_push := PackedFloat32Array([
		reflection_strength, 0.0,
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

static func _compute_light_final_energy(lgt: Light3D, physical_units: bool, exposure_norm: float) -> float:
	# Physical omni/spot: (lumens / 4π) × light_energy × exposure_norm
	# Physical directional: lux × light_energy × exposure_norm
	# Non-physical: light_energy × exposure_norm
	if physical_units:
		if lgt is DirectionalLight3D:
			var lux: float = lgt.light_intensity_lux
			if lux > 0.0:
				return lux * lgt.light_energy * exposure_norm
		elif lgt is OmniLight3D or lgt is SpotLight3D:
			var lumens: float = lgt.light_intensity_lumens
			if lumens > 0.0:
				return (lumens / (4.0 * PI)) * lgt.light_energy * exposure_norm
	return lgt.light_energy * exposure_norm

func _build_local_lights_bytes(lights: Array, physical_units: bool, exposure_norm: float) -> PackedByteArray:
	const MAX_LOCAL := 32
	var bytes := PackedByteArray()
	bytes.resize(16 + MAX_LOCAL * 64)
	bytes.fill(0)
	var count := 0
	for lgt in lights:
		if count >= MAX_LOCAL: break
		if not is_instance_valid(lgt): continue
		var off := 16 + count * 64
		var pos: Vector3 = lgt.global_position
		var range_val : float = lgt.omni_range if lgt is OmniLight3D else lgt.spot_range
		bytes.encode_float(off +  0, pos.x)
		bytes.encode_float(off +  4, pos.y)
		bytes.encode_float(off +  8, pos.z)
		bytes.encode_float(off + 12, range_val)
		var col: Color = lgt.light_color
		bytes.encode_float(off + 16, col.r)
		bytes.encode_float(off + 20, col.g)
		bytes.encode_float(off + 24, col.b)
		var final_energy := _compute_light_final_energy(lgt, physical_units, exposure_norm)
		bytes.encode_float(off + 28, final_energy)
		var fwd     := Vector3.ZERO
		var cos_out := 0.0
		var cos_in  := 0.0
		var ltype   := 0.0
		var decay   := 1.0
		if lgt is SpotLight3D:
			ltype   = 1.0
			fwd     = -lgt.global_transform.basis.z
			var oa  : float = lgt.spot_angle
			cos_out = cos(deg_to_rad(oa))
			cos_in  = cos(deg_to_rad(oa * 0.85))
			decay   = lgt.spot_attenuation
		elif lgt is OmniLight3D:
			decay   = lgt.omni_attenuation
		bytes.encode_float(off + 32, fwd.x)
		bytes.encode_float(off + 36, fwd.y)
		bytes.encode_float(off + 40, fwd.z)
		bytes.encode_float(off + 44, lgt.light_size)
		bytes.encode_float(off + 48, cos_out)
		bytes.encode_float(off + 52, cos_in)
		bytes.encode_float(off + 56, ltype)
		bytes.encode_float(off + 60, decay)
		count += 1
	bytes.encode_u32(0, count)
	return bytes

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

	# PC layout: int stride, int pad, ivec2 full_size — 16 bytes.
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
		if _reflection_mask_tex.is_valid():      _rd.free_rid(_reflection_mask_tex)
		if _reflection_mask_ping_tex.is_valid(): _rd.free_rid(_reflection_mask_ping_tex)
		if _light_buf.is_valid():                _rd.free_rid(_light_buf)
		if _local_lights_buf.is_valid():         _rd.free_rid(_local_lights_buf)
		if _gi_grid_buf.is_valid():              _rd.free_rid(_gi_grid_buf)
		if _gi_probes_fallback.is_valid():       _rd.free_rid(_gi_probes_fallback)
		if _sampler.is_valid():                  _rd.free_rid(_sampler)
		if _repeat_sampler.is_valid():           _rd.free_rid(_repeat_sampler)
		if _rt_pipeline.is_valid():              _rd.free_rid(_rt_pipeline)
		if _rt_shader.is_valid():                _rd.free_rid(_rt_shader)
		if _atrous_pipeline.is_valid():          _rd.free_rid(_atrous_pipeline)
		if _atrous_shader.is_valid():            _rd.free_rid(_atrous_shader)
		if _comp_pipeline.is_valid():            _rd.free_rid(_comp_pipeline)
		if _comp_shader.is_valid():              _rd.free_rid(_comp_shader)
