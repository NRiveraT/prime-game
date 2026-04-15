@tool
class_name RTScene
extends Node

var tlas: RID
var _tlas_next: RID   # TLAS being built; swapped into `tlas` once TLAS_PENDING resolves
var is_ready: bool = false

var _rd: RenderingDevice

# Per-mesh entry tracking BLAS and source node
class BLASEntry:
	var blas: RID
	var vertex_buf: RID
	var index_buf: RID
	var source: MeshInstance3D
	var last_xform: Transform3D
	var gi_mode: int
	var vertex_count: int
	var index_count: int
	var vertex_data: PackedFloat32Array  # CPU copy of object-space vertices (xyz triples)
	var uv_data: PackedFloat32Array      # CPU copy of UVs (uv pairs, same vertex order)
	# Per-surface material data; each element: {start_tri, end_tri, albedo, metallic, roughness, texture_res}
	var surface_mats: Array = []

var _entries: Array[BLASEntry] = []
var _deferred_frees: Array[RID] = []  # RIDs freed after new TLAS is swapped in
@export var directional_light: DirectionalLight3D = null

# Geometry SSBOs — double-buffered alongside the TLAS so the hit shader always
# sees vertex data that matches the currently-bound TLAS instance ordering.
var geom_vertex_ssbo:    RID  # flat packed xyz floats for all instances (object space)
var geom_instance_ssbo:  RID  # per-instance {uint base_vertex, uint vertex_count}
var _geom_vertex_ssbo_next:   RID
var _geom_instance_ssbo_next: RID

# Material SSBOs — double-buffered alongside TLAS.
# mat_inst_ssbo: per-instance header {uint surf_offset, uint surf_count} = 8 bytes each.
# mat_surf_ssbo: flat per-surface data {uvec4 range, vec4 albedo, vec4 props} = 48 bytes each.
#                range.z = texture index into the bindless array (0 = fallback white).
var mat_inst_ssbo:       RID
var mat_surf_ssbo:       RID
var _mat_inst_ssbo_next: RID
var _mat_surf_ssbo_next: RID

# UV SSBO — flat (u,v) pairs per vertex, same ordering as geom_vertex_ssbo.
var uv_ssbo:      RID
var _uv_ssbo_next: RID

# Ordered list of RD texture RIDs for the bindless sampler array; index 0 = fallback white.
# Always padded to MAX_TEXTURES entries.
const MAX_TEXTURES := 128
var texture_rids:       Array[RID] = []
var _texture_rids_next: Array[RID] = []
var _fallback_white_tex: RID

# Sky texture for miss shader — panorama RID or a 1×1 solid-color fallback we own.
var sky_texture_rid: RID
var _sky_fallback_tex: RID  # owned by us (freed on predelete); != panorama RID

static var instance : RTScene

enum BuildState { IDLE, BLAS_PENDING, TLAS_PENDING, READY }
var build_state: BuildState = BuildState.IDLE
var _last_frame_ticked: int = -1

func _ready() -> void:
	instance = self
	
	_rd = RenderingServer.get_rendering_device()
	tlas = _rd.tlas_create(1024, RenderingDevice.ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT)
	_fallback_white_tex = _create_fallback_white_texture()
	# Auto-register nodes added after the initial scan (player, streamed level chunks, etc.)
	get_tree().node_added.connect(_on_node_added)
	get_tree().node_removed.connect(_on_node_removing)
	scan_scene()

## Call once at load, or when scene geometry changes.
## Frees all existing BLASes and rebuilds from scratch.
func scan_scene() -> void:
	_free_all_blas()
	directional_light = null  # reset on rescan

	var nodes: Array = []
	_collect_all_nodes(get_tree().root, nodes)

	for node in nodes:
		if node is DirectionalLight3D and directional_light == null:
			directional_light = node
		if node is WorldEnvironment:
			_setup_sky_texture(node)
		if node is MeshInstance3D:
			if node.mesh == null:
				continue
			# Register ALL mesh instances regardless of GI mode — GI mode is
			# Godot's baked-GI hint, not an RT visibility flag. Skipping
			# GI_MODE_DISABLED here was the cause of most objects (player, props,
			# dynamic objects) missing from the BLAS on scene load.
			_register_mesh_instance(node, node.gi_mode)

	# Ensure sky_texture_rid is valid even when no WorldEnvironment is present.
	if not sky_texture_rid.is_valid():
		sky_texture_rid = _fallback_white_tex

	is_ready = false  # not ready until TLAS is built
	build_state = BuildState.BLAS_PENDING

func tick() -> void:
	var f := Engine.get_frames_drawn()
	if f == _last_frame_ticked:
		return
	_last_frame_ticked = f

	match build_state:
		BuildState.BLAS_PENDING:
			# BLAS commands were enqueued last frame, now safe to build TLAS
			_rebuild_tlas()
			build_state = BuildState.TLAS_PENDING

		BuildState.TLAS_PENDING:
			# _tlas_next is now committed on GPU — swap it in, release old.
			# Also release any BLASes/buffers deferred from mesh removals.
			for rid in _deferred_frees:
				if rid.is_valid(): _rd.free_rid(rid)
			_deferred_frees.clear()
			if tlas.is_valid(): _rd.free_rid(tlas)
			tlas = _tlas_next
			_tlas_next = RID()
			# Swap geometry SSBOs together with the TLAS so they stay in sync.
			if geom_vertex_ssbo.is_valid():   _rd.free_rid(geom_vertex_ssbo)
			if geom_instance_ssbo.is_valid(): _rd.free_rid(geom_instance_ssbo)
			geom_vertex_ssbo   = _geom_vertex_ssbo_next
			geom_instance_ssbo = _geom_instance_ssbo_next
			_geom_vertex_ssbo_next   = RID()
			_geom_instance_ssbo_next = RID()
			if mat_inst_ssbo.is_valid(): _rd.free_rid(mat_inst_ssbo)
			if mat_surf_ssbo.is_valid(): _rd.free_rid(mat_surf_ssbo)
			mat_inst_ssbo        = _mat_inst_ssbo_next
			mat_surf_ssbo        = _mat_surf_ssbo_next
			_mat_inst_ssbo_next  = RID()
			_mat_surf_ssbo_next  = RID()
			if uv_ssbo.is_valid(): _rd.free_rid(uv_ssbo)
			uv_ssbo       = _uv_ssbo_next
			_uv_ssbo_next = RID()
			texture_rids       = _texture_rids_next
			_texture_rids_next = []
			# Only mark ready if a real TLAS was built (empty scene produces invalid RID).
			is_ready = tlas.is_valid()
			build_state = BuildState.READY

		BuildState.READY:
			if _entries.is_empty():
				return
			# For rigid body motion, ONLY the TLAS instance transform changes.
			# The BLAS is object-space geometry — it never changes when an object moves.
			# Rebuilding the BLAS on transform change was wrong and caused:
			#   - 2-frame TLAS lag (BLAS_PENDING + TLAS_PENDING) → self-intersections
			#   - Unnecessary BLAS GPU work every frame for moving objects
			# Fix: update stored transforms, rebuild TLAS directly → 1-frame lag.
			var dirty := false
			for entry in _entries:
				if not is_instance_valid(entry.source):
					continue
				if entry.source.global_transform == entry.last_xform:
					continue
				entry.last_xform = entry.source.global_transform
				dirty = true
			if dirty:
				# is_ready stays true — current TLAS is valid for this frame.
				_rebuild_tlas()
				build_state = BuildState.TLAS_PENDING


func _on_node_added(node: Node) -> void:
	if not node is MeshInstance3D:
		return
	# Defer: node_added fires before _ready(), so global_transform isn't valid yet
	call_deferred("_register_late_mesh", node)

func _register_late_mesh(mi: MeshInstance3D) -> void:
	if not is_instance_valid(mi) or mi.mesh == null:
		return
	for entry in _entries:
		if entry.source == mi:
			return  # already registered by scan_scene or a prior call
	_register_mesh_instance(mi, mi.gi_mode)
	if build_state == BuildState.READY:
		build_state = BuildState.BLAS_PENDING  # is_ready stays true

func _on_node_removing(node: Node) -> void:
	if not node is MeshInstance3D:
		return
	for i in range(_entries.size() - 1, -1, -1):
		if _entries[i].source == node:
			var e := _entries[i]
			# Defer: current `tlas` still references this BLAS; freeing now would invalidate it.
			if e.blas.is_valid():       _deferred_frees.append(e.blas)
			if e.vertex_buf.is_valid(): _deferred_frees.append(e.vertex_buf)
			if e.index_buf.is_valid():  _deferred_frees.append(e.index_buf)
			_entries.remove_at(i)
			if build_state == BuildState.READY:
				build_state = BuildState.BLAS_PENDING
			break

func _rebuild_tlas() -> void:
	if _entries.is_empty(): return
	var instances: Array[RDAccelerationStructureInstance] = []

	# Geometry + material SSBO data built in the same pass, same ordering as TLAS.
	var all_verts      := PackedFloat32Array()
	var all_uvs        := PackedFloat32Array()
	var inst_bytes     := PackedByteArray()  # geom instance headers
	var mat_inst_bytes := PackedByteArray()  # material instance headers: {surf_offset, surf_count}
	var mat_surf_bytes := PackedByteArray()  # flat surface materials: {range uvec4, albedo vec4, props vec4}
	var surf_offset_accum := 0

	# Bindless texture list: index 0 = fallback white, subsequent = unique albedo textures.
	var tex_dedup := {}           # Texture2D resource → int index
	var tex_rids_local: Array[RID] = [_fallback_white_tex]

	for entry in _entries:
		if not entry.blas.is_valid(): continue
		if not is_instance_valid(entry.source): continue
		var inst := RDAccelerationStructureInstance.new()
		inst.blas      = entry.blas
		inst.transform = entry.source.global_transform
		inst.mask      = 0xFF
		inst.flags     = RenderingDevice.ACCELERATION_STRUCTURE_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT
		inst.id        = 0
		instances.append(inst)

		# Geometry: base vertex index + count.
		var base: int = all_verts.size() / 3
		inst_bytes.resize(inst_bytes.size() + 8)
		inst_bytes.encode_u32(inst_bytes.size() - 8, base)
		inst_bytes.encode_u32(inst_bytes.size() - 4, entry.vertex_count)
		all_verts.append_array(entry.vertex_data)
		all_uvs.append_array(entry.uv_data)

		# Material instance header: surf_offset + surf_count = 8 bytes.
		var surf_count := entry.surface_mats.size()
		mat_inst_bytes.resize(mat_inst_bytes.size() + 8)
		var mio := mat_inst_bytes.size() - 8
		mat_inst_bytes.encode_u32(mio + 0, surf_offset_accum)
		mat_inst_bytes.encode_u32(mio + 4, surf_count)

		# Per-surface material data: 48 bytes each.
		# Layout: uvec4 range (start_prim, end_prim, tex_idx, pad) = 16B
		#         vec4  albedo                                      = 16B
		#         vec4  props  (metallic, roughness, pad, pad)      = 16B
		for surf in entry.surface_mats:
			# Resolve texture index: deduplicate across all instances.
			var tex_idx := 0
			var tex_res: Texture2D = surf.get("texture_res", null)
			if tex_res != null:
				if tex_res in tex_dedup:
					tex_idx = tex_dedup[tex_res]
				elif tex_rids_local.size() < MAX_TEXTURES:
					# Try sRGB view first (correct gamma for albedo); fall back to linear
					# view if the texture format doesn't expose an sRGB variant.
					var rd_rid := RenderingServer.texture_get_rd_texture(tex_res.get_rid(), true)
					if not rd_rid.is_valid():
						rd_rid = RenderingServer.texture_get_rd_texture(tex_res.get_rid(), false)
					if rd_rid.is_valid():
						tex_idx = tex_rids_local.size()
						tex_dedup[tex_res] = tex_idx
						tex_rids_local.append(rd_rid)

			mat_surf_bytes.resize(mat_surf_bytes.size() + 48)
			var so := mat_surf_bytes.size() - 48
			mat_surf_bytes.encode_u32(so +  0, surf.start_tri)
			mat_surf_bytes.encode_u32(so +  4, surf.end_tri)
			mat_surf_bytes.encode_u32(so +  8, tex_idx)  # range.z = texture array index
			mat_surf_bytes.encode_u32(so + 12, 0)
			mat_surf_bytes.encode_float(so + 16, surf.albedo.r)
			mat_surf_bytes.encode_float(so + 20, surf.albedo.g)
			mat_surf_bytes.encode_float(so + 24, surf.albedo.b)
			mat_surf_bytes.encode_float(so + 28, surf.albedo.a)
			mat_surf_bytes.encode_float(so + 32, surf.metallic)
			mat_surf_bytes.encode_float(so + 36, surf.roughness)
			mat_surf_bytes.encode_float(so + 40, 0.0)
			mat_surf_bytes.encode_float(so + 44, 0.0)
		surf_offset_accum += surf_count

	# Build TLAS into _tlas_next — old `tlas` stays live for this frame's RT pass.
	if _tlas_next.is_valid(): _rd.free_rid(_tlas_next)
	_tlas_next = _rd.tlas_create(instances.size(), RenderingDevice.ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT)
	var err := _rd.tlas_build(_tlas_next, instances)
	if err != OK:
		push_error("[RTScene] tlas_build failed: " + str(err))

	# Pad texture list to MAX_TEXTURES so the shader always sees a full-size array.
	while tex_rids_local.size() < MAX_TEXTURES:
		tex_rids_local.append(_fallback_white_tex)
	_texture_rids_next = tex_rids_local

	# Build geometry + material SSBOs into _*_next — swapped alongside TLAS in TLAS_PENDING.
	if _geom_vertex_ssbo_next.is_valid():   _rd.free_rid(_geom_vertex_ssbo_next)
	if _geom_instance_ssbo_next.is_valid(): _rd.free_rid(_geom_instance_ssbo_next)
	if _mat_inst_ssbo_next.is_valid():      _rd.free_rid(_mat_inst_ssbo_next)
	if _mat_surf_ssbo_next.is_valid():      _rd.free_rid(_mat_surf_ssbo_next)
	if _uv_ssbo_next.is_valid():            _rd.free_rid(_uv_ssbo_next)
	if not all_verts.is_empty():
		var vbytes := all_verts.to_byte_array()
		_geom_vertex_ssbo_next   = _rd.storage_buffer_create(vbytes.size(), vbytes)
		_geom_instance_ssbo_next = _rd.storage_buffer_create(inst_bytes.size(), inst_bytes)
	if not all_uvs.is_empty():
		var uv_bytes := all_uvs.to_byte_array()
		_uv_ssbo_next = _rd.storage_buffer_create(uv_bytes.size(), uv_bytes)
	if not mat_inst_bytes.is_empty():
		_mat_inst_ssbo_next = _rd.storage_buffer_create(mat_inst_bytes.size(), mat_inst_bytes)
	if not mat_surf_bytes.is_empty():
		_mat_surf_ssbo_next = _rd.storage_buffer_create(mat_surf_bytes.size(), mat_surf_bytes)

func _register_mesh_instance(mi: MeshInstance3D, mode: int) -> void:
	print("[RTScene] registering: ", mi.name, " gi_mode: ", mode)
	var result = _upload_mesh_buffers(mi.mesh)
	if result == null: 
		return

	var blas: RID = _build_blas_from_buffers(
	result.vbuf, result.ibuf,
	result.vertex_count, result.index_count
	)
	if not blas.is_valid():
		_rd.free_rid(result.vbuf)
		_rd.free_rid(result.ibuf)
		return

	var entry := BLASEntry.new()
	entry.blas         = blas
	entry.vertex_buf   = result.vbuf
	entry.index_buf    = result.ibuf
	entry.source       = mi
	entry.last_xform   = mi.global_transform
	entry.gi_mode      = mode
	entry.vertex_count = result.vertex_count
	entry.index_count  = result.index_count
	entry.vertex_data  = result.vertex_data
	entry.uv_data      = result.uv_data
	entry.surface_mats = _extract_surface_materials(mi, result.surface_prim_counts)
	_entries.append(entry)

func _upload_mesh_buffers(mesh: Mesh):
	var all_verts          := PackedFloat32Array()
	var all_uvs            := PackedFloat32Array()
	var vert_count         := 0
	var surface_prim_counts: Array[int] = []

	for s in mesh.get_surface_count():
		var arrays = mesh.surface_get_arrays(s)
		if arrays.is_empty():
			surface_prim_counts.append(0)
			continue
		var verts: PackedVector3Array = arrays[Mesh.ARRAY_VERTEX]
		if verts == null or verts.is_empty():
			surface_prim_counts.append(0)
			continue
		var indices  = arrays[Mesh.ARRAY_INDEX]
		var uvs_raw = arrays[Mesh.ARRAY_TEX_UV]
		var surf_tris := 0

		if indices != null and not indices.is_empty():
			for i in indices:
				var v: Vector3 = verts[i]
				all_verts.append(v.x); all_verts.append(v.y); all_verts.append(v.z)
				if uvs_raw != null and i < uvs_raw.size():
					all_uvs.append(uvs_raw[i].x); all_uvs.append(uvs_raw[i].y)
				else:
					all_uvs.append(0.0); all_uvs.append(0.0)
				vert_count += 1
			surf_tris = indices.size() / 3
		else:
			for vi in range(verts.size()):
				var v: Vector3 = verts[vi]
				all_verts.append(v.x); all_verts.append(v.y); all_verts.append(v.z)
				if uvs_raw != null and vi < uvs_raw.size():
					all_uvs.append(uvs_raw[vi].x); all_uvs.append(uvs_raw[vi].y)
				else:
					all_uvs.append(0.0); all_uvs.append(0.0)
				vert_count += 1
			surf_tris = verts.size() / 3
		surface_prim_counts.append(surf_tris)

	var v_bytes := all_verts.to_byte_array()
	var vbuf: RID = _rd.vertex_buffer_create(
		v_bytes.size(),
		v_bytes,
		RenderingDevice.BUFFER_CREATION_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT
	)

	return { "vbuf": vbuf, "ibuf": RID(), "vertex_data": all_verts, "uv_data": all_uvs,
			 "vertex_count": vert_count, "index_count": 0,
			 "surface_prim_counts": surface_prim_counts }

func _build_blas_from_buffers(vbuf: RID, ibuf: RID, vertex_count: int, index_count: int) -> RID:
	var geom := RDAccelerationStructureGeometry.new()
	geom.vertex_buffer = vbuf
	geom.vertex_offset = 0
	geom.vertex_stride = 12
	geom.vertex_count  = vertex_count
	geom.vertex_format = RenderingDevice.DATA_FORMAT_R32G32B32_SFLOAT
	# Leave index_buffer unset — unindexed geometry
	geom.flags = RenderingDevice.ACCELERATION_STRUCTURE_GEOMETRY_OPAQUE_BIT

	var geoms: Array[RDAccelerationStructureGeometry] = []
	geoms.append(geom)
	var blas: RID = _rd.blas_create(geoms, RenderingDevice.ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT)
	
	if blas.is_valid():
		_rd.blas_build(blas)
		#print("[RTScene] blas_build: ", )
	return blas

# Returns one dict per surface: {start_tri, end_tri, albedo, metallic, roughness}.
# surface_prim_counts[s] = number of triangles in surface s (from _upload_mesh_buffers).
static func _extract_surface_materials(mi: MeshInstance3D, surface_prim_counts: Array) -> Array:
	var result := []
	var tri_offset := 0
	for s in mi.mesh.get_surface_count():
		var count: int = surface_prim_counts[s] if s < surface_prim_counts.size() else 0
		var albedo    := Color.WHITE
		var metallic  := 0.0
		var roughness := 1.0
		# Surface override takes priority over the mesh's own material.
		var mat: Material = mi.get_surface_override_material(s)
		if mat == null:
			mat = mi.mesh.surface_get_material(s)
		var texture_res: Texture2D = null
		if mat is BaseMaterial3D:
			albedo      = mat.albedo_color
			metallic    = mat.metallic
			roughness   = mat.roughness
			texture_res = mat.albedo_texture
		result.append({
			"start_tri":   tri_offset,
			"end_tri":     tri_offset + count,
			"albedo":      albedo,
			"metallic":    metallic,
			"roughness":   roughness,
			"texture_res": texture_res   # null if no texture (fallback white used)
		})
		tri_offset += count
	# Always provide at least one entry so gl_InstanceID lookups never go out of bounds.
	if result.is_empty():
		result.append({"start_tri": 0, "end_tri": 0x7FFFFFFF,
					   "albedo": Color.WHITE, "metallic": 0.0, "roughness": 1.0})
	return result

func _create_fallback_white_texture() -> RID:
	var fmt := RDTextureFormat.new()
	fmt.format     = RenderingDevice.DATA_FORMAT_R8G8B8A8_SRGB
	fmt.width      = 1
	fmt.height     = 1
	fmt.usage_bits = RenderingDevice.TEXTURE_USAGE_SAMPLING_BIT | RenderingDevice.TEXTURE_USAGE_CAN_UPDATE_BIT
	return _rd.texture_create(fmt, RDTextureView.new(), [PackedByteArray([255, 255, 255, 255])])

# Sets sky_texture_rid from a WorldEnvironment node.
# Priority: PanoramaSkyMaterial panorama → environment background_color → sky blue fallback.
func _setup_sky_texture(we: WorldEnvironment) -> void:
	var env := we.environment
	if env == null:
		return

	# Try PanoramaSkyMaterial first — direct texture we can pass to the miss shader.
	var sky := env.sky
	if sky != null and sky.sky_material is PanoramaSkyMaterial:
		var panorama: Texture2D = (sky.sky_material as PanoramaSkyMaterial).panorama
		if panorama != null:
			var rd_rid := RenderingServer.texture_get_rd_texture(panorama.get_rid(), false)
			if rd_rid.is_valid():
				sky_texture_rid = rd_rid
				return  # panorama found — don't create a fallback texture

	# Fall back to a 1×1 solid color using the background or sky color.
	var sky_color := Color(0.53, 0.81, 0.98)  # default sky blue
	if env.background_mode == Environment.BG_COLOR:
		sky_color = env.background_color
	elif env.background_mode == Environment.BG_SKY and sky != null:
		sky_color = Color(0.53, 0.81, 0.98)  # sky procedural — no direct color access
	_make_sky_fallback(sky_color)

func _make_sky_fallback(color: Color) -> void:
	if _sky_fallback_tex.is_valid():
		_rd.free_rid(_sky_fallback_tex)
	var fmt := RDTextureFormat.new()
	fmt.format     = RenderingDevice.DATA_FORMAT_R8G8B8A8_SRGB
	fmt.width      = 1
	fmt.height     = 1
	fmt.usage_bits = RenderingDevice.TEXTURE_USAGE_SAMPLING_BIT | RenderingDevice.TEXTURE_USAGE_CAN_UPDATE_BIT
	var c := PackedByteArray([
		int(clamp(color.r, 0.0, 1.0) * 255),
		int(clamp(color.g, 0.0, 1.0) * 255),
		int(clamp(color.b, 0.0, 1.0) * 255), 255
	])
	_sky_fallback_tex = _rd.texture_create(fmt, RDTextureView.new(), [c])
	sky_texture_rid   = _sky_fallback_tex

func _collect_all_nodes(node: Node, out: Array) -> void:
	out.append(node)
	for child in node.get_children():
		_collect_all_nodes(child, out)

func _free_all_blas() -> void:
	for e in _entries:
		if e.blas.is_valid():       _rd.free_rid(e.blas)
		if e.vertex_buf.is_valid(): _rd.free_rid(e.vertex_buf)
		if e.index_buf.is_valid():  _rd.free_rid(e.index_buf)
	_entries.clear()
	for rid in _deferred_frees:
		if rid.is_valid(): _rd.free_rid(rid)
	_deferred_frees.clear()

func _notification(what: int) -> void:
	if what == NOTIFICATION_PREDELETE:
		_free_all_blas()  # also drains _deferred_frees
		if tlas.is_valid():                    _rd.free_rid(tlas)
		if _tlas_next.is_valid():              _rd.free_rid(_tlas_next)
		if geom_vertex_ssbo.is_valid():        _rd.free_rid(geom_vertex_ssbo)
		if geom_instance_ssbo.is_valid():      _rd.free_rid(geom_instance_ssbo)
		if _geom_vertex_ssbo_next.is_valid():  _rd.free_rid(_geom_vertex_ssbo_next)
		if _geom_instance_ssbo_next.is_valid():_rd.free_rid(_geom_instance_ssbo_next)
		if mat_inst_ssbo.is_valid():           _rd.free_rid(mat_inst_ssbo)
		if mat_surf_ssbo.is_valid():           _rd.free_rid(mat_surf_ssbo)
		if _mat_inst_ssbo_next.is_valid():     _rd.free_rid(_mat_inst_ssbo_next)
		if _mat_surf_ssbo_next.is_valid():     _rd.free_rid(_mat_surf_ssbo_next)
		if uv_ssbo.is_valid():                 _rd.free_rid(uv_ssbo)
		if _uv_ssbo_next.is_valid():           _rd.free_rid(_uv_ssbo_next)
		if _fallback_white_tex.is_valid():     _rd.free_rid(_fallback_white_tex)
		if _sky_fallback_tex.is_valid():       _rd.free_rid(_sky_fallback_tex)
