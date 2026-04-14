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

var _entries: Array[BLASEntry] = []
var _deferred_frees: Array[RID] = []  # RIDs freed after new TLAS is swapped in
var directional_light: DirectionalLight3D = null

static var instance : RTScene

enum BuildState { IDLE, BLAS_PENDING, TLAS_PENDING, READY }
var build_state: BuildState = BuildState.IDLE

func _ready() -> void:
	instance = self
	print("Ready")
	_rd = RenderingServer.get_rendering_device()
	tlas = _rd.tlas_create(1024, RenderingDevice.ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT)
	# Auto-register nodes added after the initial scan (player, streamed level chunks, etc.)
	get_tree().node_added.connect(_on_node_added)
	get_tree().node_removed.connect(_on_node_removing)
	scan_scene()
	print("[RTScene] BUFFER_CREATION_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT value: ", RenderingDevice.BUFFER_CREATION_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT)

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
			print("assigned directional light")
		if node is MeshInstance3D:
			var mode: int = node.gi_mode
			if mode == GeometryInstance3D.GI_MODE_DISABLED:
				continue
			if node.mesh == null:
				continue
			_register_mesh_instance(node, mode)
	
	is_ready = false  # not ready until TLAS is built
	build_state = BuildState.BLAS_PENDING
	print("[RTScene] BLAS built, waiting for GPU commit before TLAS")

func tick() -> void:
	match build_state:
		BuildState.BLAS_PENDING:
			# BLAS commands were enqueued last frame, now safe to build TLAS
			_rebuild_tlas()
			build_state = BuildState.TLAS_PENDING

		BuildState.TLAS_PENDING:
			# _tlas_next is now committed on GPU — swap it in, release old.
			# Also release any BLASes/buffers deferred from the previous dirty cycle;
			# it is safe to free them now because the new TLAS no longer references them.
			for rid in _deferred_frees:
				if rid.is_valid(): _rd.free_rid(rid)
			_deferred_frees.clear()
			if tlas.is_valid(): _rd.free_rid(tlas)
			tlas = _tlas_next
			_tlas_next = RID()
			is_ready = true
			build_state = BuildState.READY
			#print("[RTScene] TLAS ready, RT shadows active")

		BuildState.READY:
			# Rebuild BLASes for any entry whose transform changed this frame.
			# Covers all gi_mode values — player and other late-added nodes typically
			# have GI_MODE_DISABLED (the default), not GI_MODE_DYNAMIC.
			var dirty := false
			for entry in _entries:
				if not is_instance_valid(entry.source):
					continue
				if entry.source.global_transform == entry.last_xform:
					continue
				# Defer freeing the old BLAS — freeing it immediately would invalidate
				# the current `tlas` (which still references it) before the new TLAS is ready.
				_deferred_frees.append(entry.blas)
				entry.blas = _build_blas_from_buffers(entry.vertex_buf, entry.index_buf, entry.vertex_count, entry.index_count)
				entry.last_xform = entry.source.global_transform
				dirty = true
			# dirty check is outside the loop — was previously inside (bug: fired on every
			# entry after the first dirty one, re-setting build_state each iteration)
			if dirty:
				# Keep is_ready = true: the current TLAS is still valid for this frame's
				# ray trace pass. The rebuilt TLAS will be ready two frames from now.
				build_state = BuildState.BLAS_PENDING


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

	# Build into _tlas_next — old `tlas` stays live for this frame's RT pass.
	# It will be swapped in on the TLAS_PENDING → READY transition next frame.
	if _tlas_next.is_valid(): _rd.free_rid(_tlas_next)
	_tlas_next = _rd.tlas_create(instances.size(), RenderingDevice.ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT)
	var err := _rd.tlas_build(_tlas_next, instances)
	if err != OK:
		push_error("[RTScene] tlas_build failed: " + str(err))
	#print("[RTScene] TLAS rebuilding: ", instances.size(), " instances")

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
	_entries.append(entry)

func _upload_mesh_buffers(mesh: Mesh):
	var all_verts := PackedFloat32Array()
	var vert_count := 0

	for s in mesh.get_surface_count():
		var arrays = mesh.surface_get_arrays(s)
		if arrays.is_empty(): continue
		var verts: PackedVector3Array = arrays[Mesh.ARRAY_VERTEX]
		if verts == null or verts.is_empty(): continue
		var indices = arrays[Mesh.ARRAY_INDEX]

		if indices != null and not indices.is_empty():
			for i in indices:
				var v: Vector3 = verts[i]
				all_verts.append(v.x)
				all_verts.append(v.y)
				all_verts.append(v.z)
				vert_count += 1
		else:
			for v in verts:
				all_verts.append(v.x)
				all_verts.append(v.y)
				all_verts.append(v.z)
				vert_count += 1

	# Verify first triangle
	print("[RTScene] tri0: (",
	all_verts[0], ",", all_verts[1], ",", all_verts[2], ") (",
	all_verts[3], ",", all_verts[4], ",", all_verts[5], ") (",
	all_verts[6], ",", all_verts[7], ",", all_verts[8], ")"
	)
	print("[RTScene] byte count: ", all_verts.to_byte_array().size(), " expected: ", vert_count * 12)

	var v_bytes := all_verts.to_byte_array()
	var vbuf: RID = _rd.vertex_buffer_create(
	v_bytes.size(),
	v_bytes,
	RenderingDevice.BUFFER_CREATION_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT
	)

	return { "vbuf": vbuf, "ibuf": RID(), "vertex_count": vert_count, "index_count": 0 }

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
		if tlas.is_valid():       _rd.free_rid(tlas)
		if _tlas_next.is_valid(): _rd.free_rid(_tlas_next)
