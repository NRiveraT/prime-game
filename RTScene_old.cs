using System.Collections.Generic;
using Godot;

namespace Prime;

[Tool]
[GlobalClass]
public partial class RTScene_old : Node
{
     // Public TLAS RID — bound as uniform in the RT shader each frame
    public Rid Tlas { get; private set; }
    public bool IsReady { get; private set; }

    private RenderingDevice _rd;

 
    private readonly Dictionary<MeshInstance3D, BLASEntry> _entries = new();

    public override void _Ready()
    {
        _rd = RenderingServer.GetRenderingDevice();
        // Allocate an empty TLAS RID now so it's always valid to bind
        Tlas = _rd.TlasCreate();
        ScanScene();
    }

    /// Call once at startup or when scene geometry changes significantly.
    public void ScanScene()
    {
        FreeAllBlas();

        var instances = GetAllMeshInstances(GetTree().Root);
        foreach (var mi in instances)
        {
            // Respect GI mode as RT participation flag
            var mode = mi.GiMode;
            if (mode == GeometryInstance3D.GIModeEnum.Disabled) continue;
            if (mi.Mesh == null) continue;

            RegisterMeshInstance(mi, mode);
        }

        IsReady = _entries.Count > 0;
    }

    /// Call every frame before dispatching rays. Rebuilds TLAS.
    /// For dynamic geometry, detects transform changes and rebuilds BLAS.
    public void UpdateTlas()
    {
        if (!IsReady) return;

        // Refit dynamic BLAS entries whose transforms changed
        foreach (var key in _entries.Keys)
        {
            var entry = _entries[key];
            if (entry.GiMode != GeometryInstance3D.GIModeEnum.Dynamic) continue;
            if (entry.Source.GlobalTransform != entry.LastTransform)
            {
                // Rebuild BLAS with updated vertex positions
                // For a refit we'd need ALLOW_UPDATE flag — rebuild for simplicity
                _rd.FreeRid(entry.Blas);
                entry.Blas = BuildBlas(entry.Source.Mesh, entry.VertexBuffer, entry.IndexBuffer);
                entry.LastTransform = entry.Source.GlobalTransform;
                _entries[key] = entry;
            }
        }

        // Rebuild TLAS from all current entries with their world transforms
        var instances = new Array<RDAccelerationStructureInstance>();
        foreach (var entry in _entries.Values)
        {
            var inst = new RDAccelerationStructureInstance();
            inst.Blas      = entry.Blas;
            inst.Transform = entry.Source.GlobalTransform;
            inst.Mask      = 0xFF;   // all rays see all geometry
            instances.Add(inst);
        }

        _rd.TlasBuild(Tlas, instances);
    }

    private void RegisterMeshInstance(MeshInstance3D mi, GeometryInstance3D.GIModeEnum mode)
    {
        var mesh = mi.Mesh;

        // Extract vertex and index data from the mesh resource (CPU side)
        // This is the C# tradeoff vs GDExtension — one-time cost for statics
        (Rid vbuf, Rid ibuf) = UploadMeshBuffers(mesh);
        if (!vbuf.IsValid || !ibuf.IsValid) return;

        Rid blas = BuildBlas(mesh, vbuf, ibuf);
        if (!blas.IsValid) { _rd.FreeRid(vbuf); _rd.FreeRid(ibuf); return; }

        _entries[mi] = new BLASEntry
        {
            Blas          = blas,
            VertexBuffer  = vbuf,
            IndexBuffer   = ibuf,
            GiMode        = mode,
            LastTransform = mi.GlobalTransform,
            Source        = mi,
        };
    }

    private (Rid, Rid) UploadMeshBuffers(Mesh mesh)
    {
        // Accumulate vertices and indices across all surfaces
        var allVerts   = new System.Collections.Generic.List<float>();
        var allIndices = new System.Collections.Generic.List<uint>();
        uint vertOffset = 0;

        for (int s = 0; s < mesh.GetSurfaceCount(); s++)
        {
            var arrays = mesh.SurfaceGetArrays(s);
            if (arrays.Count == 0) continue;

            var verts   = (Vector3[]) arrays[(int)Mesh.ArrayType.Vertex];
            var indices = (int[])     arrays[(int)Mesh.ArrayType.Index];
            if (verts == null) continue;

            foreach (var v in verts)
            {
                allVerts.Add(v.X);
                allVerts.Add(v.Y);
                allVerts.Add(v.Z);
            }

            if (indices != null)
                foreach (var i in indices)
                    allIndices.Add((uint)(i + vertOffset));
            else
                for (uint i = 0; i < (uint)verts.Length; i++)
                    allIndices.Add(i + vertOffset);

            vertOffset += (uint)verts.Length;
        }

        if (allVerts.Count == 0) return (default, default);

        // Upload as raw byte arrays to RenderingDevice storage buffers
        var vBytes = MemoryMarshal.Cast<float, byte>(allVerts.ToArray()).ToArray();
        var iBytes = MemoryMarshal.Cast<uint, byte>(allIndices.ToArray()).ToArray();

        // Buffers need ACCELERATION_STRUCTURE_BUILD_INPUT usage bit
        var vbuf = _rd.VertexBufferCreate(
            (uint)vBytes.Length, vBytes,
            RenderingDevice.StorageBufferUsage.AccelerationStructureBuildInput);

        var ibuf = _rd.IndexBufferCreate(
            (uint)allIndices.Count,
            RenderingDevice.IndexBufferFormat.Uint32,
            iBytes,
            false,
            RenderingDevice.StorageBufferUsage.AccelerationStructureBuildInput);

        return (vbuf, ibuf);
    }

    private Rid BuildBlas(Mesh mesh, Rid vbuf, Rid ibuf)
    {
        var geom = new RDAccelerationStructureGeometry();
        geom.VertexArray  = vbuf;
        geom.IndexArray   = ibuf;
        geom.VertexCount  = (uint)(vbuf.Id > 0 ? GetVertexCount(mesh) : 0);
        geom.IndexCount   = (uint)GetIndexCount(mesh);
        geom.IsOpaque     = true;   // no any-hit shader needed for opaque shadows

        var geoms = new Array<RDAccelerationStructureGeometry> { geom };
        return _rd.BlasCreate(geoms, RenderingDevice.AccelerationStructureFlagBits.PreferFastTrace);
    }

    private static int GetVertexCount(Mesh mesh)
    {
        int total = 0;
        for (int s = 0; s < mesh.GetSurfaceCount(); s++)
        {
            var a = mesh.SurfaceGetArrays(s);
            if (a.Count > 0 && a[(int)Mesh.ArrayType.Vertex] is Vector3[] v)
                total += v.Length;
        }
        return total;
    }

    private static int GetIndexCount(Mesh mesh)
    {
        int total = 0;
        for (int s = 0; s < mesh.GetSurfaceCount(); s++)
        {
            var a = mesh.SurfaceGetArrays(s);
            if (a.Count > 0 && a[(int)Mesh.ArrayType.Index] is int[] i)
                total += i.Length;
            else if (a.Count > 0 && a[(int)Mesh.ArrayType.Vertex] is Vector3[] v)
                total += v.Length;
        }
        return total;
    }

    private static List<MeshInstance3D> GetAllMeshInstances(Node root)
    {
        var result = new List<MeshInstance3D>();
        Walk(root, result);
        return result;
    }
    private static void Walk(Node n, List<MeshInstance3D> out_)
    {
        if (n is MeshInstance3D mi) out_.Add(mi);
        foreach (Node child in n.GetChildren()) Walk(child, out_);
    }

    private void FreeAllBlas()
    {
        foreach (var e in _entries.Values)
        {
            if (e.Blas.IsValid)        _rd.FreeRid(e.Blas);
            if (e.VertexBuffer.IsValid) _rd.FreeRid(e.VertexBuffer);
            if (e.IndexBuffer.IsValid)  _rd.FreeRid(e.IndexBuffer);
        }
        _entries.Clear();
    }

    public override void _Notification(int what)
    {
        if (what == NotificationPredelete)
        {
            FreeAllBlas();
            if (Tlas.IsValid) _rd.FreeRid(Tlas);
        }
    }
}