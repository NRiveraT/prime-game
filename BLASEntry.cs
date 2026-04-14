using Godot;

namespace Prime;

public partial class BLASEntry : RefCounted
{
    public Rid Blas;
    public Rid VertexBuffer;
    public Rid IndexBuffer;
    public GeometryInstance3D.GIModeEnum GIMode;
    public Transform3D LastTransform;
    public MeshInstance3D Source;
}