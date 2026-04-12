using Godot;
using Prime.Source.Core.Actor;

namespace Prime.Source.Core.Types;

public readonly struct HitInfo(Vector3 impactPoint, Vector3 normal, Node instigator)
{
    public Vector3 ImpactPoint { get; } = impactPoint;
    public Vector3 Normal { get; } = normal;
    public Actor.Actor Instigator { get; } = instigator as Actor.Actor;
}