using System.Numerics;
using Godot;
using Prime.Source.Core.Types;
using Prime.Source.Interfaces;
using Vector3 = Godot.Vector3;

namespace Prime.Source.Core.Actor;

[GlobalClass] [Tool]
public abstract partial class TriggerActor : Actor
{
    [Export] protected Area3D AreaBody { get; set; }
    [Export] protected CollisionShape3D CollisionShape { get; set; }
    
    public override void InitializeComponents()
    {
        if (Engine.IsEditorHint())
        {
            GD.Print("Initializing Components");
        }
    }

    public override void ResolveActorCollision(Node body)
    {
        GD.Print("Resolve Actor Collision");
        GD.Print("Body: " + body.GetClass() + " " + body.Name);
        if (body.Owner is Node3D other)
        {
            GD.Print("Other: " + other.GetClass() + " " + other.Name);
            OnActorHit(new HitInfo(other.GlobalPosition, Vector3.Zero, other));
        }
    }

    public override void OnActorHit(HitInfo hitInfo)
    {
        GD.Print("TriggerActor hit by: " + hitInfo.Instigator.GetClass() + " " + hitInfo.Instigator.Name);
    }
}