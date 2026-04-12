using System.Numerics;
using Godot;
using Prime.Source.Core.Types;
using Prime.Source.Interfaces;
using Vector3 = Godot.Vector3;

namespace Prime.Source.Core.Actor;

public abstract partial class TriggerActor : Area3D, IActor
{
    public StringName UniqueId { get; set; }
    public StringName[] Tags { get; set; }

    public override void _EnterTree()
    {
        Ready += ActorIsReady;
    }

    public override void _ExitTree()
    {
        Ready -= ActorIsReady;
        GD.Print("Exiting Tree");
    }

    public virtual void ActorIsReady()
    {
        GD.Print("Actor " + Name + " is ready");
        Connect(Area3D.SignalName.BodyEntered, new Callable(this, MethodName.ResolveActorCollision));
    }

    public Vector3 GetActorForwardVector()
    {
        return this.GlobalTransform.Basis.Z;
    }

    public Vector3 GetActorRightVector()
    {
        return this.GlobalTransform.Basis.X;
    }

    public Vector3 GetActorUpVector()
    {
        return this.GlobalTransform.Basis.Y;
    }

    public virtual Vector3 GetVelocity()
    {
        return Vector3.Zero;
    }

    public virtual void ResolveActorCollision(Node body)
    {
        GD.Print("Resolve Actor Collision");
        GD.Print("Body: " + body.GetClass() + " " + body.Name);
        if (body.Owner is Node3D other)
        {
            GD.Print("Other: " + other.GetClass() + " " + other.Name);
            OnActorHit(new HitInfo(other.GlobalPosition, Vector3.Zero, other));
        }
    }

    public virtual void OnActorHit(HitInfo hitInfo)
    {
        GD.Print("TriggerActor hit by: " + hitInfo.Instigator.GetClass() + " " + hitInfo.Instigator.Name);
    }
}