using Godot;
using Prime.Source.Core.Types;
using Prime.Source.Interfaces;

namespace Prime.Source.Core.Actor;

[GlobalClass]
public partial class Actor : Node3D, IActor
{
	[Export] public StringName UniqueId { get; set; }
	[Export(PropertyHint.ArrayType)] public StringName[] Tags { get; set; }
	
	public override void _EnterTree()
	{
		Ready += ActorIsReady;
	}

	public override void _ExitTree()
	{
		Ready -= ActorIsReady;
	}

	public virtual void ActorIsReady()
	{
		GD.Print("Actor " + Name + " is ready");
		
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
	}

	public virtual void OnActorHit(HitInfo hitInfo)
	{
		GD.Print("Actor hit by: " + hitInfo.Instigator.GetClass() + " " + hitInfo.Instigator.Name);
	}
}