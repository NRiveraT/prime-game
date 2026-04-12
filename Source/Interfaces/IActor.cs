using Godot;
using Prime.Source.Core.Types;

namespace Prime.Source.Interfaces;

public interface IActor
{
    protected StringName UniqueId { get; set; }
    protected StringName[] Tags{get; set;}
    
    protected void ActorIsReady();
    
    public Vector3 GetActorForwardVector();
    public Vector3 GetActorRightVector();
    public Vector3 GetActorUpVector();
    public Vector3 GetVelocity();
    
    void ResolveActorCollision(Node body);
    void OnActorHit(HitInfo hitInfo);
}
