using Godot;
using Prime.Source.Interfaces;

namespace Prime.Source.Core.Actor;

[GlobalClass]
public abstract partial class Pawn : Actor, IPawn
{
    [Export(PropertyHint.Flags)]public bool UseControllerYaw { get; set; }
    [Export(PropertyHint.Flags)]public bool UseControllerPitch { get; set; }
    [Export(PropertyHint.Flags)]public bool UseControllerRoll { get; set; }

    public override void _PhysicsProcess(double delta)
    {
        base._PhysicsProcess(delta);
        
    }

    public virtual bool IsPlayerControlled()
    {
        return false;
    }

    public void MoveForward(float value, double delta)
    {
        SetPosition(GetPosition() + GetActorForwardVector() * value * (float)delta);
    }
}