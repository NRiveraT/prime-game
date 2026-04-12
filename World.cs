using Godot;
using Prime.Autoloads;
using Prime.Source.Interfaces;
using Prime.Source.Player;
using Actor = Prime.Source.Core.Actor.Actor;
using Pawn = Prime.Source.Core.Actor.Pawn;

namespace Prime;


[GlobalClass]
public partial class World : Node3D
{
    [Signal] public delegate void WorldReadyEventHandler();
    
    // Called when the node enters the scene tree for the first time.
    public override void _Ready()
    {
        // Reparent(GameManagerClass.Instance);
        // CallDeferred(Node.MethodName.Reparent, GameManagerClass.Instance);
        GameBase.ActiveWorld = this;

        Ready += PostReady;
    }
    
    void PostReady()
    {
        CallDeferred(Node.MethodName.Reparent, GameBase.Instance);
        EmitSignal(SignalName.WorldReady);
    }
}