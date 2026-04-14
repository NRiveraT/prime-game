using Godot;
using Prime.Source.Core.Actor;
using Prime.Source.Core.Types;

namespace Prime.Source.Level.Loaders;

[GlobalClass] [Tool]
public partial class LoadTrigger : TriggerActor
{
    public override void InitializeComponents()
    {
        base.InitializeComponents();
        GD.Print("Initializing Load Trigger");
        
    }

    public override void OnActorHit(HitInfo hitInfo)
    {
        GD.Print("Load Trigger hit");
        
        if (hitInfo.Instigator != null)
        {
            GD.Print("Load Trigger hit by: " + hitInfo.Instigator.GetClass() + " " + hitInfo.Instigator.Name);
        }
    }
}