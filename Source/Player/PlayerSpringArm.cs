using Godot;
using Prime.Autoloads;
using Prime.Source.Interfaces;

namespace Prime.Source.Player;

[GlobalClass]
public partial class PlayerSpringArm : SpringArm3D, IActorComponent
{
    [Export(PropertyHint.Flags)] public bool UseControlRotation { get; set; }

    public override void _Process(double delta)
    {
        PlayerController pc = GameBase.GetPlayerController();
        if (pc != null)
        {
            if (UseControlRotation)
            {
                GlobalRotation = pc.GlobalRotation;
            }
        }
    }
}