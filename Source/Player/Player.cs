using Godot;
using Prime.Autoloads;
using Prime.Source.Core.Actor;

namespace Prime.Source.Player;

public partial class Player : Character
{
    [Export] public float Speed { get; set; } = 3;
    [Export] public float JumpVelocity { get; set; } = 5;

    public override void _Ready()
    {
        DisplayServer.MouseSetMode(DisplayServer.MouseMode.Captured);
    }
    
    public override void _PhysicsProcess(double delta)
    {
        if (CharacterBody == null) return;
        
        DebugDraw3D.DrawArrow(GlobalPosition, GlobalPosition + GetActorForwardVector() * 2, Colors.Red, 0.1f);
        
        Vector3 currentVelocity = GetVelocity();
        
        // Add the gravity.
        if (!CharacterBody.IsOnFloor())
        {
            currentVelocity += CharacterBody.GetGravity() * (float)delta;
        }
        
        // Handle Jump.
        if (Input.IsActionJustPressed("ui_accept") && CharacterBody.IsOnFloor())
        {
            currentVelocity.Y = JumpVelocity;
        }
        
        // Get the input direction and handle the movement/deceleration.
        // As good practice, you should replace UI actions with custom gameplay actions.
        Vector2 inputDir = Input.GetVector("Left", "Right", "Up", "Down");
        
        Vector3 direction = ((GameBase.GetPlayerController().GetActorForwardVector() * inputDir.Y) + (GameBase.GetPlayerController().GetActorRightVector() * inputDir.X)).Normalized();
        if (direction != Vector3.Zero)
        {
            currentVelocity.X = (float)(direction.X * Speed);
            currentVelocity.Z = (float)(direction.Z * Speed);
        }
        else
        {
            currentVelocity.X = Mathf.MoveToward(CharacterBody.Velocity.X, 0, Speed);
            currentVelocity.Z = Mathf.MoveToward(CharacterBody.Velocity.Z, 0, Speed);
        }

        CharacterBody.Velocity = currentVelocity;
        CharacterBody.MoveAndSlide();
        
        base._PhysicsProcess(delta);
    }

    public override void _UnhandledInput(InputEvent @event)
    {
        base._UnhandledInput(@event);
        if (@event is InputEventKey key)
        {
            if (key.Pressed && key.Keycode == Key.Escape)
            {
                DisplayServer.MouseSetMode(DisplayServer.MouseMode.Visible);
            }
        }
        if (@event is InputEventMouseMotion mouseMotion)
        {
            GameBase.GetPlayerController().AddControllerPitch(mouseMotion.ScreenRelative.Y);
            GameBase.GetPlayerController().AddControllerYaw(mouseMotion.ScreenRelative.X);
        }
    }

    public override bool IsPlayerControlled()
    {
        return true;
    }
}