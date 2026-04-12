using Godot;

namespace Prime.Source.Interfaces;

public interface IPawn : IActor
{
    public bool UseControllerYaw { get; set; }
    public bool UseControllerPitch { get; set; }
    public bool UseControllerRoll { get; set; }
    
    protected bool IsPlayerControlled()
    {
        return false;
    }

    protected void Possess()
    {
        GD.Print("Possessed");
    }

    protected void UnPossess()
    {
        GD.Print("Unpossessed");
    }
    
    public void MoveForward(float value, double delta);
}