using Godot;
namespace Prime.Source.Interfaces;

public interface IState
{
    protected void Enter();
    protected void Exit();
    protected void Process(double delta);
    protected void PhysicsProcess(double delta);
}