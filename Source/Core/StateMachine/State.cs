using Godot;
using Prime.Source.Interfaces;

namespace Prime.Source.Core.StateMachine;

[GlobalClass]
public abstract partial class State : Node, IState
{
    public abstract void Enter();
    public abstract void Exit();
    public abstract void Process(double delta);
    public abstract void PhysicsProcess(double delta);
}