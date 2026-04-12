using System;
using Godot;
using Prime.Source.Core.Actor;

namespace Prime.Source.Player;

[GlobalClass]
public partial class PlayerController : Actor
{
    [Export] protected Character Player { get; set; }

    private float _controllerYaw;
    private float _controllerPitch;
    private float _controllerRoll;

    public override void _Ready()
    {
    }

    public override void _PhysicsProcess(double delta)
    {
        Quaternion currentQuaternion = GetGlobalBasis().GetRotationQuaternion();
        Quaternion yawQuat = new Quaternion(Vector3.Up, Mathf.DegToRad(_controllerYaw));
        Quaternion pitchQuat = new Quaternion(Vector3.Right, Mathf.DegToRad(_controllerPitch));
        Quaternion rollQuat = new Quaternion(Vector3.Forward, Mathf.DegToRad(_controllerRoll));

        SetGlobalBasis(new Basis(yawQuat * pitchQuat * rollQuat));
    }

    public void AddControllerYaw(float newYaw)
    {
        _controllerYaw -= newYaw;
    }

    public void AddControllerPitch(float newPitch)
    {
        _controllerPitch -= newPitch;
        _controllerPitch = Mathf.Clamp(_controllerPitch, -89, 89);
    }
}