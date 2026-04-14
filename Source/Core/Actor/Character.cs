using System;
using System.Diagnostics;
using Godot;
using Prime.Source.Core.Types;
using Prime.Source.Interfaces;

namespace Prime.Source.Core.Actor;

[Signal]
public delegate void ActorHitEventHandler(IActor actor, IActor other, Vector3 hitPoint);

[GlobalClass]
[Tool]
public partial class Character : Pawn
{
    [Export] public StateMachine.StateMachine StateMachine { get; set; }

    [Export] public CharacterBody3D Test { get; set; }

    [Export] protected CharacterBody3D CharacterBody;
    [Export] protected CollisionShape3D CollisionShape;

    public override void _EnterTree()
    {
        if (Engine.IsEditorHint())
        {
            CallDeferred(MethodName.InitializeComponents);
        }
    }

    public override void InitializeComponents()
    {
        if (CharacterBody != null) return;

        // Init CharacterBody
        CharacterBody = new CharacterBody3D();
        CharacterBody.Name = "CharacterBody";
        AddChild(CharacterBody);
        CharacterBody.Owner = GetTree().EditedSceneRoot;

        if (CollisionShape != null) return;

        // Init CharacterCollider
        CollisionShape = new CollisionShape3D();
        CollisionShape.Name = "Collider";
        CollisionShape.Shape = new CapsuleShape3D();
        CharacterBody.AddChild(CollisionShape);
        CollisionShape.Owner = GetTree().EditedSceneRoot;
    }

    // Called when the node enters the scene tree for the first time.
    public override void _Ready()
    {
        base._Ready();
    }

    public override void _Process(double delta)
    {
        base._Process(delta);

        if (CharacterBody != null)
        {
            if (!CharacterBody.IsSetAsTopLevel())
            {
                GD.Print("Forced it to be TopLevel: " + CharacterBody.IsSetAsTopLevel());

                CharacterBody.SetAsTopLevel(true);
            }

            GlobalTransform = CharacterBody.GlobalTransform;
        }
    }

    public override void ActorIsReady()
    {
        base.ActorIsReady();

        if (CharacterBody != null)
        {
            GD.Print("Setting up CharacterBody3D");

            CharacterBody.SetAsTopLevel(true);
            CharacterBody.GlobalTransform = GlobalTransform;
            GD.Print("TopLevel?: " + CharacterBody.IsSetAsTopLevel());
        }
    }

    // Called every frame. 'delta' is the elapsed time since the previous frame.
    public override void _PhysicsProcess(double delta)
    {
        if (CharacterBody != null)
        {
            ResolveActorCollision(null);
        }

        base._PhysicsProcess(delta);
    }

    public override Vector3 GetVelocity()
    {
        if (CharacterBody != null)
        {
            return CharacterBody.GetVelocity();
        }

        return Vector3.Zero;
    }

    public override void ResolveActorCollision(Node body)
    {
        if (CharacterBody == null) return;
        if (CollisionShape == null) return;

        for (int i = 0; i < CharacterBody.GetSlideCollisionCount(); i++)
        {
            KinematicCollision3D col = CharacterBody.GetSlideCollision(i);
            if (col.GetCollider() is Node3D hit)
            {
                if (hit is IActor hitActor)
                {
                    hitActor.OnActorHit(new HitInfo(col.GetPosition(), col.GetNormal(), this));
                }
                else if (hit.GetParent() is Actor actor)
                {
                    actor.OnActorHit(new HitInfo(col.GetPosition(), col.GetNormal(), this));
                }
            }
        }
    }

    public override void OnActorHit(HitInfo hitInfo)
    {
        GD.Print("Actor hit");
    }

    public override bool IsPlayerControlled()
    {
        return false;
    }
}