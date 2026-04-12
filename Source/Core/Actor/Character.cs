using System;
using System.Diagnostics;
using Godot;
using Prime.Source.Core.Types;
using Prime.Source.Interfaces;

namespace Prime.Source.Core.Actor;

[Signal]
public delegate void ActorHitEventHandler(IActor actor, IActor other, Vector3 hitPoint);

[GlobalClass]
public partial class Character : Pawn
{
    [Export] protected CharacterBody3D Body { get; private set; }

    [Export] public StateMachine.StateMachine StateMachine { get; set; }

    private CollisionShape3D _collisionShape;

    // Called when the node enters the scene tree for the first time.
    public override void _Ready()
    {
        if (Body == null)
        {
            CharacterBody3D body = GetNode<CharacterBody3D>("CharacterBody3D");
            if (body != null)
            {
                Body = body;
            }
            else
            {
                Debug.Assert(false, "Character must have a CharacterBody3D node");
            }
        }
        else
        {
            if (_collisionShape == null)
            {
                CollisionShape3D collisionShape = Body.GetNode<CollisionShape3D>("CollisionShape3D");
                if (collisionShape != null)
                {
                    _collisionShape = collisionShape;
                }
                else
                {
                    Debug.Assert(false, "Character must have a CollisionShape3D node");
                }
            }
        }
        base._Ready();
    }

    public override void _Process(double delta)
    {
        base._Process(delta);
        if (Body != null)
        {
            if (!Body.IsSetAsTopLevel())
            {
                GD.Print("Forced it to be TopLevel: " + Body.IsSetAsTopLevel());

                Body.SetAsTopLevel(true);
            }

            GlobalTransform = Body.GlobalTransform;
        }
    }

    public override void ActorIsReady()
    {
        base.ActorIsReady();

        if (Body != null)
        {
            GD.Print("Setting up CharacterBody3D");

            Body.SetAsTopLevel(true);
            Body.GlobalTransform = GlobalTransform;
            GD.Print("TopLevel?: " + Body.IsSetAsTopLevel());
        }
    }

    // Called every frame. 'delta' is the elapsed time since the previous frame.
    public override void _PhysicsProcess(double delta)
    {
        if (Body != null)
        {
            ResolveActorCollision(null);
        }

        base._PhysicsProcess(delta);
    }

    public override Vector3 GetVelocity()
    {
        if (Body != null)
        {
            return Body.GetVelocity();
        }

        return Vector3.Zero;
    }

    public override void ResolveActorCollision(Node body)
    {
        if (Body == null) return;
        if (_collisionShape == null) return;

        for (int i = 0; i < Body.GetSlideCollisionCount(); i++)
        {
            KinematicCollision3D col = Body.GetSlideCollision(i);
            if (col.GetCollider() is Node3D hit)
            {
                if (hit is Actor hitActor)
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