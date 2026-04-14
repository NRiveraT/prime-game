using Godot;
using Prime.Source.Core.Actor;
using Prime.Source.Player;

namespace Prime.Autoloads;

[GlobalClass]
public partial class GameBase : Node
{
    public static GameBase Instance { get; private set; }

    [Export(PropertyHint.File)] public PackedScene PlayerControllerScene { private get; set; }
    [Export(PropertyHint.File)] public PackedScene PlayerCharacterScene { private get; set; }

    public static World ActiveWorld { get; set; }

    private Character PlayerCharacter { get; set; }
    private static PlayerController PlayerController { get; set; }
    
    public override void _EnterTree()
    {
        Instance = this;

        // Load the player character and player controller.
        if (PlayerCharacterScene != null)
        {
            PlayerCharacter = PlayerCharacterScene.Instantiate() as Character;
        }

        if (PlayerControllerScene != null)
        {
            PlayerController = PlayerControllerScene.Instantiate() as PlayerController;
        }

        foreach (Node child in GetTree().Root.GetChildren())
        {
            if (ActiveWorld != null)
                return;

            if (child is World world)
            {
                GD.Print("Found world!");
                ActiveWorld = world;
            }
        }

        CallDeferred(MethodName.GameReady);
    }

    void GameReady()
    {
        GD.Print("Game Ready, checking world...");
        
        if (ActiveWorld != null)
        {
            GD.Print("Game Ready, prepping world...");
            ActiveWorld.InitializeWorld();
        }
        
        // Add the player character and player controller to the scene.
        if (PlayerCharacter != null && ActiveWorld != null)
        {
            PlayerCharacter.UniqueId = "Player";
            PlayerCharacter.SetPosition(new Vector3(0, 100, 0));
            ActiveWorld.AddChild(PlayerCharacter);
        }

        if (PlayerController != null && ActiveWorld != null)
        {
            // PlayerController.SetPosition(new Vector3(0, 5, 0));

            // if (PlayerCharacter != null)
            // {
            //     PlayerController.Basis = new Basis(new Quaternion(PlayerController.GetActorForwardVector(), PlayerCharacter.GetActorForwardVector()));
            // }

            ActiveWorld.AddChild(PlayerController);
        }
    }

    public Character GetPlayerCharacter()
    {
        if (Instance != null && PlayerCharacter != null)
        {
            return Instance.PlayerCharacter;
        }

        return null;
    }

    public static PlayerController GetPlayerController()
    {
        if (Instance != null && PlayerController != null)
        {
            return PlayerController;
        }

        return null;
    }
}