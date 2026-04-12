using Godot;

namespace Prime;

public partial class FPSLabel : Label
{
	// Called every frame. 'delta' is the elapsed time since the previous frame.
	public override void _Process(double delta)
	{
		Text = $"FPS: {Engine.GetFramesPerSecond():0.0}";
	}
}