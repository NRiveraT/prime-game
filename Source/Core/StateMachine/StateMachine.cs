using Godot;
using Prime.Source.Interfaces;

namespace Prime.Source.Core.StateMachine;

[GlobalClass]
public partial class StateMachine : Node
{
    [Export(PropertyHint.ArrayType)] protected State[] States { get; set; }
}