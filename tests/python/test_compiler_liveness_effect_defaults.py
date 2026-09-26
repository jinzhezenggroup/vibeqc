"""Unknown and output-free effects must survive dead-code elimination."""

import pytest
from vibeqc_compiler.common.liveness import EffectKind, LivenessNode, analyze_liveness


def test_unspecified_effect_preserves_the_operation_and_its_inputs() -> None:
    graph = (
        LivenessNode("prepare", ("x",), ("prepared",), EffectKind.PURE),
        LivenessNode("unknown_call", ("prepared",), ("unused",)),
    )
    result = analyze_liveness(graph, roots=(), external_values=("x",))
    assert result.live_node_keys == ("prepare", "unknown_call")
    assert result.live_external_values == ("x",)
    assert result.removed_node_keys == ()


@pytest.mark.parametrize("effect", (EffectKind.EFFECTFUL, EffectKind.OPAQUE))
def test_output_free_effect_preserves_its_transitive_dependencies(
    effect: EffectKind,
) -> None:
    graph = (
        LivenessNode("prepare", ("x",), ("prepared",), EffectKind.PURE),
        LivenessNode("check_or_store", ("prepared",), (), effect),
    )
    result = analyze_liveness(graph, roots=(), external_values=("x", "unused"))
    assert result.live_node_keys == ("prepare", "check_or_store")
    assert result.live_values == ("x", "prepared")
    assert result.live_external_values == ("x",)
