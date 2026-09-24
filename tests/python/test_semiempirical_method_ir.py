"""Family-neutral semiempirical method composition contract for #875."""

from dataclasses import replace

import pytest
from vibeqc_compiler.method.semiempirical import (
    InvalidSemiempiricalMethod,
    ParameterResource,
    ParameterSetRef,
    PrimitiveNode,
    ProductSpec,
    SemiempiricalMethodIR,
    StateField,
    semiempirical_from_xtb,
)
from vibeqc_compiler.method.xtb import (
    GFN2_PARAMETER_SET,
    XtbMethodSpec,
    resolve_xtb_method,
)


def _parameters() -> ParameterSetRef:
    return ParameterSetRef(
        identifier="demo",
        schema="demo-v1",
        source_identity="source-a",
        supported_atomic_numbers=(8, 1),
        resources=(
            ParameterResource("orbital:h0", "h0-v1"),
            ParameterResource("basis:minimal", "basis-v1"),
        ),
    )


def _graph(nodes: tuple[PrimitiveNode, ...]) -> SemiempiricalMethodIR:
    return SemiempiricalMethodIR(
        family="demo-family",
        model="demo-model",
        reference="restricted",
        parameter_set=_parameters(),
        state_fields=(StateField("charge", "atom"),),
        primitives=nodes,
        requested_products=(ProductSpec("energy", 0),),
    )


def test_common_ir_canonicalizes_equivalent_dags() -> None:
    basis = PrimitiveNode(
        node_id="basis",
        category="basis",
        equation_identity="eq-basis",
        produces=("basis-state",),
        parameter_resources=("basis:minimal",),
    )
    h0 = PrimitiveNode(
        node_id="h0",
        category="hamiltonian",
        equation_identity="eq-h0",
        requires=("basis",),
        produces=("h0-matrix",),
        parameter_resources=("orbital:h0",),
    )
    scc = PrimitiveNode(
        node_id="scc",
        category="interaction",
        equation_identity="eq-scc",
        requires=("h0",),
        produces=("scc-potential",),
        state_reads=("charge",),
        state_writes=("charge",),
        fixed_point=True,
    )

    left = _graph((scc, basis, h0))
    right = _graph((h0, scc, basis))

    assert tuple(node.node_id for node in left.primitives) == ("basis", "h0", "scc")
    assert left.identity == right.identity
    assert left.parameter_set.supported_atomic_numbers == (1, 8)
    assert tuple(resource.role for resource in left.parameter_set.resources) == (
        "basis:minimal",
        "orbital:h0",
    )


def test_common_ir_rejects_missing_cycles_and_ambiguous_producers() -> None:
    with pytest.raises(InvalidSemiempiricalMethod, match="missing primitives"):
        _graph(
            (
                PrimitiveNode(
                    node_id="h0",
                    category="hamiltonian",
                    equation_identity="eq-h0",
                    requires=("basis",),
                ),
            )
        )

    with pytest.raises(InvalidSemiempiricalMethod, match="cycle"):
        _graph(
            (
                PrimitiveNode(
                    node_id="a",
                    category="interaction",
                    equation_identity="eq-a",
                    requires=("b",),
                ),
                PrimitiveNode(
                    node_id="b",
                    category="interaction",
                    equation_identity="eq-b",
                    requires=("a",),
                ),
            )
        )

    with pytest.raises(InvalidSemiempiricalMethod, match="ambiguous producer"):
        _graph(
            (
                PrimitiveNode(
                    node_id="a",
                    category="basis",
                    equation_identity="eq-a",
                    produces=("shared",),
                ),
                PrimitiveNode(
                    node_id="b",
                    category="hamiltonian",
                    equation_identity="eq-b",
                    produces=("shared",),
                ),
            )
        )


def test_common_ir_rejects_unknown_state_and_parameter_resource() -> None:
    with pytest.raises(InvalidSemiempiricalMethod, match="unknown states"):
        _graph(
            (
                PrimitiveNode(
                    node_id="bad-state",
                    category="interaction",
                    equation_identity="eq",
                    state_reads=("missing",),
                ),
            )
        )

    with pytest.raises(InvalidSemiempiricalMethod, match="unknown parameter resources"):
        _graph(
            (
                PrimitiveNode(
                    node_id="bad-parameter",
                    category="basis",
                    equation_identity="eq",
                    parameter_resources=("basis:missing",),
                ),
            )
        )


def test_common_identity_tracks_scientific_equations_and_parameter_sources() -> None:
    baseline = _graph(
        (
            PrimitiveNode(
                node_id="basis",
                category="basis",
                equation_identity="eq-v1",
                parameter_resources=("basis:minimal",),
            ),
        )
    )
    changed_equation = replace(
        baseline,
        primitives=(replace(baseline.primitives[0], equation_identity="eq-v2"),),
    )
    changed_source = replace(
        baseline,
        parameter_set=replace(baseline.parameter_set, source_identity="source-b"),
    )

    assert changed_equation.identity != baseline.identity
    assert changed_source.identity != baseline.identity


@pytest.mark.parametrize("method_name", ["GFN2-xTB", "GFN1-xTB"])
def test_gfn_adapter_preserves_audited_graph_without_runtime_claim(
    method_name: str,
) -> None:
    legacy = resolve_xtb_method(
        method_name,
        requested_products=("nuclear-gradient", "energy"),
    )
    common = semiempirical_from_xtb(legacy)

    assert common.family == "gfn-xtb"
    assert common.model == legacy.model_flavor
    assert common.reference == legacy.reference
    assert {node.node_id for node in common.primitives} == {
        primitive.kind for primitive in legacy.primitives
    }
    common_by_id = {node.node_id: node for node in common.primitives}
    for primitive in legacy.primitives:
        assert common_by_id[primitive.kind].requires == primitive.requires
    assert tuple(product.name for product in common.requested_products) == (
        "energy",
        "nuclear-gradient",
    )
    assert common.parameter_set.supported_atomic_numbers == (
        legacy.parameter_set.supported_atomic_numbers
    )
    encoded = repr(common.to_payload()).lower()
    for forbidden in (
        "broyden",
        "diis",
        "iteration_limit",
        "maximum_iterations",
        "mixing_history",
        "eigensolver_library",
        "runtime_executable",
    ):
        assert forbidden not in encoded


def test_gfn_adapter_excludes_manifest_alias_from_common_identity() -> None:
    left = resolve_xtb_method(XtbMethodSpec("alias-left", "gfn2", GFN2_PARAMETER_SET))
    right = resolve_xtb_method(XtbMethodSpec("alias-right", "gfn2", GFN2_PARAMETER_SET))

    assert left.manifest_identity != right.manifest_identity
    assert (
        semiempirical_from_xtb(left).identity == semiempirical_from_xtb(right).identity
    )


def test_gfn_adapter_parameter_provenance_changes_common_identity() -> None:
    baseline = resolve_xtb_method("GFN2-xTB")
    revised = resolve_xtb_method(
        XtbMethodSpec(
            "GFN2-xTB-repacked",
            "gfn2",
            replace(
                GFN2_PARAMETER_SET,
                revision="bannwarth-ehlert-grimme-2019-repacked",
            ),
        )
    )

    assert (
        semiempirical_from_xtb(revised).identity
        != semiempirical_from_xtb(baseline).identity
    )


def test_gfn_adapter_requires_audited_xtb_ir() -> None:
    with pytest.raises(TypeError, match="audited XtbMethodIR"):
        semiempirical_from_xtb(object())
