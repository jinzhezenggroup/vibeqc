"""Canonical XtbMethodSpec -> XtbMethodIR composition gates for #503."""

import json
import typing
from dataclasses import replace

import pytest
from vibeqc_compiler.method import (
    GFN2_PARAMETER_SET,
    XTB_METHOD_CATALOG,
    UnsupportedXtbMethod,
    XtbMethodIR,
    XtbMethodSpec,
    XtbParameterSet,
    resolve_xtb_method,
)


def test_gfn2_manifest_resolves_complete_canonical_graph() -> None:
    method = resolve_xtb_method(
        "GFN2-xTB",
        requested_products=("nuclear-gradient", "energy"),
    )

    assert method.model_flavor == "gfn2"
    assert method.reference == "restricted"
    assert method.requested_products == ("energy", "nuclear-gradient")
    assert tuple(primitive.kind for primitive in method.primitives) == (
        "basis_parameters",
        "coordination_number",
        "overlap_integrals",
        "multipole_integrals",
        "h0",
        "scc_electrostatics",
        "spin_polarization",
        "hamiltonian",
        "repulsion",
        "dispersion",
    )

    requirements = method.compiler_requirements
    assert requirements["supported_atomic_numbers"] == tuple(range(1, 87))
    assert requirements["integral_operators"] == (
        "overlap",
        "dipole",
        "quadrupole",
    )
    assert requirements["state_requirements"] == (
        "charge",
        "dipole",
        "quadrupole",
        "magnetization",
    )
    assert requirements["correction_primitives"] == (
        "coordination_number",
        "repulsion",
        "dispersion",
    )
    assert requirements["requested_products"] == ("energy", "nuclear-gradient")

    scc = next(p for p in method.primitives if p.kind == "scc_electrostatics")
    assert scc.model == "gfn2-es2-es3-aes2"
    assert scc.requires == (
        "coordination_number",
        "overlap_integrals",
        "multipole_integrals",
    )
    assert "third-order-shell" in requirements["parameter_tables"]["orbital"]

    assert method.runtime_requirements == {
        "scc_fixed_point": True,
        "generalized_eigensolution": True,
        "occupations": True,
        "policy_in_compiler_ir": False,
    }
    assert method.capability == {
        "compiler_representable": True,
        "lowering_available": False,
        "runtime_executable": False,
    }


def test_equivalent_manifests_have_stable_semantic_identity() -> None:
    equivalent_parameters = XtbParameterSet(
        identifier=GFN2_PARAMETER_SET.identifier,
        revision=GFN2_PARAMETER_SET.revision,
        source=GFN2_PARAMETER_SET.source,
        supported_atomic_numbers=tuple(
            reversed(GFN2_PARAMETER_SET.supported_atomic_numbers)
        ),
        basis_tables=tuple(reversed(GFN2_PARAMETER_SET.basis_tables)),
        orbital_tables=tuple(reversed(GFN2_PARAMETER_SET.orbital_tables)),
        correction_tables=tuple(reversed(GFN2_PARAMETER_SET.correction_tables)),
        spin_tables=GFN2_PARAMETER_SET.spin_tables,
    )
    assert equivalent_parameters.identity == GFN2_PARAMETER_SET.identity

    left = resolve_xtb_method(
        XtbMethodSpec(
            "alias-left",
            "gfn2",
            GFN2_PARAMETER_SET,
            requested_products=("nuclear-gradient", "energy"),
        )
    )
    right = resolve_xtb_method(
        XtbMethodSpec(
            "alias-right",
            "gfn2",
            equivalent_parameters,
            requested_products=("energy", "nuclear-gradient"),
        )
    )
    assert left.identity == right.identity
    assert left.manifest_identity != right.manifest_identity


def test_parameter_provenance_reference_and_products_change_identity() -> None:
    baseline = resolve_xtb_method("GFN2-xTB")

    revised_parameters = replace(
        GFN2_PARAMETER_SET,
        revision="bannwarth-ehlert-grimme-2019-repacked",
    )
    revised = resolve_xtb_method(
        XtbMethodSpec("GFN2-xTB-repacked", "gfn2", revised_parameters)
    )
    unrestricted = resolve_xtb_method("GFN2-xTB", reference="unrestricted")
    gradient = resolve_xtb_method(
        "GFN2-xTB", requested_products=("energy", "nuclear-gradient")
    )

    assert revised.identity != baseline.identity
    assert unrestricted.identity != baseline.identity
    assert gradient.identity != baseline.identity


def test_runtime_requirements_do_not_smuggle_scc_policy_into_ir() -> None:
    payload = resolve_xtb_method("GFN2-xTB").to_payload()
    encoded = json.dumps(payload, sort_keys=True).lower()

    assert payload["runtime_requirements"]["scc_fixed_point"] is True
    assert payload["runtime_requirements"]["generalized_eigensolution"] is True
    assert payload["runtime_requirements"]["policy_in_compiler_ir"] is False
    for forbidden in (
        "broyden",
        "diis",
        "iteration_limit",
        "maximum_iterations",
        "convergence_tolerance",
        "mixing_history",
        "eigensolver_library",
    ):
        assert forbidden not in encoded


def test_gfn2_parameter_manifest_fails_closed() -> None:
    wrong_name = replace(GFN2_PARAMETER_SET, identifier="gfn2-like")
    with pytest.raises(UnsupportedXtbMethod, match="identified gfn2-xtb"):
        resolve_xtb_method(XtbMethodSpec("bad-name", "gfn2", wrong_name))

    truncated_elements = replace(
        GFN2_PARAMETER_SET,
        supported_atomic_numbers=tuple(range(1, 86)),
    )
    with pytest.raises(UnsupportedXtbMethod, match="atomic numbers 1..86"):
        resolve_xtb_method(XtbMethodSpec("bad-elements", "gfn2", truncated_elements))

    missing_d4 = replace(
        GFN2_PARAMETER_SET,
        correction_tables=("coordination-number", "repulsion"),
    )
    with pytest.raises(UnsupportedXtbMethod, match="not an audited manifest"):
        resolve_xtb_method(XtbMethodSpec("bad-corrections", "gfn2", missing_d4))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"supported_atomic_numbers": (1, 1)}, "duplicate"),
        ({"supported_atomic_numbers": (0, 1)}, r"\[1, 118\]"),
        ({"basis_tables": ("slater-exponents", "slater-exponents")}, "duplicate"),
    ],
)
def test_ambiguous_parameter_manifests_are_rejected_at_construction(
    kwargs: typing.Any, match: typing.Any
) -> None:
    payload = {
        "identifier": "test",
        "revision": "test-v1",
        "source": "test-source",
        "supported_atomic_numbers": (1,),
        "basis_tables": ("atomic-basis-shell-layout",),
        "orbital_tables": ("atomic-levels",),
        "correction_tables": ("repulsion",),
    }
    payload.update(kwargs)
    with pytest.raises(UnsupportedXtbMethod, match=match):
        XtbParameterSet(**payload)


def test_gfn1_is_an_extension_point_not_claimed_capability() -> None:
    spec = XtbMethodSpec(
        "GFN1-extension-only",
        "gfn1",
        GFN2_PARAMETER_SET,
    )
    with pytest.raises(UnsupportedXtbMethod, match="schema extension point only"):
        resolve_xtb_method(spec)


def test_unknown_products_and_names_fail_closed() -> None:
    with pytest.raises(UnsupportedXtbMethod, match="unknown xTB method"):
        resolve_xtb_method("gfn2-xtb")
    with pytest.raises(UnsupportedXtbMethod, match="unsupported compiler products"):
        XtbMethodSpec(
            "bad-product",
            "gfn2",
            GFN2_PARAMETER_SET,
            requested_products=("energy", "hessian"),
        )
    with pytest.raises(UnsupportedXtbMethod, match="also request energy"):
        XtbMethodSpec(
            "gradient-only",
            "gfn2",
            GFN2_PARAMETER_SET,
            requested_products=("nuclear-gradient",),
        )


def test_incomplete_direct_ir_is_rejected() -> None:
    complete = resolve_xtb_method("GFN2-xTB")
    with pytest.raises(UnsupportedXtbMethod, match="complete canonical"):
        XtbMethodIR(
            identifier="incomplete",
            model_flavor=complete.model_flavor,
            parameter_set=complete.parameter_set,
            reference=complete.reference,
            primitives=complete.primitives[:-1],
            requested_products=complete.requested_products,
        )


def test_catalog_is_read_only_and_payload_is_json_serializable() -> None:
    with pytest.raises(TypeError):
        XTB_METHOD_CATALOG["GFN2-xTB"] = XTB_METHOD_CATALOG["GFN2-xTB"]

    payload = resolve_xtb_method("GFN2-xTB").to_payload()
    round_trip = json.loads(json.dumps(payload, sort_keys=True))
    assert round_trip["identifier"] == "GFN2-xTB"
    assert round_trip["model_flavor"] == "gfn2"
    assert round_trip["parameter_set"]["source"] == "doi:10.1021/acs.jctc.8b01176"


@pytest.mark.parametrize(
    "changes",
    [
        {"self_consistent": False},
        {"model": "posthoc-d4"},
        {"requires": ()},
        {"state_requirements": ()},
        {"derivative_capabilities": ("energy",)},
    ],
)
def test_direct_gfn2_graph_cannot_change_audited_primitive_semantics(
    changes: typing.Any,
) -> None:
    method = resolve_xtb_method("GFN2-xTB")
    primitives = (*method.primitives[:-1], replace(method.primitives[-1], **changes))
    with pytest.raises(UnsupportedXtbMethod, match="audited primitive semantics"):
        replace(method, primitives=primitives)
