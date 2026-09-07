"""Design tests for operator intent and bounded, density-independent consumers."""

import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tools.vibeqc_codegen.blocks import (
    BlockRequest,
    BlockStatus,
    RawBlock,
    ShellTile,
    TensorLayout,
    WeightDescriptor,
    WeightedDerivative,
    WeightTile,
    assemble_raw_block,
    contract_weighted_derivative,
    unsupported_block_response,
)
from tools.vibeqc_codegen.cache import integral_cache_key
from tools.vibeqc_codegen.capabilities import query_integral_capability
from tools.vibeqc_codegen.cuda_schedule import CudaKernelIR, schedule_candidates
from tools.vibeqc_codegen.ir import (
    ContractionOutput,
    IntegralIR,
    NuclearCenter,
    NuclearCoordinates,
    OperatorFamily,
    OperatorSpec,
    TranslationInvariant,
    build_integral_ir,
)
from tools.vibeqc_codegen.ir_serialization import (
    integral_from_payload,
    integral_to_payload,
)
from tools.vibeqc_codegen.shell_signature import (
    BasisShell,
    CenterBinding,
    ShellSignature,
)
from tools.vibeqc_codegen.shell_spec import FUSED_SHELL_SPECS, PSPS_SPEC


def request_ir(family, *, derivative=False, atoms=None, angular=None):
    """Use separate shell positions even when every slot belongs to one atom."""
    family = OperatorFamily(family)
    roles = {
        OperatorFamily.OVERLAP: ("orbital", "orbital"),
        OperatorFamily.KINETIC: ("orbital", "orbital"),
        OperatorFamily.NUCLEAR_ATTRACTION: ("orbital", "orbital"),
        OperatorFamily.COULOMB_METRIC: ("auxiliary", "auxiliary"),
        OperatorFamily.THREE_CENTER_ERI: ("orbital", "orbital", "auxiliary"),
        OperatorFamily.FOUR_CENTER_ERI: ("orbital",) * 4,
    }[family]
    count = len(roles) + (family == OperatorFamily.NUCLEAR_ATTRACTION)
    atoms = tuple(range(count)) if atoms is None else atoms
    angular = (1,) + (0,) * (len(roles) - 1) if angular is None else angular
    signature = ShellSignature(
        shells=tuple(
            BasisShell(i, i, l, role) for i, (l, role) in enumerate(zip(angular, roles))
        ),
        center_bindings=tuple(CenterBinding(i, atom) for i, atom in enumerate(atoms)),
    )
    operator = OperatorSpec(
        family,
        tuple(range(count)),
        (TranslationInvariant(),),
        external_centers=(NuclearCenter(2, 8.0),)
        if family == OperatorFamily.NUCLEAR_ATTRACTION
        else (),
        permutations=(tuple([1, 0] + list(range(2, len(roles)))),),
    )
    deriv = operator.nuclear_derivative() if derivative else None
    shape = ((count, 3) if derivative else ()) + signature.component_shape
    indices = (("center", "xyz") if derivative else ()) + signature.tensor_indices
    return IntegralIR(
        signature, operator, deriv, (RawBlock(TensorLayout(indices, shape), 65536),)
    )


@pytest.mark.parametrize("family", list(OperatorFamily))
@pytest.mark.parametrize("derivative", [False, True])
def test_operator_inventory_and_serialization(family, derivative):
    integral = request_ir(family, derivative=derivative)
    payload = integral_to_payload(integral)
    assert integral_from_payload(json.loads(json.dumps(payload))) == integral
    assert integral_cache_key(integral_from_payload(payload)) == integral_cache_key(
        integral
    )
    capability = query_integral_capability(integral)
    assert not capability.supported
    assert capability.reasons
    if derivative:
        assert integral.recovered_derivative_centers == (integral.operator.centers[-1],)


def test_attraction_has_a_charged_external_center_and_full_translation():
    integral = request_ir("nuclear_attraction", derivative=True, atoms=(4, 4, 9))
    assert len(integral.signature.shells) == 2
    assert integral.operator.external_centers == (NuclearCenter(2, 8.0),)
    assert integral.independent_derivative_centers == (0, 1)
    assert integral.signature.atom_indices == (4, 4, 9)
    with pytest.raises(ValueError, match="all operator centers"):
        replace(
            integral.operator,
            invariants=(TranslationInvariant(NuclearCoordinates((0, 1))),),
        )


def test_raw_derivatives_recover_before_same_atom_chain_rule():
    integral = request_ir("three_center_eri", derivative=True, atoms=(5, 5, 8))
    request = BlockRequest("raw", integral, ShellTile((0, 0, 0), (3, 1, 1)))
    a = np.arange(9, dtype=float).reshape(3, 3) + 1
    b = 2 * a + 0.25
    response = assemble_raw_block(request, {0: a.ravel(), 1: b.ravel()})
    assert response.status == BlockStatus.OK
    np.testing.assert_allclose(
        np.array(response.values).reshape(3, 3, 3), [a, b, -a - b]
    )
    assert response.center_atoms == ((0, 5), (1, 5), (2, 8))
    with pytest.raises(ValueError, match="independent"):
        assemble_raw_block(request, {0: a.ravel(), 1: b.ravel(), 2: (-a - b).ravel()})


def test_arbitrary_weights_padded_strides_signs_and_atom_accumulation():
    integral = request_ir(
        "four_center_eri", derivative=True, atoms=(7, 7, 2, 2), angular=(1, 1, 0, 0)
    )
    # Two component rows, with padding; these weights have rank two and cannot
    # be replaced by one factorized density product.
    layout = TensorLayout(integral.signature.tensor_indices, (2, 3, 1, 1), (5, 1, 1, 1))
    consumer = WeightedDerivative(
        WeightDescriptor("cc_lagrangian", layout, sign=-1, prefactor=0.5),
        TensorLayout(("atom", "xyz"), (2, 3)),
        65536,
        output=ContractionOutput.ATOMIC_FORCE,
        output_sign=-1,
    )
    integral = replace(integral, contractions=(consumer,))
    request = BlockRequest("weighted", integral, ShellTile((1, 0, 0, 0), (2, 3, 1, 1)))
    weights = np.array([1.0, 2.0, 4.0, 99.0, 99.0, 3.0, 5.0, 9.0])
    inputs = {i: np.arange(18, dtype=float) + 1 + i * 0.5 for i in range(3)}
    calls = []

    def provider(descriptor, tile_request):
        calls.append((descriptor.source, tile_request.tile.offsets))
        return WeightTile(layout, weights)

    response = contract_weighted_derivative(request, inputs, provider)
    w = weights[[0, 1, 2, 5, 6, 7]]
    expected = 0.5 * ((inputs[0] + inputs[1]).reshape(3, 6) @ w)
    # Output atom rows use sorted distinct physical indices, not shell order.
    assert response.atom_indices == (2, 7)
    np.testing.assert_allclose(
        np.array(response.values).reshape(2, 3), [-expected, expected]
    )
    assert calls == [("cc_lagrangian", (1, 0, 0, 0))]
    assert integral_from_payload(integral_to_payload(integral)) == integral


def test_all_centers_on_one_atom_cancel_without_collapsing_position_slots():
    integral = request_ir("overlap", derivative=True, atoms=(3, 3))
    consumer = WeightedDerivative(
        WeightDescriptor(
            "arbitrary", TensorLayout(integral.signature.tensor_indices, (3, 1))
        ),
        TensorLayout(("atom", "xyz"), (1, 3)),
        65536,
        output=ContractionOutput.ATOMIC_FORCE,
    )
    integral = replace(integral, contractions=(consumer,))
    request = BlockRequest("same-atom", integral, ShellTile((0, 0), (3, 1)))
    response = contract_weighted_derivative(
        request,
        {0: tuple(range(9))},
        lambda descriptor, _: WeightTile(descriptor.layout, (1.0, 2.0, 7.0)),
    )
    assert response.values == (0.0, 0.0, 0.0)


def test_raw_value_tiles_are_bounded_and_return_request_metadata():
    integral = request_ir("overlap")
    consumer = RawBlock(TensorLayout(integral.signature.tensor_indices, (1, 1)), 16)
    integral = replace(integral, contractions=(consumer,))
    for offset in range(3):
        request = BlockRequest(str(offset), integral, ShellTile((offset, 0), (1, 1)))
        response = assemble_raw_block(request, (offset + 0.5,))
        assert response.values == (offset + 0.5,)
        assert response.to_payload()["request_id"] == str(offset)
        assert response.to_payload()["tile"]["offsets"] == [offset, 0]
        assert request.required_bytes == 16
    with pytest.raises(ValueError, match="memory budget"):
        BlockRequest(
            "small",
            replace(
                integral, contractions=(replace(consumer, memory_budget_bytes=15),)
            ),
            ShellTile((0, 0), (1, 1)),
        )
    with pytest.raises(ValueError, match="shell bounds"):
        BlockRequest("outside", integral, ShellTile((3, 0), (1, 1)))


def test_unsupported_is_explicit_and_does_not_call_a_provider():
    integral = request_ir("overlap")
    request = BlockRequest("unsupported", integral, ShellTile((0, 0), (3, 1)))
    response = unsupported_block_response(request, backend="cuda")
    assert response.status == BlockStatus.UNSUPPORTED
    assert response.values == () and response.reason
    assert response.to_payload()["status"] == "unsupported"
    for order in (2, 3):
        higher = request_ir("overlap", derivative=True)
        higher = replace(
            higher, derivative=higher.operator.nuclear_derivative(order=order)
        )
        assert not query_integral_capability(higher).supported
        with pytest.raises(ValueError, match="order-one"):
            BlockRequest("higher", higher, ShellTile((0, 0), (3, 1)))


@pytest.mark.parametrize(
    "factory,match",
    [
        (
            lambda: ShellSignature((BasisShell(0, 0, 0),), (CenterBinding(0, 0),)),
            "two, three, or four",
        ),
        (
            lambda: ShellSignature(
                (BasisShell(0, 0, 0), BasisShell(0, 1, 0)),
                (CenterBinding(0, 0), CenterBinding(1, 0)),
            ),
            "slots",
        ),
        (
            lambda: ShellSignature(
                (BasisShell(0, 0, 0), BasisShell(1, 0, 0)), (CenterBinding(0, 0),)
            ),
            "independent",
        ),
        (
            lambda: ShellSignature(
                (BasisShell(0, 0, 2), BasisShell(1, 1, 0, convention="real_spherical")),
                (CenterBinding(0, 0), CenterBinding(1, 1)),
            ),
            "mixed",
        ),
        (lambda: BasisShell(0, 0, True), "integer"),
        (lambda: CenterBinding(0, -1), "non-negative"),
        (lambda: NuclearCenter(2, float("nan")), "charge"),
        (
            lambda: OperatorSpec("nuclear_attraction", (0, 1, 2)),
            "external nuclear center",
        ),
        (
            lambda: OperatorSpec("overlap", (0, 1), permutations=((0, 0),)),
            "permutation",
        ),
        (
            lambda: OperatorSpec(
                "three_center_eri", (0, 1, 2), permutations=((2, 1, 0),)
            ),
            "permutation",
        ),
        (
            lambda: OperatorSpec(
                "overlap",
                (0, 1),
                (TranslationInvariant(), TranslationInvariant(dependent_center=0)),
            ),
            "translation",
        ),
        (lambda: TensorLayout(("a", "b"), (2**62, 4)), "overflow"),
        (lambda: TensorLayout(("a", "b"), (3, 3), (1, 1)), "overlap"),
        (lambda: TensorLayout(("a", "a"), (1, 1)), "unique"),
        (
            lambda: WeightDescriptor(
                "x", TensorLayout(("a",), (1,)), prefactor=float("inf")
            ),
            "finite",
        ),
    ],
)
def test_invalid_semantic_contracts(factory, match):
    with pytest.raises((TypeError, ValueError, OverflowError), match=match):
        factory()


def test_cross_record_mismatches_fail_before_execution():
    integral = request_ir("three_center_eri", derivative=True)
    signature = integral.signature
    with pytest.raises(ValueError, match="auxiliary"):
        replace(
            integral,
            spec=replace(
                signature,
                shells=tuple(replace(s, role="orbital") for s in signature.shells),
            ),
        )
    with pytest.raises(ValueError, match="center inventory"):
        replace(
            integral,
            spec=replace(
                signature,
                center_bindings=signature.center_bindings + (CenterBinding(9, 9),),
            ),
        )
    with pytest.raises(ValueError, match="tensor index"):
        replace(
            integral, contractions=(RawBlock(TensorLayout(("wrong",), (1,)), 1024),)
        )
    with pytest.raises(ValueError, match="shell count"):
        replace(integral, operator=request_ir("four_center_eri").operator)
    with pytest.raises(ValueError, match="schema"):
        integral_from_payload({**integral_to_payload(integral), "schema_version": 0})
    with pytest.raises(ValueError, match="unknown"):
        integral_from_payload({**integral_to_payload(integral), "unrecognized": True})


def test_legacy_catalog_adapter_and_cuda_boundary():
    for spec in FUSED_SHELL_SPECS:
        signature = ShellSignature.from_shell_class(spec)
        assert signature.to_shell_class() == spec
        integral = build_integral_ir(spec)
        assert integral_from_payload(integral_to_payload(integral)) == integral
        assert integral.signature == signature
    legacy = build_integral_ir(PSPS_SPEC)
    assert query_integral_capability(legacy).supported
    generic = request_ir("four_center_eri", derivative=True)
    with pytest.raises(ValueError, match="CUDA"):
        CudaKernelIR(generic, schedule_candidates(legacy)[0])
    with pytest.raises(ValueError, match="CUDA"):
        schedule_candidates(generic)
    assert integral_cache_key(legacy) != integral_cache_key(generic)


def test_weight_provider_layout_and_length_are_checked():
    integral = request_ir("overlap", derivative=True)
    layout = TensorLayout(integral.signature.tensor_indices, (3, 1))
    consumer = WeightedDerivative(
        WeightDescriptor("external", layout),
        TensorLayout(("center", "xyz"), (2, 3)),
        4096,
    )
    integral = replace(integral, contractions=(consumer,))
    request = BlockRequest("bad-weight", integral, ShellTile((0, 0), (3, 1)))
    with pytest.raises(ValueError, match="weight layout"):
        contract_weighted_derivative(
            request,
            {0: tuple(range(9))},
            lambda *_: WeightTile(TensorLayout(("wrong",), (3,)), (1, 2, 3)),
        )
    with pytest.raises(ValueError, match="buffer"):
        contract_weighted_derivative(
            request, {0: tuple(range(9))}, lambda *_: WeightTile(layout, (1, 2))
        )


def test_partial_derivatives_do_not_recover_unrequested_centers():
    integral = request_ir("nuclear_attraction", derivative=True)
    derivative = integral.operator.nuclear_derivative(
        parameters=NuclearCoordinates((2, 0))
    )
    layout = TensorLayout(
        ("center", "xyz") + integral.signature.tensor_indices, (2, 3, 3, 1)
    )
    integral = replace(
        integral, derivative=derivative, contractions=(RawBlock(layout, 4096),)
    )
    assert integral.independent_derivative_centers == (2, 0)
    assert integral.recovered_derivative_centers == ()
    request = BlockRequest("subset", integral, ShellTile((0, 0), (3, 1)))
    response = assemble_raw_block(request, {0: (2.0,) * 9, 2: (1.0,) * 9})
    assert response.values == (1.0,) * 9 + (2.0,) * 9
    assert response.to_payload()["derivative_centers"] == [2, 0]


def test_runtime_atom_bindings_and_shell_indices_are_explicit():
    integral = build_integral_ir(PSPS_SPEC)
    layout = TensorLayout(
        ("center", "xyz") + integral.signature.tensor_indices, (4, 3, 3, 1, 3, 1)
    )
    integral = replace(integral, contractions=(RawBlock(layout, 65536),))
    tile = ShellTile((0, 0, 0, 0), (3, 1, 3, 1))
    with pytest.raises(ValueError, match="resolve every physical"):
        BlockRequest("unbound", integral, tile)
    request = BlockRequest(
        "bound",
        integral,
        tile,
        center_bindings=tuple(
            CenterBinding(c, atom) for c, atom in enumerate((4, 4, 9, 9))
        ),
        shell_indices=(12, 13, 20, 21),
    )
    assert request.to_payload()["shell_indices"] == [12, 13, 20, 21]
    assert request.atom_indices == (4, 9)
    with pytest.raises(ValueError, match="fixed physical"):
        BlockRequest(
            "override",
            request_ir("overlap"),
            ShellTile((0, 0), (3, 1)),
            center_bindings=(CenterBinding(0, 9), CenterBinding(1, 1)),
        )


def test_spherical_intent_and_rys_values_are_separate_from_cuda_support():
    integral = request_ir("coulomb_metric", angular=(2, 0))
    signature = replace(
        integral.signature,
        shells=tuple(
            replace(s, convention="real_spherical") for s in integral.signature.shells
        ),
    )
    consumer = RawBlock(TensorLayout(signature.tensor_indices, (5, 1)), 4096)
    integral = replace(
        integral, spec=signature, contractions=(consumer,), recurrence="rys2"
    )
    assert integral.signature.component_shape == (5, 1)
    assert not query_integral_capability(integral).supported
    assert integral_from_payload(integral_to_payload(integral)) == integral
    with pytest.raises(ValueError, match="Coulomb operator"):
        _ = request_ir("kinetic").required_rys_roots


def test_cache_identity_includes_charge_binding_layout_weights_and_schema():
    integral = request_ir("nuclear_attraction", derivative=True)
    different_charge = replace(
        integral,
        operator=replace(integral.operator, external_centers=(NuclearCenter(2, 7.0),)),
    )
    different_atom = replace(
        integral,
        spec=replace(
            integral.signature,
            center_bindings=(
                CenterBinding(0, 1),
                CenterBinding(1, 1),
                CenterBinding(2, 2),
            ),
        ),
    )
    negative = replace(
        integral, contractions=(replace(integral.contractions[0], output_sign=-1),)
    )
    assert (
        len(
            {
                integral_cache_key(i)
                for i in (integral, different_charge, different_atom, negative)
            }
        )
        == 4
    )
    payload = integral_to_payload(integral)
    payload["contractions"][0]["layout"]["dtype"] = "float32"
    with pytest.raises(ValueError, match="scalar type"):
        integral_from_payload(payload)
    payload = integral_to_payload(integral)
    payload["operator"]["unexpected"] = True
    with pytest.raises(ValueError, match="unknown"):
        integral_from_payload(payload)


def test_serialized_examples_are_reproducible_and_report_unavailable_lowering(tmp_path):
    output = tmp_path / "examples.json"
    subprocess.run(
        [
            sys.executable,
            "tools/generate_integral_ir_examples.py",
            "--output",
            str(output),
        ],
        check=True,
    )
    assert output.read_bytes() == Path("docs/integral_ir_examples.json").read_bytes()
    examples = json.loads(output.read_text())
    assert len(examples) == 3
    for example in examples.values():
        integral = integral_from_payload(example["integral"])
        assert (
            example["cuda_lowering"] == query_integral_capability(integral).to_payload()
        )
        assert not example["cuda_lowering"]["supported"]


def test_production_artifacts_and_catalog_are_byte_identical_to_baseline(tmp_path):
    """Pin the legacy registry/shard contract across this semantic refactor."""
    from tools.vibeqc_codegen.production import write_production_bundles

    baseline = json.loads(
        Path("tests/reference_data/integral_ir_legacy_artifacts.json").read_text()
    )
    manifest = Path("tools/vibeqc_codegen/production_shell_classes.json")
    assert (
        hashlib.sha256(manifest.read_bytes()).hexdigest() == baseline["manifest_sha256"]
    )
    assert [[s.name, list(s.angular)] for s in FUSED_SHELL_SPECS] == baseline["catalog"]
    paths = write_production_bundles(
        manifest,
        tmp_path,
        baseline["shard_count"],
        baseline["architectures"],
        unit_mode=baseline["unit_mode"],
    )
    actual = {
        str(p.relative_to(tmp_path)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths
    }
    assert actual == baseline["artifacts"]


def test_incompatible_production_profiles_are_rejected(tmp_path):
    from tools.vibeqc_codegen.production import resolve_production_profile

    manifest = json.loads(
        Path("tools/vibeqc_codegen/production_shell_classes.json").read_text()
    )
    manifest["architectures"]["sm_120"]["generator_abi"] = 0
    path = tmp_path / "incompatible.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="generator ABI"):
        resolve_production_profile(path, "sm_120")
    manifest["schema_version"] = 0
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest schema"):
        resolve_production_profile(path, "sm_120")


def test_external_weight_cache_keys_include_layout_and_scale():
    integral = request_ir("four_center_eri", derivative=True)
    weights = WeightDescriptor(
        "external", TensorLayout(integral.signature.tensor_indices, (3, 1, 1, 1))
    )
    consumer = WeightedDerivative(
        weights, TensorLayout(("center", "xyz"), (4, 3)), 4096
    )
    variants = (
        weights,
        replace(weights, sign=-1),
        replace(weights, prefactor=0.5),
        replace(weights, source="different"),
        replace(weights, layout=replace(weights.layout, strides=(2, 1, 1, 1))),
    )
    keys = {
        integral_cache_key(
            replace(integral, contractions=(replace(consumer, weights=w),))
        )
        for w in variants
    }
    assert len(keys) == len(variants)
