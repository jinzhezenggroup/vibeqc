"""Versioned second-order consumer intent and fail-closed legacy boundaries."""

import copy
import json
from dataclasses import replace

import pytest
from vibeqc_compiler.integral.blocks import RawBlock, TensorLayout
from vibeqc_compiler.integral.capabilities import query_integral_capability
from vibeqc_compiler.integral.ir import four_center_eri_operator
from vibeqc_compiler.integral.ir_serialization import (
    integral_from_payload,
    integral_to_payload,
)
from vibeqc_compiler.integral.range_separation import CoulombKernel
from vibeqc_compiler.integral.second_derivatives import (
    build_eri_second_ir,
    build_one_electron_second_ir,
    build_second_derivative_kernel,
)
from vibeqc_compiler.integral.second_derivatives_execute import (
    CompiledSecondDerivative,
    SecondPrimitive,
    pack_second_primitive,
)
from vibeqc_compiler.integral.second_derivatives_inputs import (
    prepare_second_shell_stream,
)
from vibeqc_compiler.integral.second_derivatives_native import (
    emit_second_derivative_primitive,
)
from vibeqc_compiler.integral.weighted_eri import build_weighted_eri_ir


@pytest.mark.parametrize(
    "output,packing",
    [
        ("raw_hessian", "dense"),
        ("raw_hessian", "svec"),
        ("weighted_hessian", "dense"),
        ("weighted_hessian", "svec"),
        ("weighted_hvp", "dense"),
    ],
)
@pytest.mark.parametrize("family", ["overlap", "kinetic", "nuclear_attraction", "eri"])
def test_second_order_round_trip_and_first_abi_rejection(family, output, packing):
    ir = (
        build_eri_second_ir((2, 1, 0, 1), output=output, packing=packing)
        if family == "eri"
        else build_one_electron_second_ir(
            family, (1, 2), output=output, packing=packing
        )
    )
    payload = integral_to_payload(ir)
    assert payload["schema_version"] == 3
    assert integral_from_payload(json.loads(json.dumps(payload))) == ir
    assert not ir.consumers
    for backend in (
        "cuda",
        "cuda_one_electron_derivatives",
        "cuda_weighted_eri",
        "cpu_range_weighted_eri",
    ):
        check = query_integral_capability(ir, backend=backend)
        assert not check.supported and check.reasons
    for version in (1, 2, True, 4):
        with pytest.raises((ValueError, TypeError)):
            integral_from_payload({**payload, "schema_version": version})
    with pytest.raises(ValueError, match="order two"):
        replace(ir, derivative=replace(ir.derivative, order=1))


def test_strict_second_consumer_decoder_never_defaults_weights_or_direction():
    ir = build_eri_second_ir((1, 0, 0, 0))
    original = integral_to_payload(ir)
    for field in original["contractions"][0]:
        payload = copy.deepcopy(original)
        del payload["contractions"][0][field]
        with pytest.raises(ValueError):
            integral_from_payload(payload)
    for field, value in [
        ("weights", None),
        ("direction_source", None),
        ("packing", "svec"),
        ("output", "first_gradient"),
        ("output_sign", True),
    ]:
        payload = copy.deepcopy(original)
        payload["contractions"][0][field] = value
        with pytest.raises(ValueError):
            integral_from_payload(payload)
    payload = copy.deepcopy(original)
    payload["contractions"][0]["output_layout"]["indices"] = ["atom", "xyz"]
    with pytest.raises(ValueError, match="axes"):
        integral_from_payload(payload)


@pytest.mark.parametrize("kind", ["raw_first", "weighted_first", "second_first"])
def test_mixed_consumers_fail_before_artifact_or_record_access(kind):
    """Even manually forged artifacts must reject general multi-consumer IR."""
    ir = build_eri_second_ir((1, 0, 0, 0))
    raw = RawBlock(
        TensorLayout(
            ("center", "xyz", *ir.signature.tensor_indices),
            (4, 3, *ir.signature.component_shape),
        ),
        1 << 20,
    )
    extra = (
        build_weighted_eri_ir((1, 0, 0, 0)).contractions[0]
        if kind == "weighted_first"
        else raw
    )
    consumers = (
        (*ir.contractions, extra)
        if kind == "second_first"
        else (extra, *ir.contractions)
    )
    mixed = replace(ir, contractions=consumers)
    # The general representation can round trip this request, while the
    # bounded native provider must fail before accessing a native library.
    assert integral_from_payload(integral_to_payload(mixed)) == mixed
    artifact = CompiledSecondDerivative(None, mixed, (0,), (0,), "cpu", "forged")
    kernel = build_second_derivative_kernel(ir, (0,), output_indices=(0,))
    primitive = SecondPrimitive((), ())
    for operation in (
        artifact.validate,
        lambda: pack_second_primitive(artifact, primitive),
        lambda: prepare_second_shell_stream(artifact, (), ()),
        lambda: build_second_derivative_kernel(mixed, (0,), output_indices=(0,)),
        lambda: emit_second_derivative_primitive(replace(kernel, integral=mixed)),
    ):
        with pytest.raises(ValueError, match="one explicit second-order consumer"):
            operation()


def test_range_second_intent_retains_omega_but_lowering_does_not_claim_support():
    operator = four_center_eri_operator(CoulombKernel("long_range", 0.123456789))
    ir = build_eri_second_ir((1, 0, 0, 0), operator=operator)
    payload = integral_to_payload(ir)
    assert payload["operator"]["omega"] == operator.omega
    assert integral_from_payload(json.loads(json.dumps(payload))) == ir
    with pytest.raises(ValueError, match="full four-center"):
        build_second_derivative_kernel(ir, (0,))


def test_lowering_requires_explicit_raw_and_bounded_output_selections():
    ir = build_one_electron_second_ir("kinetic", (1, 2))
    with pytest.raises(ValueError, match="one explicit AO"):
        build_second_derivative_kernel(ir)
    for indices in ((), (0, 0), (-1,), (36,), (True,)):
        with pytest.raises(ValueError, match="output indices"):
            build_second_derivative_kernel(ir, (0,), output_indices=indices)
    ir = replace(ir, contractions=(replace(ir.contractions[0], memory_budget_bytes=7),))
    with pytest.raises(ValueError, match="memory budget"):
        build_second_derivative_kernel(ir, (0,), output_indices=(0,))


def test_native_capability_is_separate_from_first_force_and_requires_coordinate_tiles():
    from vibeqc_compiler.integral.second_order_layout import second_coordinate_tiles

    raw = build_eri_second_ir((3, 0, 0, 0), output="raw_hessian")
    for backend in ("cpu_second_derivatives", "cuda_second_derivatives"):
        assert not query_integral_capability(
            raw, backend=backend, component_indices=(0,)
        ).supported
        assert query_integral_capability(
            raw, backend=backend, component_indices=(0,), output_indices=(0, 13)
        ).supported
        assert not query_integral_capability(
            raw, backend=backend, component_indices=(0,), output_indices=(True,)
        ).supported
    tiles = second_coordinate_tiles(raw.requested_derivative_centers)
    assert tuple(i for tile in tiles for i in tile) == tuple(range(144))
    assert max(map(len, tiles)) == 6
    assert not query_integral_capability(raw, output_indices=(0,)).supported
