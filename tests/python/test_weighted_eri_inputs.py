"""Public cotangent pullbacks and bounded density-independent primitive streams."""

from dataclasses import replace

import numpy as np
import pytest

from tools.vibeqc_codegen.blocks import (
    BlockRequest,
    ShellTile,
    TensorLayout,
    WeightTile,
)
from tools.vibeqc_codegen.capabilities import query_integral_capability
from tools.vibeqc_codegen.ir import ContractionOutput
from tools.vibeqc_codegen.shell_signature import BasisConvention, CenterBinding
from tools.vibeqc_codegen.weighted_eri import (
    build_weighted_eri_ir,
    build_weighted_eri_kernel,
)
from tools.vibeqc_codegen.weighted_eri_inputs import (
    PRIMITIVE_RECORD,
    prepare_weighted_eri_stream,
    weighted_eri_response,
)

PRIMITIVES = (((0.6, 0.7), (1.1, -0.2)),) + (((0.8, 1.0),),) * 3
CENTERS = ((0.0, 0.1, 0.2),) * 4


def request_for(integral, offsets=None, shape=None):
    """Resolve the otherwise topology-independent physical center bindings."""
    return BlockRequest(
        "weights",
        integral,
        ShellTile(offsets or (0,) * 4, shape or integral.signature.component_shape),
        center_bindings=tuple(CenterBinding(i, i) for i in range(4)),
    )


def test_padded_partial_weights_are_frozen_and_psss_is_one_record_per_primitive():
    integral = build_weighted_eri_ir((1, 0, 0, 0))
    consumer = integral.contractions[0]
    layout = TensorLayout(consumer.weights.layout.indices, (2, 1, 1, 1), (3, 1, 1, 1))
    consumer = replace(
        consumer,
        weights=replace(consumer.weights, layout=layout, sign=-1, prefactor=0.5),
        output_sign=-1,
    )
    request = request_for(
        replace(integral, contractions=(consumer,)), (1, 0, 0, 0), (2, 1, 1, 1)
    )
    storage = np.array([2.0, 999.0, 999.0, -4.0])
    calls = []

    def provider(descriptor, actual):
        calls.append(actual)
        return WeightTile(descriptor.layout, storage)

    stream = prepare_weighted_eri_stream(
        request, PRIMITIVES, CENTERS, provider, output_tile=7
    )
    first = list(stream.records())
    storage[:] = 17
    assert list(stream.records()) == first
    assert calls == [request]
    assert stream.record_count == 2
    assert stream.host_peak_bytes <= consumer.memory_budget_bytes
    for record, coefficient in zip(first, (0.7, -0.2)):
        fields = PRIMITIVE_RECORD.unpack(record)
        assert fields[:2] == (1, 7)
        np.testing.assert_allclose(
            fields[-3:], coefficient * np.array([0.0, 1.0, -2.0])
        )


def test_spherical_pullback_and_angular_normalization_are_applied_once():
    integral = build_weighted_eri_ir((2, 0, 0, 0))
    signature = replace(
        integral.signature,
        legacy_class=None,
        shells=tuple(
            replace(s, convention=BasisConvention.REAL_SPHERICAL)
            for s in integral.signature.shells
        ),
    )
    consumer = integral.contractions[0]
    layout = TensorLayout(signature.tensor_indices, signature.component_shape)
    integral = replace(
        integral,
        spec=signature,
        contractions=(
            replace(consumer, weights=replace(consumer.weights, layout=layout)),
        ),
    )
    request = request_for(integral)
    matrix = np.arange(30, dtype=float).reshape(6, 5) / 13
    projections = (matrix, np.ones((1, 1)), np.ones((1, 1)), np.ones((1, 1)))
    weights = np.arange(5, dtype=float) - 1.2
    provider = lambda descriptor, _: WeightTile(layout, weights)
    with pytest.raises(ValueError, match="explicit Cartesian"):
        prepare_weighted_eri_stream(request, PRIMITIVES, CENTERS, provider)
    stream = prepare_weighted_eri_stream(
        request, PRIMITIVES, CENTERS, provider, projections=projections
    )
    expected = (matrix @ weights) / np.sqrt([3, 1, 1, 3, 1, 3])
    records = list(stream.records())
    assert len(records) == stream.record_count == 12
    np.testing.assert_allclose(
        [PRIMITIVE_RECORD.unpack(r)[-3] for r in records[:6]], 0.7 * expected
    )


def test_zero_weights_and_fallback_record_count():
    request = request_for(build_weighted_eri_ir((1, 0, 0, 0)))
    for values, count in (([0.0] * 3, 0), ([1.0, 0.0, -2.0], 4)):
        stream = prepare_weighted_eri_stream(
            request,
            PRIMITIVES,
            CENTERS,
            lambda descriptor, _, values=values: WeightTile(descriptor.layout, values),
            generated=False,
        )
        assert len(list(stream.records())) == stream.record_count == count
        assert stream.fused_weights is None


def test_budget_rejection_precedes_provider_and_dense_allocation(monkeypatch):
    integral = build_weighted_eri_ir((3, 3, 3, 3), memory_budget_bytes=65536)
    consumer = integral.contractions[0]
    layout = TensorLayout(consumer.weights.layout.indices, (1, 1, 1, 1))
    integral = replace(
        integral,
        contractions=(
            replace(consumer, weights=replace(consumer.weights, layout=layout)),
        ),
    )
    request = request_for(integral, shape=(1, 1, 1, 1))

    def forbidden(*args, **kwargs):
        raise AssertionError("allocation/provider reached before preflight")

    monkeypatch.setattr(np, "zeros", forbidden)
    with pytest.raises(ValueError, match="budget"):
        prepare_weighted_eri_stream(request, PRIMITIVES, CENTERS, forbidden)


def test_response_maps_physical_atoms_after_translation_with_padded_output():
    integral = build_weighted_eri_ir((1, 0, 0, 0))
    consumer = replace(
        integral.contractions[0],
        output=ContractionOutput.ATOMIC_FORCE,
        output_layout=TensorLayout(("atom", "xyz"), (2, 3), (5, 1)),
    )
    request = BlockRequest(
        "atoms",
        replace(integral, contractions=(consumer,)),
        ShellTile((0,) * 4, (3, 1, 1, 1)),
        center_bindings=tuple(
            CenterBinding(i, atom) for i, atom in enumerate((7, 7, 2, 2))
        ),
    )
    gradient = np.arange(12, dtype=float).reshape(4, 3)
    gradient[3] = -gradient[:3].sum(axis=0)
    response = weighted_eri_response(request, np.r_[9.0, gradient.flat])
    assert response.atom_indices == (2, 7)
    np.testing.assert_allclose(response.values, [-3, -5, -7, 0, 0, 3, 5, 7])


def test_executor_capability_does_not_widen_direct_hf_lowering():
    integral = build_weighted_eri_ir((3, 0, 0, 0))
    assert query_integral_capability(integral, backend="cuda_weighted_eri").supported
    assert not query_integral_capability(integral).supported


def test_noncanonical_center_slots_fail_before_response_or_native_execution():
    integral = build_weighted_eri_ir((1, 0, 0, 0))
    signature = replace(
        integral.signature,
        legacy_class=None,
        shells=tuple(
            replace(s, center=3 - s.center) for s in integral.signature.shells
        ),
    )
    integral = replace(integral, spec=signature)
    request = request_for(integral)
    assert not query_integral_capability(
        integral, backend="cuda_weighted_eri"
    ).supported
    with pytest.raises(ValueError, match="center slots"):
        build_weighted_eri_kernel(integral)
    with pytest.raises(ValueError, match="center slots"):
        prepare_weighted_eri_stream(request, PRIMITIVES, CENTERS, None)
    with pytest.raises(ValueError, match="quartet request"):
        weighted_eri_response(request, np.zeros(13))
