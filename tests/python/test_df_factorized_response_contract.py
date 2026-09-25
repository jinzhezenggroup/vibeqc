"""Exact contract for the #419 factorized occupied-response candidate."""

from pathlib import Path

import numpy as np


def test_factorized_packed_exchange_matches_materialized_matrix() -> None:
    rng = np.random.default_rng(419)
    nbf, rank, panels = 11, 3, 5
    coefficients = rng.normal(size=(nbf, rank))
    projected_u = rng.normal(size=(panels, rank, rank))
    projected_u = 0.5 * (projected_u + projected_u.transpose(0, 2, 1))
    prefactor = -0.375

    for u in projected_u:
        cu = coefficients @ u
        materialized = prefactor * cu @ coefficients.T
        for hi in range(nbf):
            for lo in range(hi + 1):
                folded = materialized[hi, lo] * (1 if hi == lo else 2)
                direct = prefactor * np.dot(coefficients[lo], cu[hi])
                direct *= 1 if hi == lo else 2
                np.testing.assert_allclose(direct, folded, rtol=2e-15, atol=2e-15)


def test_factorized_fusion_is_explicit_ablation_not_default() -> None:
    root = Path(__file__).resolve().parents[2]
    bridge = (root / "src/scf/cuda/df_gradient_bridge.cu").read_text()
    force_owner = (root / "src/scf/cuda/df_force_response.cpp").read_text()
    producer = (root / "src/scf/cuda/df_response_weights.cu").read_text()
    consumer = (root / "src/scf/cuda/df_shell_kernel.cuh").read_text()

    assert 'fusion_control ? fusion_control : "off"' in bridge
    assert 'fusion_policy != "off" && fusion_policy != "factorized"' in bridge
    assert "factorized_exchange && packed_pairs && terms.size() == 1" in producer
    assert "response_factorized_exchange_panels" in producer
    assert "const bool response_packed_pairs =" in bridge
    assert "packed_pairs || (packed_request && owned_occupied)" in bridge
    streamed = producer.split("if (streamed_occupied) {", 1)[1].split(
        "if (single_fitted_tensor)", 1
    )[0]
    # Check argument/guard semantics, not clang-format's line wrapping.
    streamed = "".join(streamed.split())
    assert "read_fitted||packed_pairs" not in streamed
    assert "consume,packed_pairs,auxiliary_shell_offsets" in streamed
    assert "read_values,factorized_exchange" in streamed
    assert (
        '!borrow && plan->integral_source && metric.full_rank && space != "dense"'
        in force_owner
    )
    assert "owned_factors.owner_identity ? &owned_factors : nullptr" in force_owner
    compact_bridge = "".join(bridge.split())
    assert "whitened&&!borrowed&&!owned_occupied" in compact_bridge
    assert "!borrowed&&!whitened&&!owned_occupied" in compact_bridge
    assert "factorized.coefficients[lo + k * o.nbf]" in consumer
    assert "factorized.projected + panel * o.nbf * factorized.rank" in consumer
