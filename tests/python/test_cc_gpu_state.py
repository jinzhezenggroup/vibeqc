"""Pre-device CC contracts: physical equations, shifts, and warm-start identity."""

from dataclasses import replace
from itertools import pairwise

import numpy as np
import pytest
from vibeqc_compiler.integral.cuda_target import cuda_target_info
from vibeqc_compiler.tensor import execute

from tools.vibeqc_cc.gpu_state import (
    AmplitudeSnapshot,
    denominators,
    iteration_program,
    solver_plans,
    state_reservation,
)
from tools.vibeqc_cc.oracle import dense_feeds, random_case
from tools.vibeqc_cc.solver import SolverOptions
from tools.vibeqc_posthf.fixtures import fixture_snapshot, load_fixture


@pytest.mark.parametrize("shape", [(1, 3), (2, 3)])
def test_iteration_denominators_do_not_change_physical_residual(shape):
    feeds = dense_feeds(*random_case(*shape, seed=149))
    o, v = shape
    d1 = -np.arange(1, o * v + 1, dtype=np.float64).reshape(o, v)
    d2 = d1[:, None, :, None] + d1[None, :, None, :]
    program = iteration_program(*shape, damping=0.125)
    unshifted = execute(program, {**feeds, "d1": d1, "d2": d2}).outputs
    shifted = execute(program, {**feeds, "d1": d1 - 0.4, "d2": d2 - 0.8}).outputs
    for name in ("correlation_energy", "singles_residual", "doubles_residual"):
        np.testing.assert_array_equal(unshifted[name], shifted[name])
    for k, r, d in ((1, "singles_residual", d1), (2, "doubles_residual", d2)):
        np.testing.assert_allclose(
            unshifted[f"next_t{k}"], feeds[f"t{k}"] + 0.875 * unshifted[r] / d
        )
        assert np.max(np.abs(unshifted[f"next_t{k}"] - shifted[f"next_t{k}"])) > 1e-7


def test_warm_start_is_owned_and_invalidates_same_shape_reference_change():
    snapshot = fixture_snapshot(*load_fixture("water"))
    o, v = snapshot.nocc, snapshot.nmo - snapshot.nocc
    t1, t2 = random_case(o, v)[2:]
    warm = AmplitudeSnapshot(snapshot.identity, t1, t2)
    t1[:] = 99
    assert np.max(np.abs(warm.for_reference(snapshot)[0])) < 1
    with pytest.raises(ValueError, match="WRITEABLE"):
        warm.t2.setflags(write=True)
    rotated = snapshot.coefficients.copy()
    rotated[:, 0] *= -1
    for changed in (
        replace(snapshot, geometry_hash="changed"),
        replace(snapshot, generation_id="changed"),
        replace(snapshot, coefficients=rotated),
    ):
        with pytest.raises(ValueError, match="invalidated"):
            warm.for_reference(changed)
    bad = np.zeros_like(t2)
    bad.flat[1] = 1
    with pytest.raises(ValueError):
        AmplitudeSnapshot(snapshot.identity, np.zeros_like(t1), bad)
    with pytest.raises(ValueError, match="FP64"):
        AmplitudeSnapshot(snapshot.identity, np.zeros_like(t1, dtype=np.float32), t2)


def test_denominators_preserve_physical_gaps_and_shift_multiplicity():
    snapshot = fixture_snapshot(*load_fixture("water"))
    d1, d2 = denominators(snapshot, SolverOptions())
    shifted1, shifted2 = denominators(snapshot, SolverOptions(level_shift=0.4))
    np.testing.assert_allclose(shifted1, d1 - 0.4)
    np.testing.assert_allclose(shifted2, d2 - 0.8)
    with pytest.raises(ValueError, match="physical"):
        denominators(snapshot, SolverOptions(denominator_threshold=100))


def test_solver_plan_keeps_independent_replay_and_all_state_under_budget():
    target = cuda_target_info("sm_90")
    primary, replay, diagnostic = solver_plans(
        2, 3, target, SolverOptions(), provider_peak_bytes=1 << 20
    )
    assert "doubles_residual" in replay.program.outputs
    assert "next_t2" in primary.program.outputs
    assert diagnostic["combined_peak_bytes"] <= 256 << 20
    assert diagnostic[
        "combined_peak_bytes"
    ] == primary.peak_bytes + replay.peak_bytes + (1 << 20)
    reservation, segments = state_reservation(2, 3, 6)
    assert primary.reservations == reservation
    assert segments["diis_vectors"]["bytes"] == 6 * (2 * 3 + 2**2 * 3**2) * 8
    spans = [(s["offset"], s["offset"] + s["bytes"]) for s in segments.values()]
    assert all(right <= next_left for (_, right), (next_left, _) in pairwise(spans))
    assert spans[-1][1] <= reservation.total
    with pytest.raises(ValueError, match="budget"):
        solver_plans(2, 3, target, SolverOptions(max_bytes=128 << 20))
    with pytest.raises(ValueError, match="budget"):
        solver_plans(
            2, 3, target, SolverOptions(max_bytes=1 << 20), provider_peak_bytes=1 << 20
        )


def test_gpu_cc_reference_helpers_reject_ks_snapshots():
    """Shared CPKS snapshots must not widen the RCCSD reference contract."""
    snapshot = replace(
        fixture_snapshot(*load_fixture("water")),
        algorithm="KS",
        functional_identity="synthetic-functional",
        grid_identity="synthetic-grid",
    )
    with pytest.raises(ValueError, match="RHF reference"):
        denominators(snapshot, SolverOptions())
    t1, t2 = random_case(snapshot.nocc, snapshot.nmo - snapshot.nocc)[2:]
    warm = AmplitudeSnapshot(snapshot.identity, t1, t2)
    with pytest.raises(ValueError, match="RHF reference"):
        warm.for_reference(snapshot)
