"""Native proposal safeguards, immutable ownership and independent target checks."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np
import pytest
from vibeqc import Calculator

from tools.vibeqc_numerics.audit import ProbeControls, StrictHFAudit, probe_hf
from tools.vibeqc_posthf.sources import NativeSource
from tools.vibeqc_scf import DensityProposal, ScfItem, solve
from tools.vibeqc_scf.proposals import (
    OccupiedProposal,
    RotationProposal,
    diis_density,
    mixing,
)
from tools.vibeqc_scf.replay import (
    TargetOperator,
    counterfactual,
    export_trace,
    load_trace,
    restore_source,
)
from tools.vibeqc_scf.state import metric_root

ATOMS = [(2, (0.0, 0.0, -0.7)), (1, (0.0, 0.0, 0.7))]


def setup(method="rhf", fitted=False):
    charge, multiplicity = (1, 1) if method == "rhf" else (0, 2)
    calc = Calculator(
        method=method,
        density_fitting="cpu" if fitted else "none",
        auxiliary_basis="sto-3g" if fitted else None,
    )
    return NativeSource(
        ATOMS, charge=charge, auxiliary_basis="sto-3g" if fitted else None
    ), calc.resolved_model(ATOMS, charge=charge, multiplicity=multiplicity)


@pytest.mark.parametrize("method", ["rhf", "uhf"])
@pytest.mark.parametrize("fitted", [False, True])
def test_disabled_and_observed_paths_match_legacy_and_physical_fock(method, fitted):
    source, model = setup(method, fitted)
    with source:
        legacy = probe_hf(source, model)
        disabled = solve(source, model)
        traced = solve(source, model, capture=True)
        for result in (disabled, traced):
            assert result.converged, result.error
            assert result.energy == legacy.energy
            assert result.iterations == legacy.iterations
            np.testing.assert_array_equal(result.density, legacy.density)
            np.testing.assert_array_equal(result.forces, legacy.forces)
            assert result.fock_builds == result.iterations + 2
        assert disabled.snapshots == disabled.decisions == ()
        assert traced.trace_complete
        operator = TargetOperator(source, model)
        for state in traced.snapshots:
            energy, fock, residual = operator.evaluate(state.density)
            assert abs(energy - state.energy) < 1e-11
            np.testing.assert_allclose(fock, state.fock, atol=1e-11, rtol=0)
            np.testing.assert_allclose(residual, state.residual, atol=1e-11, rtol=0)
            with pytest.raises(ValueError):
                state.density.setflags(write=True)
        assert traced.snapshots[-1].diagnostics()["stability"] == "not_evaluated"
        assert (
            StrictHFAudit(source, model).evaluate(traced.as_probe())[
                "orthonormal_commutator_max"
            ]
            < 1e-8
        )


@pytest.mark.parametrize("method", ["rhf", "uhf"])
@pytest.mark.parametrize(
    "fault,reason",
    [
        ("charge", "electron_count"),
        ("nan", "nonfinite"),
        ("asymmetric", "symmetry"),
        ("stale", "stale_state"),
        ("negative", "occupation_bounds"),
    ],
)
def test_malformed_outputs_are_rejected_by_native_guard(method, fault, reason):
    source, model = setup(method)

    def propose(state):
        if state.iteration != 1:
            return None
        density = state.density.copy()
        parent = state.identity
        if fault == "charge":
            density *= 2
        elif fault == "nan":
            density[0, 0, 0] = np.nan
        elif fault == "asymmetric":
            density[0, 0, 1] += 1
        elif fault == "stale":
            parent = "another-solve"
        elif fault == "negative":
            # In the orthonormal metric this preserves the trace but has a
            # negative occupation. Testing traces alone would accept it.
            root = metric_root(state.overlap)
            x = np.linalg.inv(root)
            count = state.model.electron_count if method == "rhf" else 2
            density[0] = x @ np.diag([-1.0, count + 1.0]) @ x
        return DensityProposal(parent, density)

    with source:
        result = solve(source, model, proposer=propose, capture=True)
        assert result.converged, result.error
        decision = result.decisions[0]
        assert decision["action"] == "rejected"
        assert decision["reason"] == reason
        assert decision["trials"] == 0
        assert result.fock_builds == result.iterations + 2


def test_ensemble_and_determinant_domains_and_wrong_spin_counts():
    source, model = setup()
    with source:

        def fractional(s):
            return (
                DensityProposal(
                    s.identity, 0.5 * (s.density + s.baseline), "determinant_density"
                )
                if s.iteration == 1
                else None
            )

        result = solve(source, model, proposer=fractional, capture=True)
        assert result.decisions[0]["reason"] == "nonidempotent"
        assert result.converged
    source, model = setup("uhf")
    with source:

        def swapped(s):
            return (
                DensityProposal(s.identity, s.density[::-1])
                if s.iteration == 1
                else None
            )

        result = solve(source, model, proposer=swapped, capture=True)
        assert result.decisions[0]["reason"] == "electron_count"


def test_returning_current_density_cannot_manufacture_convergence():
    source, model = setup()
    with source:
        result = solve(
            source,
            model,
            proposer=lambda s: DensityProposal(s.identity, s.density),
            capture=True,
        )
        assert result.converged and result.iterations > 2
        assert result.decisions[0]["reason"] == "target_operator_no_descent"
        assert result.decisions[0]["trials"] == 4
        assert result.fock_builds == result.iterations + 2 + sum(
            d["trials"] for d in result.decisions
        )
        assert (
            StrictHFAudit(source, model).evaluate(result.as_probe())[
                "orthonormal_commutator_max"
            ]
            < 1e-8
        )


@pytest.mark.parametrize("fault", ["shape", "exception", "type"])
def test_callback_errors_reset_without_escaping_ctypes(fault):
    source, model = setup()

    def bad(state):
        if state.iteration != 1:
            return None
        if fault == "exception":
            raise RuntimeError("deliberate inference failure")
        if fault == "type":
            return np.zeros(4)
        return DensityProposal(state.identity, np.zeros((2, 2)))

    with source:
        result = solve(source, model, proposer=bad, capture=True)
        assert result.converged
        assert result.decisions[0]["action"] == "reset"
        assert result.decisions[0]["adapter_error"]


def test_occupied_and_rotation_invariants_and_timed_projection():
    source, model = setup()
    with source:
        state = solve(source, model, capture=True).snapshots[0]
        root = metric_root(state.overlap)
        _, natural = np.linalg.eigh(root @ state.density[0] @ root)
        c = np.linalg.solve(root, natural[:, ::-1])
        occupied = OccupiedProposal(state.identity, (c[:, :1],)).materialize(state)
        occupied.validate(state)
        assert occupied.repair_seconds > 0
        assert occupied.repair == "occupied_columns_to_density"
        with pytest.raises(ValueError, match="orthonormality"):
            OccupiedProposal(state.identity, (2 * c[:, :1],)).materialize(state)
        rotated = RotationProposal(
            state.identity, (c,), (np.array([[0.4]]),)
        ).materialize(state)
        rotated.validate(state)
        with pytest.raises(ValueError, match="stale"):
            replace(rotated, parent_id="old").validate(state)


@pytest.mark.parametrize("fitted", [False, True])
def test_native_safeguard_matches_independent_counterfactual(fitted):
    source, model = setup(fitted=fitted)
    with source:
        result = solve(source, model, proposer=diis_density, capture=True)
        operator = TargetOperator(source, model)
        assert result.converged
        for state, decision in zip(result.snapshots, result.decisions, strict=True):
            if decision["reason"] in (
                "traditional_convergence",
                "traditional_fallback",
            ):
                continue
            replay = counterfactual(state, diis_density, operator)
            assert replay["action"] == decision["action"]
            assert replay["proposal_trials"] == decision["trials"]
            assert replay["fraction"] == decision["fraction"]
            assert not replay["complete_solve_measurement"]


def test_failed_solve_trace_and_lifetime_roundtrip(tmp_path):
    source, model = setup()
    with source:
        failed = solve(source, model, ProbeControls(max_iterations=2), capture=True)
        assert failed.status == "nonconverged" and failed.density is not None
        assert failed.forces is None
        with pytest.raises(ValueError):
            failed.as_probe()
        path = export_trace(
            failed,
            source,
            tmp_path / "failed.json",
            provenance={"family": "helium-hydride"},
        )
    record, states = load_trace(path)
    assert record["result"]["status"] == "nonconverged"
    assert len(states) == 2
    assert states[0].identity == failed.snapshots[0].identity
    with restore_source(record["source"]) as restored:
        replay = counterfactual(states[0], mixing(0.5), TargetOperator(restored, model))
        assert replay["fock_builds"] >= 2
    with pytest.raises(ValueError, match="new .json"):
        export_trace(failed, source, path)


def test_trace_tampering_and_truncation_are_visible(tmp_path):
    source, model = setup()
    with source:
        result = solve(source, model, capture=True, max_trace_iterations=1)
        assert result.converged and not result.trace_complete
        path = export_trace(result, source, tmp_path / "trace.json")
    path.with_suffix(".npz").write_bytes(b"damaged")
    with pytest.raises(ValueError, match="checksum"):
        load_trace(path)


def test_batch_ownership_rebinding_and_concurrent_results():
    source, model = setup()
    with source:
        items = [ScfItem(source, model), ScfItem(source, model)]
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda item: item.solve(capture=True), items))
        assert all(r.converged for r in results)
        assert results[0].owner != results[1].owner
        old = results[0].snapshots[0]
        assert old.generation != results[1].snapshots[0].generation
        stale = lambda s: (
            DensityProposal(old.identity, old.density) if s.iteration == 1 else None
        )
        result = items[1].solve(proposer=stale, capture=True)
        assert result.decisions[0]["reason"] == "stale_state"
        items[0].rebind(source, model)
        assert items[0].last_result is None
        assert items[0].solve(capture=True).owner != results[0].owner

        def recursive(s):
            items[0].rebind(source, model)

        result = items[0].solve(proposer=recursive, capture=True)
        assert result.converged
        assert "active SCF item" in result.decisions[0]["adapter_error"]
        with pytest.raises(ValueError, match="identity"):
            items[0].rebind(source, replace(model, geometry_hash="stale"))


def test_invalid_initial_guess_is_not_silently_rescaled():
    source, model = setup()
    with source:
        result = solve(source, model)
        with pytest.raises(ValueError, match="electron_count"):
            solve(source, model, initial_density=result.density * 2)
        warm = solve(source, model, initial_density=result.density)
        assert warm.converged and warm.iterations == 2
        assert abs(warm.energy - result.energy) < 1e-12


@pytest.mark.parametrize("gauge_seed", [None, 7, 19])
def test_real_operator_damps_a_valid_but_overlarge_orbital_rotation(gauge_seed):
    atoms = [(8, (0.0, 0.0, 0.0)), (1, (0.0, -1.43, 1.1)), (1, (0.0, 1.43, 1.1))]
    model = Calculator().resolved_model(atoms)

    def proposal(state):
        if state.iteration != 1:
            return None
        root = metric_root(state.overlap)
        _, v = np.linalg.eigh(root @ state.density[0] @ root)
        c = np.linalg.solve(root, v[:, ::-1])
        # Density eigenvalues are degenerate inside each occupied/virtual
        # subspace, so LAPACK can return any orthogonal gauge. Derive the
        # rotation covariantly from the physical Fock; fixed OV coefficients
        # would propose different AO densities on different BLAS builds.
        if gauge_seed is not None:
            rng = np.random.default_rng(gauge_seed)
            c[:, :5] = c[:, :5] @ np.linalg.qr(rng.normal(size=(5, 5)))[0]
            c[:, 5:] = c[:, 5:] @ np.linalg.qr(rng.normal(size=(2, 2)))[0]
        rotation = -0.1 * (c[:, :5].T @ state.fock[0] @ c[:, 5:])
        return RotationProposal(state.identity, (c,), (rotation,))

    with NativeSource(atoms) as source:
        result = solve(source, model, proposer=proposal, capture=True)
        assert result.converged
        decision = result.decisions[0]
        assert decision["action"] == "damped"
        assert decision["fraction"] == 0.25
        assert decision["trials"] == 3
        assert decision["repair_seconds"] > 0
        replay = counterfactual(
            result.snapshots[0], proposal, TargetOperator(source, model)
        )
        assert replay["action"] == "damped" and replay["fraction"] == 0.25


def test_spin_change_invalidates_previous_proposals_and_supports_empty_beta():
    source, singlet = setup()
    triplet = Calculator(method="uhf").resolved_model(ATOMS, charge=1, multiplicity=3)
    with source:
        item = ScfItem(source, singlet)
        old = item.solve(capture=True).snapshots[0]
        item.rebind(source, triplet)
        result = item.solve(
            capture=True, proposer=lambda s: DensityProposal(old.identity, old.density)
        )
        assert result.converged
        assert result.owner != old.owner
        assert result.decisions[0]["adapter_error"] == "ValueError: shape"
        assert result.density.shape == (2, 2, 2)
        np.testing.assert_allclose(result.density[1], 0, atol=1e-12)
        audit = StrictHFAudit(source, triplet).evaluate(result.as_probe())
        assert audit["electron_trace_error_max"] < 1e-10
        with pytest.raises(ValueError, match="minimum-spin"):
            probe_hf(source, triplet)


def test_repeated_bad_model_is_disabled_so_traditional_diis_can_recover():
    source, model = setup()
    calls = []

    def broken(state):
        calls.append(state.iteration)
        return DensityProposal(state.identity, 2 * state.density)

    with source:
        result = solve(source, model, proposer=broken, capture=True)
        assert result.converged
        assert calls == [1, 2, 3]
        assert any(d["reason"] == "traditional_fallback" for d in result.decisions)
        assert result.fock_builds == result.iterations + 2


def test_transport_and_extrapolation_are_explicit_new_validated_states():
    from tools.vibeqc_scf.baselines import transported_density

    source, model = setup()
    with source:
        old = solve(source, model, capture=True).snapshots[-1]
    moved_atoms = [(z, (x, y, 1.02 * zcoord)) for z, (x, y, zcoord) in ATOMS]
    moved_model = Calculator().resolved_model(moved_atoms, charge=1)
    with NativeSource(moved_atoms, charge=1) as moved:
        seed, repair = transported_density(old, moved_model, moved.one_electron()[0])
        assert repair["repair_seconds"] > 0
        assert repair["target_model_id"] != old.model.identity
        projected = solve(moved, moved_model, initial_density=seed, capture=True)
        assert projected.converged
        extrapolated, info = transported_density(
            projected.snapshots[-1], moved_model, moved.one_electron()[0], older=old
        )
        assert info["repair"] == "projected_density_extrapolation"
        assert solve(moved, moved_model, initial_density=extrapolated).converged
        with pytest.raises(ValueError, match="compatible"):
            transported_density(
                old, replace(moved_model, method="uhf"), moved.one_electron()[0]
            )


def test_dataset_split_rejects_adjacent_geometry_leakage(tmp_path):
    import json

    from vibeqc.profiles import canonical_hash

    from tools.validate_scf_proposals import REFERENCES, load_references

    record = json.loads(REFERENCES.read_text())
    record["cases"][0]["split"] = "training"
    record.pop("record_hash")
    record["record_hash"] = canonical_hash(record)
    path = tmp_path / "leaked.json"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="leakage"):
        load_references(path)


def test_snapshot_altered_operator_is_rejected_before_counterfactual():
    source, model = setup()
    with source:
        state = solve(source, model, capture=True).snapshots[0]
        shifted_fock = state.fock + 0.1 * state.overlap
        # Adding S leaves the commutator unchanged. Snapshot-local invariants
        # alone cannot establish that this is the declared physical operator.
        shifted = replace(state, fock=shifted_fock)
        with pytest.raises(ValueError, match="differs from the target"):
            counterfactual(shifted, None, TargetOperator(source, model))


def test_malformed_model_output_is_bounded_before_ownership_copy():
    with pytest.raises(ValueError, match="output budget"):
        DensityProposal("parent", np.zeros((2, 13, 13)))
    with pytest.raises(ValueError, match="output budget"):
        OccupiedProposal("parent", (np.zeros((13, 13)),))


def test_failed_batch_item_preserves_its_trace_without_poisoning_neighbor():
    source, model = setup()
    with source:
        items = (ScfItem(source, model), ScfItem(source, model))
        controls = (ProbeControls(max_iterations=1), ProbeControls())
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = tuple(
                pool.map(
                    lambda args: args[0].solve(controls=args[1], capture=True),
                    zip(items, controls, strict=True),
                )
            )
        assert results[0].status == "nonconverged"
        assert results[0].fock_builds == 1 and len(results[0].snapshots) == 1
        assert results[1].converged and results[1].owner != results[0].owner
        recovered = items[0].solve(capture=True)
        assert recovered.converged and recovered.energy == results[1].energy
        assert recovered.snapshots[0].generation != results[0].snapshots[0].generation
