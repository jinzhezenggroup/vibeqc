"""Public native standard canonical RCCSD(T) energy-owner acceptance for #155 C."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
from generativeqc import Calculator, method_capabilities

from tools.generativeqc_cc.triples import INVENTORY_HASH
from tools.validate_ccsd_t_gradient import analytic_oracle

ROOT = Path(__file__).resolve().parents[2]
GRADIENTS = ROOT / "tests/reference_data/cc/gradients"
TRIPLES = ROOT / "tests/reference_data/cc/rccsd-t.json"


def test_force_rejects_unqualified_larger_cluster() -> None:
    """Larger reference fixtures do not silently broaden the public domain."""
    from benchmarks.readme_hf_scaling import scaling_cases

    atoms = scaling_cases()["water-48"].atoms  # 112 AOs exceeds the 56-AO limit.
    with pytest.raises(
        NotImplementedError, match="force AO dimension exceeds its qualified domain"
    ):
        _calculator().singlepoint(atoms, properties=("energy", "forces"))


@pytest.mark.parametrize(
    "atoms_count",
    (6, 12, 24)
    if os.environ.get("GENERATIVEQC_RCCSDT_FRONTIER_TEST") == "1"
    else (6, 12)
    if os.environ.get("GENERATIVEQC_RCCSDT_LARGE_TEST") == "1"
    else (6,),
)
def test_cluster_force_reference_warm_state_and_directional_energy(
    energy_device: str, atoms_count: int
) -> None:
    """Qualify larger relaxed forces with corrected independent triples Lambda.

    The 14-AO case runs routinely. Explicit large qualification also exercises
    28 AOs. The separate frontier opt-in also covers the internally degenerate
    56-AO reference; these complete response endpoints can take several minutes.
    Regenerate the independent records with benchmarks/ccsdt_cluster_oracle.py.
    """
    oracle = json.loads((GRADIENTS / "water_clusters_ccsdt.json").read_text())
    pack = ROOT / "python/generativeqc/data/basis_pack.json"
    assert hashlib.sha256(pack.read_bytes()).hexdigest() == oracle["basis_pack_sha256"]
    rows = {
        row["geometry"]: row for row in oracle["rows"] if row["atoms"] == atoms_count
    }
    atoms = rows["original"]["inputs"]
    calc = _calculator(
        device=energy_device,
        basis_representation="spherical",
        correlation_memory_budget_bytes=8 << 30,
    )
    with calc.prepare_batch([atoms]) as prepared:
        for geometry, coordinates in (
            ("original", None),
            ("original", None),
            ("changed", [xyz for _, xyz in rows["changed"]["inputs"]]),
        ):
            result = prepared.execute(
                properties=("energy", "forces"), coordinates=[coordinates], strict=True
            ).items[0]
            reference = rows[geometry]
            assert result.converged
            assert result.energy == pytest.approx(reference["energy"], abs=3e-9)
            np.testing.assert_allclose(
                result.forces, reference["forces"], atol=1e-6, rtol=0
            )
            assert result.correlation is not None
            assert result.correlation.ccsd_t_triples_energy == pytest.approx(
                reference["triples"], abs=2e-9
            )
            assert result.correlation.response_absolute_residual < 1e-9
            if geometry == "original":
                original_force = np.asarray(result.forces)

    if atoms_count in (6, 24):
        direction = np.random.default_rng(155214).normal(size=(atoms_count, 3))
        direction -= direction.mean(axis=0)
        direction /= np.linalg.norm(direction)
        analytic = -float(np.vdot(original_force, direction))
        coordinates = np.asarray([xyz for _, xyz in atoms])
        for step in (2e-4, 1e-4):
            energies = []
            for sign in (-1, 1):
                displaced = coordinates + sign * step * direction
                moved = [
                    (z, xyz.tolist())
                    for (z, _), xyz in zip(atoms, displaced, strict=True)
                ]
                energies.append(calc.singlepoint(moved, properties=("energy",)).energy)
            assert abs((energies[1] - energies[0]) / (2 * step) - analytic) < 2e-6


def _reference_case(name: str) -> tuple[list[tuple[int, list[float]]], dict, float]:
    record = json.loads(
        (GRADIENTS / f"{name if name != 'h2' else 'h2_shifted'}.json").read_text()
    )
    triples = {
        item["name"]: item["et_ground_truth"]
        for item in json.loads(TRIPLES.read_text())["molecules"]
    }
    inputs = record["inputs"]
    atoms = list(zip(inputs["atomic_numbers"], inputs["coordinates"], strict=True))
    return atoms, record, triples[name]


@pytest.mark.parametrize("displacement", (0.0, 1e-9))
def test_degenerate_methane_force_and_directional_energy(
    energy_device: str, displacement: float
) -> None:
    """Exact/near internal degeneracy retains the full molecular response.

    Tetrahedral methane has repeated occupied and virtual eigenvalues. The
    independent corrected-Lambda PySCF oracle and complete-energy differences
    guard against a zero, clipped, or double-counted same-space Fock seed.
    """
    reference = json.loads((GRADIENTS / "ch4_degenerate.json").read_text())
    pack = ROOT / "python/generativeqc/data/basis_pack.json"
    assert (
        hashlib.sha256(pack.read_bytes()).hexdigest() == reference["basis_pack_sha256"]
    )
    inputs = reference["inputs"]
    coords = np.asarray([xyz for _, xyz in inputs])
    coords[1, 2] += displacement
    atoms = [(z, xyz.tolist()) for (z, _), xyz in zip(inputs, coords, strict=True)]
    calc = _calculator(device=energy_device, basis_representation="spherical")
    result = calc.singlepoint(atoms, properties=("energy", "forces"))
    assert result.converged and result.correlation is not None
    assert result.energy == pytest.approx(reference["energy"], abs=3e-9)
    assert result.correlation.ccsd_t_triples_energy == pytest.approx(
        reference["triples"], abs=2e-9
    )
    np.testing.assert_allclose(result.forces, reference["forces"], atol=1e-6, rtol=0)
    assert result.correlation.response_absolute_residual < 1e-9
    if displacement == 0:
        direction = np.random.default_rng(1749).normal(size=coords.shape)
        direction -= direction.mean(axis=0)
        direction /= np.linalg.norm(direction)
        analytic = -float(np.vdot(result.forces, direction))
        for step in (2e-4, 1e-4):
            energies = []
            for sign in (-1, 1):
                displaced = coords + sign * step * direction
                moved = [
                    (z, xyz.tolist())
                    for (z, _), xyz in zip(inputs, displaced, strict=True)
                ]
                energies.append(calc.singlepoint(moved, properties=("energy",)).energy)
            assert abs((energies[1] - energies[0]) / (2 * step) - analytic) < 2e-6


def _calculator(**kwargs: object) -> Calculator:
    options: dict[str, object] = {
        "method": "ccsd(t)",
        "basis": "sto-3g",
        "device": "cpu",
        "max_iterations": 200,
        "energy_tolerance": 1e-13,
        "density_tolerance": 1e-11,
        "ccsd_max_iterations": 150,
        "ccsd_energy_tolerance": 1e-13,
        "ccsd_residual_tolerance": 1e-11,
    }
    options.update(kwargs)
    return Calculator(**options)


@pytest.fixture(params=("cpu", "cuda"))
def energy_device(request: pytest.FixtureRequest) -> str:
    if (
        request.param == "cuda"
        and os.environ.get("GENERATIVEQC_RCCSDT_CUDA_TEST") != "1"
    ):
        pytest.skip("requires explicitly allocated CUDA native library")
    return request.param


@pytest.fixture
def cuda_device() -> str:
    if os.environ.get("GENERATIVEQC_RCCSDT_CUDA_TEST") != "1":
        pytest.skip("requires explicitly allocated CUDA native library")
    return "cuda"


def test_native_rccsdt_capability_is_energy_forces_batch() -> None:
    caps = method_capabilities("rccsd(t)")
    alias = method_capabilities("ccsd(t)")
    assert caps.available and caps.supports_batch
    assert caps.family == "coupled_cluster"
    assert caps.supported_properties == frozenset({"energy", "forces"})
    assert alias.available and alias.supported_properties == caps.supported_properties


@pytest.mark.parametrize("case", ("h2", "h2o", "nh3"))
def test_public_native_rccsdt_matches_pinned_standard_triples(
    energy_device: str, case: str
) -> None:
    atoms, reference, triples = _reference_case(case)
    result = _calculator(device=energy_device).singlepoint(
        atoms, properties=("energy",)
    )
    diag = result.correlation
    assert result.converged and result.forces is None
    assert result.executed_backend == (
        "cuda" if energy_device == "cuda" else "cpu_reference"
    )
    assert diag is not None
    assert diag.ccsd_t_triples_energy == pytest.approx(triples, abs=2e-9)
    assert result.energy == pytest.approx(reference["total_energy"] + triples, abs=3e-9)
    assert diag.ccsd_correlation_energy == pytest.approx(
        reference["correlation_energy"], abs=2e-9
    )
    assert diag.ccsd_t_equation_hash == INVENTORY_HASH
    assert diag.ccsd_t_virtual_triples > 0
    assert diag.ccsd_t_workspace_bytes > 0
    perf = result.cc_performance
    assert perf is not None
    assert perf.triples_seconds >= 0.0
    assert perf.reference_seconds >= 0.0
    assert perf.problem_seconds >= perf.provider_seconds >= 0.0
    assert perf.source_scans > 0
    assert perf.transform_stages > 0
    assert perf.iteration_graph_calls >= diag.ccsd_iterations
    assert perf.replay_graph_calls == 1
    assert diag.ccsd_replay_singles_residual_max <= 1e-11
    assert diag.ccsd_replay_doubles_residual_max <= 1e-11
    if energy_device == "cuda":
        assert diag.correlation_owned_device_bytes >= diag.ccsd_t_workspace_bytes
        assert diag.ccsd_setup_h2d_bytes > 0
        assert diag.ccsd_amplitude_d2h_bytes > 0
        assert diag.mo_host_staging


@pytest.mark.parametrize("case", ("h2o", "nh3"))
def test_public_native_rccsdt_force_matches_pyscf_analytic_gradient(case: str) -> None:
    pytest.importorskip(
        "pyscf", reason="independent RCCSD(T) gradient reference requires PySCF"
    )
    pytest.importorskip("threadpoolctl")
    atoms, _, _ = _reference_case(case)
    expected = np.asarray(
        analytic_oracle(case)["analytic"]["gradient"], dtype=np.float64
    )
    result = _calculator().singlepoint(atoms, properties=("energy", "forces"))
    assert result.converged
    assert result.forces is not None
    np.testing.assert_allclose(result.forces, -expected, atol=1.0e-6, rtol=0)
    diag = result.correlation
    assert diag is not None
    assert diag.response_absolute_residual <= 1.0e-9
    assert diag.response_iterations > 0
    assert diag.force_provenance_flags & 0x1


def test_public_native_rccsdt_cuda_force_matches_pyscf_analytic_gradient(
    cuda_device: str,
) -> None:
    pytest.importorskip(
        "pyscf", reason="independent RCCSD(T) gradient reference requires PySCF"
    )
    pytest.importorskip("threadpoolctl")
    atoms, _, _ = _reference_case("h2o")
    expected = np.asarray(
        analytic_oracle("h2o")["analytic"]["gradient"], dtype=np.float64
    )
    result = _calculator(device=cuda_device).singlepoint(
        atoms, properties=("energy", "forces")
    )
    assert result.converged
    assert result.forces is not None
    np.testing.assert_allclose(result.forces, -expected, atol=1.0e-6, rtol=0)
    diag = result.correlation
    assert diag is not None
    assert diag.force_provenance_flags & 0x8
    assert diag.response_absolute_residual <= 1.0e-9
    assert diag.response_iterations > 0


def test_public_native_rccsdt_force_matches_three_step_energy_finite_difference() -> (
    None
):
    atoms, _, _ = _reference_case("h2o")
    rng = np.random.default_rng(15521)
    direction = rng.normal(size=(len(atoms), 3))
    direction -= direction.mean(axis=0, keepdims=True)
    direction /= np.linalg.norm(direction)

    calc = _calculator()
    center = calc.singlepoint(atoms, properties=("energy", "forces"))
    assert center.forces is not None
    analytic = -float(np.vdot(np.asarray(center.forces), direction))
    errors: list[float] = []
    for step in (1.0e-3, 3.0e-4, 1.0e-4):
        energies: list[float] = []
        for sign in (-1.0, 1.0):
            displaced = [
                (z, (np.asarray(xyz) + sign * step * delta).tolist())
                for (z, xyz), delta in zip(atoms, direction, strict=True)
            ]
            energies.append(calc.singlepoint(displaced, properties=("energy",)).energy)
        finite = (energies[1] - energies[0]) / (2.0 * step)
        errors.append(abs(finite - analytic))
    assert min(errors[1:]) < 1.0e-6, errors
    assert errors[-1] < 2.0e-6, errors


def test_public_native_rccsdt_rejects_df_and_frozen_core() -> None:
    with pytest.raises(NotImplementedError, match=r"density fitting"):
        _calculator(density_fitting="cpu")
    with pytest.raises(NotImplementedError, match=r"frozen-core"):
        _calculator(ccsd_frozen_core=1)


def test_public_native_rccsdt_nonconvergence_never_publishes_triples(
    energy_device: str,
) -> None:
    atoms, _, _ = _reference_case("h2o")
    calc = _calculator(device=energy_device, ccsd_max_iterations=1)
    with pytest.raises(RuntimeError, match=r"maximum RCCSD iterations|conver"):
        calc.singlepoint(atoms, properties=("energy",))


def test_public_native_rccsdt_homogeneous_batch_repeats_and_moves_geometry() -> None:
    atoms, reference, triples = _reference_case("h2")
    moved = [(z, [xyz[0], xyz[1], xyz[2] + 0.02]) for z, xyz in atoms]
    calc = _calculator()
    with calc.prepare_batch([atoms, moved]) as prepared:
        first = prepared.execute(strict=True)
        assert all(item.converged for item in first.items)
        assert all(not item.warm_start_used for item in first.items)
        assert first.items[0].energy == pytest.approx(
            reference["total_energy"] + triples, abs=3e-9
        )
        assert all(
            item.correlation is not None
            and item.correlation.ccsd_t_equation_hash == INVENTORY_HASH
            for item in first.items
        )
        forced = prepared.execute(properties=("energy", "forces"), strict=True)
        assert all(item.forces is not None for item in forced.items)
        assert all(
            item.warm_start_used and not item.warm_start_fallback
            for item in forced.items
        )
        repeated = prepared.execute(strict=True)
        assert all(
            item.warm_start_used and not item.warm_start_fallback
            for item in repeated.items
        )
        np.testing.assert_allclose(
            [item.energy for item in repeated.items],
            [item.energy for item in first.items],
            atol=2e-10,
            rtol=0,
        )
        prepared.clear_warm_starts()
        cleared = prepared.execute(strict=True)
        assert all(not item.warm_start_used for item in cleared.items)
        np.testing.assert_allclose(
            [item.energy for item in cleared.items],
            [item.energy for item in first.items],
            atol=2e-10,
            rtol=0,
        )


def test_public_native_rccsdt_checkpoint_restores_hf_warm_state(tmp_path: Path) -> None:
    atoms, _, _ = _reference_case("h2")
    moved = [(z, [xyz[0], xyz[1], xyz[2] + 0.02]) for z, xyz in atoms]
    calc = _calculator()
    checkpoint = tmp_path / "rccsdt-warm.vqcp"

    with calc.prepare_batch([atoms, moved]) as source:
        baseline = source.execute(strict=True)
        source.save_checkpoint(checkpoint)

    with calc.prepare_batch([atoms, moved]) as target:
        report = target.load_checkpoint(checkpoint)
        assert all(item["restored_fields"] == ["density"] for item in report["items"])
        replay = target.execute(strict=True)

    np.testing.assert_allclose(
        [item.energy for item in replay.items],
        [item.energy for item in baseline.items],
        atol=2e-10,
        rtol=0,
    )
    assert all(
        item.warm_start_used and not item.warm_start_fallback for item in replay.items
    )


def test_public_native_rccsdt_cuda_batch_rebuild_and_failure_isolation(
    cuda_device: str,
) -> None:
    atoms, _, _ = _reference_case("h2o")
    moved = [
        (z, [xyz[0], xyz[1], xyz[2] + (0.03 if i == 1 else 0.0)])
        for i, (z, xyz) in enumerate(atoms)
    ]
    calc = _calculator(device=cuda_device)
    expected = calc.singlepoint(moved, properties=("energy",)).energy
    with calc.prepare_batch([atoms, atoms]) as batch:
        first = batch.execute(properties=("energy",), strict=True)
        assert all(item.succeeded and item.forces is None for item in first.items)
        updated = batch.execute(
            coordinates=[None, [xyz for _, xyz in moved]],
            properties=("energy",),
            strict=True,
        )
        assert updated.items[1].energy == pytest.approx(expected, abs=2e-9)
        invalid = batch.execute(
            coordinates=[None, [0.0]], properties=("energy",), strict=False
        )
        assert invalid.items[0].succeeded and not invalid.items[1].succeeded
        assert invalid.items[1].correlation is None


def test_public_native_rccsdt_cuda_batch_forces(
    cuda_device: str,
) -> None:
    atoms, _, _ = _reference_case("h2")
    calc = _calculator(device=cuda_device)
    with calc.prepare_batch([atoms, atoms]) as batch:
        result = batch.execute(properties=("energy", "forces"), strict=True)
    assert all(item.succeeded and item.forces is not None for item in result.items)
    # Independent CUDA derivative reductions may differ by a few FP64 ulps.
    np.testing.assert_allclose(
        result.items[0].forces, result.items[1].forces, atol=1e-12, rtol=0
    )
    assert all(
        item.correlation is not None and item.correlation.force_provenance_flags & 0x8
        for item in result.items
    )


def test_public_force_admits_reported_endpoint_budget() -> None:
    """Admission charges the selected schedule; a tighter cap can shrink MO tiles."""
    atoms, _, _ = _reference_case("h2o")
    reference = _calculator().singlepoint(atoms, properties=("energy", "forces"))
    peak = reference.correlation.planned_endpoint_peak_bytes
    assert peak > 0
    exact = _calculator(correlation_memory_budget_bytes=peak).singlepoint(
        atoms, properties=("energy", "forces")
    )
    assert exact.correlation.numeric_capacity_bytes <= peak
    np.testing.assert_allclose(exact.forces, reference.forces, rtol=0, atol=1e-12)
    # The roomy peak now includes an optional whole-basis MO tile. One byte less
    # may select a smaller source tile instead of rejecting a valid force. The
    # native allocation probe separately enforces the force stage's exact cap
    # and one-byte-short rejection when that unavoidable stage is dominant.
    tighter = _calculator(correlation_memory_budget_bytes=peak - 1).singlepoint(
        atoms, properties=("energy", "forces")
    )
    assert tighter.correlation.numeric_capacity_bytes <= peak - 1
    assert tighter.correlation.planned_endpoint_peak_bytes <= peak - 1
    assert tighter.cc_performance.source_reads > reference.cc_performance.source_reads
    np.testing.assert_allclose(tighter.forces, reference.forces, rtol=0, atol=1e-12)
    with pytest.raises(RuntimeError, match="error 7|host budget|memory budget"):
        _calculator(correlation_memory_budget_bytes=1).singlepoint(
            atoms, properties=("energy", "forces")
        )
