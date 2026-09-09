"""Persistent restart tests use new processes and real native fleet execution.

Set VIBEQC_CHECKPOINT_DEVICE=cuda and select a CUDA build under Slurm to run
these same scientific/robustness contracts on the actual accelerator.
"""

import ctypes
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from vibeqc import Calculator, _native
from vibeqc.checkpoint import _HEADER, CheckpointError, _json, _read, inspect_checkpoint
from vibeqc.resources import ResourceBudget

DEVICE = os.environ.get("VIBEQC_CHECKPOINT_DEVICE", "cpu")
H2 = [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))]
H2_MOVED = [("H", (0.0, 0.0, -0.8)), ("H", (0.0, 0.0, 0.8))]
HE = [("He", (0.0, 0.0, 0.0))]
ROOT = Path(__file__).resolve().parents[2]


def calc(**kwargs):
    return Calculator(device=DEVICE, **kwargs)


def rewrite(path, mutate_manifest=lambda doc: None, mutate_arrays=lambda arrays: None):
    """Generate semantically bad yet checksum-correct files to reach validators."""
    manifest, arrays, _ = _read(path, 1 << 20)
    doc = manifest.to_dict()
    arrays = {name: array.copy() for name, array in arrays.items()}
    mutate_arrays(arrays)
    offset = 0
    data = []
    for blob in doc["blobs"]:
        raw = arrays[blob["name"]].astype("<f8").tobytes()
        blob.update(
            sha256=hashlib.sha256(raw).hexdigest(), offset=offset, bytes=len(raw)
        )
        offset += len(raw)
        data.append(raw)
    mutate_manifest(doc)
    raw = _json(doc)
    path.write_bytes(
        _HEADER.pack(b"VQHFCP01", len(raw), hashlib.sha256(raw).digest())
        + raw
        + b"".join(data)
    )


def save(path, systems=None, **kwargs):
    with calc(**kwargs).prepare_batch(systems or [H2]) as batch:
        result = batch.execute()
        batch.save_checkpoint(path)
        return result


@pytest.mark.parametrize("method,charge,multiplicity", [("rhf", 0, 1), ("uhf", 1, 2)])
@pytest.mark.parametrize("fitted", [False, True])
def test_fresh_process_restart_and_independent_reference(
    tmp_path, method, charge, multiplicity, fitted
):
    checkpoint = tmp_path / "hf.vqcp"
    options = {
        "method": method,
        "device": DEVICE,
        "density_fitting": DEVICE if fitted else "none",
    }
    # Producer and consumer have separate library contexts, streams and heaps.
    script = """
import json, sys
from vibeqc import Calculator
options=json.loads(sys.argv[1]); systems=json.loads(sys.argv[2]); charge=int(sys.argv[3]); multiplicity=int(sys.argv[4])
with Calculator(**options).prepare_batch(systems,charges=[charge],multiplicities=[multiplicity]) as b:
    if sys.argv[6]=='read': b.load_checkpoint(sys.argv[5])
    r=b.execute(strict=True).items[0]
    if sys.argv[6]=='write': b.save_checkpoint(sys.argv[5])
    print(json.dumps(dict(energy=r.energy,forces=r.forces.tolist(),origin=r.restart_origin,iterations=r.iterations,converged=r.converged)))
"""
    args = [
        sys.executable,
        "-c",
        script,
        json.dumps(options),
        json.dumps([H2]),
        str(charge),
        str(multiplicity),
        str(checkpoint),
    ]
    outputs = [
        json.loads(subprocess.check_output(args + [mode], text=True))
        for mode in ("write", "read")
    ]
    cold, restored = outputs
    assert cold["origin"] == "cold" and restored["origin"] == "persistent_restart"
    assert restored["converged"] and restored["iterations"] > 0
    assert restored["energy"] == pytest.approx(cold["energy"], abs=2e-10)
    np.testing.assert_allclose(restored["forces"], cold["forces"], atol=1e-8, rtol=0)
    manifest = inspect_checkpoint(checkpoint).to_dict()
    assert manifest["items"][0]["model"]["method"] == method
    assert {b["name"] for b in manifest["blobs"]} == {"density_0", "coordinates_0"}
    # Direct HF also compares complete energies/forces to independent PySCF.
    # The fitted provider's exact auxiliary/metric semantics have separate
    # established DF gates; this test preserves its cold Hamiltonian result.
    if not fitted:
        pyscf = pytest.importorskip("pyscf")
        mol = pyscf.gto.M(
            atom=H2,
            unit="Bohr",
            basis="sto-3g",
            charge=charge,
            spin=multiplicity - 1,
            verbose=0,
        )
        mf = (pyscf.scf.RHF(mol) if method == "rhf" else pyscf.scf.UHF(mol)).run(
            conv_tol=1e-12
        )
        assert restored["energy"] == pytest.approx(mf.e_tot, abs=2e-9)
        np.testing.assert_allclose(
            restored["forces"], -mf.nuc_grad_method().kernel(), atol=1e-8, rtol=0
        )


@pytest.mark.parametrize("method,multiplicities", [("rhf", [1]), ("uhf", [1])])
def test_changed_geometry_is_explicit_warm_start_and_rechecks_target(
    tmp_path, method, multiplicities
):
    path = tmp_path / "state"
    with calc(method=method).prepare_batch(
        [H2], multiplicities=multiplicities
    ) as source:
        source.execute(strict=True)
        source.save_checkpoint(path)
    with calc(method=method).prepare_batch(
        [H2_MOVED], multiplicities=multiplicities
    ) as target:
        with pytest.raises(CheckpointError, match="geometry or numerical"):
            target.load_checkpoint(path)
        report = target.load_checkpoint(path, allow_warm=True)
        assert report["items"][0]["compatibility"] == "warm_start_compatible"
        assert report["target_verification"] == "pending_execute"
        result = target.execute(strict=True).items[0]
        cold = calc(method=method).singlepoint(H2_MOVED)
        assert result.energy == pytest.approx(cold.energy, abs=2e-10)
        np.testing.assert_allclose(result.forces, cold.forces, atol=1e-8, rtol=0)
        assert result.iterations > 0 and result.restart_origin == "persistent_restart"
        assert target.checkpoint_diagnostics["target_verification"] == "executed"
        assert target.checkpoint_diagnostics["target_results"][0]["converged"]


def test_frozen_source_geometry_and_controls_survive_reexport(tmp_path):
    path, again = tmp_path / "source", tmp_path / "again"
    with calc().prepare_batch([H2]) as source:
        source.execute(strict=True)
        source.set_warm_start_updates(False)
        source.execute([np.array([a[1] for a in H2_MOVED])], strict=True)
        source.save_checkpoint(path)
    with calc(energy_tolerance=1e-11).prepare_batch([H2_MOVED]) as target:
        target.load_checkpoint(path, allow_warm=True)
        target.set_warm_start_updates(False)
        target.execute(strict=True)
        assert (
            target.execute(strict=True).items[0].restart_origin == "persistent_restart"
        )
        target.save_checkpoint(again)
    before, after = (
        inspect_checkpoint(path).items[0],
        inspect_checkpoint(again).items[0],
    )
    assert before["model"] == after["model"]
    assert before["controls"] == after["controls"]
    assert before["seed"] == after["seed"]
    with calc().prepare_batch([H2]) as target:
        assert (
            target.load_checkpoint(again)["items"][0]["compatibility"]
            == "exact_restart"
        )


def test_failed_slots_order_and_partial_compatibility(tmp_path):
    path = tmp_path / "batch"
    save(path, [H2, HE])
    with calc().prepare_batch([H2, H2]) as target:
        with pytest.raises(CheckpointError, match="1: ordered nuclei"):
            target.load_checkpoint(path)
        assert not any(i.warm_start_used for i in target.execute(strict=True).items)
        target.clear_warm_starts()
        report = target.load_checkpoint(path, strict=False)
        assert [i["index"] for i in report["items"]] == [0, 1]
        result = target.execute(strict=True)
        assert [i.restart_origin for i in result.items] == [
            "persistent_restart",
            "cold",
        ]
    with calc().prepare_batch([H2, HE]) as source:
        result = source.execute([None, np.full((1, 3), np.nan)])
        assert result.failure_indices == (1,)
        source.save_checkpoint(path)
    with calc().prepare_batch([H2, HE]) as target:
        report = target.load_checkpoint(path, strict=False)
        assert report["items"][1]["source_status"] != 0
        assert [i.warm_start_used for i in target.execute(strict=True).items] == [
            True,
            False,
        ]


@pytest.mark.parametrize(
    "options,pattern",
    [
        ({"method": "uhf"}, "method/core/spin/provider"),
        ({"basis": "def2-svp"}, "basis/AO transport"),
        ({"density_fitting": DEVICE}, "method/core/spin/provider"),
    ],
)
def test_incompatible_scientific_models(tmp_path, options, pattern):
    path = tmp_path / "state"
    save(path)
    with (
        calc(**options).prepare_batch([H2]) as target,
        pytest.raises(CheckpointError, match=pattern),
    ):
        target.load_checkpoint(path, allow_warm=True)


@pytest.mark.parametrize(
    "damage", ["header", "manifest", "truncate", "blob", "trailing"]
)
def test_corruption_never_partially_changes_live_seeds(tmp_path, damage):
    path, before, after = (tmp_path / p for p in ("state", "before", "after"))
    save(path, [H2, HE])
    data = bytearray(path.read_bytes())
    if damage == "header":
        data[0] ^= 1
    elif damage == "manifest":
        data[_HEADER.size + 10] ^= 1
    elif damage == "truncate":
        del data[-8:]
    elif damage == "blob":
        data[-1] ^= 1
    else:
        data += b"trailing"
    path.write_bytes(data)
    with calc().prepare_batch([H2_MOVED, HE]) as target:
        target.execute(strict=True)
        target.save_checkpoint(before)
        with pytest.raises(CheckpointError):
            target.load_checkpoint(path, allow_warm=True, strict=False)
        target.save_checkpoint(after)
        assert before.read_bytes() == after.read_bytes()


@pytest.mark.parametrize(
    "field,value",
    [("schema_version", 99), ("array_format", "native"), ("amplitudes", {})],
)
def test_unknown_required_contracts_rejected(tmp_path, field, value):
    path = tmp_path / "state"
    save(path)
    rewrite(path, lambda doc: doc.update({field: value}))
    with pytest.raises(CheckpointError):
        inspect_checkpoint(path)


@pytest.mark.parametrize(
    "fault,pattern",
    [
        ("nan", "nonfinite"),
        ("asymmetric", "symmetry"),
        ("wrong_trace", "electron_count"),
        ("negative_occupation", "occupation_bounds"),
    ],
)
def test_semantically_invalid_density_rejected_atomically(tmp_path, fault, pattern):
    path, before, after = (tmp_path / p for p in ("state", "before", "after"))
    save(path, [HE, H2])

    def damage(arrays):
        density = arrays["density_1"][0]
        if fault == "nan":
            density[0, 0] = np.nan
        elif fault == "asymmetric":
            density[0, 1] += 0.2
        elif fault == "wrong_trace":
            density *= 0.5
        else:
            # H2 overlap diagonal = 1: adding opposite diagonal entries
            # preserves trace but makes the metric density indefinite.
            density[0, 0] += 10
            density[1, 1] -= 10

    rewrite(path, mutate_arrays=damage)
    with calc().prepare_batch([HE, H2]) as target:
        target.execute(strict=True)
        target.save_checkpoint(before)
        with pytest.raises(CheckpointError, match=pattern):
            target.load_checkpoint(path)
        target.save_checkpoint(after)
        assert before.read_bytes() == after.read_bytes()


def test_manifest_dimensions_provider_spin_geometry_and_checksums(tmp_path):
    path = tmp_path / "state"
    cases = [
        lambda d: d["blobs"][0].update(bytes=2**63),
        lambda d: d["blobs"][0].update(shape=[1, 2**63, 2**63]),
        lambda d: d["items"][0]["seed"].update(nbf=2**63),
        lambda d: d["items"][0].update(provider="unknown-provider"),
        lambda d: d["items"][0].update(index=1),
        lambda d: d["items"][0]["model"].update(multiplicity=3),
        lambda d: d["items"][0]["model"].update(basis_hash="a" * 64),
        lambda d: d["items"][0]["seed"].update(density="../../blob"),
    ]
    for mutation in cases:
        save(path)
        rewrite(path, mutation)
        with pytest.raises(CheckpointError):
            inspect_checkpoint(path)
    save(path)
    rewrite(path, mutate_arrays=lambda a: a["coordinates_0"].__setitem__((0, 2), -0.9))
    with (
        calc().prepare_batch([H2]) as target,
        pytest.raises(CheckpointError, match="coordinate/model"),
    ):
        target.load_checkpoint(path)


def test_atomic_publish_failure_preserves_old_file(tmp_path, monkeypatch):
    path = tmp_path / "state"
    save(path)
    original = path.read_bytes()

    def interrupted(*args):
        raise OSError("interrupted before publish")

    monkeypatch.setattr(os, "replace", interrupted)
    with calc().prepare_batch([H2_MOVED]) as batch:
        batch.execute(strict=True)
        with pytest.raises(OSError, match="interrupted"):
            batch.save_checkpoint(path)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]
    inspect_checkpoint(path)


def test_size_limits_resource_headroom_and_disabled_warm_starts(tmp_path):
    path = tmp_path / "state"
    save(path)
    with (
        calc().prepare_batch([H2], warm_start=False) as target,
        pytest.raises(CheckpointError, match="warm-enabled"),
    ):
        target.load_checkpoint(path)
    with calc().prepare_batch([H2]) as target:
        with pytest.raises(MemoryError, match="max_bytes"):
            target.load_checkpoint(path, max_bytes=100)
        target.execute(strict=True)
        with pytest.raises(MemoryError):
            target.save_checkpoint(path, max_bytes=100)
    # Sufficient for HF alone, deliberately insufficient for checkpoint staging.
    with calc(resource_budget=ResourceBudget(host_bytes=1 << 20)).prepare_batch(
        [H2]
    ) as target:
        with pytest.raises(MemoryError, match="headroom"):
            target.load_checkpoint(path)
        assert target.execute(strict=True).items[0].restart_origin == "cold"


def test_native_abi_checks_dimensions_before_allocation(tmp_path):
    with calc().prepare_batch([H2]) as batch:
        state = _native.HfWarmState(
            struct_size=ctypes.sizeof(_native.HfWarmState),
            abi_version=_native.ABI_VERSION,
        )
        state.present = 1
        data = (ctypes.c_double * 1)(0.0)
        state.density = state.coordinates = data
        state.density_count = 2**64 - 1
        state.coordinate_count = 6
        status = batch._library.vibeqc_batch_restore_hf_warm_states(
            batch._batch, ctypes.byref(state), 1
        )
        assert status == _native.STATUS_INVALID_ARGUMENT
        assert batch.execute(strict=True).items[0].restart_origin == "cold"


@pytest.mark.parametrize("representation", ["cartesian", "spherical"])
@pytest.mark.parametrize("angular", [2, 3])
def test_custom_higher_angular_basis_roundtrip_and_gauge_independent_seed(
    tmp_path, representation, angular
):
    from vibeqc import Primitive, Shell

    path = tmp_path / "custom"
    basis = (
        Shell(0, 0, (Primitive(1.2, 1.0),)),
        Shell(0, angular, (Primitive(0.8, 1.0),)),
    )
    calculator = calc(basis=basis, basis_representation=representation)
    with calculator.prepare_batch([HE]) as source:
        cold = source.execute(strict=True).items[0]
        source.save_checkpoint(path)
    with calculator.prepare_batch([HE]) as target:
        target.load_checkpoint(path)
        restored = target.execute(strict=True).items[0]
    assert restored.energy == pytest.approx(cold.energy, abs=2e-10)
    np.testing.assert_allclose(restored.forces, cold.forces, atol=1e-8, rtol=0)
    # No orbital gauge, virtual columns or DIIS history is part of this schema.
    record = inspect_checkpoint(path).items[0]
    count = (
        (angular + 1) * (angular + 2) // 2
        if representation == "cartesian"
        else 2 * angular + 1
    )
    assert record["seed"]["nbf"] == 1 + count
    assert record["model"]["representation"] == (
        "cartesian" if representation == "cartesian" else "real_spherical"
    )


def test_source_convergence_metadata_cannot_bypass_target_solve(tmp_path):
    path = tmp_path / "state"
    save(path)

    # A valid, deliberately nonstationary density with the correct metric trace.
    # Source convergence/energy are provenance only, even when checksum-correct.
    def nonstationary(arrays):
        arrays["density_0"][0] = np.diag([2.0, 0.0])

    rewrite(
        path,
        lambda d: d["items"][0]["seed"].update(
            energy=-100, density_rms=0, iterations=1
        ),
        nonstationary,
    )
    with calc().prepare_batch([H2]) as target:
        target.load_checkpoint(path)
        result = target.execute(strict=True).items[0]
        assert result.energy == pytest.approx(calc().singlepoint(H2).energy, abs=2e-10)
        assert result.energy != -100 and result.iterations > 1


def test_no_checkpoint_cold_run_remains_available_after_clear(tmp_path):
    path = tmp_path / "state"
    expected = save(path).items[0]
    with calc().prepare_batch([H2]) as target:
        target.load_checkpoint(path)
        target.clear_warm_starts()
        cold = target.execute(strict=True).items[0]
        assert cold.restart_origin == "cold" and not cold.warm_start_used
        assert cold.energy == expected.energy
        np.testing.assert_array_equal(cold.forces, expected.forces)


@pytest.mark.skipif(
    DEVICE != "cuda", reason="real-GPU cross-backend test runs under Slurm"
)
@pytest.mark.parametrize("fitted", [False, True])
def test_backend_independent_cpu_cuda_restart_both_directions(tmp_path, fitted):
    path = tmp_path / "state"
    for source_device, target_device in (("cpu", "cuda"), ("cuda", "cpu")):
        for method, charge, mult in (("rhf", 0, 1), ("uhf", 1, 2)):
            with Calculator(
                device=source_device,
                method=method,
                density_fitting=source_device if fitted else "none",
            ).prepare_batch([H2], charges=[charge], multiplicities=[mult]) as source:
                cold = source.execute(strict=True).items[0]
                source.save_checkpoint(path)
            with Calculator(
                device=target_device,
                method=method,
                density_fitting=target_device if fitted else "none",
            ).prepare_batch([H2], charges=[charge], multiplicities=[mult]) as target:
                assert (
                    target.load_checkpoint(path)["items"][0]["compatibility"]
                    == "exact_restart"
                )
                warm = target.execute(strict=True).items[0]
            assert warm.executed_backend == (
                "cuda" if target_device == "cuda" else "cpu_reference"
            )
            assert warm.energy == pytest.approx(cold.energy, abs=2e-10)
            np.testing.assert_allclose(warm.forces, cold.forces, atol=1e-8, rtol=0)


def test_uhf_spin_identity_and_source_spin_trace(tmp_path):
    path = tmp_path / "spin"
    with calc(method="uhf").prepare_batch(
        [H2], charges=[1], multiplicities=[2]
    ) as source:
        source.execute(strict=True)
        source.save_checkpoint(path)
    with (
        calc(method="uhf").prepare_batch([H2]) as target,
        pytest.raises(CheckpointError, match="spin/provider"),
    ):
        target.load_checkpoint(path, allow_warm=True)
    rewrite(
        path,
        mutate_arrays=lambda arrays: arrays["density_0"].__setitem__(
            slice(None), arrays["density_0"][::-1].copy()
        ),
    )
    with (
        calc(method="uhf").prepare_batch(
            [H2], charges=[1], multiplicities=[2]
        ) as target,
        pytest.raises(CheckpointError, match="electron_count"),
    ):
        target.load_checkpoint(path)


def test_duplicate_json_keys_are_rejected(tmp_path):
    path = tmp_path / "state"
    save(path)
    data = path.read_bytes()
    magic, length, _ = _HEADER.unpack(data[: _HEADER.size])
    raw = data[_HEADER.size : _HEADER.size + length]
    raw = raw.replace(
        b'"schema_version":1', b'"schema_version":1,"schema_version":1', 1
    )
    path.write_bytes(
        _HEADER.pack(magic, len(raw), hashlib.sha256(raw).digest())
        + raw
        + data[_HEADER.size + length :]
    )
    with pytest.raises(CheckpointError, match="duplicate"):
        inspect_checkpoint(path)


def test_checkpoint_respects_a_feasible_current_resource_plan(tmp_path):
    path = tmp_path / "state"
    expected = save(path).items[0]
    budget = ResourceBudget(host_bytes=64 << 20, device_bytes=128 << 20)
    with calc(resource_budget=budget).prepare_batch([H2]) as target:
        identity = target.resource_plan.identity
        target.load_checkpoint(path)
        actual = target.execute(strict=True).items[0]
        assert target.resource_plan.identity == identity
        assert actual.restart_origin == "persistent_restart"
        assert actual.energy == pytest.approx(expected.energy, abs=2e-10)
        target.save_checkpoint(path)


@pytest.mark.parametrize(
    "variable",
    [
        "VIBEQC_FINAL_FOCK_REBUILD",
        "VIBEQC_FORCE_DENSITY_PRODUCT_SCREENING",
        "VIBEQC_MIXED_PRECISION_FOCK_THRESHOLD",
    ],
)
def test_runtime_numerical_policy_changes_require_explicit_warm_restart(
    tmp_path, monkeypatch, variable
):
    path = tmp_path / "state"
    monkeypatch.delenv(variable, raising=False)
    save(path)
    monkeypatch.setenv(variable, "1")
    with calc().prepare_batch([H2]) as target:
        with pytest.raises(CheckpointError, match="numerical controls"):
            target.load_checkpoint(path)
        report = target.load_checkpoint(path, allow_warm=True)
        assert report["items"][0]["compatibility"] == "warm_start_compatible"
        assert os.environ[variable] == "1"
        target.save_checkpoint(tmp_path / "reexport")
    source = inspect_checkpoint(tmp_path / "reexport").items[0]
    assert source["controls"]["runtime_policy"][variable] is None
