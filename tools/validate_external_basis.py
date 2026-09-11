"""Independent public HF endpoints for imported bases; CUDA requires Slurm.

Timings include synchronous energy and forces returned to host. Each trial
rebuilds its native plan, then checks fixed and changed geometry replays. This
is interface evidence, with no schedule replacement or speed promotion.
"""

from __future__ import annotations

# Source-tree CLI bootstrap for transitive compiler clients.
import sys as _compiler_sys
from pathlib import Path as _CompilerPath

_compiler_sys.path.insert(
    0, str(_CompilerPath(__file__).resolve().parents[1] / "python")
)

import argparse
import ctypes
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]
import numpy as np
from vibeqc import Calculator, import_bse
from vibeqc.profiles import probe_device

from tools.vibeqc_validation.schema import (
    block_error,
    canonical_hash,
    file_hash,
    new_evidence,
    outcome,
    validate_evidence,
)

DATA = ROOT / "tests/data/external_basis"
SCF_OPTIONS = {
    "energy_tolerance": 1e-12,
    "density_tolerance": 1e-10,
    "screening_tolerance": 1e-14,
    "max_iterations": 150,
}


def fixtures():
    """Verify every archived input before accepting independent oracle arrays."""
    manifest = json.loads((DATA / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        if file_hash(DATA / name) != digest:
            raise ValueError(f"fixture checksum mismatch: {name}")
    with np.load(DATA / "oracles.npz") as source:
        arrays = {name: source[name] for name in source.files}
    return manifest, arrays


def basis_for(spec):
    """Load the exact local source and explicitly select the oracle convention."""
    return import_bse(
        DATA / spec["source"],
        source="BSE pinned checkout; VibeQC original synthetic Fe diagnostic data",
        source_version="4adaf1372c7101620ca1a9f3130be9ae97fb8f30",
        license="BSD-3-Clause AND GPL-3.0-or-later"
        if spec["source"] == "synthetic-fe-h.json"
        else "BSD-3-Clause",
        representation=spec["representation"],
    )


def check_result(result, spec, arrays, device, *, changed=False):
    """Gate independent total energy/force values and the actual native backend."""
    expected_backend = "cuda" if device == "cuda" else "cpu_reference"
    assert result.converged and result.executed_backend == expected_backend
    if hasattr(result, "succeeded"):
        assert result.succeeded
    key = spec["name"] + ("_changed" if changed else "")
    errors = {
        "energy": block_error(
            np.asarray(result.energy), arrays[key + "_energy"], atol=1e-8, rtol=0
        ),
        "forces": block_error(
            result.forces, arrays[key + "_forces"], atol=1e-7, rtol=0
        ),
    }
    assert all(error["passed"] for error in errors.values()), (key, errors)
    assert result.basis_metadata["orbital"]["electrons"]["electron_count"] == sum(
        spec["electrons"]
    )
    return {
        "energy": result.energy,
        "forces": result.forces.tolist(),
        "iterations": result.iterations,
        "energy_change": result.energy_change,
        "density_rms": result.density_rms,
        "executed_backend": result.executed_backend,
        "model_identity": result.basis_metadata["model_identity"],
        "warm_start_used": getattr(result, "warm_start_used", False),
        "errors": errors,
    }


def case_endpoints(spec, arrays, devices, samples):
    """Interleave complete plan construction, cold execution and retained replays."""
    started = time.perf_counter()
    basis = basis_for(spec)
    import_seconds = time.perf_counter() - started
    inputs = canonical_hash({"case": spec, "basis": basis.identity, "scf": SCF_OPTIONS})
    record = new_evidence(tier="endpoint", subject=spec["name"], inputs_hash=inputs)
    record["import_seconds"] = import_seconds
    record["case"] = spec
    record["basis_metadata"] = Calculator(basis=basis, **SCF_OPTIONS).basis_metadata(
        spec["atoms"], charge=spec["charge"]
    )
    record["details"] = []
    coordinates = np.array([atom[1] for atom in spec["changed_atoms"]])
    for trial in range(samples):
        order = devices if trial % 2 == 0 else tuple(reversed(devices))
        for device in order:
            calculator = Calculator(basis=basis, device=device, **SCF_OPTIONS)
            started = time.perf_counter()
            one_shot = calculator.singlepoint(spec["atoms"], charge=spec["charge"])
            one_shot_seconds = time.perf_counter() - started
            details = {
                "trial": trial,
                "device": device,
                "one_shot_seconds": one_shot_seconds,
                "one_shot": check_result(one_shot, spec, arrays, device),
            }
            started = time.perf_counter()
            with calculator.prepare_batch(
                [spec["atoms"]], charges=[spec["charge"]]
            ) as prepared:
                details["prepare_seconds"] = time.perf_counter() - started
                for phase, update, changed in (
                    ("prepared_cold", None, False),
                    ("warm", None, False),
                    ("changed", [coordinates], True),
                    ("changed_replay", [coordinates], True),
                ):
                    # Passing the updated coordinates on every replay is
                    # essential: execute(None) restores the original geometry.
                    started = time.perf_counter()
                    result = prepared.execute(update, strict=True).items[0]
                    seconds = time.perf_counter() - started
                    details[phase] = check_result(
                        result, spec, arrays, device, changed=changed
                    )
                    details[phase]["seconds"] = seconds
                    record["timings"].append(
                        {
                            "trial": trial,
                            "selection": "candidate"
                            if device == "cuda"
                            else "baseline",
                            "phase": phase,
                            "workload": "changed-geometry"
                            if changed
                            else (
                                "cold-start"
                                if phase == "prepared_cold"
                                else "unchanged-geometry"
                            ),
                            "seconds": seconds
                            + (
                                details["prepare_seconds"]
                                if phase == "prepared_cold"
                                else 0
                            ),
                            "inputs_hash": canonical_hash(
                                {"model": inputs, "changed": changed}
                            ),
                        }
                    )
                assert not details["prepared_cold"]["warm_start_used"]
                assert details["warm"]["warm_start_used"]
                assert prepared.basis_metadata[0] == one_shot.basis_metadata
            record["details"].append(details)
    if spec["name"].startswith("water"):
        record["bundled_equivalence"] = {}
        for device in devices:
            bundled = Calculator(
                basis="sto-3g",
                basis_representation=spec["representation"],
                device=device,
                **SCF_OPTIONS,
            ).singlepoint(spec["atoms"])
            imported = Calculator(
                basis=basis, device=device, **SCF_OPTIONS
            ).singlepoint(spec["atoms"])
            assert (
                bundled.basis_metadata["orbital"]["mathematical_identity"]
                == imported.basis_metadata["orbital"]["mathematical_identity"]
            )
            if device == "cpu":
                assert bundled.energy == imported.energy
                np.testing.assert_array_equal(bundled.forces, imported.forces)
                record["bundled_equivalence"][device] = (
                    "identical inputs, energy and forces"
                )
            else:
                # Atomic GPU force reductions can change the last few bits
                # between identical calls. Input identity stays exact; this
                # replay gate is far tighter than the independent HF gates.
                errors = {
                    "energy": block_error(
                        np.asarray(imported.energy),
                        np.asarray(bundled.energy),
                        atol=1e-12,
                        rtol=0,
                    ),
                    "forces": block_error(
                        imported.forces, bundled.forces, atol=1e-12, rtol=0
                    ),
                }
                assert all(error["passed"] for error in errors.values()), errors
                record["bundled_equivalence"][device] = {
                    "identical_inputs": True,
                    "errors": errors,
                }
    return record


def ragged_endpoints(manifest, arrays, device, samples):
    """Preserve order, independent ions, basis metadata and isolated shape failures."""
    records = []
    for representation in ("cartesian", "spherical"):
        index = {spec["name"]: spec for spec in manifest["cases"]}
        specs = [
            index[name + "_" + representation]
            for name in ("fe_ion", "fe_h_ion", "fe_ion", "fe_h_ion")
        ]
        calculator = Calculator(basis=basis_for(specs[0]), device=device, **SCF_OPTIONS)
        with calculator.prepare_batch(
            [s["atoms"] for s in specs], charges=[s["charge"] for s in specs]
        ) as batch:
            for trial in range(samples):
                changed = bool(trial % 2)
                coordinates = (
                    [np.array([a[1] for a in s["changed_atoms"]]) for s in specs]
                    if changed
                    else None
                )
                started = time.perf_counter()
                result = batch.execute(coordinates, strict=True)
                seconds = time.perf_counter() - started
                records.append(
                    {
                        "representation": representation,
                        "trial": trial,
                        "seconds": seconds,
                        "items": [
                            check_result(r, s, arrays, device, changed=changed)
                            for r, s in zip(result.items, specs, strict=True)
                        ],
                        "basis_metadata": batch.basis_metadata,
                    }
                )
            failed = batch.execute([None, np.zeros((1, 3)), None, None])
            assert failed.failure_indices == (1,)
            assert failed.items[1].basis_metadata == batch.basis_metadata[1]
            for i in (0, 2, 3):
                check_result(failed.items[i], specs[i], arrays, device)
            records.append(
                {
                    "representation": representation,
                    "isolated_failure_indices": list(failed.failure_indices),
                }
            )
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 5:
        parser.error("evidence requires at least five trials")
    if args.cuda and not os.environ.get("SLURM_JOB_ID"):
        parser.error("CUDA evidence requires a finite Slurm allocation")
    manifest, arrays = fixtures()
    devices = ("cpu", "cuda") if args.cuda else ("cpu",)
    calculator = Calculator()
    library = calculator._library
    library.vibeqc_get_source_identity.restype = ctypes.c_char_p
    source_identity = library.vibeqc_get_source_identity().decode()
    device = (
        probe_device(library)
        if args.cuda
        else {"kind": "cpu", "name": platform.processor()}
    )
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=ROOT,
            text=True,
        )
    )
    args.output.mkdir(parents=True, exist_ok=True)
    for spec in manifest["cases"]:
        record = case_endpoints(spec, arrays, devices, args.samples)
        record.update(
            {
                "revision": revision,
                "dirty": dirty,
                "device": device,
                "backend_selected": "cuda" if args.cuda else "cpu_reference",
                "settings": {
                    "device": devices[-1],
                    "scf": SCF_OPTIONS,
                    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                },
                "toolchain": {
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "native_library": str(library._name),
                },
                "hardware": outcome("pass"),
            }
        )
        record["hashes"]["source"] = source_identity
        record["native_library_sha256"] = file_hash(library._name)
        record["runner_sha256"] = file_hash(__file__)
        record["manifest_sha256"] = file_hash(DATA / "manifest.json")
        record["hash_reasons"] = {
            key: "existing HF executor; no new generated equation or schedule"
            for key in ("equation", "ir", "schedule")
        }
        for stage in ("representation", "source", "numerical", "endpoint"):
            record["stages"][stage] = outcome("pass")
        record["compilation"]["reason"] = (
            "native CMake build; build log archived separately"
        )
        record["memory"]["reason"] = (
            "public HF API does not expose peak allocations; unchanged executor, no memory claim"
        )
        record["performance"] = outcome(
            "not-run", "basis input contract; no speed or schedule promotion"
        )
        validate_evidence(record)
        (args.output / (spec["name"] + ".json")).write_text(
            json.dumps(record, indent=2) + "\n"
        )
        print(spec["name"], "pass", flush=True)
    for device in devices:
        records = ragged_endpoints(manifest, arrays, device, args.samples)
        (args.output / ("ragged-" + device + ".json")).write_text(
            json.dumps(records, indent=2) + "\n"
        )
        print("ragged", device, "pass", flush=True)


if __name__ == "__main__":
    main()
