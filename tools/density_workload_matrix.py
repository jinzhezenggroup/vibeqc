"""Larger fixed-input extension of benchmark_density_sources, using its gates."""

import json
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np
from vibeqc.autotune import dft_density_candidates
from vibeqc_compiler.common.provenance import canonical_hash, file_hash
from vibeqc_compiler.common.resources import (
    ResourceBudget,
    ResourceCandidate,
    ResourceEstimate,
    ResourceIdentity,
    ResourceRequest,
    plan_resources,
)
from vibeqc_compiler.dft import DensitySource, ExplicitGrid, NativeAO
from vibeqc_compiler.dft.fixtures import basis_arguments


def validate_matrix_errors(errors):
    """Require every declared numerical gate, including equal-count key swaps.

    This is the fixed acceptance matrix, independent of which keys a measured
    loop happened to emit. Repeated samples deliberately reduce into one worst
    block record, so checking only that the surviving records pass is unsafe.
    """
    from tools.benchmark_density_sources import BUDGETS, FUNCTIONALS, NAMES, ROUTES
    from tools.generate_density_workload_references import workloads

    expected = {
        f"features/{name}/{route}/{feature}"
        for name in NAMES
        for route in ROUTES
        for feature in ("ao_jets", "rho", "gradient", "sigma", "tau")
    }
    for name, *_ in workloads():
        modes = (
            ("dense", "absolute_ao_jet")
            if name in ("water4_svp", "water8_svp")
            else ("dense",)
        )
        for functional in FUNCTIONALS:
            for cap in BUDGETS:
                for mode in modes:
                    label = f"{name}/{functional}/{mode}/{cap}"
                    for route in ROUTES:
                        expected.update(
                            f"{label}/{route}/{key}" for key in ("energy", "potential")
                        )
                        if mode == "dense":
                            features = (
                                ("ao", "rho", "gradient")
                                if functional == "PBE"
                                else ("ao", "rho")
                            )
                            expected.update(
                                f"{label}/{route}/{key}" for key in features
                            )
                            if name in ("water_svp", "water4_svp", "oh_diffuse"):
                                expected.update(
                                    f"{label}/batch4/{route}/{key}"
                                    for key in ("energy", "potential")
                                )
    if set(errors) != expected:
        raise AssertionError("incomplete registered workload/error inventory")


def load_workloads(directory):
    """Validate exporter, input and every numeric block before measurement."""
    directory = Path(directory)
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((directory / "manifest.json").read_text())
    for entry in manifest:
        name = entry["name"]
        if not name.replace("_", "").isalnum():
            raise ValueError("invalid workload name")
        meta = json.loads((directory / f"{name}.json").read_text())
        identity = meta.pop("identity")
        if (
            meta["schema"] != "vibeqc.density-workload-reference.v1"
            or identity != entry["identity"]
            or canonical_hash(meta) != identity
            or canonical_hash(meta["inputs"]) != meta["inputs_hash"]
        ):
            raise ValueError("density workload identity mismatch")
        for key, filename in (
            ("exporter_sha256", "generate_density_workload_references.py"),
            ("basis_adapter_sha256", "generate_validation_references.py"),
        ):
            if meta["reference"][key] != file_hash(root / "tools" / filename):
                raise ValueError("density workload exporter mismatch")
        if "array_files" in entry:
            # Large reference bundles retain their original NPY member bytes
            # separately. The existing numeric hashes still bind every value.
            from tools.vibeqc_validation.retention import safe_relative

            arrays = {}
            for key, record in entry["array_files"].items():
                path = directory / safe_relative(record["path"])
                if (
                    not path.resolve().is_relative_to(directory.resolve())
                    or path.stat().st_size != record["bytes"]
                    or file_hash(path) != record["sha256"]
                ):
                    raise ValueError("density workload array file mismatch")
                arrays[key] = np.load(path, allow_pickle=False)
        else:
            if file_hash(directory / f"{name}.npz") != entry["archive_sha256"]:
                raise ValueError("density workload archive identity mismatch")
            with np.load(directory / f"{name}.npz", allow_pickle=False) as archive:
                arrays = dict(archive)
        if arrays.keys() != meta["arrays"].keys():
            raise ValueError("density workload block mismatch")
        for key, value in arrays.items():
            record = meta["arrays"][key]
            if (
                list(value.shape) != record["shape"]
                or str(value.dtype) != record["dtype"]
                or sha256(value.tobytes()).hexdigest() != record["sha256"]
            ):
                raise ValueError("density workload numeric hash mismatch")
        grid = ExplicitGrid(
            arrays["points"],
            arrays["weights"],
            tuple(map(int, arrays["owners"])),
            meta["grid_provenance"],
        )
        if grid.identity != meta["grid_identity"]:
            raise ValueError("density workload grid mismatch")
        meta["identity"] = identity
        yield name, meta, arrays, grid


def original_source(basis, meta, data, *, generation=0):
    """Attach the actual occupied columns; never factorize or diagonalize D."""
    source = DensitySource(
        data["density_spin"].sum(axis=0)
        if meta["layout"] == "total"
        else data["density_spin"],
        basis_identity=basis.identity,
        density_generation=generation,
    )
    return source.with_orbitals(
        tuple(data[f"coefficients_{s}"] for s in range(2)),
        tuple(data[f"occupations_{s}"] for s in range(2)),
        stamp=source.stamp,
    )


def batch_resources(endpoint, sources):
    """Compose caller-owned replica inputs and returned output capacity under #203.

    Four systems replay serially through one prepared owner. All independent
    source snapshots stay live; their storage is additional to that owner's
    input-staging allowance. One complete output set is retained per item until timing ends.
    """
    source_bytes = sum(
        s.density.nbytes
        + sum(c.nbytes for c in s.coefficients)
        + sum(f.nbytes for f in s.occupations)
        for s in sources
    )
    output_bytes = len(sources) * 8 * (2 * endpoint.basis.nao**2 + 3)
    last_phase = max(
        e.last_phase
        for request in endpoint.resource_plan.requests
        for candidate in request.candidates
        for e in candidate.estimates
    )
    request = ResourceRequest(
        "density_replica_inputs",
        ResourceIdentity(
            "dft",
            "density_replica_inputs",
            "cpu",
            "fp64",
            json.dumps({"sources": [s.stamp.identity for s in sources]}),
            ("potential",),
            "serial_shared_owner",
        ),
        (
            ResourceCandidate(
                "retained",
                "resident",
                (
                    ResourceEstimate(
                        "source_snapshots",
                        source_bytes,
                        "pageable",
                        0,
                        last_phase,
                        kind="persistent",
                    ),
                    ResourceEstimate(
                        "returned_output", output_bytes, "pageable", 0, last_phase
                    ),
                ),
            ),
        ),
        ("Python headers and allocator rounding",),
    )
    return plan_resources(
        (*endpoint.resource_plan.requests, request), endpoint.budget
    ).require_feasible()


def measure_matrix(directory, artifact, programs, samples, errors, timings):
    """Interleave actual registered routes and preserve every failed gate/sample."""
    # Reuse the established #138-compatible runner and its full-block gates.
    from tools.benchmark_density_sources import (
        BUDGETS,
        FUNCTIONALS,
        checked_metrics,
        endpoint_owner,
        gate,
        masked_endpoint,
        record_worst,
    )

    rows, batches = [], []
    for name, meta, data, grid in load_workloads(directory):
        with NativeAO(**basis_arguments(meta)) as basis:
            started = perf_counter()
            source = original_source(basis, meta, data)
            source_seconds = perf_counter() - started
            if source.source_kind != "orbitals":
                raise AssertionError(source.fallback_reason)
            spin = "unpolarized" if meta["layout"] == "total" else "polarized"
            modes = (
                ("dense", "absolute_ao_jet")
                if name in ("water4_svp", "water8_svp")
                else ("dense",)
            )
            for functional in FUNCTIONALS:
                native = programs[functional, spin]
                ingredients = (
                    ("rho",)
                    if functional == "LDA_XC_PW"
                    else ("rho", "gradient", "sigma")
                )
                for cap in BUDGETS:
                    for mode in modes:
                        label = f"{name}/{functional}/{mode}/{cap}"
                        budget = ResourceBudget(host_bytes=256 << 20, device_bytes=cap)
                        started = perf_counter()
                        with endpoint_owner(
                            basis,
                            grid,
                            artifact,
                            native,
                            ingredients,
                            budget,
                            mode,
                            tile_points=256,
                            orbital_tile=16,
                            region_points=512,
                            orbital_capacity=tuple(map(len, source.occupations)),
                        ) as (cuda, endpoint, spatial):
                            construction = perf_counter() - started
                            expected = (
                                float(data[functional + "_energy"][0]),
                                data[functional + "_potential"],
                            )
                            screening = None
                            if spatial is not None:
                                masked = masked_endpoint(
                                    basis,
                                    grid,
                                    source,
                                    spatial,
                                    native.spec,
                                    meta["layout"],
                                )
                                screening = {
                                    "energy_difference": abs(masked[0] - expected[0]),
                                    "potential_max_difference": float(
                                        np.max(abs(masked[1] - expected[1]))
                                    ),
                                    "scope": "screening difference from independent unmasked PySCF; not an arithmetic gate",
                                }
                                expected = masked
                            candidates = dft_density_candidates(
                                endpoint, source, stamp=source.stamp
                            )
                            if not all(c.available for c in candidates):
                                raise AssertionError([c.describe() for c in candidates])
                            if spatial is None:
                                for candidate in candidates:
                                    cuda.set_source(
                                        source,
                                        stamp=source.stamp,
                                        route=candidate.route,
                                    )
                                    values = cuda.evaluate(
                                        grid.points[data["sample_ids"]],
                                        stamp=source.stamp,
                                        download_jets=True,
                                    )
                                    record_worst(
                                        errors,
                                        label + "/" + candidate.route + "/ao",
                                        values["ao_jets"],
                                        data["sample_jets"][: len(values["ao_jets"])],
                                    )
                                    record_worst(
                                        errors,
                                        label + "/" + candidate.route + "/rho",
                                        values["rho"],
                                        data["sample_rho_gradient"][:, 0],
                                    )
                                    if "gradient" in ingredients:
                                        record_worst(
                                            errors,
                                            label + "/" + candidate.route + "/gradient",
                                            values["gradient"],
                                            data["sample_rho_gradient"][
                                                :, 1:
                                            ].transpose(0, 2, 1),
                                        )
                            for candidate in candidates:
                                candidate.execute(stamp=source.stamp)
                            for repeat in range(samples):
                                for candidate in (
                                    candidates if repeat % 2 == 0 else candidates[::-1]
                                ):
                                    started = perf_counter()
                                    value, execution = candidate.execute(
                                        stamp=source.stamp
                                    )
                                    elapsed = perf_counter() - started
                                    potential = (
                                        value["potential"].mean(axis=0)
                                        if meta["layout"] == "total"
                                        else value["potential"]
                                    )
                                    record_worst(
                                        errors,
                                        label + "/" + candidate.route + "/energy",
                                        value["energy"],
                                        expected[0],
                                    )
                                    record_worst(
                                        errors,
                                        label + "/" + candidate.route + "/potential",
                                        potential,
                                        expected[1],
                                    )
                                    if spatial is None:
                                        electrons = (
                                            value["electrons"].sum()
                                            if meta["layout"] == "total"
                                            else value["electrons"]
                                        )
                                        gate(electrons, data[functional + "_electrons"])
                                    timings.append(
                                        {
                                            "case": label,
                                            "repeat": repeat,
                                            "selection": "baseline"
                                            if candidate.route == "density_matrix"
                                            else "candidate",
                                            "workload": "unchanged-geometry",
                                            "inputs_hash": candidate.workload.identity,
                                            "seconds": elapsed,
                                            "diagnostics": {
                                                "batch_size": 1,
                                                **execution,
                                            },
                                        }
                                    )
                            row = {
                                "case": label,
                                "batch_size": 1,
                                "inputs_hash": candidates[0].workload.identity,
                                "reference_identity": meta["identity"],
                                "producer": meta["reference"],
                                "npoint": endpoint.npoint,
                                "nao": basis.nao,
                                "source_construction_seconds": source_seconds,
                                "owner_construction_seconds": construction,
                                "source": asdict(source.stamp),
                                "factor_identity": source.factor_identity,
                                "tile_plan": asdict(cuda.plan),
                                "resource_plan": endpoint.resource_plan.to_dict(),
                                "native_metrics": checked_metrics(cuda),
                                "candidates": [c.describe() for c in candidates],
                                "screening": screening,
                            }
                            rows.append(row)
                            # Batch throughput here is deliberately serial prepared replay,
                            # with independently owned current states and one bounded owner.
                            if mode == "dense" and name in (
                                "water_svp",
                                "water4_svp",
                                "oh_diffuse",
                            ):
                                replicas = tuple(
                                    original_source(basis, meta, data, generation=i + 1)
                                    for i in range(4)
                                )
                                registered = tuple(
                                    dft_density_candidates(endpoint, s, stamp=s.stamp)
                                    for s in replicas
                                )
                                plan = batch_resources(endpoint, replicas)
                                batch_id = canonical_hash(
                                    [c[0].workload.identity for c in registered]
                                )
                                for repeat in range(samples):
                                    for route in (0, 1) if repeat % 2 == 0 else (1, 0):
                                        started = perf_counter()
                                        results = []
                                        for item, s in zip(
                                            registered, replicas, strict=True
                                        ):
                                            results.append(
                                                item[route].execute(stamp=s.stamp)
                                            )
                                        elapsed = perf_counter() - started
                                        records = []
                                        for item, (value, execution) in zip(
                                            registered, results, strict=True
                                        ):
                                            potential = (
                                                value["potential"].mean(axis=0)
                                                if meta["layout"] == "total"
                                                else value["potential"]
                                            )
                                            record_worst(
                                                errors,
                                                label
                                                + "/batch4/"
                                                + item[route].route
                                                + "/energy",
                                                value["energy"],
                                                expected[0],
                                            )
                                            record_worst(
                                                errors,
                                                label
                                                + "/batch4/"
                                                + item[route].route
                                                + "/potential",
                                                potential,
                                                expected[1],
                                            )
                                            records.append(execution)
                                        del results, value, potential
                                        timings.append(
                                            {
                                                "case": label + "/batch4",
                                                "repeat": repeat,
                                                "selection": "baseline"
                                                if route == 0
                                                else "candidate",
                                                "workload": "unchanged-geometry",
                                                "inputs_hash": batch_id,
                                                "seconds": elapsed,
                                                "diagnostics": {
                                                    "batch_size": 4,
                                                    "items": records,
                                                },
                                            }
                                        )
                                batches.append(
                                    {
                                        "case": label + "/batch4",
                                        "batch_size": 4,
                                        "inputs_hash": batch_id,
                                        "execution": "serial independent replica states through one shared prepared owner; no batched-kernel claim",
                                        "resource_plan": plan.to_dict(),
                                        "source_identities": [
                                            s.stamp.identity for s in replicas
                                        ],
                                    }
                                )
                            print(label, flush=True)
    if len(rows) != 36 or len(batches) != 12:
        raise AssertionError("incomplete larger workload/batch acceptance matrix")
    return rows, batches
