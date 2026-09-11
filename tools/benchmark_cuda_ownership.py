"""Compare complete CUDA ownership endpoints from explicit clean builds.

Run ``compare`` inside a finite Slurm GPU allocation. Each worker loads one
build in a fresh process; alternating order reduces cache/order bias. Timings
include preparation, cold execution, unchanged replay, changed geometry and
restoration, always through final forces. This tool records evidence and never
changes a production selector. Raw scalar correctness is validated separately.
Case isolation pairs each workload in the shared ABBA order before advancing
to another workload, reducing drift when old/new inventories take many minutes.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict
from itertools import product
from pathlib import Path
from time import perf_counter


def capture(argv):
    """Run a checked command without shell interpolation."""
    return subprocess.check_output(argv, text=True).strip()


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")


def endpoint_inventory(domain):
    """Return the unchanged workload matrix, independently of process grouping."""
    inventory = [
        ("spf", *case)
        for case in product(
            ("rhf", "uhf"), ("cartesian", "spherical"), (False, True), (1, 3)
        )
    ]
    inventory += [
        ("sdf", "rhf", "cartesian", False, 1),
        ("sdf", "uhf", "spherical", False, 3),
        ("sdf", "rhf", "spherical", True, 1),
        ("sdf", "uhf", "spherical", True, 3),
    ]
    if domain == "df":
        inventory = [row for row in inventory if row[3]]
        inventory += [
            (family, method, rep, True, count)
            for family, method in (
                ("water-def2-svp", "rhf"),
                ("oh-def2-svp-uhf", "uhf"),
            )
            for rep in ("cartesian", "spherical")
            for count in (1, 3)
        ]
    return inventory


def case_id(row):
    """Name a physical workload identically in workers and the driver."""
    family, method, representation, fitted, count = row
    return f"{family}/{method}/{representation}/{'df' if fitted else 'direct'}/batch{count}"


def worker(args):
    """Measure synchronized public endpoints and retain actual plan diagnostics."""
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("real GPU measurements require Slurm")
    if capture(["git", "-C", str(args.root), "status", "--porcelain"]):
        raise RuntimeError("endpoint evidence requires a clean source checkout")
    sys.path.insert(0, str(args.root / "python"))
    import numpy as np
    from vibeqc import Calculator, Primitive, Shell
    from vibeqc.autotune import source_identity
    from vibeqc_compiler.common.resources import ResourceBudget

    os.environ["VIBEQC_LIBRARY"] = str(args.build / "libvibeqc.so")
    if args.domain == "df":
        os.environ["VIBEQC_DF_DERIVATIVES"] = args.selection
        os.environ["VIBEQC_DF_DERIVATIVE_MAPPING"] = "thread"
        os.environ["VIBEQC_DF_VALUE_MAPPING"] = "primitive"
        os.environ["VIBEQC_ONE_ELECTRON_VALUE_MAPPING"] = "shell_warp"
        # Keep the independent one-electron force route identical on both sides.
        os.environ.pop("VIBEQC_ONE_ELECTRON_DERIVATIVES", None)
    else:
        os.environ["VIBEQC_ONE_ELECTRON_VALUES"] = args.selection
        os.environ["VIBEQC_ONE_ELECTRON_VALUE_MAPPING"] = args.mapping
    library = ctypes.CDLL(os.environ["VIBEQC_LIBRARY"])
    library.vibeqc_get_source_identity.restype = ctypes.c_char_p
    if library.vibeqc_get_source_identity().decode() != source_identity(args.root):
        raise RuntimeError(
            "selected library does not match the measured source checkout"
        )
    if (
        args.domain == "one-electron"
        and args.selection == "reference"
        and "bool generated_one_electron_values_requested("
        not in (args.root / "src/scf/cuda/rhf_policy.cpp").read_text()
    ):
        raise RuntimeError(
            "handwritten one-electron values were retired; select an archived baseline checkout"
        )
    if (
        args.domain == "df"
        and args.selection == "reference"
        and "bool generated_df_derivatives_requested("
        not in (args.root / "src/scf/cuda/rhf_policy.cpp").read_text()
    ):
        raise RuntimeError(
            "coordinate-wise DF response was retired; select an archived baseline checkout"
        )
    cache = (args.build / "CMakeCache.txt").read_text()
    if "VIBEQC_CUDA_FAST_COMPILE:BOOL=OFF" not in cache:
        raise RuntimeError("production timing requires FAST_COMPILE=OFF")
    if "CMAKE_BUILD_TYPE:STRING=Release" not in cache:
        raise RuntimeError("production timing requires Release")
    if args.case and len(set(args.case)) != len(args.case):
        raise ValueError("duplicate endpoint requests")
    records = {
        "schema": "vibeqc.cuda-ownership-endpoints.v1",
        "revision": capture(["git", "-C", str(args.root), "rev-parse", "HEAD"]),
        "dirty": False,
        "native_source_identity": library.vibeqc_get_source_identity().decode(),
        "library_sha256": hashlib.sha256(
            (args.build / "libvibeqc.so").read_bytes()
        ).hexdigest(),
        "selection": args.selection,
        "domain": args.domain,
        "process_scope": args.process_scope,
        "benchmark_driver_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "mapping": "thread" if args.domain == "df" else args.mapping,
        "python": sys.version,
        "numpy": np.__version__,
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": capture(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version",
                "--format=csv,noheader",
            ]
        ),
        "build_settings": [
            line
            for line in cache.splitlines()
            if line.startswith(
                (
                    "CMAKE_BUILD_TYPE:",
                    "CMAKE_CUDA_ARCHITECTURES:",
                    "VIBEQC_CUDA_FAST_COMPILE:",
                    "VIBEQC_AOT_PROFILE:",
                )
            )
        ],
        "endpoints": [],
    }
    # Load the CUDA context/library before either candidate's cold plan timing.
    Calculator(device="cuda").singlepoint([("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))])
    # Keep the main Cartesian inventory at 15 AOs, inside the existing <=16-AO
    # direct-HF resource inventory. Larger legacy endpoints below explicitly
    # disclose where that shared total-budget guarantee is unavailable.
    spf_basis = (
        Shell(0, 0, (Primitive(1.5, 1.0), Primitive(0.7, -0.1))),
        Shell(0, 1, (Primitive(0.8, 1.0),)),
        Shell(0, 3, (Primitive(0.6, 1.0),)),
        Shell(1, 0, (Primitive(1.2, 1.0),)),
    )
    inventory = endpoint_inventory(args.domain)
    for family, method, representation, fitted, count in inventory:
        basis = (
            spf_basis
            if family == "spf"
            else (spf_basis[0], Shell(0, 2, (Primitive(0.8, 1.0),)), *spf_basis[2:])
        )
        real_molecule = family not in ("spf", "sdf")
        key = case_id((family, method, representation, fitted, count))
        if args.case and key not in args.case:
            continue
        # Pair-policy execution itself introduces no allocation, tile buffer
        # or retained cache. Record both budgets where the HF inventory applies.
        df_budget = (1 if count == 1 else 4) << 20
        if args.domain == "df" and real_molecule and count == 1:
            df_budget = 0
        # The shared HF planner reserves 512 MiB for opaque CUDA libraries,
        # in addition to explicit DF/force workspace; the DF sub-budget alone
        # is not a bound on the complete endpoint.
        budget = ResourceBudget(
            host_bytes=512 << 20, device_bytes=(1 if count == 1 else 2) << 30
        )
        resource_note = None
        if family == "sdf" and representation == "cartesian" and not fitted:
            budget = None
            resource_note = (
                "Existing 18-AO direct endpoint: shared HF inventory v1 supports only <=16 public AOs. "
                "Measure the legacy unbudgeted endpoint explicitly; no total-budget guarantee is claimed. "
                "The one-electron pair policy introduces no allocation and its native resources are measured separately."
            )
        atoms = [
            [("He", (0.0, 0.0, -0.7 - 0.1 * i)), ("H", (0.1, 0.0, 0.7))]
            for i in range(count)
        ]
        charge, multiplicity = (1, 1) if method == "rhf" else (0, 2)
        if real_molecule:
            sys.path.insert(0, str(args.root))
            from benchmarks._cases import benchmark_cases

            fixture = benchmark_cases()[family]
            basis = fixture.vibeqc_basis
            atoms = [
                [
                    (element, tuple(np.asarray(r) * (1 + 0.01 * i)))
                    for element, r in fixture.atoms
                ]
                for i in range(count)
            ]
            charge, multiplicity = fixture.charge, fixture.multiplicity
            budget = None
            resource_note = (
                "Named molecular basis exceeds the <=16-orbital-AO whole-HF resource inventory. "
                "Record actual DF plan diagnostics; scoped derivative allocations are validated separately. "
                "No total-budget guarantee is claimed for this existing larger endpoint."
            )
        calc = Calculator(
            method=method,
            device="cuda",
            basis=basis,
            basis_representation=representation,
            density_fitting="cuda" if fitted else "none",
            density_fitting_memory_budget_bytes=df_budget,
            resource_budget=budget,
            energy_tolerance=1e-12,
            density_tolerance=1e-10,
            screening_tolerance=1e-14,
        )
        row = {
            "case": key,
            "atoms": atoms,
            "basis": basis
            if isinstance(basis, str)
            else [asdict(shell) for shell in basis],
            "df_budget_bytes": df_budget,
            "resource_scope_note": resource_note,
            "seconds": {},
            "results": {},
        }
        start = perf_counter()
        prepared = calc.prepare_batch(
            atoms,
            charges=[charge] * count,
            multiplicities=[multiplicity] * count,
        )
        row["seconds"]["prepare"] = perf_counter() - start
        original = [np.array([r for _, r in system]) for system in atoms]
        moved = [r.copy() for r in original]
        for i, positions in enumerate(moved):
            positions[1, 0] += 0.013 * (i + 1)
        with prepared:
            row["resource_plan"] = (
                None
                if prepared.resource_plan is None
                else prepared.resource_plan.to_dict()
            )
            for phase, positions in (
                ("cold", None),
                ("warm", None),
                ("moved", moved),
                ("restored", original),
            ):
                start = perf_counter()
                result = prepared.execute(positions, strict=True)
                row["seconds"][phase] = perf_counter() - start
                if phase == "cold":
                    # Keep the same density seed for each replay workload.
                    prepared.set_warm_start_updates(False)
                if any(
                    not r.converged
                    or r.executed_backend != "cuda"
                    or not np.isfinite([r.energy, r.energy_change, r.density_rms]).all()
                    or not np.isfinite(r.forces).all()
                    for r in result.items
                ):
                    raise RuntimeError(
                        "wrong backend, unconverged or nonfinite endpoint"
                    )
                row["results"][phase] = [
                    {
                        "energy": r.energy,
                        "forces": r.forces.tolist(),
                        "iterations": r.iterations,
                        "energy_change": r.energy_change,
                        "density_rms": r.density_rms,
                    }
                    for r in result.items
                ]
            row["density_fitting"] = (
                [
                    d.to_dict()
                    for d in prepared.last_density_fitting_metric_diagnostics()
                ]
                if fitted
                else []
            )
            row["observed_resources"] = prepared.resource_diagnostics
        row["seconds"]["complete"] = sum(row["seconds"].values())
        if args.domain == "df":
            # Sharing value traversal must also preserve callers that omit the
            # force consumer. These are fresh public singlepoints, not prepared
            # replay timings, and are kept separate from the four force phases.
            start = perf_counter()
            values = [
                calc.singlepoint(
                    system,
                    charge=charge,
                    multiplicity=multiplicity,
                    properties=("energy",),
                )
                for system in atoms
            ]
            elapsed = perf_counter() - start
            if any(
                not r.converged
                or r.executed_backend != "cuda"
                or r.forces is not None
                or not np.isfinite([r.energy, r.energy_change, r.density_rms]).all()
                for r in values
            ):
                raise RuntimeError("invalid energy-only endpoint")
            row["energy_only"] = {
                "seconds": elapsed,
                "properties": ["energy"],
                "results": [
                    {
                        "energy": r.energy,
                        "iterations": r.iterations,
                        "energy_change": r.energy_change,
                        "density_rms": r.density_rms,
                    }
                    for r in values
                ],
            }
        records["endpoints"].append(row)
        write(args.output, records)
        print(json.dumps({"case": key, "seconds": row["seconds"]}), flush=True)

    if not records["endpoints"] or (
        args.case and len(records["endpoints"]) != len(args.case)
    ):
        raise ValueError("empty or unknown endpoint inventory")


def measure_worker(args, label, cases, path):
    """Retain one fresh process's record without overwriting an earlier attempt."""
    if path.exists():
        raise FileExistsError(f"refusing to overwrite measured sample {path}")
    argv = [
        sys.executable,
        str(Path(__file__).resolve()),
        "worker",
        "--domain",
        args.domain,
        "--process-scope",
        args.process_scope,
        "--root",
        str(getattr(args, f"{label}_root")),
        "--build",
        str(getattr(args, f"{label}_build")),
        "--selection",
        "reference" if label == "baseline" else "generated",
        "--mapping",
        "thread" if label == "baseline" else "shell_warp",
        "--output",
        str(path),
    ]
    for case in cases:
        argv += ["--case", case]
    subprocess.run(argv, check=True)
    return json.loads(path.read_text())


def collect_runs(args):
    """Collect every case/sample with either historical or case-isolated ABBA.

    In case mode, individual process files remain immutable. Aggregated sample
    files only join endpoints whose complete source/build/device/driver metadata
    is identical. They retain all values and can reconstruct each process record
    losslessly; no samples are dropped, averaged or selected during aggregation.
    """
    from vibeqc_compiler.common.timing import interleaved_selection_order

    if args.process_scope not in ("inventory", "case") or args.samples < 5:
        raise ValueError("known process scope and at least five samples are required")
    available = [case_id(row) for row in endpoint_inventory(args.domain)]
    requested = args.case or available
    if len(set(requested)) != len(requested) or set(requested) - set(available):
        raise ValueError("duplicate or unknown endpoint requests")
    cases = [case for case in available if case in requested]
    args.output.mkdir(parents=True, exist_ok=True)
    if (
        (args.output / "case-workers").exists()
        or any(args.output.glob("*-*.json"))
        or (args.output / "comparison.json").exists()
    ):
        raise FileExistsError("refusing to reuse an existing measurement directory")
    order = interleaved_selection_order(args.samples)
    runs = {label: [None] * args.samples for label in ("baseline", "candidate")}
    units = [[case] for case in cases] if args.process_scope == "case" else [cases]
    for index, unit in enumerate(units):
        counters = {"baseline": 0, "candidate": 0}
        for label in order:
            sample = counters[label]
            counters[label] += 1
            name = f"{label}-{sample}.json"
            path = args.output / name
            if args.process_scope == "case":
                path = args.output / "case-workers" / str(index) / name
            run = measure_worker(args, label, unit, path)
            if [row["case"] for row in run["endpoints"]] != unit:
                raise ValueError("worker omitted or changed a requested endpoint")
            previous = runs[label][sample]
            if previous is None:
                runs[label][sample] = run
            else:
                old_metadata = {k: v for k, v in previous.items() if k != "endpoints"}
                new_metadata = {k: v for k, v in run.items() if k != "endpoints"}
                if old_metadata != new_metadata:
                    raise ValueError(
                        "case worker source/build/device/driver provenance changed"
                    )
                previous["endpoints"].extend(run["endpoints"])
            if args.process_scope == "case":
                write(args.output / name, runs[label][sample])
    counters = {"baseline": 0, "candidate": 0}
    measured = []
    for label in order:
        measured.append((label, runs[label][counters[label]]))
        counters[label] += 1
    return runs, measured


def compare(args):
    """Retain all samples; use unchanged numerical and 2% endpoint gates.

    The 2% gate is a non-regression ceiling, not a significant-speedup claim.
    Require non-regression for each cold/replay/changed-geometry endpoint.
    The shared significance/noise assessment is retained separately. Additional samples may resolve a noisy failed gate;
    the threshold is not widened after observing data.
    """
    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
    from vibeqc_compiler.common.evidence import canonical_hash
    from vibeqc_compiler.common.performance import assess_comparison

    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("compare requires a finite Slurm GPU allocation")
    runs, measured = collect_runs(args)
    expected_cases = [r["case"] for r in runs["baseline"][0]["endpoints"]]
    for label, group in runs.items():
        expected_source = (
            group[0]["revision"],
            group[0]["native_source_identity"],
            group[0]["library_sha256"],
        )
        for run in group:
            if [r["case"] for r in run["endpoints"]] != expected_cases:
                raise ValueError("endpoint inventories differ")
            if (
                run["revision"],
                run["native_source_identity"],
                run["library_sha256"],
            ) != expected_source:
                raise ValueError(f"{label} source or binary changed between samples")
    rows = []
    for index, base in enumerate(runs["baseline"][0]["endpoints"]):
        row = {
            "case": base["case"],
            "max_energy_error": 0.0,
            "max_force_error": 0.0,
            "phase_ratios": {},
        }
        for group in runs.values():
            for run in group:
                other = run["endpoints"][index]
                if other["case"] != base["case"]:
                    raise ValueError("endpoint inventories differ")
                for phase in base["results"]:
                    for got, want in zip(
                        other["results"][phase], base["results"][phase], strict=True
                    ):
                        row["max_energy_error"] = max(
                            row["max_energy_error"], abs(got["energy"] - want["energy"])
                        )
                        row["max_force_error"] = max(
                            row["max_force_error"],
                            float(
                                np.max(np.abs(np.array(got["forces"]) - want["forces"]))
                            ),
                        )
        samples = []
        phase_names = {
            "cold": "cold-start",
            "warm": "unchanged-geometry",
            "moved": "changed-geometry",
            "restored": "restored-geometry",
        }
        for phase, workload in phase_names.items():
            for label, run in measured:
                endpoint = run["endpoints"][index]
                seconds = endpoint["seconds"][phase]
                if phase == "cold":
                    seconds += endpoint["seconds"]["prepare"]
                samples.append(
                    {
                        "selection": label,
                        "seconds": seconds,
                        "inputs_hash": canonical_hash(
                            {
                                "case": endpoint["case"],
                                "atoms": endpoint["atoms"],
                                "basis": endpoint["basis"],
                                "df_budget_bytes": endpoint["df_budget_bytes"],
                            }
                        ),
                        "workload": workload,
                        "synchronized": True,
                    }
                )
        comparison = assess_comparison(samples)
        if "workloads" not in comparison:
            raise ValueError(f"shared timing validation failed: {comparison}")
        row["shared_comparison"] = comparison
        for workload, summary in comparison["workloads"].items():
            row["phase_ratios"][workload] = 1 - summary["relative_improvement"]
        if args.domain == "df":
            energy_samples = []
            for label, run in measured:
                endpoint = run["endpoints"][index]
                value = endpoint["energy_only"]
                for got, want in zip(
                    value["results"], base["energy_only"]["results"], strict=True
                ):
                    row["max_energy_error"] = max(
                        row["max_energy_error"], abs(got["energy"] - want["energy"])
                    )
                energy_samples.append(
                    {
                        "selection": label,
                        "seconds": value["seconds"],
                        "inputs_hash": canonical_hash(
                            {
                                "case": endpoint["case"],
                                "atoms": endpoint["atoms"],
                                "basis": endpoint["basis"],
                                "df_budget_bytes": endpoint["df_budget_bytes"],
                                "properties": value["properties"],
                            }
                        ),
                        "workload": "energy-only-singlepoints",
                        "synchronized": True,
                    }
                )
            energy_comparison = assess_comparison(energy_samples)
            if "workloads" not in energy_comparison:
                raise ValueError(f"invalid energy-only timings: {energy_comparison}")
            row["energy_only_comparison"] = energy_comparison
            row["phase_ratios"]["energy-only-singlepoints"] = (
                1
                - energy_comparison["workloads"]["energy-only-singlepoints"][
                    "relative_improvement"
                ]
            )
        row["numerical_passed"] = (
            row["max_energy_error"] <= 3e-10 and row["max_force_error"] <= 3e-9
        )
        row["endpoint_passed"] = all(
            ratio <= 1.02 for ratio in row["phase_ratios"].values()
        )
        rows.append(row)
    report = {
        "schema": "vibeqc.cuda-ownership-comparison.v1",
        "samples": args.samples,
        "domain": args.domain,
        "process_scope": args.process_scope,
        "benchmark_driver_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "energy_atol": 3e-10,
        "force_atol": 3e-9,
        "endpoint_ceiling": 1.02,
        "endpoints": rows,
        "passed": all(r["numerical_passed"] and r["endpoint_passed"] for r in rows),
    }
    write(args.output / "comparison.json", report)
    print(json.dumps(report), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    w = commands.add_parser("worker")
    for name in ("root", "build", "output"):
        w.add_argument(f"--{name}", type=Path, required=True)
    w.add_argument("--selection", choices=("reference", "generated"), required=True)
    w.add_argument("--mapping", choices=("thread", "shell_warp"), required=True)
    c = commands.add_parser("compare")
    for name in (
        "baseline-root",
        "baseline-build",
        "candidate-root",
        "candidate-build",
        "output",
    ):
        c.add_argument(f"--{name}", type=Path, required=True)
    c.add_argument("--samples", type=int, default=5)
    for p in (w, c):
        p.add_argument("--case", action="append")
        p.add_argument(
            "--process-scope",
            choices=("inventory", "case"),
            default="inventory",
            help="case pairs complete workloads closely in time; inventory reproduces historical ordering",
        )
        p.add_argument(
            "--domain", choices=("one-electron", "df"), default="one-electron"
        )
    args = parser.parse_args()
    if args.command == "compare" and args.samples < 5:
        parser.error("at least five samples are required")
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    (worker if args.command == "worker" else compare)(args)


if __name__ == "__main__":
    main()
