"""Interleave DF execution policies from one frozen engine-local warm density.

Every policy transition receives an untimed replay to prime/rebuild its plan;
that cost is retained separately. Clean samples exclude instrumentation. A
separate traced pass records executed branches and device component counters.
Run only in a finite Slurm GPU allocation.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import subprocess
import time
import typing
from pathlib import Path

import numpy as np

try:
    from benchmarks._retention import raw_output_path
except ModuleNotFoundError:
    from _retention import raw_output_path
from vibeqc import Calculator, _native

from benchmarks._cases import benchmark_cases
from benchmarks.compare_gpu4pyscf_batch import (
    convergence_payload,
    load_comparison_basis,
    scaled_geometries,
)
from benchmarks.df_component_ledger import aggregate, read_host_trace, read_trace

CASES = {
    96: "water-tetramer-def2-svp-spherical",
    192: "water-octamer-s4-def2-svp-spherical",
    384: "water-hexadecamer-2s4-def2-svp-spherical",
    648: "water-27mer-water27-derived-def2-svp-spherical",
    768: "water-32mer-4s4-def2-svp-spherical",
    864: "water-36mer-water27-derived-def2-svp-spherical",
}


def independent_reference(
    reference: typing.Any,
    aos: typing.Any,
    energies: typing.Any,
    forces: typing.Any,
    basis_metadata: typing.Any,
) -> typing.Any:
    """Reject stale scientific metadata and broadcasting before numerical gates.

    Geometry uses the comparison runner's deterministic batch-one round trip;
    basis fingerprints bind the retained reference to the active basis pack.
    No native library or GPU is needed to validate an already loaded result.
    """
    case = benchmark_cases()[CASES[aos]]
    expected = {
        "case": CASES[aos],
        "ao_count": aos,
        "batch_size": 1,
        "method": case.method,
        "charge": case.charge,
        "multiplicity": case.multiplicity,
        "basis_representation": case.basis_representation,
        "auxiliary_basis": "same as orbital basis",
        "properties": ["energy", "forces"],
        "density_fitting": "cuda",
        "density_fitting_relative_threshold": 1e-10,
        "density_fitting_memory_budget_bytes": 0,
        "energy_tolerance": 1e-12,
        "density_tolerance": 1e-10,
        "reference_gradient_tolerance": 1e-10,
        "max_iterations": 100,
        "vibeqc_screening_tolerance": 1e-12,
        "direct_scf_tolerance": 1e-14,
        "geometries": [
            [
                {"element": element, "coordinates_bohr": list(position)}
                for element, position in atoms
            ]
            for atoms in scaled_geometries(case.atoms, 1)
        ],
    }
    workload = reference["workload"]
    for key, value in expected.items():
        if workload.get(key) != value:
            raise RuntimeError(f"independent reference workload differs: {key}")
    retained_basis = [
        row.get("basis_metadata") for row in reference["vibeqc"]["cold_convergence"]
    ]
    # Live electron metadata contains tuples; JSON necessarily retains lists.
    # Canonicalize that representation without weakening identity comparisons.
    actual_basis = json.loads(json.dumps(basis_metadata))
    if not all(actual_basis) or retained_basis != actual_basis:
        raise RuntimeError("independent reference basis metadata differs")
    expected_energy = np.asarray(reference["gpu4pyscf"]["energies_hartree"])
    expected_forces = np.asarray(reference["gpu4pyscf"]["forces_hartree_per_bohr"])
    for label, actual, retained in (
        ("energy", energies, expected_energy),
        ("force", forces, expected_forces),
    ):
        if actual.shape != retained.shape:
            raise RuntimeError(f"independent reference {label} shape differs")
        if not np.all(np.isfinite(retained)):
            raise RuntimeError(f"independent reference {label} is not finite")
    return expected_energy, expected_forces


def cpu_reference(
    case: typing.Any, orbital_basis: typing.Any, auxiliary_basis: typing.Any
) -> typing.Any:
    """Build an independent explicit-basis oracle outside all native timers."""
    import pyscf
    from pyscf import gto, scf

    molecule = gto.M(
        atom=case.atoms,
        unit="Bohr",
        basis=orbital_basis,
        cart=case.basis_representation == "cartesian",
        spin=case.multiplicity - 1,
        charge=case.charge,
        verbose=0,
    )
    oracle = (scf.UHF if case.method == "uhf" else scf.RHF)(molecule).density_fit(
        auxbasis=auxiliary_basis
    )
    oracle.conv_tol, oracle.conv_tol_grad, oracle.max_cycle = 1e-13, 1e-12, 200
    oracle.direct_scf_tol = 1e-14
    oracle.kernel()
    if not oracle.converged:
        raise RuntimeError("independent CPU reference did not converge")
    gradient = oracle.nuc_grad_method()
    gradient.auxbasis_response = True
    energy, force = np.array([oracle.e_tot]), -gradient.kernel()[None, :, :]
    if not np.isfinite(energy).all() or not np.isfinite(force).all():
        raise RuntimeError("independent CPU reference is nonfinite")
    return (
        energy,
        force,
        {
            "pyscf_version": pyscf.__version__,
            "ao_count": molecule.nao,
            "auxiliary_count": oracle.with_df.auxmol.nao,
            "orbital_basis": orbital_basis,
            "auxiliary_basis": auxiliary_basis,
            "energies_hartree": energy.tolist(),
            "forces_hartree_per_bohr": force.tolist(),
            "energy_tolerance": oracle.conv_tol,
            "gradient_tolerance": oracle.conv_tol_grad,
            "direct_scf_tolerance": oracle.direct_scf_tol,
            "auxbasis_response": True,
        },
    )


def endpoint_errors(
    energy: typing.Any,
    force: typing.Any,
    reference_energy: typing.Any,
    reference_force: typing.Any,
) -> typing.Any:
    """Keep every reference check shape-strict, including cold and prime calls."""
    energy, reference_energy = np.asarray(energy), np.asarray(reference_energy)
    pairs = [(energy, reference_energy)]
    if force is not None:
        pairs.append((np.asarray(force), np.asarray(reference_force)))
    if any(
        a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all()
        for a, b in pairs
    ):
        raise RuntimeError("endpoint/reference energy or force arrays are invalid")
    differences = [float(np.max(np.abs(a - b))) for a, b in pairs]
    return differences[0], differences[1] if force is not None else None


def main() -> None:
    """Retain each numerical result before enforcing unchanged strict gates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aos", type=int, choices=CASES, required=True)
    parser.add_argument("--output", type=raw_output_path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--df-budget",
        type=int,
        default=0,
        help="Total native DF value/response budget in bytes (zero selects resource policy)",
    )
    parser.add_argument("--control", default="VIBEQC_DF_EXCHANGE")
    parser.add_argument("--policies", nargs="+", default=["dense", "occupied"])
    parser.add_argument(
        "--policy-controls",
        type=json.loads,
        default={},
        help="JSON mapping each policy to additional CUDA controls, applied before its prime",
    )
    parser.add_argument("--trace", action="store_true")
    parser.add_argument(
        "--components-after",
        action="store_true",
        help="Run a separate component pass after all clean samples",
    )
    parser.add_argument("--cuda-profile", action="store_true")
    parser.add_argument(
        "--shell-work",
        action="store_true",
        help="Capture detailed shell work only in the separate component pass",
    )
    parser.add_argument("--host-trace", action="store_true")
    parser.add_argument("--journal", action="store_true")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--cpu-reference", action="store_true")
    parser.add_argument("--orbital-basis-file", type=Path)
    parser.add_argument("--auxiliary-basis-file", type=Path)
    parser.add_argument("--source-patch", type=Path)
    parser.add_argument("--warm-checkpoint-in", type=Path)
    parser.add_argument("--warm-checkpoint-out", type=Path)
    parser.add_argument(
        "--cold-control",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Freeze a common post-cold density using declared execution controls",
    )
    parser.add_argument(
        "--skip-cold",
        action="store_true",
        help="Initialize from the checkpoint; requires an independent reference",
    )
    parser.add_argument(
        "--energy-only",
        action="store_true",
        help="Measure SCF after a complete cold reference call",
    )
    parser.add_argument(
        "--expected-iterations",
        type=int,
        help="Reject samples outside the declared fixed SCF update count",
    )
    args = parser.parse_args()
    if (
        not os.environ.get("SLURM_JOB_ID")
        or args.repeats < 1
        or args.df_budget < 0
        or (args.expected_iterations is not None and args.expected_iterations < 1)
    ):
        parser.error("requires Slurm, positive repeats and a nonnegative DF budget")
    if args.shell_work and not args.components_after:
        parser.error("--shell-work requires --components-after")
    if args.cpu_reference and args.reference:
        parser.error("choose one independent reference source")
    if (
        args.orbital_basis_file or args.auxiliary_basis_file
    ) and not args.cpu_reference:
        parser.error("explicit bases require a fresh --cpu-reference")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        parser.error("refusing to overwrite evidence")
    if args.warm_checkpoint_out and args.warm_checkpoint_out.exists():
        parser.error("refusing to overwrite the frozen warm checkpoint")
    if args.skip_cold and (
        not args.warm_checkpoint_in or not (args.reference or args.cpu_reference)
    ):
        parser.error("skip-cold requires a frozen checkpoint and independent reference")
    from vibeqc.resources_hf import _CUDA_SCHEDULE_VARIABLES

    if not isinstance(args.policy_controls, dict) or any(
        policy not in args.policies
        or not isinstance(controls, dict)
        or any(
            name not in _CUDA_SCHEDULE_VARIABLES
            or name == args.control
            or not isinstance(value, str)
            or not value
            for name, value in controls.items()
        )
        for policy, controls in args.policy_controls.items()
    ):
        parser.error(
            "policy-controls must map selected policies to known CUDA controls"
        )
    # Every arm must set the same extra controls. Otherwise an interleaved arm
    # could silently inherit the preceding arm's settings.
    control_sets = [set(args.policy_controls.get(p, {})) for p in args.policies]
    if any(names != control_sets[0] for names in control_sets):
        parser.error("policy-controls must set the same controls for every policy")

    def select_policy(policy: typing.Any) -> None:
        """Apply the complete declared arm before rebuilding/priming its owner."""
        os.environ[args.control] = policy
        os.environ.update(args.policy_controls.get(policy, {}))

    cold_controls = {}
    for assignment in args.cold_control:
        name, separator, value = assignment.partition("=")
        if name not in _CUDA_SCHEDULE_VARIABLES or not separator or not value:
            parser.error("cold-control requires a known CUDA schedule NAME=VALUE")
        cold_controls[name] = value
    case = benchmark_cases()[CASES[args.aos]]
    orbital, cpu_orbital = case.vibeqc_basis, case.pyscf_basis
    if args.orbital_basis_file:
        orbital, cpu_orbital = load_comparison_basis(
            args.orbital_basis_file, case, role="orbital", compute_forces=True
        )
    auxiliary, cpu_auxiliary = orbital, cpu_orbital
    if args.auxiliary_basis_file:
        auxiliary, cpu_auxiliary = load_comparison_basis(
            args.auxiliary_basis_file, case, role="auxiliary", compute_forces=True
        )
    # The historical explicit JKFIT fixtures have stricter retained gates.
    energy_gate, force_gate = (
        (3e-11, 3e-11) if args.auxiliary_basis_file else (1e-9, 1e-8)
    )
    screening_tolerance = 1e-14 if args.auxiliary_basis_file else 1e-12
    fresh_reference = (
        cpu_reference(case, cpu_orbital, cpu_auxiliary) if args.cpu_reference else None
    )
    if fresh_reference is not None and fresh_reference[2]["ao_count"] != args.aos:
        parser.error("explicit orbital basis AO count differs from --aos")
    library = Path(os.environ["VIBEQC_LIBRARY"]).resolve()
    native = _native.load_library()
    native.vibeqc_get_source_identity.restype = ctypes.c_char_p
    # A pinned library may outlive subsequent local edits. Prefer its frozen
    # source patch so queued runs describe the code actually linked.
    frozen_patch = args.source_patch or library.with_name("source.patch")
    patch = (
        frozen_patch.read_bytes()
        if frozen_patch.exists()
        else subprocess.check_output(
            ["git", "diff", "HEAD", "--binary", "--", "src", "python"]
        )
    )
    args.output.with_suffix(".source.patch").write_bytes(patch)
    payload = {
        "case": CASES[args.aos],
        "aos": args.aos,
        "scope": "intrusive diagnostic"
        if args.trace or args.host_trace or args.journal or args.cuda_profile
        else "clean endpoint",
        "native_source_identity": native.vibeqc_get_source_identity().decode(),
        "source_patch_sha256": hashlib.sha256(patch).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "controls": {k: v for k, v in os.environ.items() if k.startswith("VIBEQC_")},
        "scientific_settings": {
            "density_fitting_memory_budget_bytes": args.df_budget,
            "basis": cpu_orbital,
            "basis_representation": case.basis_representation,
            "auxiliary_basis": cpu_auxiliary,
            "metric_relative_threshold": 1e-10,
            "method": case.method,
            "energy_tolerance": 1e-12,
            "density_tolerance": 1e-10,
            "max_iterations": 100,
            "screening_tolerance": screening_tolerance,
            "geometries_bohr": case.atoms,
        },
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "control": args.control,
        "policies": args.policies,
        "policy_controls": args.policy_controls,
        "warm_policy": "one frozen post-cold density, prime every policy transition",
        "measured_properties": ["energy"] if args.energy_only else ["energy", "forces"],
        "expected_iterations": args.expected_iterations,
        "cold_controls": cold_controls,
        "gates": {"energy": energy_gate, "force": force_gate},
        "cpu_reference": None if fresh_reference is None else fresh_reference[2],
        "basis_file_sha256": {
            role: hashlib.sha256(path.read_bytes()).hexdigest()
            for role, path in (
                ("orbital", args.orbital_basis_file),
                ("auxiliary", args.auxiliary_basis_file),
            )
            if path is not None
        },
        "samples": [],
    }

    def save() -> None:
        """Keep raw numerical evidence even if a subsequent gate fails."""
        args.output.write_text(json.dumps(payload, indent=2) + "\n")

    def execute(batch: typing.Any, *, cold: typing.Any = False) -> typing.Any:
        """Time the complete strict energy-and-force endpoint."""
        start = time.perf_counter()
        result = batch.execute(
            strict=True,
            properties=("energy",)
            if args.energy_only and not cold
            else ("energy", "forces"),
        )
        seconds = time.perf_counter() - start
        return result, seconds

    select_policy(args.policies[0])
    previous_controls = {name: os.environ.get(name) for name in cold_controls}
    os.environ.update(cold_controls)
    calculator = Calculator(
        method=case.method,
        basis=orbital,
        basis_representation=case.basis_representation,
        device="cuda",
        density_fitting="cuda",
        auxiliary_basis=auxiliary,
        density_fitting_memory_budget_bytes=args.df_budget,
        screening_tolerance=screening_tolerance,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        max_iterations=100,
    )
    prepare_start = time.perf_counter()
    with calculator.prepare_batch([case.atoms]) as batch:
        payload["prepare_seconds"] = time.perf_counter() - prepare_start
        if args.skip_cold:
            # This is a complete untimed checkpoint replay, not a cold solve.
            # Freeze before replay so it cannot replace the declared input D.
            payload["warm_checkpoint_restore"] = batch.load_checkpoint(
                args.warm_checkpoint_in, allow_warm=True
            )
            batch.set_warm_start_updates(False)
        cold, initial_seconds = execute(batch, cold=True)
        payload["cold_seconds"] = None if args.skip_cold else initial_seconds
        payload["complete_cold_seconds"] = (
            None if args.skip_cold else payload["prepare_seconds"] + initial_seconds
        )
        payload["cold_convergence"] = (
            None if args.skip_cold else convergence_payload(cold)
        )
        payload["initialization"] = "checkpoint replay" if args.skip_cold else "cold"
        payload["initialization_seconds"] = initial_seconds
        payload["initialization_convergence"] = convergence_payload(cold)
        # Checkpoints carry the exact density and scientific restart identity.
        # Loading after cold preserves cold cost/reference reporting while
        # making cross-library warm replays use byte-identical input arrays.
        if args.warm_checkpoint_in and not args.skip_cold:
            # Runtime ablation controls intentionally differ. The restart
            # loader still validates basis/model identity and imports only D;
            # the target performs its complete normal convergence/force gates.
            payload["warm_checkpoint_restore"] = batch.load_checkpoint(
                args.warm_checkpoint_in, allow_warm=True
            )
        if args.warm_checkpoint_out:
            args.warm_checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
            batch.save_checkpoint(args.warm_checkpoint_out)
        checkpoint = args.warm_checkpoint_out or args.warm_checkpoint_in
        if checkpoint:
            from vibeqc.checkpoint import inspect_checkpoint

            payload["warm_checkpoint_sha256"] = hashlib.sha256(
                checkpoint.read_bytes()
            ).hexdigest()
            payload["warm_density_sha256"] = {
                blob["name"]: blob["sha256"]
                for blob in inspect_checkpoint(checkpoint).blobs
                if blob["name"].startswith("density_")
            }
        batch.set_warm_start_updates(False)
        for name, value in previous_controls.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        expected_energy = cold.energies
        expected_forces = np.array([item.forces for item in cold.items])
        if fresh_reference is not None:
            expected_energy, expected_forces = fresh_reference[:2]
            cold_errors = endpoint_errors(
                cold.energies,
                np.array([item.forces for item in cold.items]),
                expected_energy,
                expected_forces,
            )
            payload["initialization_errors"] = cold_errors
            save()
            if cold_errors[0] > energy_gate or cold_errors[1] > force_gate:
                raise RuntimeError(
                    "initialization failed independent energy/force gates"
                )
        if args.reference:
            reference_bytes = args.reference.read_bytes()
            expected_energy, expected_forces = independent_reference(
                json.loads(reference_bytes),
                args.aos,
                expected_energy,
                expected_forces,
                [item.basis_metadata for item in cold.items],
            )
            payload["reference_sha256"] = hashlib.sha256(reference_bytes).hexdigest()
        save()
        # Every clean replay precedes instrumentation. Component records live
        # in a separate collection and reuse the same frozen density/owner.
        jobs = [
            (repeat, policy, False)
            for repeat in range(args.repeats)
            for policy in (args.policies if repeat % 2 == 0 else args.policies[::-1])
        ]
        if args.components_after:
            jobs += [(0, policy, True) for policy in args.policies]
        for repeat, policy, diagnostic in jobs:
            traced = args.trace or diagnostic
            phase = "diagnostic-" if diagnostic else ""
            select_policy(policy)
            prime, prime_seconds = execute(batch)
            if fresh_reference is not None:
                prime_errors = endpoint_errors(
                    prime.energies,
                    None
                    if args.energy_only
                    else np.array([item.forces for item in prime.items]),
                    expected_energy,
                    expected_forces,
                )
                if prime_errors[0] > energy_gate or (
                    prime_errors[1] is not None and prime_errors[1] > force_gate
                ):
                    payload["failed_prime"] = {
                        "policy": policy,
                        "repeat": repeat,
                        "errors": prime_errors,
                    }
                    save()
                    raise RuntimeError("prime failed independent energy/force gates")
            trace = args.output.with_suffix(f".{phase}{repeat}-{policy}.jsonl")
            host_trace = args.output.with_suffix(
                f".{phase}{repeat}-{policy}.host.jsonl"
            )
            if traced:
                os.environ["VIBEQC_DF_TRACE"] = str(trace.resolve())
            if traced or args.host_trace:
                os.environ["VIBEQC_DF_HOST_TRACE"] = str(host_trace.resolve())
            journal = args.output.with_suffix(
                f".{phase}{repeat}-{policy}.journal.jsonl"
            )
            if args.journal or diagnostic:
                os.environ["VIBEQC_DF_PROGRESS_TRACE"] = str(journal.resolve())
            if args.cuda_profile:
                cudart = ctypes.CDLL("libcudart.so.12")
                if cudart.cudaProfilerStart() != 0:
                    raise RuntimeError("cudaProfilerStart failed")
            previous_work = os.environ.get("VIBEQC_DF_SHELL_WORK")
            previous_counters = os.environ.get("VIBEQC_DF_SHELL_COUNTERS")
            if diagnostic:
                os.environ["VIBEQC_DF_SHELL_COUNTERS"] = "1"
                if args.shell_work:
                    os.environ["VIBEQC_DF_SHELL_WORK"] = "1"
            result, seconds = execute(batch)
            if diagnostic:
                if previous_counters is None:
                    os.environ.pop("VIBEQC_DF_SHELL_COUNTERS", None)
                else:
                    os.environ["VIBEQC_DF_SHELL_COUNTERS"] = previous_counters
            if previous_work is None:
                os.environ.pop("VIBEQC_DF_SHELL_WORK", None)
            else:
                os.environ["VIBEQC_DF_SHELL_WORK"] = previous_work
            if args.cuda_profile and cudart.cudaProfilerStop() != 0:
                raise RuntimeError("cudaProfilerStop failed")
            os.environ.pop("VIBEQC_DF_PROGRESS_TRACE", None)
            os.environ.pop("VIBEQC_DF_TRACE", None)
            os.environ.pop("VIBEQC_DF_HOST_TRACE", None)
            forces = (
                None
                if args.energy_only
                else np.array([item.forces for item in result.items])
            )
            if (
                result.energies.shape != expected_energy.shape
                or (forces is not None and forces.shape != expected_forces.shape)
                or not np.all(np.isfinite(result.energies))
                or (forces is not None and not np.all(np.isfinite(forces)))
            ):
                raise RuntimeError("endpoint returned invalid energy/force arrays")
            sample = {
                "policy": policy,
                "scope": "intrusive diagnostic"
                if traced or args.host_trace or args.journal or args.cuda_profile
                else "clean endpoint",
                "repeat": repeat,
                "seconds": seconds,
                "prime_seconds": prime_seconds,
                "prime_iterations": [item.iterations for item in prime.items],
                "iterations": [item.iterations for item in result.items],
                "convergence": convergence_payload(result),
                "energies_hartree": result.energies.tolist(),
                "forces_hartree_per_bohr": None if forces is None else forces.tolist(),
                "maximum_energy_error": float(
                    np.max(np.abs(result.energies - expected_energy))
                ),
                "maximum_force_error": None
                if forces is None
                else float(np.max(np.abs(forces - expected_forces))),
                "metric": [
                    d.to_dict() for d in batch.last_density_fitting_metric_diagnostics()
                ],
            }
            sample["peak_memory"] = {
                "df_plan_peak_device_bytes": max(
                    (item["peak_device_bytes"] for item in sample["metric"]), default=0
                ),
                "df_plan_peak_host_bytes": max(
                    (item["peak_host_bytes"] for item in sample["metric"]), default=0
                ),
            }
            if traced:
                sample["components"] = aggregate(read_trace(trace))
                # Query after the timed call, while the prepared owner is
                # still live. A separate sampler measures process peaks;
                # a post-close reading would instead measure teardown.
                processes = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-compute-apps=pid,used_memory",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                )
                process_memory = {
                    int(pid): int(mib) * 1024**2
                    for line in processes.splitlines()
                    for pid, mib in [line.split(",")]
                }
                sample["process_device_resident_bytes"] = process_memory[os.getpid()]
                sample["peak_memory"]["process_device_resident_bytes"] = sample[
                    "process_device_resident_bytes"
                ]
            if traced or args.host_trace:
                # Inclusive host force scope begins after the physical
                # final state is selected and includes one-electron/Pulay,
                # DF response, transfers and the completed atom gradient.
                # These are intrusive stage timings, not clean endpoints
                # or an energy-only subtraction with a different SCF solve.
                force_regions = [
                    region
                    for record in read_host_trace(host_trace)
                    for region in record["regions"]
                    if region["name"] == "force_response"
                ]
                if args.energy_only:
                    if force_regions:
                        raise RuntimeError(
                            "energy-only endpoint executed a force stage"
                        )
                else:
                    if len(force_regions) != 1 or force_regions[0]["failed"]:
                        raise RuntimeError("expected one completed force stage")
                    sample["force_stage_seconds"] = force_regions[0]["wall_ms"] / 1000
            if args.journal or diagnostic:
                journal_rows = [
                    json.loads(line) for line in journal.read_text().splitlines()
                ]
                sample["final_state_observations"] = [
                    row
                    for row in journal_rows
                    if row.get("key", "").startswith("final_")
                ]
                resource_keys = {
                    "resource_policy_version",
                    "resolved_total_budget_bytes",
                    "resolved_value_budget_bytes",
                    "resolved_response_budget_bytes",
                    "resource_reserved_headroom_bytes",
                    "resource_observed_free_bytes",
                    "resource_observed_total_bytes",
                    "resource_probe_live",
                }
                resource_policy = {
                    row["key"]: row["value"]
                    for row in journal_rows
                    if row.get("event") == "VALUE" and row.get("key") in resource_keys
                }
                if diagnostic and resource_keys - resource_policy.keys():
                    raise RuntimeError("DF resource-policy evidence is incomplete")
                sample["resource_policy"] = resource_policy
            payload.setdefault("diagnostics" if diagnostic else "samples", []).append(
                sample
            )
            save()
            print(phase + policy, repeat, seconds, sample["iterations"], flush=True)
            if sample["maximum_energy_error"] > energy_gate or (
                sample["maximum_force_error"] is not None
                and sample["maximum_force_error"] > force_gate
            ):
                raise RuntimeError("unchanged DF energy/force gate failed")
            if args.expected_iterations is not None and any(
                count != args.expected_iterations for count in sample["iterations"]
            ):
                raise RuntimeError(
                    "SCF iteration branch differs from the declared fixed work"
                )


if __name__ == "__main__":
    main()
