"""Host-eigensolve diagnostic workloads used by the existing #206 matrix CLI.

By default A/B configurations are identical protocol controls. The explicit
eager-core ablation restores discarded warm frames in the baseline selection.
Separate clean and profiled invocations retain setup, destruction and every
changed-geometry call. Same-model ablations retain their iteration branches;
external-engine parity remains with the existing matched #206 matrix.
"""

from __future__ import annotations

import os
import time
import typing
from pathlib import Path

import numpy as np

from benchmarks._cases import benchmark_cases
from benchmarks.compare_gpu4pyscf_batch import convergence_payload, scaled_geometries
from benchmarks.df_component_ledger import (
    aggregate_host,
    read_host_trace,
    trace_identity,
)
from benchmarks.issue206_df_force_probe import _source_metadata
from benchmarks.validation_gate import _cuda
from tools.vibeqc_validation.performance import assess_comparison, measure_interleaved
from tools.vibeqc_validation.schema import canonical_hash


class AblationBranchMismatch(ValueError):
    """Retain completed samples without authorizing an unmatched timing claim.

    The CLI publishes this diagnostic payload separately from successful
    endpoints. All workloads remain available, including the differing SCF
    branches that caused rejection; no timing assessment is produced.
    """

    def __init__(self, rows: typing.Any, mismatches: typing.Any) -> None:
        super().__init__(
            "preparation timing requires matching SCF iteration/retry branches: "
            + repr(mismatches)
        )
        self.evidence = {
            "schema": "vibeqc.issue206.rejected_host_workloads",
            "version": 1,
            "status": "rejected",
            "reason": "SCF iteration/retry branch mismatch",
            "mismatched_branches": mismatches,
            "samples": rows,
            "timing_assessment": None,
        }


def validate_ablation_branches(rows: typing.Any) -> None:
    """Reject timing promotion when either selection took different SCF work.

    A frozen seed must give one iteration/retry branch per workload and item.
    Equal energies alone do not justify an iteration-unmatched speed claim.
    """
    mismatches = {}
    for workload in sorted({row["workload"] for row in rows}):
        branches = {
            selection: {
                tuple(
                    (
                        item["iterations"],
                        item["warm_start_used"],
                        item["warm_start_fallback"],
                    )
                    for item in row["diagnostics"]["convergence"]
                )
                for row in rows
                if row["workload"] == workload and row["selection"] == selection
            }
            for selection in ("baseline", "candidate")
        }
        if (
            len(branches["baseline"]) != 1
            or branches["baseline"] != branches["candidate"]
        ):
            mismatches[workload] = {
                selection: sorted(values) for selection, values in branches.items()
            }
    if mismatches:
        raise AblationBranchMismatch(rows, mismatches)


def preparation_policies(ablation: typing.Any) -> typing.Any:
    """Return eager-core/rebuild-overlap switches for a causal A/B comparison.

    All three increments share the original eager/rebuilt baseline. Comparing
    the separate candidates yields baseline, lazy-only, cache-only and combined
    measurements without changing the binary or frozen SCF seed.
    """
    candidates = {
        "lazy-core": (False, True),
        "overlap-cache": (True, False),
        "combined": (False, False),
    }
    if ablation not in candidates:
        raise ValueError("unknown preparation ablation")
    return {"baseline": (True, True), "candidate": candidates[ablation]}


def validate_preparation_counts(
    components: typing.Any,
    *,
    batch_size: typing.Any,
    workload: typing.Any,
    eager: typing.Any,
    rebuild: typing.Any,
) -> None:
    """Require actual solves and matching cache scopes, including partial rebuilds.

    The changed-geometry workload moves only the last item and restores the
    original geometry before every sample. Other items must retain their X.
    """
    cold = workload == "cold-start"
    changed = workload == "changed-geometry"
    expected_core = batch_size if cold or eager else 0
    expected_overlap = batch_size if cold or rebuild else int(changed)
    expected_misses = 0 if rebuild else expected_overlap
    expected_hits = 0 if rebuild else batch_size - expected_misses
    reference = components["eigensolves_by_reason"]
    device = components.get("device_eigensolves_by_reason", {})
    solves = {
        name: {
            "calls": reference.get(name, {}).get("calls", 0)
            + device.get(name, {}).get("calls", 0)
        }
        for name in ("overlap", "core_guess")
    }
    phases = components["exclusive_phases"]
    expected = (
        (solves, "core_guess", expected_core),
        (solves, "overlap", expected_overlap),
        (phases, "initial_density", batch_size),
        (phases, "overlap_cache_miss", expected_misses),
        (phases, "overlap_cache_hit", expected_hits),
    )
    if any(
        rows.get(name, {}).get("calls", 0) != count for rows, name, count in expected
    ):
        raise RuntimeError(
            "preparation ablation did not execute its declared solve/cache policy"
        )


def validate_final_eigen_counts(
    components: typing.Any,
    *,
    batch_size: typing.Any,
    method: typing.Any,
    reference: typing.Any,
    strict_final_state: typing.Any = False,
) -> None:
    """Require actual finalizer leaves; a flag or omitted observer is insufficient.

    Provider ablations force rebuilding after retained-state integration. Cold
    strict correction may need more than one solve; all physical evaluations
    and corrections must agree with the actual provider leaves. UHF corrects
    both spin frames jointly, including an empty occupation channel.
    """
    spins = 2 if method == "uhf" else 1
    expected = batch_size * spins
    solves = components["eigensolves_by_reason"]
    device = components["device_eigensolves_by_reason"]
    phases = components.get("exclusive_phases", {})
    physical = phases.get("final_state_fock_build", {}).get("calls", 0)
    if strict_final_state or physical:
        corrections = phases.get("strict_final_correction", {}).get("calls", 0)
        if (
            physical != batch_size + corrections
            or not batch_size <= corrections <= 16 * batch_size
            or phases.get("final_state_reuse", {}).get("calls", 0)
            or phases.get("final_state_corrected", {}).get("calls", 0) != batch_size
            or phases.get("final_state_validation", {}).get("calls", 0) != physical
        ):
            raise RuntimeError(
                "provider ablation did not perform bounded strict rebuilding"
            )
        checks = phases.get("final_state_fixed_point", {}).get("calls", 0)
        promoted = phases.get("final_state_fixed_point_promotion", {}).get("calls", 0)
        outputs = phases.get("final_state_weighted_density", {}).get("calls", 0)
        if (checks or promoted) and (
            checks != outputs + promoted or not 0 <= promoted <= corrections
        ):
            raise RuntimeError(
                "provider ablation miscounted physical fixed-point solves"
            )
        expected = spins * (corrections + checks - promoted)
    if (
        solves.get("final_fock", {}).get("calls", 0) != (expected if reference else 0)
        or device.get("final_fock", {}).get("calls", 0)
        != (0 if reference else expected)
        or solves.get("fallback", {}).get("calls", 0)
    ):
        raise RuntimeError("final eigen ablation did not execute its declared provider")


def validate_final_state_counts(
    components: typing.Any,
    *,
    batch_size: typing.Any,
    method: typing.Any,
    force: typing.Any,
    reference: typing.Any,
    compute_forces: typing.Any,
) -> None:
    """Distinguish retained work from real correction, even at zero solves.

    Every item must still read and validate its candidate against current F.
    Count joint UHF corrections separately from their two provider leaves;
    rejected candidates may legitimately correct on either ablation side.
    """
    phases = components.get("exclusive_phases", {})

    def calls(name: typing.Any) -> typing.Any:
        return phases.get(name, {}).get("calls", 0)

    corrections = calls("strict_final_correction")
    corrected = calls("final_state_corrected")
    reused = calls("final_state_reuse")
    physical = calls("final_state_fock_build")
    if (
        calls("final_state_read") != batch_size
        or physical != batch_size + corrections
        or calls("final_state_validation") != physical
        or reused + corrected != batch_size
        or not corrected <= corrections <= 16 * corrected
        or (force and (reused or corrected != batch_size))
        or calls("final_state_weighted_density")
        != (batch_size if compute_forces else 0)
        or calls("force_response") != (batch_size if compute_forces else 0)
    ):
        raise RuntimeError(
            "final state ablation omitted current-F work or misreported correction/output selection"
        )
    checks = calls("final_state_fixed_point")
    promoted = calls("final_state_fixed_point_promotion")
    # Historical ledgers have neither probe field. In the new protocol, every
    # successful force item contributes one accepted probe; rejected probes
    # are promoted into corrections without another eigensolve.
    if (checks or promoted) and (
        checks != (batch_size if compute_forces else 0) + promoted
        or not 0 <= promoted <= corrections
    ):
        raise RuntimeError(
            "final state ablation miscounted physical fixed-point solves"
        )
    expected = (corrections + checks - promoted) * (2 if method == "uhf" else 1)
    host = components["eigensolves_by_reason"]
    device = components["device_eigensolves_by_reason"]
    if (
        host.get("final_fock", {}).get("calls", 0) != (expected if reference else 0)
        or device.get("final_fock", {}).get("calls", 0)
        != (0 if reference else expected)
        or host.get("fallback", {}).get("calls", 0)
    ):
        raise RuntimeError(
            "final state ablation did not execute the declared spin providers"
        )


def validate_setup_eigen_counts(
    components: typing.Any,
    *,
    batch_size: typing.Any,
    workload: typing.Any,
    reference: typing.Any,
    eager: typing.Any = False,
    rebuild: typing.Any = False,
) -> None:
    """Gate setup substitution separately from work elimination and final solves."""
    expected = {
        "overlap": batch_size
        if workload == "cold-start" or rebuild
        else int(workload == "changed-geometry"),
        "core_guess": batch_size if workload == "cold-start" or eager else 0,
    }
    for name, count in expected.items():
        for key, selected in (
            ("eigensolves_by_reason", reference),
            ("device_eigensolves_by_reason", not reference),
        ):
            if components[key].get(name, {}).get("calls", 0) != (
                count if selected else 0
            ):
                raise RuntimeError(
                    "setup eigen ablation did not execute its declared provider"
                )
    if components["eigensolves_by_reason"].get("fallback", {}).get("calls", 0):
        raise RuntimeError(
            "setup eigen ablation unexpectedly used a reference fallback"
        )


def host_workloads(
    *,
    case_name: typing.Any,
    batch_size: typing.Any,
    library: typing.Any,
    repeats: typing.Any,
    memory_budget_bytes: typing.Any,
    energy_only: typing.Any,
    trace_directory: typing.Any = None,
    eager_core_ablation: typing.Any = False,
    preparation_ablation: typing.Any = None,
    final_eigen_ablation: typing.Any = False,
    setup_eigen_ablation: typing.Any = False,
    final_state_ablation: typing.Any = False,
    combined_host_ablation: typing.Any = False,
) -> typing.Any:
    """Measure one source-bound cold/replay/rebuild domain without hiding setup.

    Warm updates are frozen after cold convergence. Every changed sample starts
    from the original geometry, restored outside its measured region. A trace
    is collected inside that region and includes all failed attempts; its wall
    time is diagnostic and must never be compared to a clean sample as a gain.
    """
    from vibeqc import Calculator

    if repeats < 5 or batch_size < 1:
        raise ValueError(
            "at least five paired samples and a positive batch are required"
        )
    if any(
        os.environ.get(k)
        for k in ("VIBEQC_DF_TRACE", "VIBEQC_DF_HOST_TRACE", "VIBEQC_DF_PROGRESS_TRACE")
    ):
        raise ValueError(
            "provide trace_directory explicitly; ambient profiling is not clean timing"
        )
    if any(
        os.environ.get(k)
        for k in (
            "VIBEQC_DF_EAGER_CORE_GUESS",
            "VIBEQC_DF_REBUILD_OVERLAP",
            "VIBEQC_DF_REFERENCE_FINAL_EIGEN",
            "VIBEQC_DF_FORCE_FINAL_REBUILD",
            "VIBEQC_DF_REFERENCE_SETUP_EIGEN",
        )
    ):
        raise ValueError(
            "use an explicit ablation; ambient preparation policy is ambiguous"
        )
    if (
        sum(
            map(
                bool,
                (
                    preparation_ablation,
                    eager_core_ablation,
                    final_eigen_ablation,
                    setup_eigen_ablation,
                    final_state_ablation,
                    combined_host_ablation,
                ),
            )
        )
        > 1
    ):
        raise ValueError("select one host ablation")
    policies = (
        preparation_policies(preparation_ablation) if preparation_ablation else None
    )
    if eager_core_ablation:
        policies = {"baseline": (True, False), "candidate": (False, False)}
    if combined_host_ablation:
        policies = preparation_policies("combined")
    library = Path(library).resolve(strict=True)
    source = _source_metadata(library)
    device, synchronize = _cuda()
    case = benchmark_cases()[case_name]
    systems = scaled_geometries(case.atoms, batch_size)
    properties = ("energy",) if energy_only else ("energy", "forces")
    inputs = {
        "case": case_name,
        "systems": systems,
        "method": case.method,
        "basis": case.vibeqc_basis,
        "representation": case.basis_representation,
        "charge": case.charge,
        "multiplicity": case.multiplicity,
        "auxiliary_basis": case.vibeqc_basis,
        "properties": properties,
        "energy_tolerance": 1e-12,
        "density_tolerance": 1e-10,
        "screening_tolerance": 1e-12,
        "metric_relative_threshold": 1e-10,
        "memory_budget_bytes": memory_budget_bytes,
        "eager_core_ablation": eager_core_ablation,
        "preparation_ablation": preparation_ablation,
        "preparation_policies": policies,
        "setup_eigen_ablation": setup_eigen_ablation,
        "setup_eigen_policies": {
            "baseline": "cpu_reference",
            "candidate": "ordinary_xsyevd",
        }
        if setup_eigen_ablation or combined_host_ablation
        else None,
        "final_eigen_ablation": final_eigen_ablation,
        "final_state_ablation": final_state_ablation,
        "combined_host_ablation": combined_host_ablation,
        "final_state_policies": {
            "baseline": "forced_reference_rebuild"
            if combined_host_ablation
            else "forced_device_rebuild",
            "candidate": "verified_retention_or_device_correction",
        }
        if final_state_ablation or combined_host_ablation
        else None,
        "forced_final_rebuild": final_eigen_ablation or setup_eigen_ablation,
        "final_eigen_policies": {
            "baseline": "cpu_reference",
            "candidate": "ordinary_xsyevd",
        }
        if final_eigen_ablation or combined_host_ablation
        else None,
    }
    input_hash = canonical_hash(inputs)
    calculator = Calculator(
        method=case.method,
        basis=case.vibeqc_basis,
        basis_representation=case.basis_representation,
        device="cuda",
        max_iterations=100,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        screening_tolerance=1e-12,
        density_fitting="cuda",
        auxiliary_basis=case.vibeqc_basis,
        density_fitting_relative_threshold=1e-10,
        density_fitting_memory_budget_bytes=memory_budget_bytes,
    )
    if Path(calculator._library._name).resolve() != library:
        raise RuntimeError(
            "loaded library differs from the recorded source-bound binary"
        )
    if trace_directory is not None:
        trace_directory = Path(trace_directory)
        trace_directory.mkdir(parents=True, exist_ok=False)
    sequence = 0

    def measured(workload: typing.Any, evaluate: typing.Any) -> typing.Any:
        def sample(_selection: typing.Any) -> typing.Any:
            nonlocal sequence
            # Both selections replay the same frozen density on one plan.
            # Diagnostic switches restore discarded work, preserving numerical
            # inputs. Separate actual-count gates reject ineffective controls.
            path = None
            if trace_directory is not None:
                path = trace_directory / f"{sequence:04d}-{workload}.jsonl"
                with path.open("x"):
                    pass
                os.environ["VIBEQC_DF_HOST_TRACE"] = str(path.resolve())
            sequence += 1
            try:
                if policies:
                    eager, rebuild = policies[_selection]
                    os.environ["VIBEQC_DF_EAGER_CORE_GUESS"] = "1" if eager else "0"
                    os.environ["VIBEQC_DF_REBUILD_OVERLAP"] = "1" if rebuild else "0"
                if (
                    final_eigen_ablation
                    or setup_eigen_ablation
                    or final_state_ablation
                    or combined_host_ablation
                ):
                    os.environ["VIBEQC_DF_FORCE_FINAL_REBUILD"] = (
                        "1"
                        if final_eigen_ablation
                        or setup_eigen_ablation
                        or _selection == "baseline"
                        else "0"
                    )
                if final_eigen_ablation or combined_host_ablation:
                    os.environ["VIBEQC_DF_REFERENCE_FINAL_EIGEN"] = (
                        "1" if _selection == "baseline" else "0"
                    )
                if setup_eigen_ablation or combined_host_ablation:
                    os.environ["VIBEQC_DF_REFERENCE_SETUP_EIGEN"] = (
                        "1" if _selection == "baseline" else "0"
                    )
                result = evaluate()
            finally:
                if (
                    final_eigen_ablation
                    or setup_eigen_ablation
                    or final_state_ablation
                    or combined_host_ablation
                ):
                    os.environ.pop("VIBEQC_DF_FORCE_FINAL_REBUILD", None)
                if setup_eigen_ablation or combined_host_ablation:
                    os.environ.pop("VIBEQC_DF_REFERENCE_SETUP_EIGEN", None)
                if final_eigen_ablation or combined_host_ablation:
                    os.environ.pop("VIBEQC_DF_REFERENCE_FINAL_EIGEN", None)
                if policies:
                    os.environ.pop("VIBEQC_DF_EAGER_CORE_GUESS", None)
                    os.environ.pop("VIBEQC_DF_REBUILD_OVERLAP", None)
                if path is not None:
                    os.environ.pop("VIBEQC_DF_HOST_TRACE")
            if path is not None:
                components = aggregate_host(read_host_trace(path))
                if eager_core_ablation:
                    expected = (
                        batch_size
                        if _selection == "baseline" or workload == "cold-start"
                        else 0
                    )
                    actual = sum(
                        components[key].get("core_guess", {}).get("calls", 0)
                        for key in (
                            "eigensolves_by_reason",
                            "device_eigensolves_by_reason",
                        )
                    )
                    if (
                        actual != expected
                        or components["exclusive_phases"]
                        .get("initial_density", {})
                        .get("calls", 0)
                        != batch_size
                    ):
                        raise RuntimeError(
                            "eager/lazy ablation did not execute its declared core-guess policy"
                        )
                if preparation_ablation:
                    validate_preparation_counts(
                        components,
                        batch_size=batch_size,
                        workload=workload,
                        eager=eager,
                        rebuild=rebuild,
                    )
                if setup_eigen_ablation:
                    validate_preparation_counts(
                        components,
                        batch_size=batch_size,
                        workload=workload,
                        eager=False,
                        rebuild=False,
                    )
                    validate_setup_eigen_counts(
                        components,
                        batch_size=batch_size,
                        workload=workload,
                        reference=_selection == "baseline",
                    )
                    validate_final_eigen_counts(
                        components,
                        batch_size=batch_size,
                        method=case.method,
                        reference=False,
                        strict_final_state=True,
                    )
                if final_eigen_ablation:
                    validate_preparation_counts(
                        components,
                        batch_size=batch_size,
                        workload=workload,
                        eager=False,
                        rebuild=False,
                    )
                    validate_final_eigen_counts(
                        components,
                        batch_size=batch_size,
                        method=case.method,
                        reference=_selection == "baseline",
                        strict_final_state=True,
                    )
                if final_state_ablation or combined_host_ablation:
                    baseline = combined_host_ablation and _selection == "baseline"
                    validate_preparation_counts(
                        components,
                        batch_size=batch_size,
                        workload=workload,
                        eager=baseline,
                        rebuild=baseline,
                    )
                    validate_setup_eigen_counts(
                        components,
                        batch_size=batch_size,
                        workload=workload,
                        reference=baseline,
                        eager=baseline,
                        rebuild=baseline,
                    )
                    validate_final_state_counts(
                        components,
                        batch_size=batch_size,
                        method=case.method,
                        force=_selection == "baseline",
                        reference=baseline,
                        compute_forces=not energy_only,
                    )
                result["host_components"] = {
                    **components,
                    "raw_trace": trace_identity(path),
                }
            return result

        return sample

    def prepare(geometries: typing.Any = None) -> typing.Any:
        return calculator.prepare_batch(
            systems if geometries is None else geometries,
            charges=[case.charge] * batch_size,
            multiplicities=[case.multiplicity] * batch_size,
            warm_start=True,
        )

    def execute(batch: typing.Any, coordinates: typing.Any = None) -> typing.Any:
        result = batch.execute(coordinates, strict=True, properties=properties)
        if any(item.executed_backend != "cuda" for item in result.items):
            raise RuntimeError("a CUDA workload cannot pass with a substituted backend")
        return {
            "energies": result.energies.tolist(),
            "convergence": convergence_payload(result),
            "fock_builds": [item.fock_builds for item in result.items],
            "forces": None
            if energy_only
            else [item.forces.tolist() for item in result.items],
            "metric": [
                d.to_dict() for d in batch.last_density_fitting_metric_diagnostics()
            ],
        }

    def cold() -> typing.Any:
        # Both ownership creation and destruction are within the timed call.
        with prepare() as batch:
            return execute(batch)

    rows = measure_interleaved(
        measured("cold-start", cold),
        synchronize,
        workload="cold-start",
        inputs_hash=input_hash,
        repeats=repeats,
    )
    synchronize()
    setup_start = time.perf_counter()
    batch = prepare()
    try:
        seed = execute(batch)
        batch.set_warm_start_updates(False)
        synchronize()
        setup_seconds = time.perf_counter() - setup_start
        for workload in (
            "unchanged-geometry",
            "energy-only" if energy_only else "energy-plus-force",
        ):
            rows += measure_interleaved(
                measured(workload, lambda: execute(batch)),
                synchronize,
                workload=workload,
                inputs_hash=input_hash,
                repeats=repeats,
            )
        coordinates = [np.array([atom[1] for atom in system]) for system in systems]
        changed = [xyz.copy() for xyz in coordinates]
        changed[-1][-1, 0] += 0.01
        changed_systems = [
            [(atom[0], tuple(xyz)) for atom, xyz in zip(system, positions, strict=True)]
            for system, positions in zip(systems, changed, strict=True)
        ]
        # Rebuild from independent ownership, outside measured replay, to
        # detect a stale geometry cache even when all repeated samples agree.
        with prepare(changed_systems) as changed_batch:
            changed_seed = execute(changed_batch)
        changed_hash = canonical_hash(
            {**inputs, "coordinates": [xyz.tolist() for xyz in changed]}
        )
        rows += measure_interleaved(
            measured("changed-geometry", lambda: execute(batch, changed)),
            synchronize,
            workload="changed-geometry",
            inputs_hash=changed_hash,
            repeats=repeats,
            prepare=lambda _: execute(batch, coordinates),
        )
    finally:
        synchronize()
        start = time.perf_counter()
        batch.close()
        synchronize()
        destruction_seconds = time.perf_counter() - start
    if _source_metadata(library) != source:
        raise RuntimeError("source/library changed during workload measurement")
    if policies or final_eigen_ablation or setup_eigen_ablation or final_state_ablation:
        try:
            validate_ablation_branches(rows)
        except AblationBranchMismatch as error:
            error.evidence.update(
                source=source,
                device=device,
                inputs=inputs,
                inputs_hash=input_hash,
                profiled=trace_directory is not None,
                cold_endpoints={"original": seed, "changed": changed_seed},
                prepared_setup_seconds=setup_seconds,
                prepared_destruction_seconds=destruction_seconds,
                endpoint_integrity="not_checked_after_branch_rejection",
            )
            raise
    # Same-model cold endpoints check cache/replay integrity. They are not an
    # independent scientific oracle or a substitute for the matched #206 gate.
    for row in rows:
        expected = changed_seed if row["workload"] == "changed-geometry" else seed
        if not np.allclose(
            row["diagnostics"]["energies"],
            expected["energies"],
            atol=1e-9,
            rtol=0,
        ):
            raise RuntimeError("cold/replay energy endpoint changed")
        if not energy_only and not np.allclose(
            row["diagnostics"]["forces"], expected["forces"], atol=1e-8, rtol=0
        ):
            raise RuntimeError("cold/replay force endpoint changed")
    return {
        "schema": "vibeqc.issue206.df_host_workloads",
        "version": 1,
        "source": source,
        "device": device,
        "inputs": inputs,
        "inputs_hash": input_hash,
        "cold_endpoints": {"original": seed, "changed": changed_seed},
        "profiled": trace_directory is not None,
        "samples": rows,
        "prepared_setup_seconds": setup_seconds,
        "prepared_destruction_seconds": destruction_seconds,
        "comparison": "eager/rebuilt reference setup and forced reference finalization versus lazy cached device setup and verified final-state selection; both sides retain strict physical-state gates"
        if combined_host_ablation
        else "forced ordinary device final rebuilding versus verified retention with necessary device correction; identical preparation"
        if final_state_ablation
        else "reference versus ordinary device setup eigensolves with identical lazy cached preparation and device finalization"
        if setup_eigen_ablation
        else "reference versus ordinary device final eigensolve with identical lazy cached preparation"
        if final_eigen_ablation
        else f"original eager/rebuilt preparation versus {preparation_ablation} on one native library"
        if preparation_ablation
        else "eager core frame versus lazy warm initialization on one native library"
        if eager_core_ablation
        else "identical native configurations as an ABBA protocol control; no speedup claim",
        "timing_assessment": assess_comparison(rows)
        if (
            policies
            or final_eigen_ablation
            or setup_eigen_ablation
            or final_state_ablation
        )
        and trace_directory is None
        else None,
        "limitations": [
            "This probe records actual host solves; complete device work/traffic requires the separate CUDA/Nsight ledger.",
            "Reported legacy Fock counts can omit finalizer work; actual reference-eigensolve leaves remain complete within traced scopes.",
            "External DF numerical/performance parity remains the existing matched #206 matrix gate.",
        ],
    }
