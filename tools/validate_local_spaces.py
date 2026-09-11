"""Measure native-reference local spaces, full recovery and truncation costs."""

# Source-tree CLI bootstrap for transitive compiler clients.
import sys as _compiler_sys
from pathlib import Path as _CompilerPath

_compiler_sys.path.insert(
    0, str(_CompilerPath(__file__).resolve().parents[1] / "python")
)

import argparse
import ctypes as ct
import json
import os
import platform
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from vibeqc import Calculator
from vibeqc.profiles import canonical_hash, file_hash

from tools.vibeqc_local_cc.localization import localize_occupied
from tools.vibeqc_local_cc.mp2 import build_local_mp2, recover_canonical_amplitudes
from tools.vibeqc_local_cc.spaces import projected_virtual_space
from tools.vibeqc_posthf.df import DFProvider, MetricFactor
from tools.vibeqc_posthf.export import export_rhf
from tools.vibeqc_posthf.fixtures import load_fixture, source_arguments
from tools.vibeqc_posthf.mp2 import restricted_mp2
from tools.vibeqc_posthf.providers import ConventionalProvider
from tools.vibeqc_posthf.sources import NativeSource


def run(output, names=("h2", "water", "lih", "f_heh")):
    """Archive complete setup costs and rejected states without changing defaults."""
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    rows, states = [], {}
    for name in names:
        metadata, _ = load_fixture(name)
        with NativeSource(**source_arguments(metadata)) as source:
            ao_atoms = tuple(
                sh.atom_index
                for sh in source.shells
                for _ in range(
                    (sh.angular_momentum + 1) * (sh.angular_momentum + 2) // 2
                    if source.representation == "cartesian"
                    else 2 * sh.angular_momentum + 1
                )
            )
            for label in ("conventional", "df"):
                case_started = time.perf_counter()
                case = {
                    "name": name,
                    "hamiltonian": label,
                    "reference_array_hash": metadata["array_hash"],
                    "rows": [],
                }
                try:
                    metric = MetricFactor.from_source(source) if label == "df" else None
                    snapshot, export = export_rhf(
                        source,
                        metric=metric,
                        generation_id=f"local-spaces-182-{name}-{label}",
                    )
                    local = localize_occupied(snapshot, ao_atoms)
                    domain_started = time.perf_counter()
                    domain = projected_virtual_space(snapshot)
                    domain_seconds = time.perf_counter() - domain_started
                    provider = (
                        ConventionalProvider(snapshot, source)
                        if metric is None
                        else DFProvider(snapshot, source, metric)
                    )
                    reference_started = time.perf_counter()
                    with provider:
                        canonical = restricted_mp2(snapshot, provider)
                        canonical_provider = dict(provider.statistics)
                    reference_seconds = time.perf_counter() - reference_started
                    independent_error = abs(
                        canonical.correlation_energy
                        - metadata["records"][label]["correlation_energy"]
                    )
                    hf_error = abs(
                        snapshot.reference_energy
                        - metadata["records"][label]["hf_energy"]
                    )
                    if max(independent_error, hf_error) > 1e-9:
                        raise RuntimeError(
                            "native reference failed independent HF/MP2 energy gates"
                        )
                    case.update(
                        status="pass",
                        snapshot_id=snapshot.identity,
                        hamiltonian_id=snapshot.hamiltonian_id,
                        export=export,
                        native_hf_reference_error=hf_error,
                        native_mp2_reference_error=independent_error,
                        canonical_correlation_energy=canonical.correlation_energy,
                        canonical_provider=canonical_provider,
                        canonical_mp2_seconds=reference_seconds,
                        localization={
                            "identity": local.identity,
                            "objective": local.objective,
                            "initial_objective": local.initial_objective,
                            "gradient_max": local.gradient_max,
                            "sweeps": local.sweeps,
                            "seconds": local.seconds,
                            "numeric_peak_bytes": local.numeric_peak_bytes,
                            "occupied_fock_offdiagonal_max": float(
                                np.max(
                                    np.abs(
                                        local.occupied_fock
                                        - np.diag(np.diag(local.occupied_fock))
                                    )
                                )
                            ),
                        },
                        projected_virtual={
                            "rank": domain.rank,
                            "input_columns": len(domain.source_ao_indices),
                            "gram_eigenvalues": domain.gram_eigenvalues.tolist(),
                            "cutoff": domain.absolute_cutoff,
                            "rank_crossing": domain.rank_crossing,
                            "seconds": domain_seconds,
                        },
                    )
                    prefix = f"{name}__{label}"
                    states[prefix + "__occupied_rotation"] = local.rotation
                    states[prefix + "__occupied_fock"] = local.occupied_fock
                    states[prefix + "__virtual_domain"] = domain.columns
                    for budget in (9 << 20, 128 << 20):
                        for threshold in (0.0, 1e-6, 1e-4, 1e-2):
                            row = {
                                "budget_bytes": budget,
                                "occupation_threshold": threshold,
                                "axis_tile": 2,
                                "keep_full_space": threshold == 0.0,
                                "cluster_tolerance": 1e-12,
                            }
                            try:
                                result = build_local_mp2(
                                    snapshot,
                                    source,
                                    local,
                                    domain,
                                    metric=metric,
                                    occupation_threshold=threshold,
                                    keep_full_space=threshold == 0.0,
                                    budget_bytes=budget,
                                )
                                row.update(
                                    status="pass",
                                    correlation_energy=result.correlation_energy,
                                    observed_energy_difference=result.observed_energy_difference,
                                    full_space_recovery=result.full_space_recovery,
                                    minimum_absolute_denominator=result.minimum_absolute_denominator,
                                    plan=asdict(result.plan),
                                    provider_peak_bytes=result.provider_peak_bytes,
                                    retained_numeric_bytes=result.retained_numeric_bytes,
                                    provider_seconds=result.provider_seconds,
                                    pair_transform_seconds=result.pair_transform_seconds,
                                    total_seconds=result.total_seconds,
                                    total_with_reference_export_and_space_setup_seconds=result.total_seconds
                                    + export["export_seconds"]
                                    + local.seconds
                                    + domain_seconds,
                                    derivative_status=result.derivative_status,
                                    pairs=[
                                        {
                                            "pair": p.space.pair,
                                            "rank": p.space.rank,
                                            "occupation_eigenvalues": p.space.occupation_eigenvalues.tolist(),
                                            "retained_indices": p.space.retained_indices,
                                            "discarded_weight": p.space.discarded_weight,
                                            "rank_crossing": p.space.rank_crossing,
                                            "energy": p.energy,
                                            "full_virtual_pair_energy": p.full_virtual_pair_energy,
                                            "projector_identity": p.space.identity,
                                            "gauge_identity": p.space.gauge_identity,
                                        }
                                        for p in result.pairs
                                    ],
                                )
                                if result.full_space_recovery:
                                    recovered = recover_canonical_amplitudes(
                                        snapshot, local, result
                                    )
                                    row["amplitude_recovery_max"] = float(
                                        np.max(np.abs(recovered - canonical.amplitudes))
                                    )
                                    if row["amplitude_recovery_max"] > 1e-11:
                                        row["status"] = "failed_amplitude_gate"
                                for p in result.pairs:
                                    key = (
                                        prefix
                                        + f"__budget_{budget}__threshold_{threshold:g}__pair_{p.space.pair[0]}_{p.space.pair[1]}"
                                    )
                                    states[key + "__columns"] = p.space.columns
                                    states[key + "__amplitudes"] = p.amplitudes
                                    states[key + "__integrals"] = p.integrals
                            except (ValueError, RuntimeError, MemoryError) as error:
                                row.update(status="failed", reason=str(error))
                            case["rows"].append(row)
                    case["case_seconds"] = time.perf_counter() - case_started
                except (ValueError, RuntimeError, MemoryError) as error:
                    case.update(
                        status="failed_reference_or_localization", reason=str(error)
                    )
                rows.append(case)
                print(
                    json.dumps(
                        {"case": name, "hamiltonian": label, "status": case["status"]}
                    ),
                    flush=True,
                )
    archive = output / "states.npz"
    np.savez_compressed(archive, **states)
    library = Calculator()._library
    library.vibeqc_get_source_identity.restype = ct.c_char_p
    result = {
        "schema": "vibeqc.local_space_experiment",
        "version": 1,
        "cases": rows,
        "source_identity": library.vibeqc_get_source_identity().decode(),
        "runner_sha256": file_hash(Path(__file__)),
        "module_sha256": {
            str(path.relative_to(Path(__file__).parent)): file_hash(path)
            for package in ("vibeqc_local_cc", "vibeqc_posthf")
            for path in sorted((Path(__file__).parent / package).glob("*.py"))
        },
        "python": platform.python_version(),
        "numpy": np.__version__,
        "thread_policy": {
            k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
        "total_seconds": time.perf_counter() - started,
        "states": {
            "file": archive.name,
            "sha256": file_hash(archive),
            "bytes": archive.stat().st_size,
            "array_shapes": {k: list(v.shape) for k, v in states.items()},
        },
        "boundaries": {
            "method": "canonical MP2 projected into localized occupied/PNO pair spaces; no local CC equations",
            "derivatives": "unsupported: localization, projected domains and rank response are not implemented",
            "screening": "all occupied pairs; no pair-distance screening",
            "error": "actual total differences; discarded pair occupations are not independent energy bounds",
            "memory": "numeric arrays plus bounded provider reservations; interpreter and library overhead excluded",
            "backend": "native CPU raw tiles with NumPy transforms; no GPU acceleration claim",
        },
    }
    result["record_hash"] = canonical_hash(result)
    (output / "report.json").write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cases", nargs="+", default=["h2", "water", "lih", "f_heh"])
    arguments = parser.parse_args()
    result = run(arguments.output, arguments.cases)
    if any(
        c["status"] != "pass" or any(r["status"] != "pass" for r in c["rows"])
        for c in result["cases"]
    ):
        raise SystemExit("local-space validation failed; see retained report")
