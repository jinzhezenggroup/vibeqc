"""Build the issue-41 current-head component ledger from Nsight summaries.

The benchmark endpoint records synchronized host intervals and convergence,
while ``nsys stats --report nvtx_kern_sum`` records device kernel time inside
the same named warm ranges.  This tool joins those independent sources with
the exact production shell-class task ledger.  It deliberately labels the
host-interval remainder as unattributed rather than pretending that kernel
summaries cover CUDA API waits, copies, Python/C++ work, or idle gaps.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

VIBEQC_RANGE = ":vibeqc/warm/energy-plus-force"
GPU4PYSCF_SCF_RANGE = ":gpu4pyscf/warm/scf"
GPU4PYSCF_FORCE_RANGE = ":gpu4pyscf/warm/force"

_GENERATED_FORCE = re.compile(r"generated_sm\d+_([spdfg]+)_shell_class_force_")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _median(values: Iterable[float]) -> float:
    materialized = tuple(values)
    if not materialized:
        raise ValueError("cannot take the median of an empty sample set")
    return float(statistics.median(materialized))


def _component(name: str) -> str:
    """Classify one VibeQC kernel without relying on launch order."""

    if "one_electron_force" in name:
        return "one_electron_force"
    if (
        "shell_class_force" in name
        or "two_electron_force" in name
        or "ppps_resident_bra_force" in name
    ):
        return "two_electron_force"
    if any(
        marker in name
        for marker in (
            "compact_active_shell_quartet",
            "compact_generic_order5",
            "generated_shell_task",
            "ppps_resident",
            "low_order_signature",
            "shell_pair_density_bound",
            "select_final_fock_rebuild",
        )
    ):
        return "screening_and_queue_preparation"
    if "fock" in name.lower():
        return "direct_jk_and_fock_transforms"
    if any(
        marker in name
        for marker in (
            "sytrd",
            "laed",
            "steqr",
            "lansy",
            "larft",
            "setup_vhat",
            "merge_ker",
            "lacpy",
            "lascl",
            "zero_lower",
            "copy_info",
            "xx_set_info",
            "scale_max",
        )
    ):
        return "diagonalization"
    if "cutlass::" in name or "syherk_" in name or "cuds_" in name:
        return "matrix_and_diis_transforms"
    if any(
        marker in name
        for marker in ("compute_energy", "nuclear_force", "weighted_density")
    ):
        return "energy_and_other_force"
    return "other_device_kernels"


def _shell_class(name: str) -> str | None:
    generated = _GENERATED_FORCE.search(name)
    if generated is not None:
        return generated.group(1)
    if "ppps_resident_bra_force" in name:
        return "ppps"
    if "two_electron_force_psss_resident_bra" in name:
        return "psss"
    return None


def _kernel_ledger(
    path: Path,
) -> tuple[int, dict[str, float], dict[str, float], list[dict[str, Any]]]:
    component_ns: defaultdict[str, float] = defaultdict(float)
    shell_class_ns: defaultdict[str, float] = defaultdict(float)
    retained: list[tuple[float, int, str, str]] = []
    replay_counts: set[int] = set()
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["NVTX Range"] != VIBEQC_RANGE:
                continue
            replay_count = int(row["NVTX Inst"])
            if replay_count <= 0:
                raise ValueError("VibeQC kernel row has no containing NVTX range")
            replay_counts.add(replay_count)
            total_ns = float(row["Total Time (ns)"])
            name = row["Kernel Name"]
            component = _component(name)
            component_ns[component] += total_ns
            shell_class = _shell_class(name)
            if shell_class is not None:
                shell_class_ns[shell_class] += total_ns
            retained.append((total_ns, int(row["Kern Inst"]), component, name))
    if len(replay_counts) != 1:
        raise ValueError(
            f"expected one VibeQC replay count, found {sorted(replay_counts)}"
        )
    replay_count = replay_counts.pop()

    def milliseconds(total_ns: float) -> float:
        return total_ns / replay_count / 1.0e6

    components = {
        name: milliseconds(total_ns) for name, total_ns in sorted(component_ns.items())
    }
    shell_classes = {
        name: milliseconds(total_ns)
        for name, total_ns in sorted(shell_class_ns.items())
    }
    top_kernels = [
        {
            "component": component,
            "kernel": name,
            "kernel_time_milliseconds_per_replay": milliseconds(total_ns),
            "launches_per_replay": launches // replay_count,
        }
        for total_ns, launches, component, name in sorted(retained, reverse=True)[:30]
    ]
    return replay_count, components, shell_classes, top_kernels


def _range_medians(path: Path) -> dict[str, float]:
    medians: dict[str, float] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            medians[row["Range"]] = float(row["Med (ns)"]) / 1.0e6
    return medians


def _gpu4pyscf_component_medians(endpoint: dict[str, Any]) -> dict[str, float]:
    samples = endpoint["gpu4pyscf"]["warm_samples"]
    return {
        "scf": 1.0e3
        * _median(float(sample["component_seconds"]["scf"]) for sample in samples),
        "force": 1.0e3
        * _median(float(sample["component_seconds"]["force"]) for sample in samples),
    }


def _shell_class_ledger(
    profile: dict[str, Any], device_milliseconds: dict[str, float]
) -> list[dict[str, Any]]:
    rows = []
    for source in profile["shell_classes"]:
        shell_class = str(source["class"])
        rows.append(
            {
                **source,
                "force_kernel_milliseconds_per_replay": device_milliseconds.get(
                    shell_class, 0.0
                ),
            }
        )
    return sorted(
        rows,
        key=lambda row: row["force_kernel_milliseconds_per_replay"],
        reverse=True,
    )


def build_ledger(
    endpoint: dict[str, Any],
    kernel_summary_path: Path,
    nvtx_summary_path: Path,
    shell_profile: dict[str, Any],
) -> dict[str, Any]:
    replay_count, components, shell_device_ms, top_kernels = _kernel_ledger(
        kernel_summary_path
    )
    range_medians = _range_medians(nvtx_summary_path)
    profiled_vibeqc_ms = range_medians[VIBEQC_RANGE]
    device_total_ms = sum(components.values())
    raw_unattributed_ms = profiled_vibeqc_ms - device_total_ms
    # Nsight projects GPU timestamps into the host NVTX range. Small clock
    # conversion and graph-node attribution differences can make summed
    # kernel durations slightly exceed the enclosing host interval; never
    # publish that measurement artifact as a negative residual.
    unattributed_ms = max(raw_unattributed_ms, 0.0)
    projection_excess_ms = max(-raw_unattributed_ms, 0.0)
    if components.get("direct_jk_and_fock_transforms", 0.0) > 0.0:
        fock_rebuild_observation = (
            "CUDA Graph node tracing exposed the captured direct-J/K Fock "
            "kernels, so their device time is reported explicitly instead "
            "of being folded into the host/API/synchronization remainder."
        )
    else:
        fock_rebuild_observation = (
            "Only the final-Fock-rebuild selector was visible in the "
            "captured one-iteration warm range. CUDA Graph node tracing was "
            "not available, so graph-contained Fock work remains in the "
            "unattributed remainder."
        )
    timing = endpoint["timing_summary"]["iteration_matched"]
    vibeqc_seconds = float(timing["vibeqc_median_seconds"])
    gpu4pyscf_seconds = float(timing["gpu4pyscf_median_seconds"])
    issue_baseline_vibeqc_seconds = 3.181492
    issue_baseline_gpu4pyscf_seconds = 2.143182
    return {
        "benchmark": "issue_41_current_head_component_ledger",
        "schema_version": 1,
        "environment": endpoint["environment"],
        "workload": {
            key: endpoint["workload"][key]
            for key in (
                "ao_count",
                "basis_representation",
                "batch_size",
                "case",
                "density_tolerance",
                "direct_scf_tolerance",
                "energy_tolerance",
                "method",
            )
        },
        "protocol": {
            "measurement_order": endpoint["settings"]["measurement_order"],
            "repeats_per_engine": endpoint["settings"]["repeats_per_engine"],
            "iteration_branch": timing["iteration_branch"],
            "profiled_replays": replay_count,
            "profiled_nvtx_range": VIBEQC_RANGE.removeprefix(":"),
            "warning": (
                "Nsight component values are average device kernel time per "
                "replay. Headline endpoint medians come from the separate "
                "unprofiled interleaved run."
            ),
        },
        "headline": {
            "vibeqc_median_seconds": vibeqc_seconds,
            "gpu4pyscf_median_seconds": gpu4pyscf_seconds,
            "gap_seconds": vibeqc_seconds - gpu4pyscf_seconds,
            "vibeqc_over_gpu4pyscf": vibeqc_seconds / gpu4pyscf_seconds,
            "maximum_energy_error_hartree": endpoint["accuracy"]["gate_selection"][
                "maximum_energy_error_hartree"
            ],
            "maximum_force_error_hartree_per_bohr": endpoint["accuracy"][
                "gate_selection"
            ]["maximum_force_error_hartree_per_bohr"],
            "gate_passed": endpoint["gate"]["passed"],
        },
        "change_from_issue_baseline": {
            "issue_vibeqc_seconds": issue_baseline_vibeqc_seconds,
            "issue_gpu4pyscf_seconds": issue_baseline_gpu4pyscf_seconds,
            "issue_gap_seconds": (
                issue_baseline_vibeqc_seconds - issue_baseline_gpu4pyscf_seconds
            ),
            "vibeqc_endpoint_saving_seconds": (
                issue_baseline_vibeqc_seconds - vibeqc_seconds
            ),
            "gap_reduction_seconds": (
                issue_baseline_vibeqc_seconds
                - issue_baseline_gpu4pyscf_seconds
                - (vibeqc_seconds - gpu4pyscf_seconds)
            ),
        },
        "component_ledger": {
            "vibeqc_profiled_host_interval_milliseconds": profiled_vibeqc_ms,
            "vibeqc_device_kernel_milliseconds": device_total_ms,
            "vibeqc_host_api_sync_and_idle_unattributed_milliseconds": unattributed_ms,
            "vibeqc_device_projection_excess_milliseconds": projection_excess_ms,
            "vibeqc_device_components_milliseconds": components,
            "gpu4pyscf_unprofiled_host_component_medians_milliseconds": _gpu4pyscf_component_medians(
                endpoint
            ),
            "gpu4pyscf_profiled_host_ranges_milliseconds": {
                "scf": range_medians[GPU4PYSCF_SCF_RANGE],
                "force": range_medians[GPU4PYSCF_FORCE_RANGE],
            },
            "fock_rebuild_observation": fock_rebuild_observation,
        },
        "shell_classes": _shell_class_ledger(shell_profile, shell_device_ms),
        "top_kernels": top_kernels,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="join issue-41 endpoint, Nsight, and shell-class ledgers"
    )
    parser.add_argument("--endpoint", type=Path, required=True)
    parser.add_argument("--nvtx-kernel-summary", type=Path, required=True)
    parser.add_argument("--nvtx-summary", type=Path, required=True)
    parser.add_argument("--shell-profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    payload = build_ledger(
        _load_json(arguments.endpoint),
        arguments.nvtx_kernel_summary,
        arguments.nvtx_summary,
        _load_json(arguments.shell_profile),
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"JSON result: {arguments.output}")


if __name__ == "__main__":
    main()
