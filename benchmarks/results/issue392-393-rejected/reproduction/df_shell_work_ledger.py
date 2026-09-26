"""Reduce intrusive shell diagnostics and validate their independent host domain.

Run after a VIBEQC_DF_SHELL_WORK=1 component capture. Counts describe emitted
source operations, not hardware instructions or elapsed-time percentages.
The supported headline cases share orbital/auxiliary bases; general sparse
counter correctness is tested by the native shell-pair oracle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

from vibeqc import Atom
from vibeqc.calculator import _named_basis_shells
from vibeqc_compiler.integral.df_shell_derivatives import (
    shell_schedule,
    shell_work_model,
)

from benchmarks._cases import benchmark_cases
from benchmarks.df_component_ledger import read_trace
from benchmarks.df_policy_endpoint import CASES

RYS_PROTOTYPE = {
    (0, 0, 0),
    (0, 0, 1),
    (0, 0, 2),
    (1, 0, 0),
    (1, 0, 1),
    (1, 1, 0),
    (2, 0, 0),
}

# Required source-operation fields form the diagnostic report contract. Zero
# counts must still be present so a truncated capture cannot look like no work.
WORK_FIELDS = (
    "shell_tasks",
    "active_shell_tasks",
    "primitive_products",
    "geometry_preparations",
    "boys_evaluations",
    "boys_order_sum",
    "boys_series_iterations",
    "boys_series",
    "boys_small_argument",
    "boys_large_argument",
    "axis_polynomial_calls",
    "specialized_prepare_axis_calls",
    "cache_coefficient_values",
    "convolution_iterations",
    "active_component_products",
    "public_weight_loads",
    "public_nonzero_weights",
    "expansion_term_products",
    "folding_shared_atomics",
    "folding_direct_stores",
    "gradient_atomics_a",
    "gradient_atomics_b",
    "gradient_atomics_c",
    "gradient_atomics_shared_atom",
    "gradient_atomics_distinct_atom",
    "subgroup_rendezvous",
    "orbital_local_accumulations",
)


def reconstruct_domain(shells, panels, pair_mode):
    """Count implicit signature products using only public host shell metadata.

    Each shell row is (angular, primitives, public AO offset, public AO count).
    A panel may split an auxiliary shell; each intersection is an executed
    visit. Signature ordering is angular then primitive count, retaining
    triangles only when the two orbital groups are identical.
    """
    if pair_mode not in (0, 1, 2):
        raise ValueError("unknown public pair mode")
    groups = Counter((row[0], row[1]) for row in shells)
    result = Counter()
    for first, nfirst in sorted(groups.items()):
        for second, nsecond in sorted(groups.items()):
            if pair_mode and first < second:
                continue
            pairs = (
                nfirst * (nfirst + 1) // 2
                if pair_mode and first == second
                else nfirst * nsecond
            )
            for begin, count, repetitions in panels:
                third_groups = Counter(
                    (angular, primitives)
                    for angular, primitives, offset, width in shells
                    if offset < begin + count and offset + width > begin
                )
                for third, nthird in third_groups.items():
                    signature = (
                        first[0],
                        second[0],
                        third[0],
                        first[1],
                        second[1],
                        third[1],
                    )
                    result[signature] += pairs * nthird * repetitions
    return result


def kernel_activity(database_path, record):
    """Read actual Nsight durations, requiring the traced class launch domain.

    A signature packet shares one duration among its slices. No per-signature
    timing is inferred from work counts, and CUDA event intervals stay separate
    because they may include host gaps between launches.
    """
    expected = Counter()
    for region in record["regions"]:
        match = re.fullmatch(
            r"shell_([0-3])([0-3])([0-3])_(?:packet|p\d+_\d+_\d+)", region["name"]
        )
        if match:
            expected[tuple(map(int, match.groups()))] += 1
    observed, durations, resources = Counter(), Counter(), defaultdict(set)
    schedules = {}
    with sqlite3.connect(database_path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        rows = db.execute(
            "SELECT s.value, k.end-k.start, k.registersPerThread, "
            "k.staticSharedMemory, k.dynamicSharedMemory, k.localMemoryPerThread, "
            "k.blockX, k.blockY, k.blockZ "
            "FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id "
            "WHERE s.value LIKE '%::shell_panel<%' OR s.value LIKE '%::shell_packet<%'"
        )
        for name, ns, registers, shared, dynamic, local, bx, by, bz in rows:
            match = re.search(r"::shell_(?:panel|packet)<([^>]+)>", name)
            if not match or ns < 0:
                raise ValueError("invalid shell kernel activity")
            template = tuple(
                int(value.replace("(unsigned int)", "").strip().removesuffix("u"))
                for value in match[1].split(",")[:4]
            )
            angular, variant = template[:3], template[3]
            schedule = shell_schedule(angular, variant)
            # Reconstruct either candidate generation: scalar schedules named
            # these triples, while C-grouping schedules named them subgroups.
            subgroups = getattr(schedule, "subgroups_per_block", None)
            if subgroups is None:
                subgroups = schedule.triples_per_block
            if bx * by * bz != schedule.component_lanes * subgroups:
                raise ValueError(
                    "Nsight block size disagrees with the generated schedule"
                )
            # Old captures precede aggregation. Read the executed limit from
            # the trace; the current compiler policy cannot describe old binaries.
            c_group = record["counters"].get(
                f"shell_{''.join(map(str, angular))}_auxiliary_group_limit", 1
            )
            if type(c_group) is not int or c_group < 1:
                raise ValueError("invalid auxiliary owner grouping limit")
            parallel = record["counters"].get(
                f"shell_{''.join(map(str, angular))}_auxiliary_parallel", 0
            )
            if parallel not in (0, 1) or (parallel and subgroups % c_group):
                raise ValueError("invalid parallel auxiliary owner layout")
            warp = record["counters"].get(
                f"shell_{''.join(map(str, angular))}_auxiliary_warp", 0
            )
            owner_lanes = schedule.component_lanes * c_group
            if warp not in (0, 1) or (
                warp and (not parallel or owner_lanes > 32 or 32 % owner_lanes)
            ):
                raise ValueError("invalid warp auxiliary owner layout")
            owners = subgroups // (c_group if parallel else 1)
            current = {
                "variant": variant,
                "component_lanes": schedule.component_lanes,
                "owner_groups_per_block": owners,
                "maximum_shell_tasks_per_block": owners * c_group,
                "maximum_auxiliary_shells_per_owner": c_group,
            }
            if parallel:
                current["auxiliary_shells_run_in_parallel"] = True
            if warp:
                current["owner_rendezvous"] = "warp"
            if schedules.setdefault(angular, current) != current:
                raise ValueError(
                    "a class mixes schedules within one force-call capture"
                )
            observed[angular] += 1
            durations[angular] += ns / 1e6
            resources[angular].add((registers, shared, dynamic, local, bx * by * bz))
    if not expected or observed != expected:
        raise ValueError("Nsight class launch counts differ from the diagnostic trace")
    return {
        angular: {
            "gpu_ms": durations[angular],
            "launches": observed[angular],
            "schedule": schedules[angular],
            "resources": [
                dict(
                    zip(
                        (
                            "registers_per_thread",
                            "static_shared_bytes",
                            "dynamic_shared_bytes",
                            "local_bytes_per_thread",
                            "block_threads",
                        ),
                        values,
                        strict=True,
                    )
                )
                for values in sorted(resources[angular])
            ],
        }
        for angular in sorted(observed)
    }


def reduce_work(record, shells):
    """Require complete class/signature counters and conserved primitive work."""
    counters = record["counters"]
    if counters.get("shell_work_diagnostics_enabled") != 1:
        raise ValueError("requires one complete detailed shell diagnostic")
    panels = []
    signatures, classes = defaultdict(dict), defaultdict(dict)
    for name, value in counters.items():
        match = re.fullmatch(r"shell_work_panel_(\d+)_(\d+)", name)
        if match:
            panels.append((*map(int, match.groups()), value))
        match = re.fullmatch(
            r"shell_([0-3])([0-3])([0-3])_p(\d+)_(\d+)_(\d+)_work_(.+)", name
        )
        if match:
            signatures[tuple(map(int, match.groups()[:6]))][match[7]] = value
        match = re.fullmatch(r"shell_([0-3])([0-3])([0-3])_work_(.+)", name)
        if match:
            classes[tuple(map(int, match.groups()[:3]))][match[4]] = value
    if not panels or not signatures:
        raise ValueError("missing panel/signature diagnostics")
    expected = reconstruct_domain(shells, panels, counters["shell_work_pair_mode"])
    if set(expected) != set(signatures):
        raise ValueError("device signature domain differs from the host reconstruction")
    if set(classes) != {signature[:3] for signature in signatures}:
        raise ValueError("device class domain differs from its signatures")
    # Older captures accounted for every expansion product with an atomic or
    # direct store. The new register remainder may be absent only when it is
    # zero; conservation below rejects unexplained work in a truncated capture.
    for rows in (signatures, classes):
        for values in rows.values():
            values.setdefault("folding_register_values", 0)
    totals = Counter()
    signature_sums = defaultdict(Counter)
    for signature, values in signatures.items():
        if not set(WORK_FIELDS).issubset(values):
            raise ValueError(f"incomplete signature work counters: {signature}")
        if any(type(value) is not int or value < 0 for value in values.values()):
            raise ValueError("work counters must be nonnegative integers")
        if values["expansion_term_products"] != sum(
            values[field]
            for field in (
                "folding_shared_atomics",
                "folding_direct_stores",
                "folding_register_values",
            )
        ):
            raise ValueError("weight folding does not conserve expansion products")
        tasks = expected[signature]
        if values["shell_tasks"] != tasks:
            raise ValueError(f"host/device shell count differs: {signature}")
        active = values["active_shell_tasks"]
        if active > tasks:
            raise ValueError("active shell count exceeds visited shells")
        primitives = active * math.prod(signature[3:])
        if primitives != values["primitive_products"]:
            raise ValueError(f"host/device primitive count differs: {signature}")
        if not (
            values["geometry_preparations"] == values["boys_evaluations"] == primitives
        ):
            raise ValueError("geometry/Boys counts differ from active primitive work")
        if values["boys_order_sum"] != primitives * (sum(signature[:3]) + 1):
            raise ValueError("requested Boys order disagrees with angular class")
        if (
            values["boys_evaluations"]
            != values["boys_series"] + values["boys_large_argument"]
        ):
            raise ValueError("Boys branch counts do not conserve evaluations")
        if values["boys_small_argument"] > values["boys_series"]:
            raise ValueError("small-argument subdomain exceeds its series branch")
        if values["boys_series_iterations"] < values["boys_series"]:
            raise ValueError("series iterations cannot be less than series evaluations")
        if (
            values["gradient_atomics_a"]
            + values["gradient_atomics_b"]
            + values["gradient_atomics_c"]
            != values["gradient_atomics_shared_atom"]
            + values["gradient_atomics_distinct_atom"]
        ):
            raise ValueError("gradient center/physical-atom counts disagree")
        signature_sums[signature[:3]].update(values)
    for angular, values in sorted(classes.items()):
        if dict(signature_sums[angular]) != values:
            raise ValueError(f"signature/class counts disagree: {angular}")
        model = shell_work_model(angular)
        primitives = values["primitive_products"]
        for field in (
            "axis_polynomial_calls",
            "specialized_prepare_axis_calls",
            "cache_coefficient_values",
        ):
            if values[field] != model[field] * primitives:
                raise ValueError(f"generated work model disagrees: {angular} {field}")
        totals.update(values)
    for detailed, existing in (
        ("shell_tasks", "shell_triples_visited"),
        ("active_shell_tasks", "shell_triples_nonzero"),
        ("primitive_products", "shell_primitive_products"),
        ("public_weight_loads", "shell_public_weights_consumed"),
        ("public_nonzero_weights", "shell_public_weights_nonzero"),
        ("active_component_products", "shell_cartesian_component_products"),
    ):
        if totals[detailed] != counters[existing]:
            raise ValueError(f"detailed/existing counter mismatch: {detailed}")
    gpu = Counter()
    for region in record["regions"]:
        match = re.fullmatch(
            r"shell_([0-3])([0-3])([0-3])_(?:packet|p\d+_\d+_\d+)", region["name"]
        )
        if match:
            gpu[tuple(map(int, match.groups()))] += region["gpu_ms"]
    rows = [
        {
            "angular": angular,
            "work": values,
            "maximum_boys_order": sum(angular) + 1
            if values["boys_evaluations"]
            else None,
            "component_gpu_inclusive_ms": gpu[angular],
            "resources": {
                name.removeprefix(f"shell_{''.join(map(str, angular))}_"): value
                for name, value in counters.items()
                if name.startswith(f"shell_{''.join(map(str, angular))}_")
                and any(
                    name.endswith(suffix)
                    for suffix in (
                        "registers",
                        "shared_bytes",
                        "thread_limit",
                        "block_limit",
                        "group_limit",
                        "auxiliary_parallel",
                        "auxiliary_warp",
                    )
                )
            },
            "signatures": [
                {"primitives": key[3:], "work": work}
                for key, work in sorted(signatures.items())
                if key[:3] == angular
            ],
        }
        for angular, values in sorted(classes.items())
    ]
    grouped = {}
    for label, predicate in (
        ("pure_sp", lambda a: max(a) <= 1),
        ("d_containing", lambda a: max(a) == 2),
        ("rys_prototype", lambda a: a in RYS_PROTOTYPE),
    ):
        selected = [row for row in rows if predicate(tuple(row["angular"]))]
        work = Counter()
        for row in selected:
            work.update(row["work"])
        grouped[label] = {
            "work": dict(work),
            "component_gpu_inclusive_ms": sum(
                row["component_gpu_inclusive_ms"] for row in selected
            ),
        }
    reconstruction = {
        "shells": shells,
        "panels": sorted(panels),
        "signature_tasks": [
            {"signature": key, "tasks": value}
            for key, value in sorted(expected.items())
        ],
    }
    return {
        "classes": rows,
        "totals": dict(totals),
        "groups": grouped,
        "host_reconstruction": reconstruction,
        "host_reconstruction_sha256": hashlib.sha256(
            json.dumps(reconstruction, sort_keys=True).encode()
        ).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--measurement", type=Path, required=True)
    parser.add_argument("--generated-header", type=Path, required=True)
    parser.add_argument(
        "--nsys",
        type=Path,
        help="SQLite export of the same measured force-call capture",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    measurement = json.loads(args.measurement.read_text())
    records = [
        row
        for row in read_trace(args.trace)
        if row["operation"] == "force_response" and row["execution"] == "stream"
    ]
    if len(records) != 1:
        raise ValueError("select one complete force-call trace")
    aos = records[0]["nbf"]
    case = benchmark_cases()[CASES[aos]]
    metadata = _named_basis_shells(
        case.vibeqc_basis, [Atom.from_value(atom) for atom in case.atoms]
    )
    shells, offset = [], 0
    for shell in metadata:
        angular = shell.angular_momentum
        width = (
            2 * angular + 1
            if case.basis_representation == "spherical"
            else (angular + 1) * (angular + 2) // 2
        )
        shells.append((angular, len(shell.primitives), offset, width))
        offset += width
    if offset != aos or measurement["aos"] != aos or records[0]["naux"] != aos:
        raise ValueError("headline basis/model shape mismatch")
    result = reduce_work(records[0], shells)
    if args.nsys:
        activities = kernel_activity(args.nsys, records[0])
        for row in result["classes"]:
            row["nsys"] = activities[tuple(row["angular"])]
        for label, predicate in (
            ("pure_sp", lambda a: max(a) <= 1),
            ("d_containing", lambda a: max(a) == 2),
            ("rys_prototype", lambda a: a in RYS_PROTOTYPE),
        ):
            result["groups"][label]["nsys_gpu_ms"] = sum(
                row["gpu_ms"] for a, row in activities.items() if predicate(a)
            )
        result["nsys_sqlite_sha256"] = hashlib.sha256(
            args.nsys.read_bytes()
        ).hexdigest()
    result.update(
        {
            "scope": "Intrusive source-operation ledger; CUDA event intervals can include stream idle time. Counts are not hardware instructions, DRAM transactions, or time fractions.",
            "aos": aos,
            "trace_sha256": hashlib.sha256(args.trace.read_bytes()).hexdigest(),
            "measurement_sha256": hashlib.sha256(
                args.measurement.read_bytes()
            ).hexdigest(),
            "generated_schedule_sha256": hashlib.sha256(
                args.generated_header.read_bytes()
            ).hexdigest(),
            "native_source_identity": measurement["native_source_identity"],
            "library_sha256": measurement["library_sha256"],
            "source_patch_sha256": measurement["source_patch_sha256"],
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
