"""Reduce intrusive shell diagnostics and validate their independent host domain.

Run after a VIBEQC_DF_SHELL_WORK=1 component capture. Counts describe emitted
source operations, not hardware instructions or elapsed-time percentages.
Explicit unequal auxiliary bases retain separate orbital/auxiliary shell
domains; sparse counter correctness is tested by the native shell-pair oracle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sqlite3
import typing
from collections import Counter, defaultdict
from pathlib import Path

try:
    from benchmarks._retention import raw_output_path
except ModuleNotFoundError:
    from _retention import raw_output_path
from vibeqc import Atom
from vibeqc.calculator import _named_basis_shells
from vibeqc_compiler.integral.df_rys_shell import shell_rys_work_model
from vibeqc_compiler.integral.df_shell_derivatives import (
    shell_schedule,
    shell_work_model,
)

from benchmarks._cases import benchmark_cases
from benchmarks.compare_gpu4pyscf_batch import load_comparison_basis
from benchmarks.df_component_ledger import read_trace
from benchmarks.df_policy_endpoint import CASES

# Historical seven-class coverage grouping from #394. This is not an
# availability mask: availability is owned by the compiler, not this report.
RYS_PROTOTYPE = {
    (0, 0, 0),
    (0, 0, 1),
    (0, 0, 2),
    (1, 0, 0),
    (1, 0, 1),
    (1, 1, 0),
    (2, 0, 0),
}

# #437 screening observability. Distance bins are in Bohr; exponent bins are
# primitive Gaussian exponents in Bohr^-2. These are descriptive evidence, not
# force-screening thresholds.
DISTANCE_EDGES_BOHR = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
EXPONENT_EDGES = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 1e2, 1e3, 1e4, 1e5)

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


def reconstruct_domain(
    shells: typing.Any,
    panels: typing.Any,
    pair_mode: typing.Any,
    auxiliary_shells: typing.Any = None,
) -> typing.Any:
    """Count implicit signature products using only public host shell metadata.

    Each shell row is (angular, primitives, public AO offset, public AO count).
    A panel may split an auxiliary shell; each intersection is an executed
    visit. Signature ordering is angular then primitive count, retaining
    triangles only when the two orbital groups are identical.
    """
    if pair_mode not in (0, 1, 2):
        raise ValueError("unknown public pair mode")
    auxiliary_shells = shells if auxiliary_shells is None else auxiliary_shells
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
                    for angular, primitives, offset, width in auxiliary_shells
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


def _screening_bin(value: float, edges: tuple[float, ...], *, zero_bin: bool) -> int:
    if not math.isfinite(value) or value < 0:
        raise ValueError("screening feature must be finite and nonnegative")
    if zero_bin and value == 0:
        return 0
    index = 1 if zero_bin else 0
    for edge in edges:
        if value < edge:
            return index
        index += 1
    return index


def screening_feature_bins(
    shells: typing.Sequence[typing.Sequence[int]],
    auxiliary_shells: typing.Sequence[typing.Sequence[int]],
    orbital_metadata: typing.Sequence[typing.Any],
    auxiliary_metadata: typing.Sequence[typing.Any],
    atoms: typing.Sequence[typing.Any],
    panels: typing.Sequence[typing.Sequence[int]],
    pair_mode: int,
) -> dict:
    """Reconstruct #437 distance/exponent distributions over considered primitives.

    Every entry is weighted by primitive-product work before zero-weight pruning.
    Panel intersections are repeated exactly, matching reconstruct_domain.
    Distances use the three physical shell centers; exponent histograms retain
    the alpha/beta/gamma roles of the executed shell orientation.
    """
    if pair_mode not in (0, 1, 2):
        raise ValueError("unknown public pair mode")
    if len(shells) != len(orbital_metadata) or len(auxiliary_shells) != len(
        auxiliary_metadata
    ):
        raise ValueError("screening feature shell metadata mismatch")

    visits = []
    for row in auxiliary_shells:
        offset, width = int(row[2]), int(row[3])
        visits.append(
            sum(
                int(repeats)
                for begin, count, repeats in panels
                if offset < int(begin) + int(count) and offset + width > int(begin)
            )
        )

    distance_bins = len(DISTANCE_EDGES_BOHR) + 2
    exponent_bins = len(EXPONENT_EDGES) + 1
    classes: dict[tuple[int, int, int], dict[str, typing.Any]] = {}

    def row_for(angular: tuple[int, int, int]) -> dict[str, typing.Any]:
        return classes.setdefault(
            angular,
            {
                "primitive_products_considered": 0,
                "distance_bohr": {
                    key: [0] * distance_bins for key in ("ab", "ac", "bc")
                },
                "exponents": {
                    key: [0] * exponent_bins for key in ("alpha", "beta", "gamma")
                },
            },
        )

    positions = [tuple(float(value) for value in atom.position) for atom in atoms]
    exponent_hists = []
    for metadata in (*orbital_metadata, *auxiliary_metadata):
        histogram = [0] * exponent_bins
        for primitive in metadata.primitives:
            exponent = float(primitive.exponent)
            if not math.isfinite(exponent) or exponent <= 0:
                raise ValueError("primitive exponent must be finite and positive")
            histogram[_screening_bin(exponent, EXPONENT_EDGES, zero_bin=False)] += 1
        exponent_hists.append(histogram)
    orbital_exponent_hists = exponent_hists[: len(orbital_metadata)]
    auxiliary_exponent_hists = exponent_hists[len(orbital_metadata) :]

    for ia, (a_row, a_shell) in enumerate(zip(shells, orbital_metadata, strict=True)):
        pa = int(a_row[1])
        if pa != len(a_shell.primitives):
            raise ValueError("orbital primitive count mismatch")
        a_key = ((int(a_row[0]), pa), ia)
        ra = positions[a_shell.atom_index]
        for ib, (b_row, b_shell) in enumerate(
            zip(shells, orbital_metadata, strict=True)
        ):
            pb = int(b_row[1])
            b_key = ((int(b_row[0]), pb), ib)
            if pair_mode and a_key < b_key:
                continue
            if pb != len(b_shell.primitives):
                raise ValueError("orbital primitive count mismatch")
            rb = positions[b_shell.atom_index]
            ab = math.dist(ra, rb)
            for ic, (c_row, c_shell) in enumerate(
                zip(auxiliary_shells, auxiliary_metadata, strict=True)
            ):
                repetitions = visits[ic]
                if not repetitions:
                    continue
                pc = int(c_row[1])
                if pc != len(c_shell.primitives):
                    raise ValueError("auxiliary primitive count mismatch")
                rc = positions[c_shell.atom_index]
                angular = (int(a_row[0]), int(b_row[0]), int(c_row[0]))
                output = row_for(angular)
                products = pa * pb * pc * repetitions
                output["primitive_products_considered"] += products
                for key, distance in (
                    ("ab", ab),
                    ("ac", math.dist(ra, rc)),
                    ("bc", math.dist(rb, rc)),
                ):
                    output["distance_bohr"][key][
                        _screening_bin(distance, DISTANCE_EDGES_BOHR, zero_bin=True)
                    ] += products

                for bin_index, count in enumerate(orbital_exponent_hists[ia]):
                    output["exponents"]["alpha"][bin_index] += (
                        count * pb * pc * repetitions
                    )
                for bin_index, count in enumerate(orbital_exponent_hists[ib]):
                    output["exponents"]["beta"][bin_index] += (
                        count * pa * pc * repetitions
                    )
                for bin_index, count in enumerate(auxiliary_exponent_hists[ic]):
                    output["exponents"]["gamma"][bin_index] += (
                        count * pa * pb * repetitions
                    )

    expected = reconstruct_domain(shells, panels, pair_mode, auxiliary_shells)
    expected_by_class: Counter = Counter()
    for signature, tasks in expected.items():
        expected_by_class[signature[:3]] += tasks * math.prod(signature[3:])
    if set(classes) != set(expected_by_class):
        raise ValueError(
            "screening feature class domain differs from host reconstruction"
        )

    rows = []
    for angular, values in sorted(classes.items()):
        total = values["primitive_products_considered"]
        if total != expected_by_class[angular]:
            raise ValueError(
                "screening feature primitive domain does not conserve work"
            )
        for histogram in (
            *values["distance_bohr"].values(),
            *values["exponents"].values(),
        ):
            if sum(histogram) != total:
                raise ValueError(
                    "screening feature histogram does not conserve primitive work"
                )
        rows.append({"angular": list(angular), **values})
    return {
        "weighting": "considered primitive products before zero-response pruning",
        "distance_bohr": {
            "positive_edges": list(DISTANCE_EDGES_BOHR),
            "bins": "zero,(0,e0),[e0,e1),...,[e_last,+inf)",
        },
        "exponents": {
            "positive_edges": list(EXPONENT_EDGES),
            "bins": "(0,e0),[e0,e1),...,[e_last,+inf)",
        },
        "classes": rows,
    }


def _primitive_work_domains(
    expected: typing.Any, signature_policy: typing.Any
) -> typing.Any:
    """Map host signatures to the actual trace grouping, retaining exact costs.

    Angular-only launches report the documented p0_0_0 sentinel: they do not
    claim that their primitives have zero length. A sparse angular group only
    reports how many shell tasks were active, so its possible primitive work
    is bounded by the cheapest/most expensive host tasks with that count.
    """
    if signature_policy not in (0, 1, 2):
        raise ValueError("unknown primitive signature policy")
    domains = defaultdict(Counter)
    for signature, tasks in expected.items():
        key = signature[:3] + (0, 0, 0) if signature_policy == 0 else signature
        domains[key][math.prod(signature[3:])] += tasks
    return domains


def _active_primitive_bounds(costs: typing.Any, active: typing.Any) -> typing.Any:
    if active < 0 or active > sum(costs.values()):
        raise ValueError("active shell count exceeds visited shells")

    def bound(reverse: typing.Any) -> typing.Any:
        remaining, work = active, 0
        for cost, count in sorted(costs.items(), reverse=reverse):
            used = min(remaining, count)
            work += cost * used
            remaining -= used
        return work

    return bound(False), bound(True)


def kernel_activity(database_path: typing.Any, record: typing.Any) -> typing.Any:
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
            if bx * by * bz != schedule.component_lanes * schedule.triples_per_block:
                raise ValueError(
                    "Nsight block size disagrees with the generated schedule"
                )
            current = {
                "variant": variant,
                "component_lanes": schedule.component_lanes,
                "shell_tasks_per_block": schedule.triples_per_block,
            }
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


def reduce_work(
    record: typing.Any, shells: typing.Any, auxiliary_shells: typing.Any = None
) -> typing.Any:
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
    expected = reconstruct_domain(
        shells, panels, counters["shell_work_pair_mode"], auxiliary_shells
    )
    work_domains = _primitive_work_domains(
        expected, counters.get("shell_primitive_signature_policy", 1)
    )
    if set(work_domains) != set(signatures):
        raise ValueError("device signature domain differs from the host reconstruction")
    if set(classes) != {signature[:3] for signature in signatures}:
        raise ValueError("device class domain differs from its signatures")
    totals = Counter()
    signature_sums = defaultdict(Counter)
    for signature, values in signatures.items():
        if not set(WORK_FIELDS).issubset(values):
            raise ValueError(f"incomplete signature work counters: {signature}")
        if any(type(value) is not int or value < 0 for value in values.values()):
            raise ValueError("work counters must be nonnegative integers")
        costs = work_domains[signature]
        tasks = sum(costs.values())
        if values["shell_tasks"] != tasks:
            raise ValueError(f"host/device shell count differs: {signature}")
        active = values["active_shell_tasks"]
        if active > tasks:
            raise ValueError("active shell count exceeds visited shells")
        lower, upper = _active_primitive_bounds(costs, active)
        primitives = values["primitive_products"]
        if not lower <= primitives <= upper:
            raise ValueError(f"host/device primitive count differs: {signature}")
        roots = values.get("rys_evaluations", 0)
        if not (
            values["geometry_preparations"]
            == values["boys_evaluations"] + roots
            == primitives
        ):
            raise ValueError("geometry/Boys counts differ from active primitive work")
        rys_model = shell_rys_work_model(signature[:3]) if roots else None
        if roots and roots != primitives:
            raise ValueError("Rys work must cover the entire primitive signature")
        states = values.get("recurrence_states", 0)
        if rys_model:
            component_states = rys_model["component_recurrence_states"]
            component_work = states - primitives * rys_model.get(
                "shared_recurrence_states", 0
            )
            products = values["active_component_products"]
            # Zero folded components skip their moments. The aggregate ledger
            # cannot identify which sparse components survived, so enforce the
            # exact dense count or the documented bounds for a sparse domain.
            valid_states = (
                component_work == primitives * sum(component_states)
                if products == primitives * len(component_states)
                else min(component_states) * products
                <= component_work
                <= max(component_states) * products
            )
        else:
            valid_states = states == 0
        if (
            values.get("rys_roots", 0)
            != roots * (rys_model["rys_roots"] if rys_model else 0)
            or not valid_states
        ):
            raise ValueError("Rys root/recurrence counts differ from generated work")
        if values["boys_order_sum"] != (primitives - roots) * (sum(signature[:3]) + 1):
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
        model = (
            shell_rys_work_model(angular)
            if values.get("rys_evaluations")
            else shell_work_model(angular)
        )
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
            "lowering": "rys" if values.get("rys_evaluations") else "polynomial",
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
        ("auxiliary_f", lambda a: a[2] == 3),
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
        "auxiliary_shells": shells if auxiliary_shells is None else auxiliary_shells,
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--measurement", type=Path, required=True)
    parser.add_argument("--generated-header", type=Path, required=True)
    parser.add_argument("--orbital-basis-file", type=Path)
    parser.add_argument("--auxiliary-basis-file", type=Path)
    parser.add_argument(
        "--nsys",
        type=Path,
        help="SQLite export of the same measured force-call capture",
    )
    parser.add_argument("--output", type=raw_output_path, required=True)
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
    atoms = [Atom.from_value(atom) for atom in case.atoms]

    def shell_rows(
        path: typing.Any, role: typing.Any, default: typing.Any = None
    ) -> typing.Any:
        if path is None:
            metadata = (
                _named_basis_shells(case.vibeqc_basis, atoms)
                if default is None
                else default
            )
        else:
            expected_hash = measurement.get("basis_file_sha256", {}).get(role)
            if expected_hash != hashlib.sha256(path.read_bytes()).hexdigest():
                raise ValueError(f"{role} basis does not match the measured snapshot")
            basis, _ = load_comparison_basis(path, case, role=role, compute_forces=True)
            metadata = basis.shells_for(atoms)
        rows, offset = [], 0
        for shell in metadata:
            angular = shell.angular_momentum
            width = (
                2 * angular + 1
                if case.basis_representation == "spherical"
                else (angular + 1) * (angular + 2) // 2
            )
            rows.append((angular, len(shell.primitives), offset, width))
            offset += width
        return rows, offset, metadata

    shells, nbf, metadata = shell_rows(args.orbital_basis_file, "orbital")
    auxiliary_shells, naux, auxiliary_metadata = shell_rows(
        args.auxiliary_basis_file, "auxiliary", metadata
    )
    if nbf != aos or measurement["aos"] != aos or records[0]["naux"] != naux:
        raise ValueError("headline basis/model shape mismatch")
    result = reduce_work(records[0], shells, auxiliary_shells)
    result["distance_exponent_bins"] = screening_feature_bins(
        shells,
        auxiliary_shells,
        metadata,
        auxiliary_metadata,
        atoms,
        result["host_reconstruction"]["panels"],
        records[0]["counters"]["shell_work_pair_mode"],
    )
    if args.nsys:
        activities = kernel_activity(args.nsys, records[0])
        for row in result["classes"]:
            row["nsys"] = activities[tuple(row["angular"])]
        for label, predicate in (
            ("pure_sp", lambda a: max(a) <= 1),
            ("d_containing", lambda a: max(a) == 2),
            ("auxiliary_f", lambda a: a[2] == 3),
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
