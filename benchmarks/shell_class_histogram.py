"""Report pre-screen direct-work coverage by canonical shell class."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

ANGULAR_LABELS = "spdfgh"

# GPU4PySCF's unrolled Rys gradient symbols encode the four shell angular
# momenta in the final four digits.  The same physical shell class can be
# emitted in more than one pair/center orientation (for example, ppps is
# present as both ``..._1110`` and ``..._1011``), so the suffix must not be
# used as the aggregation key directly.
_GPU4PYSCF_RYS_IP1_RE = re.compile(
    r"(?<![A-Za-z0-9])rys_(?:[ev]jk)_ip1_(?P<angular>[0-9]{4})(?![0-9])",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class ShellWork:
    """Topology data needed to reproduce VIBEQC's direct tile counts."""

    angular: int
    ao_count: int
    primitive_count: int


def shell_pair_class(first: int, second: int) -> int:
    high = max(first, second)
    low = min(first, second)
    return high * (high + 1) // 2 + low


def canonical_shell_class(
    first: tuple[int, int], second: tuple[int, int]
) -> tuple[int, int, int, int]:
    """Apply within-pair and pair-exchange angular symmetry."""

    pairs = []
    for high, low in (first, second):
        ordered = (max(high, low), min(high, low))
        pairs.append((shell_pair_class(*ordered), ordered))
    pairs.sort(reverse=True)
    return (*pairs[0][1], *pairs[1][1])


def gpu4pyscf_rys_ip1_shell_class(
    kernel_name: str,
) -> tuple[int, int, int, int] | None:
    """Decode and canonicalize one GPU4PySCF unrolled Rys IP1 symbol.

    GPU4PySCF uses the four digits after ``rys_ejk_ip1_`` (or
    ``rys_vjk_ip1_``) to describe the raw shell order.  Pair exchange and
    within-pair exchange are integral symmetries, not distinct work classes;
    applying :func:`canonical_shell_class` here makes all orientations share
    one key.  Non-Rys symbols, including the generic unspecialized
    ``rys_ejk_ip1_kernel``, return ``None`` so callers can process a complete
    Nsight kernel table without a separate name filter.

    A recognized symbol with an unsupported angular digit is rejected instead
    of being silently assigned to a misleading class.
    """

    if not isinstance(kernel_name, str):
        raise TypeError("kernel_name must be a string")
    match = _GPU4PYSCF_RYS_IP1_RE.search(kernel_name)
    if match is None:
        return None
    angular = tuple(int(value) for value in match.group("angular"))
    if any(value >= len(ANGULAR_LABELS) for value in angular):
        raise ValueError(
            f"unsupported angular digit in GPU4PySCF kernel {kernel_name!r}"
        )
    return canonical_shell_class(
        (angular[0], angular[1]), (angular[2], angular[3])
    )


def shell_class_label(shell_angular: Iterable[int]) -> str:
    """Return the conventional label for one canonical four-center class."""

    values = tuple(shell_angular)
    if len(values) != 4:
        raise ValueError("a shell class must contain exactly four centers")
    if any(value < 0 or value >= len(ANGULAR_LABELS) for value in values):
        raise ValueError(f"unsupported shell angular tuple {values!r}")
    return "".join(ANGULAR_LABELS[value] for value in values)


def aggregate_gpu4pyscf_rys_ip1_sqlite(
    database: str | Path,
) -> dict[str, dict[str, int | float | list[str]]]:
    """Aggregate unrolled GPU4PySCF Rys IP1 kernels by canonical class.

    Nsight Systems stores kernel names in ``StringIds`` and references them
    from ``CUPTI_ACTIVITY_KIND_KERNEL``.  The returned records use the same
    duration convention as the profile artifacts (milliseconds) and retain
    the raw symbols for auditability.  Both pair orientations are therefore
    included in one class total without a ppps-specific special case.

    The trace schema has used both ``demangledName`` and ``shortName`` across
    Nsight releases.  Prefer the demangled name and fall back to the short
    name so old and new captures receive identical canonical treatment.
    """

    with sqlite3.connect(str(database)) as connection:
        table_columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)"
            )
        }
        if not table_columns:
            raise ValueError(
                "SQLite trace has no CUPTI_ACTIVITY_KIND_KERNEL table"
            )
        name_column = next(
            (
                candidate
                for candidate in ("demangledName", "demangled", "shortName")
                if candidate in table_columns
            ),
            None,
        )
        if name_column is None:
            raise ValueError(
                "SQLite kernel table has no recognized name-id column"
            )
        try:
            rows = connection.execute(
                f"""
                SELECT names.value, COUNT(*),
                       COALESCE(SUM(kernels.end - kernels.start), 0)
                FROM CUPTI_ACTIVITY_KIND_KERNEL AS kernels
                JOIN StringIds AS names ON kernels.{name_column} = names.id
                GROUP BY names.value
                """
            )
        except sqlite3.OperationalError as error:
            raise ValueError(
                "SQLite trace does not expose the expected StringIds table"
            ) from error

        totals: dict[str, dict[str, int | float | set[str]]] = {}
        for kernel_name, launches, nanoseconds in rows:
            shell_angular = gpu4pyscf_rys_ip1_shell_class(kernel_name)
            if shell_angular is None:
                continue
            label = shell_class_label(shell_angular)
            row = totals.setdefault(
                label,
                {
                    "kernel_time_milliseconds": 0.0,
                    "launches": 0,
                    "kernel_names": set(),
                },
            )
            row["kernel_time_milliseconds"] += float(nanoseconds) / 1.0e6
            row["launches"] += int(launches)
            row["kernel_names"].add(kernel_name)

    return {
        label: {
            "kernel_time_milliseconds": float(row["kernel_time_milliseconds"]),
            "launches": int(row["launches"]),
            "kernel_names": sorted(row["kernel_names"]),
        }
        for label, row in sorted(totals.items())
    }


def summarize_shell_classes(
    shells: Iterable[ShellWork],
    angular_order: int | None,
    tile_size: int = 256,
) -> list[dict[str, int | float | str | list[int]]]:
    """Match the host planner's exact pre-screen AO tile enumeration."""

    shell_list = list(shells)
    pairs: list[tuple[int, int, ShellWork, ShellWork, int, int]] = []
    for first_shell, first in enumerate(shell_list):
        for second_shell in range(first_shell + 1):
            second = shell_list[second_shell]
            ao_count = (
                first.ao_count * (first.ao_count + 1) // 2
                if first_shell == second_shell
                else first.ao_count * second.ao_count
            )
            pairs.append(
                (
                    first_shell,
                    second_shell,
                    first,
                    second,
                    ao_count,
                    first.primitive_count * second.primitive_count,
                )
            )

    totals: dict[tuple[int, int, int, int], list[int]] = defaultdict(
        lambda: [0, 0, 0, 0]
    )
    for first_pair_index, first_pair in enumerate(pairs):
        for second_pair_index in range(first_pair_index + 1):
            second_pair = pairs[second_pair_index]
            angular = sum(
                (
                    first_pair[2].angular,
                    first_pair[3].angular,
                    second_pair[2].angular,
                    second_pair[3].angular,
                )
            )
            if angular_order is not None and angular != angular_order:
                continue
            shell_class = canonical_shell_class(
                (first_pair[2].angular, first_pair[3].angular),
                (second_pair[2].angular, second_pair[3].angular),
            )
            ao_quartets = (
                first_pair[4] * (first_pair[4] + 1) // 2
                if first_pair_index == second_pair_index
                else first_pair[4] * second_pair[4]
            )
            primitive_quartets = (
                ao_quartets * first_pair[5] * second_pair[5]
            )
            row = totals[shell_class]
            row[0] += 1
            row[1] += ao_quartets
            row[2] += primitive_quartets
            row[3] += (ao_quartets + tile_size - 1) // tile_size

    primitive_total = sum(row[2] for row in totals.values())
    result = []
    for shell_class, row in sorted(
        totals.items(), key=lambda item: item[1][2], reverse=True
    ):
        result.append(
            {
                "class": "".join(ANGULAR_LABELS[item] for item in shell_class),
                "shell_angular": list(shell_class),
                "shell_quartets": row[0],
                "unique_ao_quartets": row[1],
                "primitive_quartets": row[2],
                "primitive_work_fraction": (
                    row[2] / primitive_total if primitive_total else 0.0
                ),
                "tiles": row[3],
            }
        )
    return result


def summarize_active_shell_classes(
    entries: Iterable[object], angular_order: int | None
) -> list[dict[str, int | float | str | list[int]]]:
    """Format native final-density counters for one total angular order."""

    selected = [
        entry
        for entry in entries
        if (angular_order is None or sum(entry.shell_angular) == angular_order)
        and entry.primitive_quartets != 0
    ]
    primitive_total = sum(entry.primitive_quartets for entry in selected)
    tile_total = sum(entry.tiles for entry in selected)
    return [
        {
            "class": entry.label,
            "shell_angular": list(entry.shell_angular),
            "shell_quartets": entry.shell_quartets,
            "unique_ao_quartets": entry.ao_quartets,
            "primitive_quartets": entry.primitive_quartets,
            "primitive_work_fraction": (
                entry.primitive_quartets / primitive_total
                if primitive_total
                else 0.0
            ),
            "tiles": entry.tiles,
            "tile_fraction": entry.tiles / tile_total if tile_total else 0.0,
        }
        for entry in sorted(
            selected,
            key=lambda item: item.primitive_quartets,
            reverse=True,
        )
    ]


def scaled_geometries(
    atoms: tuple[tuple[str, tuple[float, float, float]], ...],
    batch_size: int,
) -> list[tuple[tuple[str, tuple[float, float, float]], ...]]:
    """Match the fixed-topology geometry perturbations in the formal gate."""

    centroid = tuple(
        sum(position[axis] for _, position in atoms) / len(atoms)
        for axis in range(3)
    )
    systems = []
    for index in range(batch_size):
        centered_index = index - 0.5 * (batch_size - 1)
        scale = 1.0 + 0.002 * centered_index
        systems.append(
            tuple(
                (
                    element,
                    tuple(
                        centroid[axis]
                        + scale * (position[axis] - centroid[axis])
                        for axis in range(3)
                    ),
                )
                for element, position in atoms
            )
        )
    return systems


def main() -> None:
    from _cases import benchmark_cases

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=tuple(benchmark_cases()),
        default="water-tetramer-def2-svp-spherical",
    )
    parser.add_argument("--angular-order", type=int, default=5)
    parser.add_argument(
        "--all-orders",
        action="store_true",
        help="report every active shell class from one SCF execution",
    )
    parser.add_argument(
        "--active",
        action="store_true",
        help="run VIBEQC CUDA and report final-density screened work",
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--warm-repeats", type=int, default=1)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--energy-tolerance", type=float, default=1.0e-12)
    parser.add_argument("--density-tolerance", type=float, default=1.0e-10)
    parser.add_argument(
        "--screening-tolerance",
        type=float,
        default=1.0e-14,
        help="VIBEQC direct-screening threshold; default matches the formal gate",
    )
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.batch < 1 or arguments.warm_repeats < 0:
        raise ValueError("--batch must be positive and --warm-repeats non-negative")

    # PySCF is a benchmark-only dependency. Importing it here keeps --help and
    # the pure topology helpers usable in the normal VIBEQC development venv.
    from pyscf import gto

    case = benchmark_cases()[arguments.case]
    molecule = gto.M(
        atom=case.atoms,
        unit="Bohr",
        basis=case.pyscf_basis,
        charge=case.charge,
        spin=case.multiplicity - 1,
        cart=True,
        verbose=0,
    )
    shells = []
    for shell in range(molecule.nbas):
        angular = molecule.bas_angular(shell)
        shells.append(
            ShellWork(
                angular=angular,
                ao_count=(angular + 1) * (angular + 2) // 2,
                primitive_count=molecule.bas_nprim(shell),
            )
        )
    payload: dict[str, object] = {
        "angular_order": None if arguments.all_orders else arguments.angular_order,
        "basis": case.pyscf_basis,
        "case": arguments.case,
        "direct_cartesian_ao_count": molecule.nao_nr(cart=True),
    }
    if arguments.active:
        from vibeqc import Calculator

        systems = scaled_geometries(case.atoms, arguments.batch)
        calculator = Calculator(
            method=case.method,
            basis=case.vibeqc_basis,
            basis_representation=case.basis_representation,
            device="cuda",
            max_iterations=arguments.max_iterations,
            energy_tolerance=arguments.energy_tolerance,
            density_tolerance=arguments.density_tolerance,
            screening_tolerance=arguments.screening_tolerance,
        )
        with calculator.prepare_batch(
            systems,
            charges=[case.charge] * arguments.batch,
            multiplicities=[case.multiplicity] * arguments.batch,
            warm_start=True,
            shell_class_profiling=True,
        ) as batch:
            result = batch.execute(strict=True)
            for _ in range(arguments.warm_repeats):
                result = batch.execute(strict=True)
            profile = batch.last_shell_class_profile()
        payload.update(
            {
                "batch_size": arguments.batch,
                "iterations": [item.iterations for item in result.items],
                "methodology": (
                    "VIBEQC final converged-density direct task compaction after "
                    "Schwarz and density screening; counters aggregate the "
                    "most recent execution across the native batch"
                ),
                "screening_tolerance": arguments.screening_tolerance,
                "shell_classes": summarize_active_shell_classes(
                    profile,
                    None if arguments.all_orders else arguments.angular_order,
                ),
            }
        )
    else:
        payload.update(
            {
                "methodology": (
                    "VIBEQC host-planner topology before Schwarz/density screening; "
                    "primitive work weights unique Cartesian AO quartets by the "
                    "four shell primitive counts"
                ),
                "shell_classes": summarize_shell_classes(
                    shells,
                    None if arguments.all_orders else arguments.angular_order,
                ),
            }
        )
    output = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(output, end="")
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
