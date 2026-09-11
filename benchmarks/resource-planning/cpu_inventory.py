"""Audit complete native CPU C++ heap peaks against the common HF inventory.

Run with a CPU library: PYTHONPATH=python:. python this_file.py --build build
--output /tmp/cpu-resources.json. The helper instruments requested new/delete
bytes, including recurrence, integral preparation, SCF/DIIS, fleet and outputs.
Malloc bookkeeping, page retention and caller inputs are outside that scope.
"""

import argparse
import json
import os
import platform
import subprocess
import tempfile
from pathlib import Path

from vibeqc import Atom, Calculator, electron_state, estimate_hf_resources
from vibeqc.profiles import file_hash

ROOT = Path(__file__).resolve().parents[2]
H2 = [(1, (0, 0, -0.7)), (1, (0, 0, 0.7))]
WATER = [(8, (0, 0, 0)), (1, (1.43, 0, 1.11)), (1, (-1.43, 0, 1.11))]


def payload(calculator, systems, *, unrestricted, fitted):
    """Serialize actual resolved primitives before the native measurement scope."""
    lines = [f"{len(systems)} {int(unrestricted)} {int(fitted)}"]
    for system in systems:
        atoms = tuple(Atom.from_value(atom) for atom in system)
        shells = calculator._shells_for_atoms(atoms)
        state = electron_state(atoms)
        lines.append(f"{len(atoms)} {len(shells)} 0 1 {state.electron_count}")
        lines.extend(
            " ".join(str(v) for v in (atom.atomic_number, *atom.position))
            for atom in atoms
        )
        for shell in shells:
            lines.append(
                f"{shell.atom_index} {shell.angular_momentum} {len(shell.primitives)}"
            )
            lines.extend(
                f"{p.exponent:.17g} {p.coefficient:.17g}" for p in shell.primitives
            )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build = args.build.resolve()
    cache = (build / "CMakeCache.txt").read_text()
    if "VIBEQC_ENABLE_CUDA:BOOL=OFF" not in cache:
        raise ValueError("the CPU heap audit requires a CPU-only library build")
    library = build / "libvibeqc.so"
    os.environ["VIBEQC_LIBRARY"] = str(library)
    compiler = os.environ.get("CXX", "c++")
    records = []
    cases = [
        ("h2-rhf", [H2], "sto-3g", "rhf", False),
        ("ragged-rhf", [H2, WATER, H2], "sto-3g", "rhf", False),
        ("ragged-uhf", [H2, WATER], "sto-3g", "uhf", False),
        ("ragged-ri-rhf", [H2, WATER, H2], "sto-3g", "rhf", True),
        ("h2-ri-uhf", [H2], "def2-svp", "uhf", True),
    ]
    with tempfile.TemporaryDirectory(prefix="vibeqc-resource-probe-") as temporary:
        executable = Path(temporary) / "probe"
        command = [
            compiler,
            "-std=c++20",
            "-O2",
            "-Wl,--export-dynamic",
            f"-I{ROOT / 'include'}",
            f"-I{ROOT / 'src'}",
            str(Path(__file__).with_name("cpu_heap_probe.cpp")),
            f"-L{build}",
            f"-Wl,-rpath,{build}",
            "-lvibeqc",
            "-o",
            str(executable),
        ]
        subprocess.run(command, check=True)
        for name, systems, basis, method, fitted in cases:
            options = {
                "basis": basis,
                "method": method,
                "density_fitting": "cpu" if fitted else "none",
            }
            calculator = Calculator(**options)
            plan = estimate_hf_resources(systems, **options).require_feasible()
            result = subprocess.run(
                [str(executable)],
                input=payload(
                    calculator, systems, unrestricted=method == "uhf", fitted=fitted
                ),
                text=True,
                capture_output=True,
                check=True,
            )
            measured = json.loads(result.stdout)
            expected = [calculator.singlepoint(atoms).energy for atoms in systems]
            maximum_error = max(
                abs(a - b) for a, b in zip(measured["energies"], expected, strict=True)
            )
            if (
                measured["peak_host_bytes"] > plan.peak_bytes["host"]
                or maximum_error > 1e-9
            ):
                raise AssertionError(
                    f"{name}: underestimated CPU allocations or changed energy: {measured}"
                )
            records.append(
                {
                    "case": name,
                    "plan": plan.to_dict(),
                    "observed": measured,
                    "maximum_energy_error": maximum_error,
                }
            )
            print(
                f"{name}: predicted={plan.peak_bytes['host']} observed={measured['peak_host_bytes']}",
                flush=True,
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "platform": platform.platform(),
                "library_sha256": file_hash(library),
                "compiler": subprocess.check_output([compiler, "--version"], text=True),
                "scope": "requested native C++ heap bytes during preparation and two serialized fleet solves; excludes caller inputs, malloc bookkeeping and page/runtime retention",
                "records": records,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
