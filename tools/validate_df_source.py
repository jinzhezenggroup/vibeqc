"""Reconstruct native raw DF tensors and compare schedules, tiles, and RI-J/K.

This manual tier must run in a finite Slurm GPU allocation. The independently
normalized libcint M/A tensors and NumPy eigendecomposition define its oracle;
the probe links the exact production native library under test.
"""

# Source-tree CLI bootstrap for transitive compiler clients.
import sys as _compiler_sys
from pathlib import Path as _CompilerPath

_compiler_sys.path.insert(
    0, str(_CompilerPath(__file__).resolve().parents[1] / "python")
)

import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from pyscf.df.incore import aux_e2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.generate_validation_references import pyscf_molecule
from tools.vibeqc_validation.f_shell_numerics import numerical_error
from tools.vibeqc_validation.schema import canonical_hash, file_hash


def fixture_systems(representation, *, long_contractions=False):
    """Two same-size systems with different geometry and primitive offsets."""
    angular = range(2 if long_contractions else 4)
    atoms = len(angular)
    systems = []
    for item in range(2):
        base = {
            "atomic_numbers": [1] * atoms,
            "coordinates": [
                [0.37 * i, (-1) ** i * 0.24, 0.21 * i + (0.08 * item if i == 1 else 0)]
                for i in angular
            ],
            "basis_representation": representation,
            "charge": 0,
            "multiplicity": 1,
        }
        pair = []
        for role in range(2):
            value = copy.deepcopy(base)
            value["shells"] = [
                {
                    "atom_index": i,
                    "angular_momentum": i,
                    "primitives": [
                        [
                            0.57 + role * 0.31 + i * 0.11 + p * 0.37,
                            0.83 if p == 0 else -0.13 / p,
                        ]
                        for p in range(
                            (9 - 2 * i)
                            if long_contractions
                            else 1 + (i + item + role) % 2
                        )
                    ],
                }
                for i in angular
            ]
            pair.append(value)
        systems.append(pair)
    return systems


def write_input(path, systems):
    """Serialize physical shell inputs, before either implementation normalizes."""
    representation = int(systems[0][0]["basis_representation"] == "spherical")
    rows = [f"{len(systems)} {representation}"]
    for orbital, auxiliary in systems:
        rows.append(
            f"{len(orbital['coordinates'])} {len(orbital['shells'])} {len(auxiliary['shells'])}"
        )
        rows.extend(
            " ".join(map(str, (z, *xyz)))
            for z, xyz in zip(orbital["atomic_numbers"], orbital["coordinates"])
        )
        for basis in (orbital, auxiliary):
            for shell in basis["shells"]:
                rows.append(
                    f"{shell['atom_index']} {shell['angular_momentum']} {len(shell['primitives'])}"
                )
                rows.extend(
                    " ".join(map(str, primitive)) for primitive in shell["primitives"]
                )
    path.write_text("\n".join(rows) + "\n")


def references(systems, *, derivatives=False):
    """Raw libcint values and an independently ordered dense RI contraction."""
    result = {key: [] for key in ("metric", "raw", "j", "k", "uj", "ka", "kb")}
    diagnostics = []
    responses = {key: [] for key in ("raw_derivative", "metric_derivative")}
    for item, (orbital, auxiliary) in enumerate(systems):
        if derivatives:
            from tools.vibeqc_validation.df_gradient import reference_df_matrices

            _, _, da, dm = reference_df_matrices(orbital, auxiliary)
            responses["raw_derivative"].append(da.reshape((-1, *da.shape[2:])))
            responses["metric_derivative"].append(dm.reshape((-1, *dm.shape[2:])))
        mol, scale, _ = pyscf_molecule(orbital)
        aux, aux_scale, _ = pyscf_molecule(auxiliary)
        metric = aux.intor("int2c2e") * np.outer(aux_scale, aux_scale)
        raw = aux_e2(mol, aux, intor="int3c2e", aosym="s1")
        raw *= np.einsum("i,j,p->ijp", scale, scale, aux_scale)
        values, vectors = np.linalg.eigh(metric)
        retained = values > values.max() * 1e-12
        inverse = (vectors[:, retained] / np.sqrt(values[retained])) @ vectors[
            :, retained
        ].T
        tensor = raw @ inverse
        n = len(scale)
        density = np.fromfunction(
            lambda i, j, item=item: (
                np.where(i == j, 0.3, 0.01 / (1 + abs(i - j))) * (item + 1)
            ),
            (n, n),
        )
        coulomb = np.einsum("ijp,klp,kl->ij", tensor, tensor, density, optimize=True)
        exchange = np.einsum("ikp,jlp,kl->ij", tensor, tensor, density, optimize=True)
        for key, value in zip(
            result,
            (metric, raw, coulomb, exchange, coulomb, 0.6 * exchange, 0.4 * exchange),
        ):
            result[key].append(value)
        diagnostics.append(
            {
                "rank": int(retained.sum()),
                "condition_number": float(values.max() / values[retained].min()),
            }
        )
    if derivatives:
        result.update(responses)
        result.update({"bulk_" + key: value for key, value in responses.items()})
    return {key: np.asarray(value) for key, value in result.items()}, diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--derivatives", action="store_true")
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=("cartesian", "spherical", "long", "dependent"),
        default=["cartesian", "spherical", "long", "dependent"],
    )
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("run this native GPU validation through srun")
    directory = args.directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    linked = subprocess.run(
        ["ldd", str(args.probe.resolve())], capture_output=True, text=True, check=True
    )
    libraries = [
        line.split()[2]
        for line in linked.stdout.splitlines()
        if line.strip().startswith("libvibeqc.so") and "=>" in line
    ]
    if len(libraries) != 1 or not Path(libraries[0]).is_file():
        raise RuntimeError(
            "cannot verify the production library linked by the native probe"
        )
    report = {
        "schema": "vibeqc.df_source_validation",
        "version": 1,
        "probe_hash": file_hash(args.probe),
        "library": libraries[0],
        "library_hash": file_hash(Path(libraries[0])),
        "production_promoted": False,
        "runs": [],
    }
    for case in args.cases:
        systems = fixture_systems(
            "cartesian" if case in ("long", "dependent") else case,
            long_contractions=case in ("long", "dependent"),
        )
        if case == "dependent":
            # Exact duplicate auxiliary charge distributions have a deficient
            # metric by construction. Compare rank/RI values at one threshold
            # rather than treating discarded eigenmodes as arithmetic errors.
            for _, auxiliary in systems:
                auxiliary["shells"].insert(0, copy.deepcopy(auxiliary["shells"][0]))
        inputs = directory / f"{case}.txt"
        write_input(inputs, systems)
        expected, diagnostics = references(systems, derivatives=args.derivatives)
        np.savez(directory / f"{case}-reference.npz", **expected)
        nbf, naux = expected["raw"].shape[1], expected["raw"].shape[-1]
        # Generated values are the sole native definition. The independent
        # Libcint/NumPy oracle above validates every mapping; historical native
        # A/B reproduction belongs to the archived promotion source checkout.
        for mapping in ("auxiliary", "component", "primitive"):
            for pair_tile, aux_tile in ((nbf * nbf, naux), (7, 3)):
                name = f"{case}-{mapping}-p{pair_tile}-a{aux_tile}"
                prefix = directory / name
                env = {
                    **os.environ,
                    "VIBEQC_DF_VALUE_MAPPING": mapping,
                }
                command = [
                    str(args.probe.resolve()),
                    str(inputs),
                    str(prefix),
                    str(pair_tile),
                    str(aux_tile),
                ]
                if args.derivatives:
                    command.append("--derivatives")
                run = subprocess.run(
                    command,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=600,
                    check=False,
                )
                (directory / f"{name}.log").write_text(run.stdout + run.stderr)
                if run.returncode:
                    raise RuntimeError(f"native source failed for {name}: {run.stderr}")
                runtime = json.loads(run.stdout)
                errors = {}
                for key, reference in expected.items():
                    actual = np.fromfile(
                        str(prefix) + f"-{key}.bin", dtype=np.float64
                    ).reshape(reference.shape)
                    errors[key] = numerical_error(
                        actual,
                        reference,
                        atol=2e-9 if key not in ("metric", "raw") else 2e-10,
                        rtol=2e-10,
                    )
                ranks_match = [d["rank"] for d in diagnostics] == [
                    d["rank"] for d in runtime["metrics"]
                ]
                row = {
                    "name": name,
                    "inputs_hash": canonical_hash(systems),
                    "runtime": runtime,
                    "reference_metrics": diagnostics,
                    "errors": errors,
                    "ranks_match": ranks_match,
                    "passed": ranks_match and all(e["passed"] for e in errors.values()),
                }
                report["runs"].append(row)
                (directory / "report.json").write_text(
                    json.dumps(report, indent=2, sort_keys=True) + "\n"
                )
                print(
                    json.dumps(
                        {
                            "name": name,
                            "passed": row["passed"],
                            "raw_ms": runtime["raw_reconstruction_ms"],
                        }
                    ),
                    flush=True,
                )
    report["passed"] = all(row["passed"] for row in report["runs"])
    (directory / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
