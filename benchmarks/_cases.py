"""Shared exact workloads for QCE and GPU4PySCF comparisons."""

from __future__ import annotations

from dataclasses import dataclass

from qce import Primitive, Shell


@dataclass(frozen=True)
class BenchmarkCase:
    """One exact common workload for QCE and PySCF/GPU4PySCF."""

    description: str
    atoms: tuple[tuple[str, tuple[float, float, float]], ...]
    qce_basis: str | tuple[Shell, ...]
    pyscf_basis: str | dict[str, list]
    charge: int = 0
    multiplicity: int = 1
    method: str = "rhf"
    basis_representation: str = "cartesian"


def benchmark_cases() -> dict[str, BenchmarkCase]:
    """Return artificial and bundled named-basis validation cases."""

    sp_atoms = (("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7)))
    return {
        "sp8": BenchmarkCase(
            description="H2, 8 Cartesian s/p AOs",
            atoms=sp_atoms,
            qce_basis=(
                Shell(0, 0, (Primitive(1.2, 1.0),)),
                Shell(0, 1, (Primitive(0.7, 1.0),)),
                Shell(1, 0, (Primitive(1.2, 1.0),)),
                Shell(1, 1, (Primitive(0.7, 1.0),)),
            ),
            pyscf_basis={"H": [[0, [1.2, 1.0]], [1, [0.7, 1.0]]]},
        ),
        "sdf18-direct": BenchmarkCase(
            description="HeH+, 18 Cartesian s/d/f AOs, screened direct J/K",
            atoms=(("He", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))),
            qce_basis=(
                Shell(0, 0, (Primitive(1.5, 1.0),)),
                Shell(0, 2, (Primitive(0.8, 1.0),)),
                Shell(0, 3, (Primitive(0.6, 1.0),)),
                Shell(1, 0, (Primitive(1.2, 1.0),)),
            ),
            pyscf_basis={
                "He": [[0, [1.5, 1.0]], [2, [0.8, 1.0]], [3, [0.6, 1.0]]],
                "H": [[0, [1.2, 1.0]]],
            },
            charge=1,
        ),
        "he3-sd21-direct": BenchmarkCase(
            description="linear He3, 21 Cartesian s/d AOs, direct J/K",
            atoms=(
                ("He", (0.0, 0.0, -2.0)),
                ("He", (0.0, 0.0, 0.0)),
                ("He", (0.0, 0.0, 2.0)),
            ),
            qce_basis=tuple(
                shell
                for atom_index in range(3)
                for shell in (
                    Shell(atom_index, 0, (Primitive(1.5, 1.0),)),
                    Shell(atom_index, 2, (Primitive(0.8, 1.0),)),
                )
            ),
            pyscf_basis={"He": [[0, [1.5, 1.0]], [2, [0.8, 1.0]]]},
        ),
        "water-def2-svp": BenchmarkCase(
            description="H2O, 25 Cartesian AOs, def2-SVP direct J/K",
            atoms=(
                ("O", (0.0, 0.0, 0.0)),
                ("H", (0.0, -1.43233673, 1.10715266)),
                ("H", (0.0, 1.43233673, 1.10715266)),
            ),
            qce_basis="def2-svp",
            pyscf_basis="def2-svp",
        ),
        "water-def2-svp-spherical": BenchmarkCase(
            description="H2O, 24 real spherical AOs, def2-SVP direct J/K",
            atoms=(
                ("O", (0.0, 0.0, 0.0)),
                ("H", (0.0, -1.43233673, 1.10715266)),
                ("H", (0.0, 1.43233673, 1.10715266)),
            ),
            qce_basis="def2-svp",
            pyscf_basis="def2-svp",
            basis_representation="spherical",
        ),
        "oh-def2-svp-uhf": BenchmarkCase(
            description=(
                "OH, 20 Cartesian AOs, def2-SVP direct UHF doublet"
            ),
            atoms=(
                ("O", (0.0, 0.0, 0.0)),
                ("H", (0.0, 0.0, 1.8323918340046244)),
            ),
            qce_basis="def2-svp",
            pyscf_basis="def2-svp",
            multiplicity=2,
            method="uhf",
        ),
        "oh-def2-svp-spherical-uhf": BenchmarkCase(
            description=(
                "OH, 19 real spherical AOs, def2-SVP direct UHF doublet"
            ),
            atoms=(
                ("O", (0.0, 0.0, 0.0)),
                ("H", (0.0, 0.0, 1.8323918340046244)),
            ),
            qce_basis="def2-svp",
            pyscf_basis="def2-svp",
            multiplicity=2,
            method="uhf",
            basis_representation="spherical",
        ),
        "h2plus-uhf2": BenchmarkCase(
            description="H2+, 2 Cartesian s AOs, UHF doublet",
            atoms=sp_atoms,
            qce_basis=(
                Shell(
                    0,
                    0,
                    (
                        Primitive(3.42525091, 0.15432897),
                        Primitive(0.62391373, 0.53532814),
                        Primitive(0.16885540, 0.44463454),
                    ),
                ),
                Shell(
                    1,
                    0,
                    (
                        Primitive(3.42525091, 0.15432897),
                        Primitive(0.62391373, 0.53532814),
                        Primitive(0.16885540, 0.44463454),
                    ),
                ),
            ),
            pyscf_basis={
                "H": [[
                    0,
                    [3.42525091, 0.15432897],
                    [0.62391373, 0.53532814],
                    [0.16885540, 0.44463454],
                ]]
            },
            charge=1,
            multiplicity=2,
            method="uhf",
        ),
        "heh-sdf18-uhf": BenchmarkCase(
            description="HeH, 18 Cartesian s/d/f AOs, direct UHF doublet",
            atoms=(("He", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))),
            qce_basis=(
                Shell(0, 0, (Primitive(1.5, 1.0),)),
                Shell(0, 2, (Primitive(0.8, 1.0),)),
                Shell(0, 3, (Primitive(0.6, 1.0),)),
                Shell(1, 0, (Primitive(1.2, 1.0),)),
            ),
            pyscf_basis={
                "He": [[0, [1.5, 1.0]], [2, [0.8, 1.0]], [3, [0.6, 1.0]]],
                "H": [[0, [1.2, 1.0]]],
            },
            multiplicity=2,
            method="uhf",
        ),
    }
