"""Shared exact workloads for VIBEQC and GPU4PySCF comparisons."""

from __future__ import annotations

from dataclasses import dataclass

from vibeqc import Primitive, Shell

_ANGSTROM_TO_BOHR = 1.8897261246257702


def _angstrom_atoms(
    atoms: tuple[tuple[str, tuple[float, float, float]], ...],
) -> tuple[tuple[str, tuple[float, float, float]], ...]:
    """Convert one published fixed conformer to the benchmark's Bohr unit."""

    return tuple(
        (
            element,
            tuple(_ANGSTROM_TO_BOHR * component for component in position),
        )
        for element, position in atoms
    )


@dataclass(frozen=True)
class BenchmarkCase:
    """One exact common workload for VIBEQC and PySCF/GPU4PySCF."""

    description: str
    atoms: tuple[tuple[str, tuple[float, float, float]], ...]
    vibeqc_basis: str | tuple[Shell, ...]
    pyscf_basis: str | dict[str, list]
    charge: int = 0
    multiplicity: int = 1
    method: str = "rhf"
    basis_representation: str = "cartesian"
    expected_ao_count: int | None = None


@dataclass(frozen=True)
class BenchmarkGatePoint:
    """One mandatory real-molecule accuracy/performance acceptance point."""

    case: str
    batch_size: int
    expected_ao_count: int
    maximum_energy_error: float
    maximum_force_error: float
    reference_gradient_tolerance: float
    minimum_speedup: float | None


def benchmark_cases() -> dict[str, BenchmarkCase]:
    """Return artificial and bundled named-basis validation cases."""

    sp_atoms = (("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7)))
    cases = {
        "sp8": BenchmarkCase(
            description="H2, 8 Cartesian s/p AOs",
            atoms=sp_atoms,
            vibeqc_basis=(
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
            vibeqc_basis=(
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
            vibeqc_basis=tuple(
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
            vibeqc_basis="def2-svp",
            pyscf_basis="def2-svp",
        ),
        "water-def2-svp-spherical": BenchmarkCase(
            description="H2O, 24 real spherical AOs, def2-SVP direct J/K",
            atoms=(
                ("O", (0.0, 0.0, 0.0)),
                ("H", (0.0, -1.43233673, 1.10715266)),
                ("H", (0.0, 1.43233673, 1.10715266)),
            ),
            vibeqc_basis="def2-svp",
            pyscf_basis="def2-svp",
            basis_representation="spherical",
        ),
        "water-def2-tzvp": BenchmarkCase(
            description="H2O, 48 Cartesian AOs, def2-TZVP direct J/K",
            atoms=(
                ("O", (0.0, 0.0, 0.0)),
                ("H", (0.0, -1.43233673, 1.10715266)),
                ("H", (0.0, 1.43233673, 1.10715266)),
            ),
            vibeqc_basis="def2-tzvp",
            pyscf_basis="def2-tzvp",
        ),
        "water-def2-tzvp-spherical": BenchmarkCase(
            description="H2O, 43 real spherical AOs, def2-TZVP direct J/K",
            atoms=(
                ("O", (0.0, 0.0, 0.0)),
                ("H", (0.0, -1.43233673, 1.10715266)),
                ("H", (0.0, 1.43233673, 1.10715266)),
            ),
            vibeqc_basis="def2-tzvp",
            pyscf_basis="def2-tzvp",
            basis_representation="spherical",
        ),
        "water-tetramer-def2-svp-spherical": BenchmarkCase(
            description=(
                "WATER27 water tetramer, 96 real spherical AOs, def2-SVP direct J/K"
            ),
            # GMTKN55/WATER27 H2O4 at revision 8d485b37. This optimized,
            # hydrogen-bonded cluster is not the separated-water scaling grid.
            atoms=_angstrom_atoms(
                (
                    ("O", (-1.45649242390384, 1.28720932656356, -0.00736889769275)),
                    ("H", (-0.49296699491671, 1.47969274920594, -0.00986060363313)),
                    ("H", (-1.81941989798974, 1.74937311874999, 0.75214359146194)),
                    ("O", (-1.28720932656356, -1.45649242390384, 0.00736889769275)),
                    ("H", (-1.47969274920594, -0.49296699491671, 0.00986060363313)),
                    ("H", (-1.74937311874999, -1.81941989798974, -0.75214359146194)),
                    ("O", (1.45649242390384, -1.28720932656356, -0.00736889769275)),
                    ("H", (0.49296699491671, -1.47969274920594, -0.00986060363313)),
                    ("H", (1.81941989798974, -1.74937311874999, 0.75214359146194)),
                    ("O", (1.28720932656356, 1.45649242390384, 0.00736889769275)),
                    ("H", (1.47969274920594, 0.49296699491671, 0.00986060363313)),
                    ("H", (1.74937311874999, 1.81941989798974, -0.75214359146194)),
                )
            ),
            vibeqc_basis="def2-svp",
            pyscf_basis="def2-svp",
            basis_representation="spherical",
            expected_ao_count=96,
        ),
        "water-octamer-s4-def2-svp-spherical": BenchmarkCase(
            description=(
                "WATER27 S4 water octamer, 192 real spherical AOs, def2-SVP direct J/K"
            ),
            # GMTKN55/WATER27 H2O8s4 at revision 8d485b37. The optimized S4
            # hydrogen-bond network is one physical cluster, not eight
            # independently translated monomers.
            atoms=_angstrom_atoms(
                (
                    ("O", (1.99059108375705, -0.10840031530352, -1.46522910651851)),
                    ("H", (1.33029276236467, -0.84678678331271, -1.53045186369272)),
                    ("H", (2.70353950286834, -0.31034517655420, -2.07624001004628)),
                    ("O", (-0.10840031530352, -1.99059108375705, 1.46522910651851)),
                    ("H", (-0.31034517655420, -2.70353950286834, 2.07624001004628)),
                    ("H", (-0.84678678331271, -1.33029276236467, 1.53045186369272)),
                    ("O", (-1.93875321600294, -0.02557562889193, 1.40160922927486)),
                    ("H", (-2.17246664587724, 0.02636079885791, 0.45836947014101)),
                    ("H", (-1.39882401939553, 0.76704414676758, 1.56446474221910)),
                    ("O", (0.02557562889193, -1.93875321600294, -1.40160922927486)),
                    ("H", (-0.02636079885791, -2.17246664587724, -0.45836947014101)),
                    ("H", (-0.76704414676758, -1.39882401939553, -1.56446474221910)),
                    ("O", (-0.02557562889193, 1.93875321600294, -1.40160922927486)),
                    ("H", (0.76704414676758, 1.39882401939553, -1.56446474221910)),
                    ("H", (0.02636079885791, 2.17246664587724, -0.45836947014101)),
                    ("O", (0.10840031530352, 1.99059108375705, 1.46522910651851)),
                    ("H", (0.84678678331271, 1.33029276236467, 1.53045186369272)),
                    ("H", (0.31034517655420, 2.70353950286834, 2.07624001004628)),
                    ("O", (-1.99059108375705, 0.10840031530352, -1.46522910651851)),
                    ("H", (-1.33029276236467, 0.84678678331271, -1.53045186369272)),
                    ("H", (-2.70353950286834, 0.31034517655420, -2.07624001004628)),
                    ("O", (1.93875321600294, 0.02557562889193, 1.40160922927486)),
                    ("H", (2.17246664587724, -0.02636079885791, 0.45836947014101)),
                    ("H", (1.39882401939553, -0.76704414676758, 1.56446474221910)),
                )
            ),
            vibeqc_basis="def2-svp",
            pyscf_basis="def2-svp",
            basis_representation="spherical",
            expected_ao_count=192,
        ),
        "oh-def2-svp-uhf": BenchmarkCase(
            description=("OH, 20 Cartesian AOs, def2-SVP direct UHF doublet"),
            atoms=(
                ("O", (0.0, 0.0, 0.0)),
                ("H", (0.0, 0.0, 1.8323918340046244)),
            ),
            vibeqc_basis="def2-svp",
            pyscf_basis="def2-svp",
            multiplicity=2,
            method="uhf",
        ),
        "oh-def2-svp-spherical-uhf": BenchmarkCase(
            description=("OH, 19 real spherical AOs, def2-SVP direct UHF doublet"),
            atoms=(
                ("O", (0.0, 0.0, 0.0)),
                ("H", (0.0, 0.0, 1.8323918340046244)),
            ),
            vibeqc_basis="def2-svp",
            pyscf_basis="def2-svp",
            multiplicity=2,
            method="uhf",
            basis_representation="spherical",
        ),
        "h2plus-uhf2": BenchmarkCase(
            description="H2+, 2 Cartesian s AOs, UHF doublet",
            atoms=sp_atoms,
            vibeqc_basis=(
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
                "H": [
                    [
                        0,
                        [3.42525091, 0.15432897],
                        [0.62391373, 0.53532814],
                        [0.16885540, 0.44463454],
                    ]
                ]
            },
            charge=1,
            multiplicity=2,
            method="uhf",
        ),
        "heh-sdf18-uhf": BenchmarkCase(
            description="HeH, 18 Cartesian s/d/f AOs, direct UHF doublet",
            atoms=(("He", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))),
            vibeqc_basis=(
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

    # This synthetic scaling point reuses the physical WATER27 S4 octamer
    # topology while keeping the two copies far enough apart to avoid atomic
    # overlap. It is intentionally labeled synthetic: the case exists to make
    # the 384-AO J/K scaling regression reproducible, not to represent
    # an optimized water-hexadecamer structure.
    octamer = cases["water-octamer-s4-def2-svp-spherical"].atoms
    half_separation = 5.0 * _ANGSTROM_TO_BOHR
    cases["water-hexadecamer-2s4-def2-svp-spherical"] = BenchmarkCase(
        description=(
            "synthetic pair of translated WATER27 S4 water octamers, "
            "384 real spherical AOs, def2-SVP J/K"
        ),
        atoms=tuple(
            (
                element,
                (position[0] + x_shift, position[1], position[2]),
            )
            for x_shift in (-half_separation, half_separation)
            for element, position in octamer
        ),
        vibeqc_basis="def2-svp",
        pyscf_basis="def2-svp",
        basis_representation="spherical",
        expected_ao_count=384,
    )
    # Issue 52's large-topology gate uses four translated copies of the same
    # octamer. Its 384 shells imply 2,732,120,160 symmetry-unique shell
    # quartets, so it must exercise bounded device enumeration rather than an
    # exact descriptor arena.
    cases["water-32mer-4s4-def2-svp-spherical"] = BenchmarkCase(
        description=(
            "synthetic 2x2 translated WATER27 S4 octamer array, "
            "768 real spherical AOs, def2-SVP bounded direct J/K"
        ),
        atoms=tuple(
            (
                element,
                (
                    position[0] + x_shift,
                    position[1] + y_shift,
                    position[2],
                ),
            )
            for x_shift in (-half_separation, half_separation)
            for y_shift in (-half_separation, half_separation)
            for element, position in octamer
        ),
        vibeqc_basis="def2-svp",
        pyscf_basis="def2-svp",
        basis_representation="spherical",
        expected_ao_count=768,
    )
    return cases


def real_molecule_gate_points() -> tuple[BenchmarkGatePoint, ...]:
    """Return the four explicit scale gates agreed for direct and DF phases."""

    accuracy_96 = {
        "maximum_energy_error": 3.0e-11,
        "maximum_force_error": 3.0e-11,
        # GPU4PySCF reaches a stable energy/force plateau here but can keep
        # cycling above its 1e-10 gradient threshold for the scaled batch
        # endpoints. The 1e-9 setting converges all four reference systems
        # while remaining well inside the explicit VIBEQC error gates.
        "reference_gradient_tolerance": 1.0e-9,
    }
    accuracy_192 = {
        "maximum_energy_error": 1.0e-10,
        "maximum_force_error": 5.0e-10,
        "reference_gradient_tolerance": 1.0e-8,
    }
    return (
        BenchmarkGatePoint(
            case="water-tetramer-def2-svp-spherical",
            batch_size=1,
            expected_ao_count=96,
            minimum_speedup=1.0,
            **accuracy_96,
        ),
        BenchmarkGatePoint(
            case="water-tetramer-def2-svp-spherical",
            batch_size=4,
            expected_ao_count=96,
            minimum_speedup=1.0,
            **accuracy_96,
        ),
        BenchmarkGatePoint(
            case="water-octamer-s4-def2-svp-spherical",
            batch_size=1,
            expected_ao_count=192,
            minimum_speedup=None,
            **accuracy_192,
        ),
        BenchmarkGatePoint(
            case="water-octamer-s4-def2-svp-spherical",
            batch_size=4,
            expected_ao_count=192,
            minimum_speedup=None,
            **accuracy_192,
        ),
    )


def density_fitting_gate_points() -> tuple[BenchmarkGatePoint, ...]:
    """Return the CUDA-DF 96/192-AO parity plus 384-AO scaling matrix.

    Direct-SCF gates intentionally keep their historical thresholds in
    :func:`real_molecule_gate_points`; this separate matrix can evolve with
    the DF implementation without weakening the direct-versus-direct gate.
    """

    accuracy_96 = {
        "maximum_energy_error": 3.0e-11,
        "maximum_force_error": 3.0e-11,
        "reference_gradient_tolerance": 1.0e-9,
    }
    accuracy_192 = {
        "maximum_energy_error": 1.0e-10,
        "maximum_force_error": 5.0e-10,
        "reference_gradient_tolerance": 1.0e-8,
    }
    return (
        BenchmarkGatePoint(
            case="water-tetramer-def2-svp-spherical",
            batch_size=1,
            expected_ao_count=96,
            minimum_speedup=1.0,
            **accuracy_96,
        ),
        BenchmarkGatePoint(
            case="water-tetramer-def2-svp-spherical",
            batch_size=4,
            expected_ao_count=96,
            minimum_speedup=1.0,
            **accuracy_96,
        ),
        BenchmarkGatePoint(
            case="water-octamer-s4-def2-svp-spherical",
            batch_size=1,
            expected_ao_count=192,
            minimum_speedup=1.0,
            **accuracy_192,
        ),
        BenchmarkGatePoint(
            case="water-octamer-s4-def2-svp-spherical",
            batch_size=4,
            expected_ao_count=192,
            minimum_speedup=1.0,
            **accuracy_192,
        ),
        # The translated WATER27 16-mer is deliberately outside the direct
        # parity matrix: its 384-AO four-center path is the scaling point at
        # which DF must remain competitive rather than merely accurate.
        BenchmarkGatePoint(
            case="water-hexadecamer-2s4-def2-svp-spherical",
            batch_size=1,
            expected_ao_count=384,
            maximum_energy_error=2.0e-9,
            maximum_force_error=2.0e-8,
            reference_gradient_tolerance=2.0e-8,
            minimum_speedup=1.0,
        ),
    )
