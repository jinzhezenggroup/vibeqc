"""Shared geometry/basis data; constructing the native fixtures needs no oracle."""

from vibeqc.calculator import Primitive, Shell


def _source_inputs(atoms, basis, charge, spin):
    shells = []
    for atom in range(len(atoms)):
        # Fixture tags carry the exact atom index, independently of element.
        tag = next(tag for tag in basis if tag.rstrip("0123456789") + str(atom) == tag)
        for angular, *primitives in basis[tag]:
            shells.append(
                Shell(atom, angular, tuple(Primitive(*p) for p in primitives))
            )
    return {
        "atoms": atoms,
        "basis": tuple(shells),
        "charge": charge,
        "multiplicity": spin + 1,
    }


def fixture_inputs(name):
    """Return one of the A2 fixtures (Bohr, Cartesian).

    ``h2``: 2 AO / 1 occ / 1 virt.  ``water``: genuine STO-3G O (7 AO /
    5 occ / 2 virt).  ``water_sdf``: custom s+p+d O (12 AO / 5 occ / 7 virt),
    the multi-virtual stress case.  These mirror A1's test fixtures so the
    same molecule is checked in both the reference and the integrated path.
    """
    _S = [
        [
            0,
            (3.425250914, 0.1543289673),
            (0.6239137298, 0.5353281423),
            (0.168855404, 0.4446345422),
        ]
    ]
    if name == "h2":
        return _source_inputs(
            [(1, [0.0, 0.0, 0.0]), (1, [0.1, 0.2, 1.4])],
            {"H0": list(_S), "H1": list(_S)},
            0,
            0,
        )
    if name == "water":
        _O = [
            [
                0,
                (130.7093214, 0.1543289673),
                (23.80886605, 0.5353281423),
                (6.443608313, 0.4446345422),
            ],
            [
                0,
                (5.033151319, -0.09996722919),
                (1.169596125, 0.3995128261),
                (0.38038896, 0.7001154689),
            ],
            [
                1,
                (5.033151319, 0.155916275),
                (1.169596125, 0.6076837186),
                (0.38038896, 0.3919573931),
            ],
        ]
        return _source_inputs(
            [(8, [0.0, 0.0, 0.0]), (1, [0.0, 0.958, 0.587]), (1, [0.0, -0.958, 0.587])],
            {"O0": list(_O), "H1": list(_S), "H2": list(_S)},
            0,
            0,
        )
    if name == "water_sdf":
        _O = [
            [
                0,
                (18.5950316763, 0.0499684015),
                (5.6203025291, 0.1505027689),
                (1.3720603452, 0.4117903918),
                (0.3434943771, 0.4106611356),
            ],
            [
                1,
                (7.1153375795, 0.0102428309),
                (2.0218099978, 0.0314286582),
                (0.5271803969, 0.0814495192),
                (0.1703101032, 0.1646776769),
                (0.0628267133, 0.2917579986),
            ],
            [2, (0.0628267133, 0.1506938663), (0.0351485943, 0.2198166377)],
        ]
        return _source_inputs(
            [(8, [0.0, 0.0, 0.0]), (1, [0.0, 0.958, 0.587]), (1, [0.0, -0.958, 0.587])],
            {"O0": list(_O), "H1": list(_S), "H2": list(_S)},
            0,
            0,
        )
    raise ValueError(f"unknown A2 fixture {name!r}")


def oracle_system(source):
    """Build the separate PySCF finite-difference oracle only when requested."""
    from pyscf import gto

    from tools.vibeqc_hessian.reference import System, build_mol

    atoms = [(a.atomic_number, a.position) for a in source.atoms]
    tags = [
        f"{gto.mole._symbol(a.atomic_number)}{i}" for i, a in enumerate(source.atoms)
    ]
    basis = {tag: [] for tag in tags}
    for shell in source.shells:
        basis[tags[shell.atom_index]].append(
            [
                shell.angular_momentum,
                *((p.exponent, p.coefficient) for p in shell.primitives),
            ]
        )
    return System(build_mol(atoms, basis, source.charge, source.multiplicity - 1))
