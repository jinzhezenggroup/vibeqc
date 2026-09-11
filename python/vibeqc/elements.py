"""Element identity and electron bookkeeping, independent of bundled basis data."""

import math
from dataclasses import dataclass
from numbers import Integral

# Names identify nuclei only. Their presence does not advertise an executable
# basis, ECP, relativistic Hamiltonian, or validated transition-metal method.
SYMBOLS = (
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
)
ATOMIC_NUMBERS = {symbol: z for z, symbol in enumerate(SYMBOLS, 1)}


def checked_integer(value, name, *, low=0, high=2**31 - 1):
    """Reject truncation and overflow before constructing native integer fields."""
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or not low <= value <= high
    ):
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return int(value)


def atomic_number(value):
    """Resolve an IUPAC symbol or an exact integer Z without basis lookup."""
    if isinstance(value, str):
        try:
            return ATOMIC_NUMBERS[value.capitalize()]
        except KeyError as error:
            raise ValueError(f"unknown element symbol {value!r}") from error
    return checked_integer(value, "atomic number", low=1, high=118)


@dataclass(frozen=True)
class ElectronState:
    """Nuclei, effective ionic cores and requested occupations are distinct."""

    atomic_numbers: tuple[int, ...]
    nuclear_charges: tuple[int, ...]
    ecp_core_electrons: tuple[int, ...]
    ionic_charge: int
    electron_count: int
    multiplicity: int
    nalpha: int
    nbeta: int


def electron_state(
    atoms, *, charge=0, multiplicity=1, element_metadata=None, electron_count=None
):
    """Compute active electrons including declared ECP cores, without execution.

    A schema can describe an ECP's active-electron count even while capability
    preflight rejects its Hamiltonian. Never pass that metadata through the
    all-electron native ABI by merely modifying the ionic charge.
    """
    from .calculator import Atom

    atoms = tuple(Atom.from_value(a) for a in atoms)
    charge = checked_integer(charge, "ionic charge", low=-(2**31), high=2**31 - 1)
    multiplicity = checked_integer(multiplicity, "multiplicity", low=1, high=2**31 - 1)
    numbers, nuclei, cores = [], [], []
    for atom in atoms:
        z = atomic_number(atom.atomic_number)
        if len(atom.position) != 3 or not all(math.isfinite(v) for v in atom.position):
            raise ValueError("atom coordinates must be three finite Bohr values")
        metadata = (element_metadata or {}).get(z)
        nucleus = z if metadata is None else metadata.nuclear_charge
        core = 0 if metadata is None else metadata.ecp_core_electrons
        numbers.append(z)
        nuclei.append(nucleus)
        cores.append(core)
    if not numbers:
        raise ValueError("electron state requires atoms")
    count = sum(nuclei) - sum(cores) - charge
    checked_integer(count, "active electron count", low=1)
    if electron_count is not None:
        checked_integer(electron_count, "requested electron count", low=1)
        if electron_count != count:
            raise ValueError(
                "requested electron count conflicts with nuclei, ECP cores and ionic charge"
            )
    unpaired = multiplicity - 1
    if unpaired > count or (count - unpaired) % 2:
        raise ValueError(
            "electron count and multiplicity require integer nonnegative spin occupations"
        )
    return ElectronState(
        tuple(numbers),
        tuple(nuclei),
        tuple(cores),
        charge,
        count,
        multiplicity,
        (count + unpaired) // 2,
        (count - unpaired) // 2,
    )
