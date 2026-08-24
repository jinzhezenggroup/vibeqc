"""Method-oriented Python API for native RHF/UHF calculations."""

from __future__ import annotations

import ctypes
import json
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import Iterable, Sequence

import numpy as np

from . import _native


_ELEMENT_SYMBOLS = (
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na",
    "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
)
_ATOMIC_NUMBERS = {
    symbol: atomic_number
    for atomic_number, symbol in enumerate(_ELEMENT_SYMBOLS, start=1)
}


@dataclass(frozen=True)
class Atom:
    atomic_number: int
    position: tuple[float, float, float]

    @classmethod
    def from_value(cls, value: "Atom | tuple[str | int, Sequence[float]]") -> "Atom":
        if isinstance(value, cls):
            return value
        element, position = value
        if isinstance(element, str):
            try:
                atomic_number = _ATOMIC_NUMBERS[element.capitalize()]
            except KeyError as error:
                raise ValueError(
                    f"element symbol {element!r} is outside the bundled H-Ar scope"
                ) from error
        else:
            atomic_number = int(element)
        xyz = tuple(float(component) for component in position)
        if len(xyz) != 3:
            raise ValueError("atom coordinates must have three components")
        return cls(atomic_number, xyz)  # type: ignore[arg-type]


@dataclass(frozen=True)
class Primitive:
    exponent: float
    coefficient: float


@dataclass(frozen=True)
class Shell:
    atom_index: int
    angular_momentum: int
    primitives: tuple[Primitive, ...]


@dataclass(frozen=True)
class Result:
    energy: float
    forces: np.ndarray
    converged: bool
    iterations: int
    energy_change: float
    density_rms: float
    executed_backend: str


@lru_cache(maxsize=1)
def _basis_pack() -> dict[str, object]:
    """Load the generated, data-only Basis Set Exchange subset once."""

    path = resources.files("qce").joinpath("data/basis_pack.json")
    with path.open("r", encoding="utf-8") as handle:
        pack = json.load(handle)
    if pack.get("schema_version") != 1:
        raise RuntimeError("unsupported bundled basis-pack schema")
    return pack


def _named_basis_shells(name: str, atoms: Sequence[Atom]) -> tuple[Shell, ...]:
    canonical_name = name.lower().replace("_", "-")
    bases = _basis_pack()["bases"]
    assert isinstance(bases, dict)
    try:
        basis = bases[canonical_name]
    except KeyError as error:
        supported = ", ".join(sorted(str(key) for key in bases))
        raise NotImplementedError(
            f"bundled basis {name!r} is unavailable; choose one of: {supported}"
        ) from error
    assert isinstance(basis, dict)
    elements = basis["elements"]
    assert isinstance(elements, dict)

    shells: list[Shell] = []
    for atom_index, atom in enumerate(atoms):
        try:
            element_shells = elements[str(atom.atomic_number)]
        except KeyError as error:
            raise NotImplementedError(
                f"{canonical_name} is not bundled for atomic number "
                f"{atom.atomic_number}"
            ) from error
        assert isinstance(element_shells, list)
        for packed_shell in element_shells:
            angular_momentum = int(packed_shell["angular_momentum"])
            exponents = packed_shell["exponents"]
            coefficients = packed_shell["coefficients"]
            primitives = tuple(
                Primitive(float(exponent), float(coefficient))
                for exponent, coefficient in zip(
                    exponents, coefficients, strict=True
                )
            )
            shells.append(Shell(atom_index, angular_momentum, primitives))
    return tuple(shells)


class Calculator:
    """Prepare and execute a native single-system electronic-structure calculation.

    Coordinates are in Bohr. The current implementation accepts RHF or UHF and
    a bundled Cartesian STO-3G/def2-SVP/def2-TZVP basis for H-Ar, or explicit
    `Shell` objects. Both the CPU reference and CUDA backend support Cartesian
    or PySCF/libcint-ordered real spherical AOs through `f` shells.
    """

    def __init__(
        self,
        method: str = "rhf",
        basis: str | Sequence[Shell] = "sto-3g",
        device: str = "cpu",
        device_id: int = 0,
        basis_representation: str = "cartesian",
        *,
        max_iterations: int = 100,
        energy_tolerance: float = 1.0e-10,
        density_tolerance: float = 1.0e-8,
        diis_history: int = 8,
        screening_tolerance: float = 1.0e-12,
    ) -> None:
        methods = {
            "rhf": _native.METHOD_RHF,
            "uhf": _native.METHOD_UHF,
            "wb97m-v": _native.METHOD_WB97M_V,
            "ccsd(t)": _native.METHOD_RCCSD_T,
        }
        if method.lower() not in methods:
            raise ValueError(f"unknown method {method!r}")
        if device not in {"cpu", "cuda"}:
            raise ValueError("device must be 'cpu' or 'cuda'")
        representations = {
            "cartesian": _native.BASIS_CARTESIAN,
            "spherical": _native.BASIS_SPHERICAL,
        }
        if basis_representation not in representations:
            raise ValueError(
                "basis_representation must be 'cartesian' or 'spherical'"
            )
        self._method = methods[method.lower()]
        self._basis = basis
        self._basis_representation = representations[basis_representation]
        self._backend = (
            _native.BACKEND_CUDA if device == "cuda" else _native.BACKEND_CPU_REFERENCE
        )
        self._device_id = int(device_id)
        self._max_iterations = int(max_iterations)
        self._energy_tolerance = float(energy_tolerance)
        self._density_tolerance = float(density_tolerance)
        self._diis_history = int(diis_history)
        self._screening_tolerance = float(screening_tolerance)
        if self._screening_tolerance <= 0.0:
            raise ValueError("screening_tolerance must be positive")
        self._library = _native.load_library()

        available = ctypes.c_int32()
        _native.check(
            self._library,
            self._library.qce_method_available(self._method, ctypes.byref(available)),
        )
        if not available.value:
            raise NotImplementedError(f"method {method!r} is reserved but not implemented")

    def _context_descriptor(self) -> _native.ContextDescriptor:
        return _native.ContextDescriptor(
            ctypes.sizeof(_native.ContextDescriptor),
            _native.ABI_VERSION,
            self._device_id,
            self._backend,
        )

    def _method_descriptor(self) -> _native.MethodDescriptor:
        return _native.MethodDescriptor(
            ctypes.sizeof(_native.MethodDescriptor),
            _native.ABI_VERSION,
            self._method,
            self._max_iterations,
            self._diis_history,
            self._energy_tolerance,
            self._density_tolerance,
            self._screening_tolerance,
        )

    def _shells_for_atoms(self, atoms: Sequence[Atom]) -> tuple[Shell, ...]:
        if isinstance(self._basis, str):
            return _named_basis_shells(self._basis, atoms)
        return tuple(self._basis)

    def _create_native_system(
        self,
        context: ctypes.c_void_p,
        atoms: Sequence[Atom],
        charge: int,
        multiplicity: int,
    ) -> ctypes.c_void_p:
        shells = self._shells_for_atoms(atoms)
        atom_array = (_native.AtomDescriptor * len(atoms))(
            *(
                _native.AtomDescriptor(atom.atomic_number, *atom.position)
                for atom in atoms
            )
        )
        flattened_primitives: list[Primitive] = []
        shell_descriptors: list[_native.ShellDescriptor] = []
        for shell in shells:
            offset = len(flattened_primitives)
            flattened_primitives.extend(shell.primitives)
            shell_descriptors.append(
                _native.ShellDescriptor(
                    shell.atom_index,
                    shell.angular_momentum,
                    offset,
                    len(shell.primitives),
                )
            )
        shell_array = (_native.ShellDescriptor * len(shell_descriptors))(
            *shell_descriptors
        )
        primitive_array = (_native.PrimitiveDescriptor * len(flattened_primitives))(
            *(
                _native.PrimitiveDescriptor(p.exponent, p.coefficient)
                for p in flattened_primitives
            )
        )
        descriptor = _native.SystemDescriptor(
            ctypes.sizeof(_native.SystemDescriptor),
            _native.ABI_VERSION,
            atom_array,
            len(atom_array),
            shell_array,
            len(shell_array),
            primitive_array,
            len(primitive_array),
            int(charge),
            int(multiplicity),
            self._basis_representation,
        )
        system = ctypes.c_void_p()
        _native.check(
            self._library,
            self._library.qce_system_create(
                context, ctypes.byref(descriptor), ctypes.byref(system)
            ),
        )
        return system

    def prepare_batch(
        self,
        systems: Sequence[Iterable[Atom | tuple[str | int, Sequence[float]]]],
        *,
        charges: Sequence[int] | None = None,
        multiplicities: Sequence[int] | None = None,
        warm_start: bool = True,
    ):  # Return annotation is deferred to avoid an import cycle.
        """Prepare a persistent native ragged batch for repeated execution."""

        from .batch import PreparedBatch

        return PreparedBatch(
            self,
            systems,
            charges=charges,
            multiplicities=multiplicities,
            warm_start=warm_start,
        )

    def batch_singlepoint(
        self,
        systems: Sequence[Iterable[Atom | tuple[str | int, Sequence[float]]]],
        *,
        charges: Sequence[int] | None = None,
        multiplicities: Sequence[int] | None = None,
        strict: bool = False,
    ):
        """Execute a one-shot native ragged batch and return per-system status."""

        with self.prepare_batch(
            systems,
            charges=charges,
            multiplicities=multiplicities,
            warm_start=False,
        ) as batch:
            return batch.execute(strict=strict)

    def singlepoint(
        self,
        atoms: Iterable[Atom | tuple[str | int, Sequence[float]]],
        *,
        charge: int = 0,
        multiplicity: int = 1,
    ) -> Result:
        native_atoms = tuple(Atom.from_value(atom) for atom in atoms)
        if not native_atoms:
            raise ValueError("at least one atom is required")
        context = ctypes.c_void_p()
        _native.check(
            self._library,
            self._library.qce_context_create(
                ctypes.byref(self._context_descriptor()), ctypes.byref(context)
            ),
        )
        system = ctypes.c_void_p()
        calculation = ctypes.c_void_p()
        try:
            system = self._create_native_system(
                context, native_atoms, charge, multiplicity
            )
            method_descriptor = self._method_descriptor()
            _native.check(
                self._library,
                self._library.qce_calculation_prepare(
                    context,
                    system,
                    ctypes.byref(method_descriptor),
                    ctypes.byref(calculation),
                ),
            )
            force_storage = (ctypes.c_double * (3 * len(native_atoms)))()
            result_descriptor = _native.ResultDescriptor(
                ctypes.sizeof(_native.ResultDescriptor),
                _native.ABI_VERSION,
                0.0,
                force_storage,
                len(force_storage),
                0,
                0.0,
                0.0,
                0,
                _native.BACKEND_CPU_REFERENCE,
            )
            _native.check(
                self._library,
                self._library.qce_calculation_execute(
                    calculation, ctypes.byref(result_descriptor)
                ),
            )
            forces = np.ctypeslib.as_array(force_storage).copy().reshape(-1, 3)
            backend = (
                "cuda"
                if result_descriptor.executed_backend == _native.BACKEND_CUDA
                else "cpu_reference"
            )
            return Result(
                energy=result_descriptor.energy,
                forces=forces,
                converged=bool(result_descriptor.converged),
                iterations=result_descriptor.iterations,
                energy_change=result_descriptor.energy_change,
                density_rms=result_descriptor.density_rms,
                executed_backend=backend,
            )
        finally:
            if calculation.value:
                self._library.qce_calculation_destroy(calculation)
            if system.value:
                self._library.qce_system_destroy(system)
            self._library.qce_context_destroy(context)
