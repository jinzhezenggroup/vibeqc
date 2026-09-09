"""Method-oriented Python API for native RHF/UHF calculations."""

from __future__ import annotations

import ctypes
import json
import math
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from functools import cache, lru_cache
from importlib import resources
from pathlib import Path

import numpy as np

from . import _native
from .accuracy import AccuracyAssessment, ResolvedModel, TargetAccuracy
from .basis import BasisProvenance, BasisSet, BasisShell, ElementBasis, load_basis
from .basis_capabilities import require_basis, resolved_basis_metadata
from .elements import atomic_number as element_number
from .elements import checked_integer
from .profiles import canonical_hash

_METHODS = {
    "rhf": _native.METHOD_RHF,
    "uhf": _native.METHOD_UHF,
    "wb97m-v": _native.METHOD_WB97M_V,
    "ccsd(t)": _native.METHOD_RCCSD_T,
}


@dataclass(frozen=True)
class Atom:
    atomic_number: int
    position: tuple[float, float, float]

    def __post_init__(self):
        """Own finite coordinates and validate nuclei independently of basis data."""
        z = element_number(self.atomic_number)
        xyz = tuple(float(v) for v in self.position)
        if len(xyz) != 3 or not all(math.isfinite(v) for v in xyz):
            raise ValueError("atom coordinates must be three finite Bohr values")
        object.__setattr__(self, "atomic_number", z)
        object.__setattr__(self, "position", xyz)

    @classmethod
    def from_value(cls, value: Atom | tuple[str | int, Sequence[float]]) -> Atom:
        if isinstance(value, cls):
            return value
        element, position = value
        atomic_number = element_number(element)
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
    basis_metadata: dict = field(default_factory=dict)
    accuracy: AccuracyAssessment | None = None
    resource_diagnostics: dict | None = None


@dataclass(frozen=True)
class MethodCapabilities:
    """Executable properties reported by the native method registry."""

    method: str
    family: str
    available: bool
    supports_batch: bool
    supported_properties: frozenset[str]


@cache
def method_capabilities(method: str) -> MethodCapabilities:
    """Query method support without constructing a calculator or system."""

    canonical = method.lower()
    try:
        method_id = _METHODS[canonical]
    except KeyError as error:
        raise ValueError(f"unknown method {method!r}") from error
    library = _native.load_library()
    native = _native.MethodCapabilitiesDescriptor(
        ctypes.sizeof(_native.MethodCapabilitiesDescriptor),
        _native.ABI_VERSION,
        0,
        0,
        0,
        0,
        0,
    )
    _native.check(
        library,
        library.vibeqc_method_get_capabilities(method_id, ctypes.byref(native)),
    )
    family = {
        _native.METHOD_FAMILY_HARTREE_FOCK: "hartree_fock",
        _native.METHOD_FAMILY_DENSITY_FUNCTIONAL: "density_functional",
        _native.METHOD_FAMILY_COUPLED_CLUSTER: "coupled_cluster",
    }[native.family]
    properties = set()
    if native.supported_properties & _native.PROPERTY_ENERGY:
        properties.add("energy")
    if native.supported_properties & _native.PROPERTY_FORCES:
        properties.add("forces")
    return MethodCapabilities(
        method=canonical,
        family=family,
        available=bool(native.available),
        supports_batch=bool(native.supports_batch),
        supported_properties=frozenset(properties),
    )


@lru_cache(maxsize=1)
def _basis_pack() -> dict[str, object]:
    """Load the generated, data-only Basis Set Exchange subset once."""

    path = resources.files("vibeqc").joinpath("data/basis_pack.json")
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
                for exponent, coefficient in zip(exponents, coefficients, strict=True)
            )
            shells.append(Shell(atom_index, angular_momentum, primitives))
    return tuple(shells)


@lru_cache(maxsize=8)
def _named_basis_record(name, representation):
    """Wrap the unchanged bundled decimal tables in the public record schema."""
    canonical = name.lower().replace("_", "-")
    pack = _basis_pack()
    if canonical not in pack["bases"]:
        raise NotImplementedError(f"bundled basis {name!r} is unavailable")
    elements = []
    for z, shells in pack["bases"][canonical]["elements"].items():
        elements.append(
            ElementBasis(
                int(z),
                tuple(
                    BasisShell(
                        s["angular_momentum"], s["exponents"], (s["coefficients"],), i
                    )
                    for i, s in enumerate(shells)
                ),
            )
        )
    return BasisSet(
        canonical,
        tuple(elements),
        BasisProvenance(
            "vibeqc/data/basis_pack.json; Basis Set Exchange",
            str(pack["source"]),
            "BSD-3-Clause",
            canonical_hash(pack),
        ),
        representation,
    )


def _snapshot_basis(basis, representation=None):
    """Load local records once and detach all caller-owned primitive storage."""
    if isinstance(basis, os.PathLike) or (
        isinstance(basis, str) and basis.endswith(".json")
    ):
        basis = load_basis(Path(basis))
    if isinstance(basis, BasisSet):
        if representation is not None and basis.representation != representation:
            raise ValueError(
                "basis representation conflicts with the loaded record; create an explicitly reinterpreted record"
            )
        return basis
    if isinstance(basis, str):
        return _named_basis_record(basis, representation or "cartesian")
    return tuple(
        Shell(
            checked_integer(s.atom_index, "shell atom index", high=2**32 - 1),
            checked_integer(
                s.angular_momentum, "shell angular momentum", high=2**32 - 1
            ),
            tuple(
                Primitive(float(p.exponent), float(p.coefficient)) for p in s.primitives
            ),
        )
        for s in basis
    )


class Calculator:
    """Prepare and execute a native single-system electronic-structure calculation.

    Coordinates are in Bohr. The current implementation accepts RHF or UHF and
    a bundled STO-3G/def2-SVP/def2-TZVP basis for H-Ar, local canonical JSON,
    immutable `BasisSet` records, or explicit `Shell` objects. Element symbols
    cover H-Og; execution depends on every actual shell and Hamiltonian. Both the CPU reference and CUDA backend support Cartesian
    or PySCF/libcint-ordered real spherical AOs through `f` shells.
    """

    def __init__(
        self,
        method: str = "rhf",
        basis: str | Path | BasisSet | Sequence[Shell] = "sto-3g",
        device: str = "cpu",
        device_id: int = 0,
        basis_representation: str | None = None,
        *,
        density_fitting: str | bool = "none",
        auxiliary_basis: str | Path | BasisSet | Sequence[Shell] | None = None,
        density_fitting_relative_threshold: float = 1.0e-10,
        density_fitting_memory_budget_bytes: int = 0,
        max_iterations: int = 100,
        energy_tolerance: float = 1.0e-10,
        density_tolerance: float = 1.0e-8,
        diis_history: int = 8,
        screening_tolerance: float = 1.0e-12,
        target_accuracy: TargetAccuracy | None = None,
        resource_budget=None,
    ) -> None:
        """Create a calculator, optionally selecting CPU or CUDA DF.

        ``density_fitting_memory_budget_bytes`` is a device-workspace hint for
        CUDA DF.  Positive values select smaller auxiliary tiles (and stream
        transformed three-center values when needed); zero uses the backend's
        default policy.

        ``target_accuracy`` requests observable diagnostics independently of
        iteration convergence. Until an explicit audit/estimator is attached,
        successful results report ``unverified`` and numerical defaults remain
        unchanged. It never certifies an error from ``energy_tolerance``.
        """
        if target_accuracy is not None and not isinstance(
            target_accuracy, TargetAccuracy
        ):
            raise TypeError("target_accuracy must be a TargetAccuracy contract")
        self._target_accuracy = target_accuracy
        if resource_budget is not None:
            from .resources import ResourceBudget

            if not isinstance(resource_budget, ResourceBudget):
                raise TypeError("resource_budget must be a ResourceBudget")
        self._resource_budget = resource_budget
        if method.lower() not in _METHODS:
            raise ValueError(f"unknown method {method!r}")
        if device not in {"cpu", "cuda"}:
            raise ValueError("device must be 'cpu' or 'cuda'")
        representations = {
            "cartesian": _native.BASIS_CARTESIAN,
            "spherical": _native.BASIS_SPHERICAL,
        }
        basis = _snapshot_basis(basis, basis_representation)
        if basis_representation is None:
            basis_representation = (
                basis.representation if isinstance(basis, BasisSet) else "cartesian"
            )
        if basis_representation not in representations:
            raise ValueError("basis_representation must be 'cartesian' or 'spherical'")
        if auxiliary_basis is not None:
            auxiliary_basis = _snapshot_basis(
                auxiliary_basis,
                None
                if isinstance(auxiliary_basis, (BasisSet, os.PathLike))
                or (
                    isinstance(auxiliary_basis, str)
                    and auxiliary_basis.endswith(".json")
                )
                else basis_representation,
            )
        if isinstance(density_fitting, bool):
            density_fitting = "cpu" if density_fitting else "none"
        density_fitting_modes = {
            "none": _native.DENSITY_FITTING_NONE,
            "cpu": _native.DENSITY_FITTING_CPU_REFERENCE,
            "cpu_reference": _native.DENSITY_FITTING_CPU_REFERENCE,
            "cuda": _native.DENSITY_FITTING_CUDA,
            "auto": _native.DENSITY_FITTING_AUTO,
        }
        try:
            density_fitting_mode = density_fitting_modes[str(density_fitting).lower()]
        except KeyError as error:
            raise ValueError(
                "density_fitting must be 'none', 'cpu', 'cuda', or 'auto'"
            ) from error
        if (
            auxiliary_basis is not None
            and density_fitting_mode == _native.DENSITY_FITTING_NONE
        ):
            raise ValueError("auxiliary_basis requires density_fitting to be enabled")
        if not (0.0 < float(density_fitting_relative_threshold) < 1.0):
            raise ValueError(
                "density_fitting_relative_threshold must lie between zero and one"
            )
        if int(density_fitting_memory_budget_bytes) < 0:
            raise ValueError("density_fitting_memory_budget_bytes must be non-negative")
        self._method = _METHODS[method.lower()]
        self._basis = basis
        self._auxiliary_basis = auxiliary_basis
        self._density_fitting_mode = density_fitting_mode
        self._density_fitting_relative_threshold = float(
            density_fitting_relative_threshold
        )
        self._density_fitting_memory_budget_bytes = int(
            density_fitting_memory_budget_bytes
        )
        self._basis_representation = representations[basis_representation]
        self._backend = (
            _native.BACKEND_CUDA if device == "cuda" else _native.BACKEND_CPU_REFERENCE
        )
        self._device_name = device
        self._representation_name = basis_representation
        self._device_id = int(device_id)
        self._max_iterations = int(max_iterations)
        self._energy_tolerance = float(energy_tolerance)
        self._density_tolerance = float(density_tolerance)
        self._diis_history = int(diis_history)
        self._screening_tolerance = float(screening_tolerance)
        if self._screening_tolerance <= 0.0:
            raise ValueError("screening_tolerance must be positive")
        self._library = _native.load_library(device=device, device_id=self._device_id)

        available = ctypes.c_int32()
        _native.check(
            self._library,
            self._library.vibeqc_method_available(
                self._method, ctypes.byref(available)
            ),
        )
        if not available.value:
            raise NotImplementedError(
                f"method {method!r} is reserved but not implemented"
            )

    @property
    def profile_diagnostics(self) -> dict:
        """Report official/local/portable selection and incompatible-cache reasons."""
        import copy

        return copy.deepcopy(
            getattr(
                self._library,
                "_vibeqc_profile_diagnostics",
                {
                    "source": "cpu",
                    "identity": None,
                    "kernels": [],
                    "rejected": [],
                },
            )
        )

    def _context_descriptor(self) -> _native.ContextDescriptor:
        return _native.ContextDescriptor(
            ctypes.sizeof(_native.ContextDescriptor),
            _native.ABI_VERSION,
            self._device_id,
            self._backend,
        )

    def _method_descriptor(
        self, auxiliary_basis: ctypes.c_void_p | None = None, *, resource_plan=None
    ) -> _native.MethodDescriptor:
        df_budget = self._density_fitting_memory_budget_bytes
        if resource_plan is not None:
            request = next(r for r in resource_plan.requests if r.name == "hf")
            chosen = dict(resource_plan.selections)["hf"]
            candidate = next(c for c in request.candidates if c.name == chosen)
            df_budget = int(
                dict(candidate.decisions).get(
                    "density_fitting_memory_budget_bytes", df_budget
                )
            )
        return _native.MethodDescriptor(
            ctypes.sizeof(_native.MethodDescriptor),
            _native.ABI_VERSION,
            self._method,
            self._max_iterations,
            self._diis_history,
            self._energy_tolerance,
            self._density_tolerance,
            self._screening_tolerance,
            self._density_fitting_mode,
            auxiliary_basis,
            self._density_fitting_relative_threshold,
            df_budget,
        )

    def _shells_for_atoms(
        self,
        atoms: Sequence[Atom],
        basis: str | Sequence[Shell] | None = None,
        *,
        operator: str = "eri",
        derivative_order: int = 0,
    ) -> tuple[Shell, ...]:
        selected_basis = self._basis if basis is None else basis
        if not isinstance(selected_basis, BasisSet):
            selected_basis = _snapshot_basis(selected_basis, self._representation_name)
        require_basis(
            selected_basis,
            atoms,
            backend=self._device_name,
            operator=operator,
            derivative_order=derivative_order,
            role="orbital" if basis is None else "auxiliary",
            representation=selected_basis.representation
            if isinstance(selected_basis, BasisSet)
            else self._representation_name,
        )
        return (
            selected_basis.shells_for(atoms)
            if isinstance(selected_basis, BasisSet)
            else tuple(selected_basis)
        )

    def _model_signature(self):
        """Detect replacement of scientific input snapshots before prepared reuse."""

        def identity(basis):
            if basis is None:
                return None
            if isinstance(basis, BasisSet):
                return basis.identity
            return canonical_hash(
                [
                    (
                        s.atom_index,
                        s.angular_momentum,
                        [
                            (float(p.exponent).hex(), float(p.coefficient).hex())
                            for p in s.primitives
                        ],
                    )
                    for s in basis
                ]
            )

        return canonical_hash(
            {
                "orbital": identity(self._basis),
                "auxiliary": identity(self._auxiliary_basis),
                "representation": self._basis_representation,
                "method": self._method,
                "density_fitting": self._density_fitting_mode,
                "df_threshold": self._density_fitting_relative_threshold,
                "screening": self._screening_tolerance,
                "energy_tolerance": self._energy_tolerance,
                "density_tolerance": self._density_tolerance,
                "max_iterations": self._max_iterations,
                "diis_history": self._diis_history,
                "target_accuracy": self._target_accuracy.to_dict()
                if self._target_accuracy
                else None,
            }
        )

    def basis_metadata(self, atoms, *, charge=0, multiplicity=1):
        """Resolve complete orbital/auxiliary scientific identities for this model."""
        atoms = tuple(Atom.from_value(a) for a in atoms)
        checked_integer(charge, "ionic charge", low=-(2**31), high=2**31 - 1)
        checked_integer(multiplicity, "multiplicity", low=1, high=2**31 - 1)
        result = {}
        for role, basis in (
            ("orbital", self._basis),
            ("auxiliary", self._auxiliary_basis),
        ):
            if basis is None:
                continue
            shells = self._shells_for_atoms(atoms, None if role == "orbital" else basis)
            mode = (
                basis.representation
                if isinstance(basis, BasisSet)
                else self._representation_name
            )
            result[role] = resolved_basis_metadata(
                basis,
                shells,
                atoms,
                representation=mode,
                charge=charge,
                multiplicity=multiplicity,
                role=role,
            )
        result["model_identity"] = canonical_hash(
            {
                "calculator": self._model_signature(),
                "basis": result,
                "charge": int(charge),
                "multiplicity": int(multiplicity),
            }
        )
        return result

    def resolved_model(self, atoms, *, charge=0, multiplicity=1) -> ResolvedModel:
        """Resolve the scientific HF identity for accuracy comparisons.

        Unlike a prepared-plan signature, this identity excludes execution
        backend, iteration tolerances, screening and schedules. Fitting and its
        actual auxiliary basis remain mathematical choices. This method only
        resolves compact basis metadata; it performs no integral/SCF work.
        """
        atoms = tuple(Atom.from_value(atom) for atom in atoms)
        self._preflight_hf_basis(atoms)
        metadata = self.basis_metadata(atoms, charge=charge, multiplicity=multiplicity)
        orbital = metadata["orbital"]
        fitted = self._density_fitting_mode != _native.DENSITY_FITTING_NONE
        # HF's native default uses the orbital system as the auxiliary system.
        # An AUTO provider may choose a backend, but never changes this model.
        auxiliary = metadata.get("auxiliary", orbital) if fitted else None
        return ResolvedModel(
            method={_native.METHOD_RHF: "rhf", _native.METHOD_UHF: "uhf"}[self._method],
            geometry_hash=canonical_hash(
                [
                    (
                        a.atomic_number,
                        tuple(float(x).hex() if x else "0x0.0p+0" for x in a.position),
                    )
                    for a in atoms
                ]
            ),
            basis_hash=orbital["mathematical_identity"],
            electron_count=orbital["electrons"]["electron_count"],
            charge=charge,
            multiplicity=multiplicity,
            representation="real_spherical"
            if self._representation_name == "spherical"
            else "cartesian",
            approximation="density_fitting" if fitted else "conventional",
            auxiliary_basis_hash=auxiliary["mathematical_identity"]
            if auxiliary
            else None,
            metric_relative_threshold=self._density_fitting_relative_threshold
            if fitted
            else None,
        )

    def _accuracy_assessment(self, atoms, charge, multiplicity, converged):
        """Attach a request without inventing operator audits or certificates."""
        if self._target_accuracy is None:
            return None
        return AccuracyAssessment(
            self.resolved_model(atoms, charge=charge, multiplicity=multiplicity),
            self._target_accuracy,
            converged=converged,
        )

    def _preflight_hf_basis(self, atoms):
        """Check the value and force operators needed by a public HF endpoint.

        Runtime shape/resource and occupation checks remain native. This data
        preflight never turns an ECP or an unsupported auxiliary shell into an
        all-electron through-f approximation.
        """
        for role, basis, operators in (
            (
                "orbital",
                self._basis,
                ("overlap", "kinetic", "nuclear_attraction", "eri"),
            ),
            ("auxiliary", self._auxiliary_basis, ("df_metric", "df_three_center")),
        ):
            if basis is None:
                continue
            for operator in operators:
                for order in (0, 1):
                    require_basis(
                        basis,
                        atoms,
                        backend=self._device_name,
                        operator=operator,
                        derivative_order=order,
                        role=role,
                        representation=basis.representation
                        if isinstance(basis, BasisSet)
                        else self._representation_name,
                    )

    def _create_native_system(
        self,
        context: ctypes.c_void_p,
        atoms: Sequence[Atom],
        charge: int,
        multiplicity: int,
        basis: str | Sequence[Shell] | None = None,
    ) -> ctypes.c_void_p:
        selected_basis = self._basis if basis is None else basis
        if not isinstance(selected_basis, BasisSet):
            selected_basis = _snapshot_basis(selected_basis, self._representation_name)
        shells = self._shells_for_atoms(atoms, basis)
        checked_integer(charge, "ionic charge", low=-(2**31), high=2**31 - 1)
        checked_integer(multiplicity, "multiplicity", low=1, high=2**31 - 1)
        checked_integer(len(atoms), "atom count", low=1, high=2**32 - 1)
        checked_integer(len(shells), "shell count", low=1, high=2**32 - 1)
        checked_integer(
            sum(len(s.primitives) for s in shells),
            "primitive count",
            low=1,
            high=2**32 - 1,
        )
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
            (
                _native.BASIS_SPHERICAL
                if selected_basis.representation == "spherical"
                else _native.BASIS_CARTESIAN
            )
            if isinstance(selected_basis, BasisSet)
            else self._basis_representation,
        )
        system = ctypes.c_void_p()
        _native.check(
            self._library,
            self._library.vibeqc_system_create(
                context, ctypes.byref(descriptor), ctypes.byref(system)
            ),
        )
        return system

    def _resource_request(self, systems, *, charges=None, multiplicities=None):
        """Resolve this calculator's exact active HF controls without executing."""
        from .resources_hf import hf_resource_request

        request = hf_resource_request(
            systems,
            charges=charges,
            multiplicities=multiplicities,
            method={_native.METHOD_RHF: "rhf", _native.METHOD_UHF: "uhf"}[self._method],
            basis=self._basis,
            auxiliary_basis=self._auxiliary_basis,
            backend=self._device_name,
            basis_representation=self._representation_name,
            density_fitting={
                _native.DENSITY_FITTING_NONE: "none",
                _native.DENSITY_FITTING_CPU_REFERENCE: "cpu",
                _native.DENSITY_FITTING_CUDA: "cuda",
                _native.DENSITY_FITTING_AUTO: "auto",
            }[self._density_fitting_mode],
            diis_history=self._diis_history,
            max_iterations=self._max_iterations,
            energy_tolerance=self._energy_tolerance,
            density_tolerance=self._density_tolerance,
            screening_tolerance=self._screening_tolerance,
            density_fitting_relative_threshold=self._density_fitting_relative_threshold,
            density_fitting_memory_budget_bytes=self._density_fitting_memory_budget_bytes,
            device_id=self._device_id,
            library=self._library,
        )
        if request.identity.backend == "cpu":
            from dataclasses import replace

            query = getattr(
                self._library, "vibeqc_cpu_resource_inventory_version_v1", None
            )
            if query is not None:
                query.argtypes = []
                query.restype = ctypes.c_int
            if query is None or query() != 1:
                return replace(
                    request,
                    unsupported_reason="native library does not implement CPU allocation inventory v1",
                )
        return request

    def estimate_resources(
        self, systems, *, charges=None, multiplicities=None, budget=None
    ):
        """Dry-run the active scientific inputs; no solve or warm-state mutation."""
        from .resources import ResourceBudget, plan_resources

        budget = self._resource_budget if budget is None else budget
        return plan_resources(
            (
                self._resource_request(
                    systems, charges=charges, multiplicities=multiplicities
                ),
            ),
            ResourceBudget() if budget is None else budget,
        )

    def prepare_batch(
        self,
        systems: Sequence[Iterable[Atom | tuple[str | int, Sequence[float]]]],
        *,
        charges: Sequence[int] | None = None,
        multiplicities: Sequence[int] | None = None,
        warm_start: bool = True,
        shell_class_profiling: bool = False,
        inactive_eigensolver_profiling: bool = False,
        resource_plan=None,
    ):  # Return annotation is deferred to avoid an import cycle.
        """Prepare a persistent native ragged batch for repeated execution.

        The profiling options are CUDA performance diagnostics and should
        remain disabled during normal endpoint timing.
        """

        from .batch import PreparedBatch

        return PreparedBatch(
            self,
            systems,
            charges=charges,
            multiplicities=multiplicities,
            warm_start=warm_start,
            shell_class_profiling=shell_class_profiling,
            inactive_eigensolver_profiling=inactive_eigensolver_profiling,
            resource_plan=resource_plan,
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
        self._preflight_hf_basis(native_atoms)
        resource_plan = None
        if self._resource_budget is not None:
            resource_plan = self.estimate_resources(
                [native_atoms], charges=[charge], multiplicities=[multiplicity]
            ).require_feasible()
        context = ctypes.c_void_p()
        _native.check(
            self._library,
            self._library.vibeqc_context_create(
                ctypes.byref(self._context_descriptor()), ctypes.byref(context)
            ),
        )
        system = ctypes.c_void_p()
        auxiliary_system = ctypes.c_void_p()
        calculation = ctypes.c_void_p()
        ledger = None
        try:
            if (
                resource_plan is not None
                and resource_plan.requests[0].identity.backend == "cuda"
            ):
                from .resources_native import NativeDeviceLedger

                ledger = NativeDeviceLedger(self._library, resource_plan)
            system = self._create_native_system(
                context, native_atoms, charge, multiplicity
            )
            if self._auxiliary_basis is not None:
                auxiliary_system = self._create_native_system(
                    context,
                    native_atoms,
                    charge,
                    multiplicity,
                    self._auxiliary_basis,
                )
            method_descriptor = self._method_descriptor(
                auxiliary_system if auxiliary_system.value else None,
                resource_plan=resource_plan,
            )
            _native.check(
                self._library,
                self._library.vibeqc_calculation_prepare(
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
            resource_diagnostics = None
            if resource_plan is None:
                status = self._library.vibeqc_calculation_execute(
                    calculation, ctypes.byref(result_descriptor)
                )
            else:
                from .resources import CpuResourceObservation

                with CpuResourceObservation(
                    self._library, cpu_workers=1, ledger=ledger
                ) as observed:
                    status = self._library.vibeqc_calculation_execute(
                        calculation, ctypes.byref(result_descriptor)
                    )
                resource_diagnostics = {
                    "plan": resource_plan.to_dict(),
                    "observation": observed.to_dict(),
                }
                observed.verify(resource_plan)
            if resource_diagnostics is None:
                _native.check(self._library, status)
            else:
                from .resources_native import check_resource_status

                check_resource_status(self._library, status, resource_diagnostics)
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
                resource_diagnostics=resource_diagnostics,
                basis_metadata=self.basis_metadata(
                    native_atoms, charge=charge, multiplicity=multiplicity
                ),
                accuracy=self._accuracy_assessment(
                    native_atoms,
                    charge,
                    multiplicity,
                    bool(result_descriptor.converged),
                ),
            )
        finally:
            if calculation.value:
                self._library.vibeqc_calculation_destroy(calculation)
            if system.value:
                self._library.vibeqc_system_destroy(system)
            if auxiliary_system.value:
                self._library.vibeqc_system_destroy(auxiliary_system)
            self._library.vibeqc_context_destroy(context)
            if ledger is not None:
                ledger.close()
