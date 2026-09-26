"""Method-oriented Python API for native mean-field calculations."""

from __future__ import annotations

import ctypes
import json
import math
import os
import typing
from dataclasses import dataclass, field, replace
from functools import cache, lru_cache
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from . import _generated_methods as _method_manifest
from . import _native
from .accuracy import AccuracyAssessment, ResolvedModel, TargetAccuracy
from .basis import BasisProvenance, BasisSet, BasisShell, ElementBasis, load_basis
from .basis_capabilities import require_basis, resolved_basis_metadata
from .elements import atomic_number as element_number
from .elements import checked_integer
from .profiles import canonical_hash

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from .ks_diagnostics import KsDiagnostic, KsTransportDiagnostic

_METHODS = _method_manifest.METHOD_NAME_TO_ID
_COMPOSITE_METHOD_ALIASES = {
    "r2scan-3c": ("R2SCAN-3c", "unpolarized"),
    "r2scan-3c-rks": ("R2SCAN-3c", "unpolarized"),
    "r2scan-3c-uks": ("R2SCAN-3c", "polarized"),
}
_HF_METHODS = _method_manifest.HF_METHOD_IDS
_COUPLED_CLUSTER_METHODS = frozenset((_native.METHOD_RCCSD, _native.METHOD_RCCSD_T))
_CORRELATED_METHODS = frozenset((_native.METHOD_MP2, *_COUPLED_CLUSTER_METHODS))


@dataclass(frozen=True)
class Atom:
    atomic_number: int
    position: tuple[float, float, float]

    def __post_init__(self) -> None:
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
class CorrelationResult:
    """Canonical MP2 components and the completed native phase diagnostics.

    Energies and denominator magnitudes are Hartree; timings are milliseconds.
    Numeric capacity counts the largest phase, excluding allocator/object
    overhead. Transfer fields describe the disclosed CUDA staging path.
    """

    reference_energy: float
    opposite_spin_energy: float
    same_spin_energy: float
    minimum_absolute_denominator: float
    reference_residual: float
    numeric_capacity_bytes: int
    energy_tile_count: int
    mo_host_staging: bool
    correlation_owned_device_bytes: int
    correlation_provider_retained_bytes: int
    mo_transfer_bytes: int
    host_to_device_ms: float
    device_to_host_ms: float
    transform_library_ms: float
    tensor_kernel_ms: float
    equation_hash: str
    response_iterations: int
    response_restarts: int
    response_absolute_residual: float
    response_relative_residual: float
    response_workspace_bytes: int
    derivative_workspace_bytes: int
    planned_endpoint_peak_bytes: int
    measured_endpoint_peak_bytes: int
    force_provenance_flags: int
    response_operator_hash: str
    measured_response_workspace_peak_bytes: int
    response_workspace_allocation_count: int
    ccsd_iterations: int
    ccsd_diis_restarts: int
    ccsd_correlation_energy: float
    ccsd_energy_change: float
    ccsd_singles_residual_max: float
    ccsd_doubles_residual_max: float
    ccsd_replay_singles_residual_max: float
    ccsd_replay_doubles_residual_max: float
    ccsd_setup_h2d_bytes: int
    ccsd_scalar_d2h_bytes: int
    ccsd_amplitude_d2h_bytes: int
    ccsd_synchronizations: int
    ccsd_replay_equation_hash: str
    ccsd_t_triples_energy: float
    ccsd_t_virtual_triples: int
    ccsd_t_workspace_bytes: int
    ccsd_t_equation_hash: str


def _read_correlation_result(
    library: ctypes.CDLL,
    owner: ctypes.c_void_p,
    *,
    index: int | None = None,
    context: ctypes.c_void_p | None = None,
) -> CorrelationResult | None:
    name = (
        "vibeqc_calculation_get_correlation_diagnostic"
        if index is None
        else "vibeqc_batch_get_correlation_diagnostic"
    )
    getter = getattr(library, name)
    diag = _native.CorrelationDiagnostic()
    diag.struct_size = ctypes.sizeof(diag)
    diag.abi_version = _native.ABI_VERSION
    args = (
        (owner, ctypes.byref(diag))
        if index is None
        else (owner, index, ctypes.byref(diag))
    )
    status = getter(*args)
    if status == _native.STATUS_NOT_IMPLEMENTED:
        return None
    _native.check(library, status, context=context)
    values = {
        name: getattr(diag, name)
        for name, _ in diag._fields_
        if name not in ("struct_size", "abi_version")
    }
    values["mo_host_staging"] = bool(values["mo_host_staging"])
    for key in (
        "equation_hash",
        "response_operator_hash",
        "ccsd_replay_equation_hash",
        "ccsd_t_equation_hash",
    ):
        values[key] = values[key].decode("ascii")
    return CorrelationResult(**values)


@dataclass(frozen=True)
class Result:
    """Calculation outputs with distinct SCF update and stationarity measures.

    ``density_rms`` retains the density-update convergence measure.
    ``physical_residual_rms`` is optional: unsupported methods or older native
    libraries report None, rather than reusing the density-update value.
    """

    energy: float
    forces: np.ndarray | None
    converged: bool
    iterations: int
    energy_change: float
    density_rms: float
    executed_backend: str
    basis_metadata: dict = field(default_factory=dict)
    accuracy: AccuracyAssessment | None = None
    resource_diagnostics: dict | None = None
    precision: dict | None = None
    correlation: CorrelationResult | None = None
    physical_residual_rms: float | None = None
    ks_diagnostic: KsDiagnostic | None = None
    ks_transport_diagnostic: KsTransportDiagnostic | None = None
    dispersion: object | None = None


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
    composite = _COMPOSITE_METHOD_ALIASES.get(canonical)
    if composite is not None:
        _, spin = composite
        electronic = "r2scan-uks" if spin == "polarized" else "r2scan-rks"
        return replace(method_capabilities(electronic), method=canonical)
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
        _native.METHOD_FAMILY_PERTURBATION: "perturbation",
        _native.METHOD_FAMILY_SEMIEMPIRICAL: "semiempirical",
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
def _named_basis_record(name: typing.Any, representation: typing.Any) -> typing.Any:
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


def _snapshot_basis(basis: typing.Any, representation: typing.Any = None) -> typing.Any:
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

    Coordinates are in Bohr. The current implementation accepts RHF, UHF, or
    energy-only LDA/PBE RKS/UKS on CPU/CUDA and
    a bundled STO-3G/def2-SVP/def2-TZVP basis for H-Ar, local canonical JSON,
    immutable `BasisSet` records, or explicit `Shell` objects. Element symbols
    cover H-Og; execution depends on every actual shell and Hamiltonian. Both the CPU reference and CUDA backend support Cartesian
    or PySCF/libcint-ordered real spherical AOs through `g` on CPU (`f` on CUDA).
    """

    def __init__(
        self,
        method: typing.Any = "rhf",
        basis: str | Path | BasisSet | Sequence[Shell] | None = None,
        device: str = "cpu",
        device_id: int = 0,
        basis_representation: str | None = None,
        *,
        density_fitting: str | bool = "none",
        auxiliary_basis: str | Path | BasisSet | Sequence[Shell] | None = None,
        density_fitting_relative_threshold: float = 1.0e-10,
        density_fitting_memory_budget_bytes: int = 0,
        correlation_memory_budget_bytes: int = 0,
        mp2_denominator_threshold: float = 1e-10,
        ccsd_max_iterations: int = 100,
        ccsd_diis_history: int = 6,
        ccsd_energy_tolerance: float = 1e-11,
        ccsd_residual_tolerance: float = 1e-9,
        ccsd_denominator_threshold: float = 1e-10,
        ccsd_damping: float = 0.0,
        ccsd_level_shift: float = 0.0,
        ccsd_frozen_core: int = 0,
        max_iterations: int = 100,
        energy_tolerance: float = 1.0e-10,
        density_tolerance: float = 1.0e-8,
        diis_history: int = 8,
        screening_tolerance: float | None = None,
        precision: str = "fp64",
        target_accuracy: TargetAccuracy | None = None,
        resource_budget: typing.Any = None,
        ks_options: typing.Any = None,
        dispersion_memory_budget_bytes: int = 256 * 1024 * 1024,
    ) -> None:
        """Create a calculator, optionally selecting CPU or CUDA DF.

        ``density_fitting_memory_budget_bytes`` is a device-workspace hint for
        CUDA DF.  Positive values select smaller auxiliary tiles (and stream
        transformed three-center values when needed); zero uses the backend's
        default policy. Energy-only DF uses the whole positive value allowance;
        force calls reserve half for response and rebuild cached value storage
        when needed. The preparation preflight also bounds its host copies.
        Use ``resource_budget`` for the composed whole-calculation inventory.

        ``target_accuracy`` requests observable diagnostics independently of
        iteration convergence. Until an explicit audit/estimator is attached,
        successful results report ``unverified`` and numerical defaults remain
        unchanged. It never certifies an error from ``energy_tolerance``.

        ``method`` may be a native selector string, the public
        ``r2scan-3c[-rks|-uks]`` composite selectors, a spin-explicit PBE-family
        MethodIR with one production D3(BJ) correction, or the canonical
        r2SCAN-3c MethodIR. The latter forms bind the exact def2-mTZVPP basis and
        composes r2SCAN + D4 + gCP without a named native scientific driver.
        ``ks_options`` snapshots the electronic composition, GridSpec and XC
        tile schedule. ``dispersion_memory_budget_bytes`` independently bounds
        the retained external-correction owner. Production two-body D3(BJ)
        also contributes its retained host/device capacity to the global
        ResourceBudget; composite D4/gCP planning remains fail-closed.
        """
        if target_accuracy is not None and not isinstance(
            target_accuracy, TargetAccuracy
        ):
            raise TypeError("target_accuracy must be a TargetAccuracy contract")
        self._target_accuracy = target_accuracy
        if resource_budget is not None:
            from vibeqc_compiler.common.resources import ResourceBudget

            if not isinstance(resource_budget, ResourceBudget):
                raise TypeError("resource_budget must be a ResourceBudget")
        self._resource_budget = resource_budget
        if (
            type(dispersion_memory_budget_bytes) is not int
            or not 0 < dispersion_memory_budget_bytes < 2**64
        ):
            raise ValueError(
                "dispersion_memory_budget_bytes must be a positive uint64 integer"
            )
        self._dispersion_memory_budget_bytes = dispersion_memory_budget_bytes
        self._dispersion_method_ir = None

        from vibeqc_compiler.method import (
            D3Spec,
            D4Spec,
            DispersionCorrectionPrimitive,
            GeometricCounterpoisePrimitive,
            MethodIR,
            SemilocalXCPrimitive,
            resolve_method,
            validate_basis_snapshot,
        )

        if isinstance(method, str):
            composite = _COMPOSITE_METHOD_ALIASES.get(method.lower())
            if composite is not None:
                identifier, spin = composite
                method = resolve_method(identifier, spin=spin)
        supplied_method_ir = method if isinstance(method, MethodIR) else None
        if supplied_method_ir is not None:
            corrections = tuple(
                node
                for node in supplied_method_ir.primitives
                if isinstance(node, DispersionCorrectionPrimitive)
            )
            gcp_nodes = tuple(
                node
                for node in supplied_method_ir.primitives
                if isinstance(node, GeometricCounterpoisePrimitive)
            )
            electronic_family = None
            if corrections:
                if len(corrections) != 1:
                    raise NotImplementedError(
                        "Calculator MethodIR execution requires exactly one supported dispersion correction"
                    )
                correction = corrections[0].specification
                if isinstance(correction, D3Spec):
                    if gcp_nodes:
                        raise NotImplementedError(
                            "Calculator D3 execution does not accept a gCP primitive"
                        )
                    electronic_family = "pbe"
                elif isinstance(correction, D4Spec):
                    expected = resolve_method("R2SCAN-3c", spin=supplied_method_ir.spin)
                    if (
                        len(gcp_nodes) != 1
                        or supplied_method_ir.manifest_identity
                        != expected.manifest_identity
                    ):
                        raise NotImplementedError(
                            "Calculator D4+gCP execution requires the canonical r2SCAN-3c MethodIR"
                        )
                    electronic_family = "r2scan"
                else:
                    raise NotImplementedError(
                        "Calculator MethodIR execution does not support this correction family"
                    )
            elif gcp_nodes:
                raise NotImplementedError(
                    "Calculator MethodIR execution does not accept gCP without its qualified composite owner"
                )

            electronic_primitives = tuple(
                node
                for node in supplied_method_ir.primitives
                if not isinstance(
                    node,
                    (
                        DispersionCorrectionPrimitive,
                        GeometricCounterpoisePrimitive,
                    ),
                )
            )
            electronic_ir = (
                supplied_method_ir
                if not corrections and not gcp_nodes
                else replace(
                    supplied_method_ir,
                    identifier=f"{supplied_method_ir.identifier}/electronic",
                    primitives=electronic_primitives,
                    basis=None,
                )
            )
            semilocal = tuple(
                node
                for node in electronic_ir.primitives
                if isinstance(node, SemilocalXCPrimitive)
            )
            components = (
                set(dict(semilocal[0].functional.components))
                if len(semilocal) == 1
                else set()
            )
            if electronic_family is None:
                if not components <= {"GGA_X_PBE", "GGA_C_PBE"}:
                    raise NotImplementedError(
                        "Calculator electronic MethodIR execution currently supports the PBE family"
                    )
                electronic_family = "pbe"

            if electronic_family == "pbe":
                if not components <= {"GGA_X_PBE", "GGA_C_PBE"}:
                    raise NotImplementedError(
                        "Calculator PBE-family MethodIR has incompatible semilocal components"
                    )
                method = (
                    "pbe-uks" if supplied_method_ir.spin == "polarized" else "pbe-rks"
                )
            else:
                if components != {"MGGA_X_R2SCAN", "MGGA_C_R2SCAN"}:
                    raise NotImplementedError(
                        "canonical r2SCAN-3c requires the audited r2SCAN electronic graph"
                    )
                method = (
                    "r2scan-uks"
                    if supplied_method_ir.spin == "polarized"
                    else "r2scan-rks"
                )

            from .ks import KsOptions

            if ks_options is None:
                ks_options = KsOptions(composition=electronic_ir)
            elif not isinstance(ks_options, KsOptions):
                raise TypeError("ks_options must be KsOptions")
            elif (
                ks_options.functional is not None or ks_options.composition is not None
            ):
                raise ValueError(
                    "a MethodIR calculator owns its KS composition; ks_options may only set execution controls"
                )
            else:
                ks_options = replace(ks_options, composition=electronic_ir)
            if corrections:
                self._dispersion_method_ir = supplied_method_ir

        if not isinstance(method, str) or method.lower() not in _METHODS:
            raise ValueError(f"unknown method {method!r}")
        if device not in {"cpu", "cuda"}:
            raise ValueError("device must be 'cpu' or 'cuda'")
        representations = {
            "cartesian": _native.BASIS_CARTESIAN,
            "spherical": _native.BASIS_SPHERICAL,
        }
        method_id = _METHODS[method.lower()]
        intrinsic_xtb_basis = method_id == _native.METHOD_GFN2_XTB
        if intrinsic_xtb_basis:
            if basis is not None:
                raise ValueError(
                    "GFN2-xTB uses its intrinsic minimal basis; omit the basis argument"
                )
            if basis_representation not in (None, "cartesian"):
                raise ValueError(
                    "GFN2-xTB does not accept a Gaussian basis representation"
                )
            if auxiliary_basis is not None:
                raise ValueError("GFN2-xTB does not accept an auxiliary Gaussian basis")
            basis_representation = "cartesian"
        else:
            if basis is None:
                if (
                    supplied_method_ir is not None
                    and supplied_method_ir.basis is not None
                ):
                    if supplied_method_ir.identifier != "R2SCAN-3c":
                        raise NotImplementedError(
                            "automatic composite basis loading is qualified only for canonical r2SCAN-3c"
                        )
                    from .r2scan3c import load_r2scan3c_basis

                    basis = load_r2scan3c_basis()
                else:
                    basis = "sto-3g"
            basis = _snapshot_basis(basis, basis_representation)
            if supplied_method_ir is not None and supplied_method_ir.basis is not None:
                validate_basis_snapshot(supplied_method_ir.basis, basis)
            if basis_representation is None:
                basis_representation = (
                    basis.representation if isinstance(basis, BasisSet) else "cartesian"
                )
            if basis_representation not in representations:
                raise ValueError(
                    "basis_representation must be 'cartesian' or 'spherical'"
                )
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
        self._method_name = method.lower()
        self._method = _METHODS[self._method_name]
        precision_modes = {
            "fp64": _native.PRECISION_FP64,
            "auto": _native.PRECISION_AUTO,
        }
        try:
            self._precision_mode = precision_modes[str(precision).lower()]
        except KeyError as error:
            raise ValueError("precision must be 'fp64' or 'auto'") from error
        if (
            self._method in (_native.METHOD_R2SCAN_RKS, _native.METHOD_R2SCAN_UKS)
            and self._precision_mode != _native.PRECISION_FP64
        ):
            raise NotImplementedError("r2SCAN currently requires strict FP64")
        if (
            self._method == _native.METHOD_PBE_D4_RKS
            and self._precision_mode != _native.PRECISION_FP64
        ):
            raise NotImplementedError("PBE-D4 currently requires strict FP64")
        self._ks_options = None
        if self._method in _method_manifest.NATIVE_DFT_METHOD_IDS:
            from .ks import KsOptions, resolve_ks_options

            if supplied_method_ir is None and isinstance(ks_options, KsOptions):
                full_graph = ks_options.composition or ks_options._method_ir
                if full_graph is not None:
                    corrections = tuple(
                        node
                        for node in full_graph.primitives
                        if isinstance(node, DispersionCorrectionPrimitive)
                    )
                    # Leave non-D3 recipes to the existing native resolver;
                    # explicit/resolved PBE-D4 options already have an owner.
                    if any(
                        isinstance(node.specification, D3Spec) for node in corrections
                    ):
                        if len(corrections) != 1 or not isinstance(
                            corrections[0].specification, D3Spec
                        ):
                            raise NotImplementedError(
                                "Calculator supports exactly one D3 correction primitive"
                            )
                        electronic_graph = replace(
                            full_graph,
                            identifier=f"{full_graph.identifier}/electronic",
                            primitives=tuple(
                                node
                                for node in full_graph.primitives
                                if not isinstance(node, DispersionCorrectionPrimitive)
                            ),
                        )
                        ks_options = replace(
                            ks_options,
                            functional=None,
                            composition=electronic_graph,
                        )
                        self._dispersion_method_ir = full_graph
            self._ks_options = resolve_ks_options(self._method_name, ks_options)
        elif ks_options is not None:
            raise ValueError("ks_options requires a supported RKS/UKS method")
        if self._dispersion_method_ir is not None and resource_budget is not None:
            correction_nodes = tuple(
                node
                for node in self._dispersion_method_ir.primitives
                if isinstance(node, DispersionCorrectionPrimitive)
            )
            if len(correction_nodes) != 1 or not isinstance(
                correction_nodes[0].specification, D3Spec
            ):
                raise NotImplementedError(
                    "global resource_budget currently supports composed D3(BJ) only; "
                    "composite D4/gCP planning remains unavailable"
                )
        if self._method == _native.METHOD_GFN2_XTB:
            # Backend-specific admission is owned by native calculation preparation.
            # Native SDK builds may include GFN2 CUDA while CUDA wheels currently do not.
            if density_fitting_mode != _native.DENSITY_FITTING_NONE:
                raise ValueError("GFN2-xTB does not use Gaussian density fitting")
            if target_accuracy is not None:
                raise NotImplementedError(
                    "target_accuracy is not implemented for GFN2-xTB yet"
                )
            if resource_budget is not None:
                raise NotImplementedError(
                    "resource_budget planning is not implemented for GFN2-xTB yet"
                )
            for name, value in (
                ("max_iterations", max_iterations),
                ("diis_history", diis_history),
            ):
                if type(value) is not int or not 1 <= value <= 2**31 - 1:
                    raise ValueError(f"{name} must be a positive int32 for GFN2-xTB")
            if diis_history > 64:
                raise ValueError("GFN2-xTB mixer history must not exceed 64")
        if self._method in _CORRELATED_METHODS:
            if target_accuracy is not None:
                raise NotImplementedError(
                    "target_accuracy is not implemented for canonical correlated methods"
                )
            if resource_budget is not None:
                raise NotImplementedError(
                    "resource_budget planning is not implemented for canonical correlated methods"
                )
            for name, value in (
                ("max_iterations", max_iterations),
                ("diis_history", diis_history),
            ):
                if type(value) is not int or not 1 <= value <= 2**32 - 1:
                    raise ValueError(
                        f"{name} must be a positive uint32 for canonical correlated methods"
                    )
        if (
            type(correlation_memory_budget_bytes) is not int
            or not 0 <= correlation_memory_budget_bytes < 2**63
        ):
            raise ValueError(
                "correlation_memory_budget_bytes must be a nonnegative signed-64-bit integer"
            )
        if not np.isfinite(mp2_denominator_threshold) or mp2_denominator_threshold <= 0:
            raise ValueError("mp2_denominator_threshold must be finite and positive")
        if self._method in _COUPLED_CLUSTER_METHODS:
            for name, value in (
                ("ccsd_max_iterations", ccsd_max_iterations),
                ("ccsd_diis_history", ccsd_diis_history),
            ):
                if type(value) is not int or not 0 <= value <= 2**32 - 1:
                    raise ValueError(f"{name} must be a non-negative uint32 for RCCSD")
            if ccsd_max_iterations == 0:
                raise ValueError("ccsd_max_iterations must be positive for RCCSD")
            if ccsd_diis_history == 1 or ccsd_diis_history > 20:
                raise ValueError("ccsd_diis_history must be 0 or 2..20 for RCCSD")
            for name, value, upper in (
                ("ccsd_energy_tolerance", ccsd_energy_tolerance, 1e-8),
                ("ccsd_residual_tolerance", ccsd_residual_tolerance, 1e-9),
            ):
                if not np.isfinite(value) or not 0.0 < value <= upper:
                    raise ValueError(
                        f"{name} must be finite, positive, and <= {upper:g}"
                    )
            if (
                not np.isfinite(ccsd_denominator_threshold)
                or ccsd_denominator_threshold <= 0.0
            ):
                raise ValueError(
                    "ccsd_denominator_threshold must be finite and positive"
                )
            if not np.isfinite(ccsd_damping) or not 0.0 <= ccsd_damping < 1.0:
                raise ValueError("ccsd_damping must be finite and in [0, 1)")
            if not np.isfinite(ccsd_level_shift) or ccsd_level_shift < 0.0:
                raise ValueError("ccsd_level_shift must be finite and non-negative")
            if type(ccsd_frozen_core) is not int or ccsd_frozen_core < 0:
                raise ValueError("ccsd_frozen_core must be a non-negative integer")
            if ccsd_frozen_core:
                raise NotImplementedError(
                    "native coupled-cluster frozen-core references are not implemented"
                )
        self._correlation_memory_budget_bytes = correlation_memory_budget_bytes
        self._mp2_denominator_threshold = float(mp2_denominator_threshold)
        self._ccsd_max_iterations = int(ccsd_max_iterations)
        self._ccsd_diis_history = int(ccsd_diis_history)
        self._ccsd_energy_tolerance = float(ccsd_energy_tolerance)
        self._ccsd_residual_tolerance = float(ccsd_residual_tolerance)
        self._ccsd_denominator_threshold = float(ccsd_denominator_threshold)
        self._ccsd_damping = float(ccsd_damping)
        self._ccsd_level_shift = float(ccsd_level_shift)
        self._ccsd_frozen_core = int(ccsd_frozen_core)
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
        if screening_tolerance is None:
            screening_tolerance = 0.0 if self._method in _CORRELATED_METHODS else 1e-12
        self._screening_tolerance = float(screening_tolerance)
        if self._method in _CORRELATED_METHODS:
            if self._screening_tolerance != 0:
                raise ValueError(
                    "canonical correlated methods require screening_tolerance=0"
                )
        elif self._screening_tolerance <= 0.0:
            raise ValueError("screening_tolerance must be positive")
        if (
            self._method in _CORRELATED_METHODS
            and self._precision_mode != _native.PRECISION_FP64
        ):
            raise ValueError("canonical correlated methods require precision='fp64'")
        self._library = _native.load_library(device=device, device_id=self._device_id)
        self._ks_options_version = 0
        if self._ks_options is not None:
            query = self._library.vibeqc_ks_options_version
            query.argtypes, query.restype = [], ctypes.c_uint32
            self._ks_options_version = query()
            from .ks import resolve_ks_options

            if self._ks_options_version != 1:
                raise NotImplementedError(
                    "native library does not support the current semantic KS execution-plan ABI"
                )

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
        self._capabilities = method_capabilities(self._method_name)
        from ._cpu_force_resources import qualified_basis

        basis_has_ecp = isinstance(self._basis, BasisSet) and any(
            element.ecp_core_electrons for element in self._basis.elements
        )
        named_cpu_all_electron_force = (
            self._device_name == "cpu"
            and self._method_name
            in (
                "pbe0-rks",
                "pbe0-uks",
                "b3lyp-rks",
                "b3lyp-uks",
                "pbe-d4-rks",
                "wb97m-v",
                "wb97m-v-rks",
                "wb97m-v-uks",
            )
            and not basis_has_ecp
            and self._ks_options is not None
            and (
                self._ks_options.coefficients[2] < 0.0
                or (
                    self._method_name == "pbe-d4-rks"
                    and self._ks_options.coefficients == (1.0, 1.0, 0.0)
                )
            )
        )
        if self._method_name.startswith("wb97m-v"):
            named_cpu_all_electron_force = named_cpu_all_electron_force and (
                self._basis == "sto-3g"
                if isinstance(self._basis, str)
                else all(
                    shell.angular_momentum <= 1
                    for element in self._basis.elements
                    for shell in element.shells
                )
            )
        semilocal_force = (
            self._ks_options is not None
            and self._ks_options.coefficients == (1.0, 1.0, 0.0)
            and not (
                self._device_name == "cuda"
                and self._ks_options.execution_plan.nonlocal_correlation is not None
            )
            and not (self._method_name == "pbe-d4-rks" and basis_has_ecp)
            and (
                self._device_name == "cuda"
                or (self._device_name == "cpu" and qualified_basis(self._basis))
            )
        )
        cuda_wb97mv_force = (
            self._device_name == "cuda"
            and self._method_name.startswith("wb97m-v")
            and not basis_has_ecp
            and self._precision_mode == _native.PRECISION_FP64
            and self._ks_options is not None
            and (
                self._basis in ("sto-3g", "def2-svp")
                if isinstance(self._basis, str)
                else all(
                    shell.angular_momentum <= 2
                    for element in self._basis.elements
                    for shell in element.shells
                )
            )
        )
        if (
            self._capabilities.family == "density_functional"
            and density_fitting_mode == _native.DENSITY_FITTING_NONE
            and (semilocal_force or named_cpu_all_electron_force or cuda_wb97mv_force)
            and not (
                self._device_name == "cuda"
                and basis_has_ecp
                and not qualified_basis(self._basis)
            )
            and self._method in _method_manifest.NATIVE_DFT_METHOD_IDS
        ):
            # Python public capability layered on the native KS prepared owner
            # plus the backend's compiled stationary gradient consumer.
            # Keep the backend-neutral C registry conservative.
            # ECP promotion admits s/p/d on CPU and s/p on CUDA, in both layouts. The shared
            # nine-source consumer also enforces shape, byte and work caps;
            # higher-angular ECP domains remain energy-only.
            self._capabilities = replace(
                self._capabilities,
                supported_properties=self._capabilities.supported_properties
                | {"forces"},
            )
        if self._method in _COUPLED_CLUSTER_METHODS:
            if density_fitting_mode != _native.DENSITY_FITTING_NONE:
                raise NotImplementedError(
                    "native coupled-cluster density fitting is not implemented"
                )
            if auxiliary_basis is not None:
                raise ValueError(
                    "conventional coupled-cluster methods do not accept an auxiliary basis"
                )
        if self._capabilities.family == "density_functional":
            if (
                self._precision_mode == _native.PRECISION_AUTO
                and self._device_name != "cuda"
            ):
                raise NotImplementedError(
                    "DFT automatic precision currently requires CUDA"
                )
            if density_fitting_mode != _native.DENSITY_FITTING_NONE:
                # DF changes the Hamiltonian. Keep its backend explicit and do
                # not advertise the conventional stationary force consumer.
                if self._precision_mode != _native.PRECISION_FP64:
                    raise NotImplementedError(
                        "DFT density fitting requires precision='fp64'"
                    )
                if (
                    density_fitting_mode == _native.DENSITY_FITTING_CPU_REFERENCE
                    and device != "cpu"
                ) or (
                    density_fitting_mode == _native.DENSITY_FITTING_CUDA
                    and device != "cuda"
                ):
                    raise ValueError(
                        "DFT density-fitting backend must match device; use 'auto' to follow it"
                    )
            if target_accuracy is not None:
                raise NotImplementedError(
                    "DFT accuracy-model identities are not implemented yet"
                )

    @property
    def method_ir(self) -> typing.Any:
        """Resolved full KS MethodIR, including an external D3 correction when present."""
        if self._dispersion_method_ir is not None:
            return self._dispersion_method_ir
        return None if self._ks_options is None else self._ks_options.method_ir

    @property
    def ks_options(self) -> typing.Any:
        """Resolved immutable KS model, or None for another method family."""
        return self._ks_options

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
                    "target": None,
                    "kernels": [],
                    "dft_schedules": [],
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
        self,
        auxiliary_basis: ctypes.c_void_p | None = None,
        *,
        resource_plan: typing.Any = None,
        ks_options: typing.Any = None,
    ) -> _native.MethodDescriptor:
        df_budget = self._density_fitting_memory_budget_bytes
        if resource_plan is not None and self._method in _HF_METHODS:
            request = next(r for r in resource_plan.requests if r.name == "hf")
            chosen = dict(resource_plan.selections)["hf"]
            candidate = next(c for c in request.candidates if c.name == chosen)
            df_budget = int(
                dict(candidate.decisions).get(
                    "density_fitting_memory_budget_bytes", df_budget
                )
            )
        descriptor = _native.MethodDescriptor(
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
            self._precision_mode,
            self._correlation_memory_budget_bytes,
            self._mp2_denominator_threshold,
        )
        active_ks_options = self._ks_options if ks_options is None else ks_options
        if active_ks_options is not None and self._ks_options_version >= 1:
            from .ks import native_ks_options

            descriptor.ks_options = ctypes.pointer(native_ks_options(active_ks_options))
        if self._method in _COUPLED_CLUSTER_METHODS:
            descriptor.ccsd_max_iterations = self._ccsd_max_iterations
            descriptor.ccsd_diis_history = self._ccsd_diis_history
            descriptor.ccsd_energy_tolerance = self._ccsd_energy_tolerance
            descriptor.ccsd_residual_tolerance = self._ccsd_residual_tolerance
            descriptor.ccsd_denominator_threshold = self._ccsd_denominator_threshold
            descriptor.ccsd_damping = self._ccsd_damping
            descriptor.ccsd_level_shift = self._ccsd_level_shift
            descriptor.ccsd_frozen_core = self._ccsd_frozen_core
        return descriptor

    def _precision_provenance(
        self, calculation: ctypes.c_void_p, index: int | None = None
    ) -> dict | None:
        """Read a calculation's policy, or an input-indexed batch item's policy.

        Returns ``None`` only when no completed execution has populated the
        record yet (the native getter reports
        :data:`STATUS_PRECISION_UNAVAILABLE`).
        """
        name = (
            "vibeqc_calculation_get_precision_provenance"
            if index is None
            else "vibeqc_batch_get_precision_provenance"
        )
        getter = getattr(self._library, name)
        provenance = _native.PrecisionProvenance(
            ctypes.sizeof(_native.PrecisionProvenance), _native.ABI_VERSION
        )
        arguments = (calculation,) if index is None else (calculation, index)
        status = getter(*arguments, ctypes.byref(provenance))
        if status == _native.STATUS_PRECISION_UNAVAILABLE:
            return None
        _native.check(self._library, status)
        return {
            "policy_version": provenance.policy_version,
            "requested_mode": (
                "auto"
                if provenance.requested_mode == _native.PRECISION_AUTO
                else "fp64"
            ),
            "effective_bits": provenance.effective_bits,
            "mixed_precision_fock_threshold": provenance.mixed_precision_fock_threshold,
            "strict_refinement_applied": bool(provenance.strict_refinement_applied),
            "mixed_precision_reserved_error": (
                provenance.mixed_precision_reserved_error
            ),
            "refinement_iterations": provenance.refinement_iterations,
            "mixed_stage_fock_builds": provenance.mixed_stage_fock_builds,
            "strict_stage_fock_builds": provenance.strict_stage_fock_builds,
            "post_scf_fock_builds": provenance.post_scf_fock_builds,
            "execution_retries": provenance.execution_retries,
            "mixed_admission_census": provenance.mixed_admission_census,
            "final_residual_audits": provenance.final_residual_audits,
            "skipped_final_fock_builds": provenance.skipped_final_fock_builds,
            "operator_work_counters_valid": bool(
                provenance.operator_work_counters_valid
            ),
        }

    def _shells_for_atoms(
        self,
        atoms: Sequence[Atom],
        basis: str | Sequence[Shell] | None = None,
        *,
        operator: str | None = None,
        derivative_order: int = 0,
    ) -> tuple[Shell, ...]:
        selected_basis = self._basis if basis is None else basis
        if not isinstance(selected_basis, BasisSet):
            selected_basis = _snapshot_basis(selected_basis, self._representation_name)
        auxiliary = basis is not None
        require_basis(
            selected_basis,
            atoms,
            backend=self._density_fitting_backend() if auxiliary else self._device_name,
            operator=operator or ("df_metric" if auxiliary else "eri"),
            derivative_order=derivative_order,
            role="auxiliary" if auxiliary else "orbital",
            representation=selected_basis.representation
            if isinstance(selected_basis, BasisSet)
            else self._representation_name,
        )
        return (
            selected_basis.shells_for(atoms)
            if isinstance(selected_basis, BasisSet)
            else tuple(selected_basis)
        )

    def _density_fitting_backend(self) -> typing.Any:
        if self._density_fitting_mode == _native.DENSITY_FITTING_CPU_REFERENCE:
            return "cpu"
        if self._density_fitting_mode == _native.DENSITY_FITTING_CUDA:
            return "cuda"
        return self._device_name

    def _model_signature(self) -> typing.Any:
        """Detect replacement of scientific input snapshots before prepared reuse."""

        def identity(basis: typing.Any) -> typing.Any:
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
                **(
                    {"ks_options": self._ks_options.to_payload()}
                    if self._ks_options
                    else {}
                ),
                **(
                    {
                        "dispersion_method_ir": self._dispersion_method_ir.to_payload(),
                        "dispersion_memory_budget_bytes": self._dispersion_memory_budget_bytes,
                    }
                    if self._dispersion_method_ir is not None
                    else {}
                ),
                **(
                    {
                        "correlation_memory_budget_bytes": self._correlation_memory_budget_bytes,
                        "mp2_denominator_threshold": self._mp2_denominator_threshold,
                    }
                    if self._method == _native.METHOD_MP2
                    else {}
                ),
                **(
                    {
                        "correlation_memory_budget_bytes": self._correlation_memory_budget_bytes,
                        "ccsd_max_iterations": self._ccsd_max_iterations,
                        "ccsd_diis_history": self._ccsd_diis_history,
                        "ccsd_energy_tolerance": self._ccsd_energy_tolerance,
                        "ccsd_residual_tolerance": self._ccsd_residual_tolerance,
                        "ccsd_denominator_threshold": self._ccsd_denominator_threshold,
                        "ccsd_damping": self._ccsd_damping,
                        "ccsd_level_shift": self._ccsd_level_shift,
                        "ccsd_frozen_core": self._ccsd_frozen_core,
                    }
                    if self._method in _COUPLED_CLUSTER_METHODS
                    else {}
                ),
                "density_fitting": self._density_fitting_mode,
                "df_threshold": self._density_fitting_relative_threshold,
                "screening": self._screening_tolerance,
                "energy_tolerance": self._energy_tolerance,
                "density_tolerance": self._density_tolerance,
                "max_iterations": self._max_iterations,
                "diis_history": self._diis_history,
                "precision": self._precision_mode,
                "target_accuracy": self._target_accuracy.to_dict()
                if self._target_accuracy
                else None,
            }
        )

    def basis_metadata(
        self, atoms: typing.Any, *, charge: typing.Any = 0, multiplicity: typing.Any = 1
    ) -> typing.Any:
        """Resolve complete orbital/auxiliary scientific identities for this model."""
        atoms = tuple(Atom.from_value(a) for a in atoms)
        checked_integer(charge, "ionic charge", low=-(2**31), high=2**31 - 1)
        checked_integer(multiplicity, "multiplicity", low=1, high=2**31 - 1)
        if self._method == _native.METHOD_GFN2_XTB:
            intrinsic = {
                "name": "GFN2-xTB intrinsic minimal basis",
                "parameter_source": "xtbloom@5a67cc59ace94c8296e873503b2ae1298e7c2861",
                "element_domain": [1, 86],
            }
            return {
                "intrinsic_basis": intrinsic,
                "model_identity": canonical_hash(
                    {
                        "method": "gfn2-xtb",
                        "intrinsic_basis": intrinsic,
                        "atoms": [a.atomic_number for a in atoms],
                        "charge": int(charge),
                        "multiplicity": int(multiplicity),
                    }
                ),
            }
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

    def resolved_model(
        self, atoms: typing.Any, *, charge: typing.Any = 0, multiplicity: typing.Any = 1
    ) -> ResolvedModel:
        """Resolve the scientific HF or MP2 identity for comparisons.

        Unlike a prepared-plan signature, this identity excludes execution
        backend, iteration tolerances, screening and schedules. Fitting and its
        actual auxiliary basis remain mathematical choices. This method only
        resolves compact basis metadata; it performs no integral/SCF work.
        """
        if self._method not in (*_HF_METHODS, _native.METHOD_MP2):
            raise NotImplementedError("accuracy model is unavailable for this method")
        atoms = tuple(Atom.from_value(atom) for atom in atoms)
        self._preflight_hf_basis(atoms)
        metadata = self.basis_metadata(atoms, charge=charge, multiplicity=multiplicity)
        orbital = metadata["orbital"]
        from .ecp import resolve_ecp

        if any(resolve_ecp(self._basis, atoms)[0]):
            # ResolvedModel currently encodes an all-electron Hamiltonian.
            # Do not label a core-replaced calculation as that different model.
            raise NotImplementedError(
                "ECP accuracy model resolution is not implemented"
            )
        fitted = self._density_fitting_mode != _native.DENSITY_FITTING_NONE
        # HF's native default uses the orbital system as the auxiliary system.
        # An AUTO provider may choose a backend, but never changes this model.
        auxiliary = metadata.get("auxiliary", orbital) if fitted else None
        try:
            resolved_method = _method_manifest.METHOD_ID_TO_NAME[self._method]
        except KeyError as error:
            raise NotImplementedError(
                "accuracy model is unavailable for this method"
            ) from error
        return ResolvedModel(
            method=resolved_method,
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

    def _accuracy_assessment(
        self,
        atoms: typing.Any,
        charge: typing.Any,
        multiplicity: typing.Any,
        converged: typing.Any,
    ) -> typing.Any:
        """Attach a request without inventing operator audits or certificates."""
        if self._target_accuracy is None:
            return None
        return AccuracyAssessment(
            self.resolved_model(atoms, charge=charge, multiplicity=multiplicity),
            self._target_accuracy,
            converged=converged,
        )

    def _preflight_hf_basis(
        self, atoms: typing.Any, *, compute_forces: typing.Any = True
    ) -> None:
        """Check operators and AO jets needed by the selected mean-field outputs.

        Runtime shape/resource and occupation checks remain native. GFN2-xTB
        owns an intrinsic minimal basis and deliberately bypasses Gaussian
        basis capability checks.
        """
        if self._method == _native.METHOD_GFN2_XTB:
            return
        if self._dispersion_method_ir is not None:
            self._dispersion_method_ir.preflight_atomic_numbers(
                tuple(atom.atomic_number for atom in atoms)
            )
        derivative_orders = (0, 1) if compute_forces else (0,)
        auxiliary_backend = self._density_fitting_backend()
        orbital_operators = ["overlap", "kinetic", "nuclear_attraction", "eri"]
        if self._capabilities.family == "density_functional":
            orbital_operators.append("ao")
        for role, basis, operators in (
            (
                "orbital",
                self._basis,
                tuple(orbital_operators),
            ),
            ("auxiliary", self._auxiliary_basis, ("df_metric", "df_three_center")),
        ):
            if basis is None:
                continue
            for operator in operators:
                # GGA energies consume first AO jets even when nuclear forces
                # are unavailable. LDA requires only AO values; integral
                # derivatives remain controlled by the requested observable.
                orders = derivative_orders
                if operator == "ao":
                    orders = (self._ks_options.ao_order,)
                for order in orders:
                    require_basis(
                        basis,
                        atoms,
                        backend=auxiliary_backend
                        if role == "auxiliary"
                        else self._device_name,
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
        if self._method == _native.METHOD_GFN2_XTB:
            if basis is not None:
                raise ValueError("GFN2-xTB does not accept a Gaussian basis")
            checked_integer(charge, "ionic charge", low=-(2**31), high=2**31 - 1)
            checked_integer(multiplicity, "multiplicity", low=1, high=2**31 - 1)
            checked_integer(len(atoms), "atom count", low=1, high=2**32 - 1)
            atom_array = (_native.AtomDescriptor * len(atoms))(
                *(
                    _native.AtomDescriptor(atom.atomic_number, *atom.position)
                    for atom in atoms
                )
            )
            descriptor = _native.SystemDescriptor(
                ctypes.sizeof(_native.SystemDescriptor),
                _native.ABI_VERSION,
                atom_array,
                len(atom_array),
                None,
                0,
                None,
                0,
                int(charge),
                int(multiplicity),
                _native.BASIS_CARTESIAN,
            )
            system = ctypes.c_void_p()
            _native.check(
                self._library,
                self._library.vibeqc_system_create(
                    context, ctypes.byref(descriptor), ctypes.byref(system)
                ),
                context=context,
            )
            return system
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
        from .ecp import resolve_ecp

        cores, ecp_terms = resolve_ecp(selected_basis, atoms)
        if ecp_terms:
            if self._density_fitting_mode != _native.DENSITY_FITTING_NONE:
                raise NotImplementedError(
                    "ECP density-fitting execution is not yet validated"
                )
            core_array = (ctypes.c_int32 * len(cores))(*cores)
            term_array = (_native.EcpTermDescriptor * len(ecp_terms))(
                *(_native.EcpTermDescriptor(*t) for t in ecp_terms)
            )
            _native.check(
                self._library,
                self._library.vibeqc_system_create_ecp(
                    context,
                    ctypes.byref(descriptor),
                    core_array,
                    term_array,
                    len(term_array),
                    ctypes.byref(system),
                ),
            )
            return system
        _native.check(
            self._library,
            self._library.vibeqc_system_create(
                context, ctypes.byref(descriptor), ctypes.byref(system)
            ),
            context=context,
        )
        return system

    def _resource_request(
        self,
        systems: typing.Any,
        *,
        charges: typing.Any = None,
        multiplicities: typing.Any = None,
        ks_options: typing.Any = None,
    ) -> typing.Any:
        """Resolve this calculator's active scientific controls without executing."""
        if self._capabilities.family == "density_functional":
            if self._density_fitting_mode != _native.DENSITY_FITTING_NONE:
                raise NotImplementedError(
                    "DFT density-fitting resource plans are not qualified; "
                    "use density_fitting_memory_budget_bytes for the native DF provider"
                )
            from .resources_ks import ks_resource_request

            return ks_resource_request(
                systems,
                charges=charges,
                multiplicities=multiplicities,
                method=self._method_name,
                basis=self._basis,
                backend=self._device_name,
                precision={
                    _native.PRECISION_FP64: "fp64",
                    _native.PRECISION_AUTO: "auto",
                }[self._precision_mode],
                basis_representation=self._representation_name,
                diis_history=self._diis_history,
                max_iterations=self._max_iterations,
                energy_tolerance=self._energy_tolerance,
                density_tolerance=self._density_tolerance,
                screening_tolerance=self._screening_tolerance,
                ks_options=self._ks_options if ks_options is None else ks_options,
                device_id=self._device_id,
                library=self._library,
            )
        if self._method == _native.METHOD_MP2:
            raise NotImplementedError(
                "resource planning is not implemented for canonical MP2"
            )
        if self._method not in _HF_METHODS:
            raise NotImplementedError(
                "resource estimation currently supports Hartree-Fock methods only"
            )
        from .resources_hf import hf_resource_request

        request = hf_resource_request(
            systems,
            charges=charges,
            multiplicities=multiplicities,
            method=_method_manifest.METHOD_ID_TO_NAME[self._method],
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
            precision={
                _native.PRECISION_FP64: "fp64",
                _native.PRECISION_AUTO: "auto",
            }[self._precision_mode],
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

    def _dispersion_resource_request(self, systems: typing.Any) -> typing.Any:
        """Return the bounded D3 owner request for global planning, when present."""
        if self._dispersion_method_ir is None:
            return None

        from vibeqc_compiler.method import D3Spec, DispersionCorrectionPrimitive

        corrections = tuple(
            node
            for node in self._dispersion_method_ir.primitives
            if isinstance(node, DispersionCorrectionPrimitive)
        )
        if len(corrections) != 1 or not isinstance(
            corrections[0].specification, D3Spec
        ):
            raise NotImplementedError(
                "global ResourcePlan currently supports composed D3(BJ) only"
            )

        from .resources_d3 import d3_resource_request

        return d3_resource_request(
            tuple(tuple(atom.atomic_number for atom in system) for system in systems),
            method=self._dispersion_method_ir,
            backend=self._device_name,
            device_id=self._device_id,
            maximum_bytes=self._dispersion_memory_budget_bytes,
        )

    def _effective_ks_selection(
        self,
        systems: typing.Any,
        *,
        charges: typing.Any = None,
        multiplicities: typing.Any = None,
    ) -> typing.Any:
        """Resolve and ABI-check one batch-local DFT09 profile selection."""

        from .ks import ProfiledKsSelection

        if (
            self._ks_options is None
            or self._device_name != "cuda"
            or self._precision_mode != _native.PRECISION_FP64
        ):
            return ProfiledKsSelection(self._ks_options)
        count = len(systems)
        charges = tuple(0 for _ in range(count)) if charges is None else tuple(charges)
        multiplicities = (
            tuple(1 for _ in range(count))
            if multiplicities is None
            else tuple(multiplicities)
        )
        if len(charges) != count or len(multiplicities) != count:
            raise ValueError("charges and multiplicities must match the batch size")
        from .ks import native_ks_options, profiled_ks_selection

        selection = profiled_ks_selection(
            self._ks_options,
            self.profile_diagnostics,
            systems,
            charges=charges,
            multiplicities=multiplicities,
        )
        if selection.options is not None:
            if self._ks_options_version != 1:
                raise NotImplementedError(
                    "native library does not support the current semantic KS execution-plan ABI"
                )
            native_ks_options(selection.options)
        return selection

    def _effective_ks_options(
        self,
        systems: typing.Any,
        *,
        charges: typing.Any = None,
        multiplicities: typing.Any = None,
    ) -> typing.Any:
        """Resolve an exact local DFT09 schedule for this batch without mutation."""

        return self._effective_ks_selection(
            systems,
            charges=charges,
            multiplicities=multiplicities,
        ).options

    def estimate_resources(
        self,
        systems: typing.Any,
        *,
        charges: typing.Any = None,
        multiplicities: typing.Any = None,
        budget: typing.Any = None,
    ) -> typing.Any:
        """Dry-run the active scientific inputs; no solve or warm-state mutation."""
        from vibeqc_compiler.common.resources import ResourceBudget, plan_resources

        systems = tuple(
            tuple(Atom.from_value(atom) for atom in system) for system in systems
        )
        count = len(systems)
        charges = tuple(0 for _ in range(count)) if charges is None else tuple(charges)
        multiplicities = (
            tuple(1 for _ in range(count))
            if multiplicities is None
            else tuple(multiplicities)
        )
        if len(charges) != count or len(multiplicities) != count:
            raise ValueError("charges and multiplicities must match the batch size")
        effective_ks_options = self._effective_ks_options(
            systems,
            charges=charges,
            multiplicities=multiplicities,
        )
        budget = self._resource_budget if budget is None else budget
        requests = [
            self._resource_request(
                systems,
                charges=charges,
                multiplicities=multiplicities,
                ks_options=effective_ks_options,
            )
        ]
        dispersion_request = self._dispersion_resource_request(systems)
        if dispersion_request is not None:
            requests.append(dispersion_request)
        return plan_resources(
            tuple(requests),
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
        resource_plan: typing.Any = None,
    ) -> typing.Any:  # Return annotation is deferred to avoid an import cycle.
        """Prepare a persistent native ragged batch for repeated execution.

        The profiling options are CUDA performance diagnostics and should
        remain disabled during normal endpoint timing.
        """

        if not self._capabilities.supports_batch:
            raise NotImplementedError(
                f"method {self._method_name!r} does not support prepared batches"
            )

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
    ) -> typing.Any:
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
        properties: Iterable[str] | None = None,
    ) -> Result:
        """Compute energy and optional analytic forces for one system.

        Energy and convergence diagnostics are always returned. Select
        ``properties=("energy",)`` to omit analytic-force evaluation; the
        returned ``Result.forces`` is then ``None``. By default each method
        requests all properties reported by its native capability record.
        """
        if properties is None:
            properties = (
                frozenset({"energy"})
                if self._method == _native.METHOD_MP2
                and self._density_fitting_mode != _native.DENSITY_FITTING_NONE
                else self._capabilities.supported_properties
            )
        if isinstance(properties, (str, bytes)):
            raise TypeError("properties must be an iterable of property names")
        try:
            requested_properties = frozenset(properties)
        except TypeError as error:
            raise TypeError(
                "properties must contain hashable property names"
            ) from error
        supported_properties = self._capabilities.supported_properties
        if not requested_properties or "energy" not in requested_properties:
            raise ValueError("properties must include 'energy'")
        unknown_properties = requested_properties - {"energy", "forces"}
        if unknown_properties:
            names = ", ".join(sorted(repr(name) for name in unknown_properties))
            raise ValueError(
                f"method {self._method_name!r} does not support properties: {names}"
            )
        unsupported_properties = requested_properties - supported_properties
        if unsupported_properties:
            names = ", ".join(sorted(unsupported_properties))
            raise ValueError(
                f"method {self._method_name!r} does not support properties: {names}"
            )
        compute_forces = "forces" in requested_properties
        native_atoms = tuple(Atom.from_value(atom) for atom in atoms)
        if not native_atoms:
            raise ValueError("at least one atom is required")
        self._preflight_hf_basis(native_atoms, compute_forces=compute_forces)
        effective_ks_options = self._effective_ks_options(
            (native_atoms,),
            charges=(charge,),
            multiplicities=(multiplicity,),
        )
        resource_plan = None
        if self._resource_budget is not None:
            resource_plan = self.estimate_resources(
                [native_atoms], charges=[charge], multiplicities=[multiplicity]
            ).require_feasible()
        if self._dispersion_method_ir is not None or (
            compute_forces and self._capabilities.family == "density_functional"
        ):
            # Reuse the prepared-batch owner because the stationary snapshot ABI
            # is intentionally tied to a live native owner.  This avoids a second
            # scientific implementation in the single-system path.
            with self.prepare_batch(
                [native_atoms],
                charges=[charge],
                multiplicities=[multiplicity],
                warm_start=False,
                resource_plan=resource_plan,
            ) as batch:
                item = batch.execute(
                    strict=True, properties=requested_properties
                ).items[0]
                return Result(
                    energy=item.energy,
                    forces=item.forces,
                    converged=item.converged,
                    iterations=item.iterations,
                    energy_change=item.energy_change,
                    density_rms=item.density_rms,
                    executed_backend=item.executed_backend,
                    basis_metadata=item.basis_metadata,
                    accuracy=item.accuracy,
                    resource_diagnostics=batch.resource_diagnostics,
                    precision=item.precision,
                    physical_residual_rms=item.physical_residual_rms,
                    ks_diagnostic=item.ks_diagnostic,
                    ks_transport_diagnostic=batch.ks_transport_diagnostics[0],
                    dispersion=item.dispersion,
                )
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
        resource_diagnostics = None
        try:
            if (
                resource_plan is not None
                and resource_plan.requests[0].identity.backend == "cuda"
            ):
                from .resources_native import NativeDeviceLedger

                ledger = NativeDeviceLedger(
                    self._library, resource_plan, owner=resource_plan.requests[0].name
                )
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
                ks_options=effective_ks_options,
            )

            def prepare() -> typing.Any:
                return self._library.vibeqc_calculation_prepare(
                    context,
                    system,
                    ctypes.byref(method_descriptor),
                    ctypes.byref(calculation),
                )

            if resource_plan is None:
                _native.check(self._library, prepare(), context=context)
            else:
                from .resources_native import check_resource_status, observe_method_call

                status, resource_diagnostics = observe_method_call(
                    self._library,
                    resource_plan,
                    ledger,
                    prepare,
                    owner=resource_plan.requests[0].name,
                    phase="preparation",
                )
                if status != _native.STATUS_SUCCESS:
                    check_resource_status(self._library, status, resource_diagnostics)
            force_storage = (
                (ctypes.c_double * (3 * len(native_atoms)))()
                if compute_forces
                else None
            )
            result_descriptor = _native.ResultDescriptor(
                ctypes.sizeof(_native.ResultDescriptor),
                _native.ABI_VERSION,
                0.0,
                force_storage,
                len(force_storage) if force_storage is not None else 0,
                0,
                0.0,
                0.0,
                0,
                _native.BACKEND_CPU_REFERENCE,
            )
            if resource_plan is None:
                status = self._library.vibeqc_calculation_execute(
                    calculation, ctypes.byref(result_descriptor)
                )
            else:
                status, resource_diagnostics = observe_method_call(
                    self._library,
                    resource_plan,
                    ledger,
                    lambda: self._library.vibeqc_calculation_execute(
                        calculation, ctypes.byref(result_descriptor)
                    ),
                    owner=resource_plan.requests[0].name,
                    previous=resource_diagnostics,
                )
            try:
                if resource_diagnostics is None:
                    _native.check(self._library, status, context=context)
                else:
                    from .resources_native import check_resource_status

                    check_resource_status(self._library, status, resource_diagnostics)
            except (RuntimeError, MemoryError) as error:
                # The ordinary checker already attaches the context detail.
                if resource_diagnostics is not None:
                    detail = self._library.vibeqc_context_get_last_detail(context)
                    if detail:
                        error.args = (f"{error}: {detail.decode('utf-8')}",)
                raise
            forces = (
                np.ctypeslib.as_array(force_storage).copy().reshape(-1, 3)
                if force_storage is not None
                else None
            )
            correlation = (
                _read_correlation_result(self._library, calculation, context=context)
                if self._method in _CORRELATED_METHODS
                else None
            )
            physical_residual_rms = None
            scf_diag = _native.ScfDiagnostic(
                ctypes.sizeof(_native.ScfDiagnostic), _native.ABI_VERSION
            )
            scf_status = self._library.vibeqc_calculation_get_scf_diagnostic(
                calculation, ctypes.byref(scf_diag)
            )
            if scf_status != _native.STATUS_NOT_IMPLEMENTED:
                _native.check(self._library, scf_status, context=context)
                physical_residual_rms = scf_diag.physical_residual_rms
            backend = (
                "cuda"
                if result_descriptor.executed_backend == _native.BACKEND_CUDA
                else "cpu_reference"
            )
            ks_diagnostic = None
            ks_transport_diagnostic = None
            if self._ks_options is not None:
                from .ks_diagnostics import (
                    read_ks_diagnostic,
                    read_ks_transport_diagnostic,
                )

                ks_diagnostic = read_ks_diagnostic(
                    self._library,
                    calculation,
                    expected_domain=self._ks_options.scf_domain,
                )
                ks_transport_diagnostic = read_ks_transport_diagnostic(
                    self._library, calculation
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
                precision=self._precision_provenance(calculation),
                basis_metadata=self.basis_metadata(
                    native_atoms, charge=charge, multiplicity=multiplicity
                ),
                accuracy=self._accuracy_assessment(
                    native_atoms,
                    charge,
                    multiplicity,
                    bool(result_descriptor.converged),
                ),
                correlation=correlation,
                physical_residual_rms=physical_residual_rms,
                ks_diagnostic=ks_diagnostic,
                ks_transport_diagnostic=ks_transport_diagnostic,
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
