"""Immutable physical KS records copied from one completed native execution."""

import ctypes
import math
import typing
from dataclasses import asdict, dataclass

from . import _native
from .ks import B3LYP_SCF_DOMAIN, SCF_DOMAIN, WB97MV_SCF_DOMAIN


@dataclass(frozen=True)
class KsEnergyComponents:
    """Physical Hartree energy terms, with XC counted exactly once."""

    nuclear: float
    one_electron: float
    hartree: float
    xc: float

    @property
    def total(self) -> typing.Any:
        return self.nuclear + self.one_electron + self.hartree + self.xc


@dataclass(frozen=True)
class KsIteration:
    """Current physical E/F/D and its proposed density change for one step.

    Energy change is None on the first step: there is no preceding energy.
    RMS maxima gate each spin separately. An occupation-stabilized proposal
    changes the iteration control only; all reported components are unshifted.
    """

    iteration: int
    components: KsEnergyComponents
    energy_change: float | None
    density_change_max: float
    physical_residual_max: float
    electrons: tuple[float, float]
    occupation_stabilized: bool


@dataclass(frozen=True)
class KsDiagnostic:
    """Final physical state plus the complete iteration history of one solve.

    CPU RKS performs additional final validation rebuilds; final components and
    residual can therefore differ from history[-1]. Neither is overwritten by
    the other. Electron counts use the AO overlap metric, not quadrature.
    A warm retry reports its final attempt; this is not a total-work counter
    for discarded attempts. Execution and transport profiling are separate.
    """

    occupations: tuple[int, int]
    electrons: tuple[float, float]
    grid_points: int
    tile_points: int
    ao_order: int
    scf_domain: str
    initial_density_used: bool
    fock_builds: int
    components: KsEnergyComponents
    density_change_max: float
    physical_residual_max: float
    history: tuple[KsIteration, ...]

    def to_payload(self) -> typing.Any:
        """Return a finite JSON-compatible snapshot, including every iteration."""
        return asdict(self)


@dataclass(frozen=True)
class KsTransportDiagnostic:
    """Cumulative measured transfers and synchronization for prepared CUDA KS."""

    setup_h2d_bytes: int
    density_h2d_bytes: int
    scalar_d2h_bytes: int
    matrix_d2h_bytes: int
    synchronizations: int
    iterations: int
    occupation_stabilized_proposals: int

    def to_payload(self) -> typing.Any:
        return asdict(self)


def _components(value: typing.Any) -> typing.Any:
    return KsEnergyComponents(
        value.nuclear_energy,
        value.one_electron_energy,
        value.hartree_energy,
        value.xc_energy,
    )


def read_ks_diagnostic(
    library: typing.Any, handle: typing.Any, index: typing.Any = None
) -> typing.Any:
    """Copy the current native record; absence never substitutes an old one."""
    name = (
        "vibeqc_calculation_get_ks_diagnostic"
        if index is None
        else "vibeqc_batch_get_ks_diagnostic"
    )
    query = getattr(library, name)
    prefix = (handle,) if index is None else (handle, index)
    summary = _native.KsDiagnosticDescriptor(
        ctypes.sizeof(_native.KsDiagnosticDescriptor), _native.ABI_VERSION
    )
    status = query(*prefix, ctypes.byref(summary), None, 0)
    if status == _native.STATUS_NOT_IMPLEMENTED:
        return None
    _native.check(library, status)
    # Domain IDs identify numerical policies, independently of method aliases.
    domains = {1: SCF_DOMAIN, 2: B3LYP_SCF_DOMAIN, 3: WB97MV_SCF_DOMAIN}
    if summary.scf_domain_version not in domains:
        raise RuntimeError("unsupported native KS diagnostic domain version")
    history = (_native.KsIterationDescriptor * summary.history_count)()
    for row in history:
        row.struct_size = ctypes.sizeof(row)
        row.abi_version = _native.ABI_VERSION
    _native.check(library, query(*prefix, ctypes.byref(summary), history, len(history)))
    rows = []
    for row in history:
        change = row.energy_change
        if row.iteration == 1 and math.isinf(change) and change > 0:
            change = None
        rows.append(
            KsIteration(
                row.iteration,
                _components(row),
                change,
                row.density_change_max,
                row.physical_residual_max,
                tuple(row.electrons),
                bool(row.occupation_stabilized),
            )
        )
    return KsDiagnostic(
        occupations=tuple(summary.occupations),
        electrons=tuple(summary.electrons),
        grid_points=summary.grid_points,
        tile_points=summary.tile_points,
        ao_order=summary.required_ao_order,
        scf_domain=domains[summary.scf_domain_version],
        initial_density_used=bool(summary.initial_density_used),
        fock_builds=summary.fock_builds,
        components=_components(summary),
        density_change_max=summary.density_change_max,
        physical_residual_max=summary.physical_residual_max,
        history=tuple(rows),
    )


def read_ks_transport_diagnostic(
    library: typing.Any, handle: typing.Any, index: typing.Any = None
) -> typing.Any:
    """Snapshot cumulative CUDA movement; unsupported/CPU execution returns None."""
    name = (
        "vibeqc_calculation_get_ks_transport_diagnostic"
        if index is None
        else "vibeqc_batch_get_ks_transport_diagnostic"
    )
    query = getattr(library, name)
    prefix = (handle,) if index is None else (handle, index)
    value = _native.KsTransportDiagnosticDescriptor(
        ctypes.sizeof(_native.KsTransportDiagnosticDescriptor), _native.ABI_VERSION
    )
    status = query(*prefix, ctypes.byref(value))
    if status == _native.STATUS_NOT_IMPLEMENTED:
        return None
    _native.check(library, status)
    return KsTransportDiagnostic(
        **{
            field: getattr(value, field)
            for field, _ in value._fields_
            if field not in ("struct_size", "abi_version")
        }
    )
