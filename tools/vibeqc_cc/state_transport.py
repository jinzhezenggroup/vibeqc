"""Fail-closed orbital-frame and RCCSD-amplitude transport contracts.

Slice A of issue #190 classifies transport and implements only exact changes of
coordinates.  A projected warm start is a diagnosed future operation: this
module can identify one, but it cannot produce amplitudes for it.  Target
equations, residual refinement, solver history, and controller integration are
deliberately outside this layer.
"""

from __future__ import annotations

import math
import typing
from dataclasses import InitVar, asdict, dataclass, field
from enum import Enum
from hashlib import sha256

import numpy as np
from vibeqc_compiler.common.provenance import canonical_hash

from tools.vibeqc_posthf.reference import ReferenceSnapshot, immutable


def _array_hash(value: np.ndarray) -> str:
    return sha256(np.ascontiguousarray(value, dtype="<f8").tobytes()).hexdigest()


def _settings(
    value: typing.Mapping[str, str] | typing.Iterable[tuple[str, str]],
) -> tuple[tuple[str, str], ...]:
    items = tuple(value.items()) if isinstance(value, typing.Mapping) else tuple(value)
    if any(
        not isinstance(key, str)
        or not key
        or not isinstance(setting, str)
        or not setting
        for key, setting in items
    ):
        raise ValueError("approximation settings require nonempty string pairs")
    normalized = tuple(sorted(items))
    if len({key for key, _ in normalized}) != len(normalized):
        raise ValueError("approximation setting names must be unique")
    return normalized


@dataclass(frozen=True)
class StateIdentity:
    """Complete endpoint identity needed before amplitudes may be compared."""

    reference_id: str
    geometry_id: str
    basis_id: str
    ao_representation: str
    reference_kind: str
    precision: str
    screening_tolerance: float
    overlap_threshold: float
    reference_energy: float
    orbital_energies: tuple[float, ...]
    functional_identity: str | None
    grid_identity: str | None
    coefficient_frame_hash: str
    spin: int
    electron_count: int
    occupations: tuple[float, ...]
    occupied_orbitals: tuple[int, ...]
    virtual_orbitals: tuple[int, ...]
    frozen_core_orbitals: tuple[int, ...]
    hamiltonian_id: str
    equation_id: str
    integral_provider_id: str
    approximation_settings: tuple[tuple[str, str], ...] = ()
    identity: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "reference_id",
            "geometry_id",
            "basis_id",
            "ao_representation",
            "reference_kind",
            "precision",
            "hamiltonian_id",
            "equation_id",
            "integral_provider_id",
            "coefficient_frame_hash",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a nonempty identity")
        if type(self.spin) is not int or self.spin < 0:
            raise ValueError("spin must be a nonnegative integer")
        if type(self.electron_count) is not int or self.electron_count <= 0:
            raise ValueError("electron_count must be a positive integer")
        for name in ("screening_tolerance", "overlap_threshold"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.overlap_threshold == 0:
            raise ValueError("overlap_threshold must be positive")
        if not math.isfinite(self.reference_energy):
            raise ValueError("reference_energy must be finite")
        for name in ("functional_identity", "grid_identity"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{name} must be None or a nonempty identity")
        if len(self.coefficient_frame_hash) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.coefficient_frame_hash
        ):
            raise ValueError(
                "coefficient_frame_hash must be a lowercase SHA-256 identity"
            )
        occupations = tuple(self.occupations)
        if not occupations or any(not math.isfinite(value) for value in occupations):
            raise ValueError("occupations must be finite and nonempty")
        object.__setattr__(self, "occupations", occupations)
        orbital_energies = tuple(self.orbital_energies)
        if len(orbital_energies) != len(occupations) or any(
            not math.isfinite(value) for value in orbital_energies
        ):
            raise ValueError(
                "orbital energies must be finite and match the orbital frame"
            )
        object.__setattr__(self, "orbital_energies", orbital_energies)
        occupied = tuple(self.occupied_orbitals)
        virtual = tuple(self.virtual_orbitals)
        core = tuple(self.frozen_core_orbitals)
        nmo = len(occupations)
        partition = {*occupied, *virtual}
        if (
            any(
                type(index) is not int or index < 0
                for index in (*occupied, *virtual, *core)
            )
            or len(occupied) + len(virtual) != nmo
            or len(partition) != nmo
            or partition != set(range(nmo))
            or set(occupied) & set(virtual)
            or not occupied
            or not virtual
        ):
            raise ValueError(
                "occupied/virtual orbitals must partition the orbital frame"
            )
        if tuple(sorted(occupied)) != occupied or tuple(sorted(virtual)) != virtual:
            raise ValueError("orbital partitions must be sorted")
        if len(set(core)) != len(core) or not set(core).issubset(occupied):
            raise ValueError("frozen core orbitals must be a unique occupied subset")
        object.__setattr__(self, "occupied_orbitals", occupied)
        object.__setattr__(self, "virtual_orbitals", virtual)
        object.__setattr__(self, "frozen_core_orbitals", core)
        settings = _settings(self.approximation_settings)
        object.__setattr__(self, "approximation_settings", settings)
        object.__setattr__(
            self,
            "identity",
            canonical_hash(
                {
                    name: getattr(self, name)
                    for name in (
                        "reference_id",
                        "geometry_id",
                        "basis_id",
                        "ao_representation",
                        "reference_kind",
                        "precision",
                        "screening_tolerance",
                        "overlap_threshold",
                        "reference_energy",
                        "orbital_energies",
                        "functional_identity",
                        "grid_identity",
                        "coefficient_frame_hash",
                        "spin",
                        "electron_count",
                        "occupations",
                        "occupied_orbitals",
                        "virtual_orbitals",
                        "frozen_core_orbitals",
                        "hamiltonian_id",
                        "equation_id",
                        "integral_provider_id",
                        "approximation_settings",
                    )
                }
            ),
        )

    @classmethod
    def from_snapshot(
        cls,
        snapshot: ReferenceSnapshot,
        *,
        equation_id: str,
        integral_provider_id: str,
        approximation_settings: typing.Mapping[str, str]
        | typing.Iterable[tuple[str, str]] = (),
    ) -> StateIdentity:
        if not isinstance(snapshot, ReferenceSnapshot):
            raise TypeError("state transport requires a validated ReferenceSnapshot")
        occupied = tuple(range(snapshot.nocc))
        virtual = tuple(range(snapshot.nocc, snapshot.nmo))
        return cls(
            reference_id=snapshot.identity,
            geometry_id=snapshot.geometry_hash,
            basis_id=snapshot.basis_hash,
            ao_representation=snapshot.representation,
            reference_kind=snapshot.algorithm,
            precision=snapshot.precision,
            screening_tolerance=snapshot.screening_tolerance,
            overlap_threshold=snapshot.overlap_threshold,
            reference_energy=snapshot.reference_energy,
            orbital_energies=tuple(float(value) for value in snapshot.orbital_energies),
            functional_identity=snapshot.functional_identity,
            grid_identity=snapshot.grid_identity,
            coefficient_frame_hash=_array_hash(snapshot.coefficients),
            spin=0,
            electron_count=snapshot.electron_count,
            occupations=tuple(float(value) for value in snapshot.occupations),
            occupied_orbitals=occupied,
            virtual_orbitals=virtual,
            frozen_core_orbitals=tuple(snapshot.frozen_mask),
            hamiltonian_id=snapshot.hamiltonian_id,
            equation_id=equation_id,
            integral_provider_id=integral_provider_id,
            approximation_settings=_settings(approximation_settings),
        )

    @property
    def nmo(self) -> int:
        return len(self.occupations)

    @property
    def nocc(self) -> int:
        return len(self.occupied_orbitals)

    @property
    def nvir(self) -> int:
        return len(self.virtual_orbitals)


@dataclass(frozen=True)
class StateTransportRequest:
    """Source/target identities and the metric bridge between their AO frames."""

    source: StateIdentity
    target: StateIdentity
    source_coefficients: np.ndarray
    target_coefficients: np.ndarray
    cross_overlap: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.source, StateIdentity) or not isinstance(
            self.target, StateIdentity
        ):
            raise TypeError("transport endpoints must be StateIdentity objects")
        source = immutable(
            self.source_coefficients, shape=(self.source.nmo, self.source.nmo)
        )
        target = immutable(
            self.target_coefficients, shape=(self.target.nmo, self.target.nmo)
        )
        overlap = immutable(
            self.cross_overlap, shape=(self.target.nmo, self.source.nmo)
        )
        object.__setattr__(self, "source_coefficients", source)
        object.__setattr__(self, "target_coefficients", target)
        object.__setattr__(self, "cross_overlap", overlap)
        if _array_hash(source) != self.source.coefficient_frame_hash:
            raise ValueError(
                "source coefficient array does not match its frame identity"
            )
        if _array_hash(target) != self.target.coefficient_frame_hash:
            raise ValueError(
                "target coefficient array does not match its frame identity"
            )

    @classmethod
    def from_snapshots(
        cls,
        source: ReferenceSnapshot,
        target: ReferenceSnapshot,
        cross_overlap: np.ndarray,
        *,
        source_equation_id: str,
        target_equation_id: str,
        source_integral_provider_id: str,
        target_integral_provider_id: str,
        source_approximation_settings: typing.Mapping[str, str]
        | typing.Iterable[tuple[str, str]] = (),
        target_approximation_settings: typing.Mapping[str, str]
        | typing.Iterable[tuple[str, str]] = (),
    ) -> StateTransportRequest:
        return cls(
            StateIdentity.from_snapshot(
                source,
                equation_id=source_equation_id,
                integral_provider_id=source_integral_provider_id,
                approximation_settings=source_approximation_settings,
            ),
            StateIdentity.from_snapshot(
                target,
                equation_id=target_equation_id,
                integral_provider_id=target_integral_provider_id,
                approximation_settings=target_approximation_settings,
            ),
            source.coefficients,
            target.coefficients,
            cross_overlap,
        )


@dataclass(frozen=True)
class StateTransportPolicy:
    """Numerical gates for rank, block mixing, and exact isometry."""

    rank_tolerance: float = 1e-10
    mixing_tolerance: float = 1e-8
    exact_tolerance: float = 1e-10

    def __post_init__(self) -> None:
        for name in ("rank_tolerance", "mixing_tolerance", "exact_tolerance"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 < value <= 1e-4:
                raise ValueError(f"invalid state-transport {name}")


class TransportCompatibility(str, Enum):
    """The four admissible outcomes; only the first two permit exact rotation."""

    identity = "identity"
    exact_orbital_rotation = "exact_orbital_rotation"
    projected_warm_start = "projected_warm_start"
    incompatible = "incompatible"


@dataclass(frozen=True)
class OrbitalFrameDiagnostics:
    occupied_rank: int
    virtual_rank: int
    occupied_min_singular: float
    virtual_min_singular: float
    occupied_unitarity_error: float | None
    virtual_unitarity_error: float | None
    occupied_energy_frame_error: float | None
    virtual_energy_frame_error: float | None
    reference_energy_difference: float
    occupied_virtual_mixing: float
    source_virtual_nullity: int
    target_virtual_nullity: int


def _rank_and_minimum(value: np.ndarray, threshold: float) -> tuple[int, float]:
    singular = np.linalg.svd(value, compute_uv=False)
    if not np.isfinite(singular).all():
        raise ValueError("orbital-frame singular values are nonfinite")
    return int(np.count_nonzero(singular > threshold)), (
        float(singular[-1]) if singular.size else 0.0
    )


def _unitarity_error(value: np.ndarray) -> float | None:
    if value.shape[0] != value.shape[1]:
        return None
    identity = np.eye(value.shape[0])
    return float(
        max(
            np.max(np.abs(value.T @ value - identity)),
            np.max(np.abs(value @ value.T - identity)),
        )
    )


def _energy_frame_error(
    value: np.ndarray,
    source: tuple[float, ...],
    target: tuple[float, ...],
) -> float | None:
    if value.shape[0] != value.shape[1]:
        return None
    transformed = value @ np.diag(source) @ value.T
    return float(np.max(np.abs(transformed - np.diag(target)), initial=0.0))


def _canonical_rhf(identity: StateIdentity) -> bool:
    return (
        identity.reference_kind == "RHF"
        and identity.precision == "float64"
        and identity.screening_tolerance == 0.0
        and identity.spin == 0
        and identity.electron_count == 2 * identity.nocc
        and all(
            identity.occupations[index] == 2.0 for index in identity.occupied_orbitals
        )
        and all(
            identity.occupations[index] == 0.0 for index in identity.virtual_orbitals
        )
    )


def _endpoint_mismatches(source: StateIdentity, target: StateIdentity) -> list[str]:
    reasons = []
    if not _canonical_rhf(source) or not _canonical_rhf(target):
        reasons.append(
            "only unscreened FP64 closed-shell RHF 2/0 occupation identities are supported"
        )
    if source.frozen_core_orbitals or target.frozen_core_orbitals:
        reasons.append("frozen-core state transport is not supported")
    for field_name, label in (
        ("geometry_id", "geometry identity"),
        ("reference_kind", "reference kind"),
        ("precision", "reference precision"),
        ("screening_tolerance", "screening identity"),
        ("overlap_threshold", "overlap-threshold identity"),
        ("functional_identity", "functional identity"),
        ("grid_identity", "grid identity"),
        ("spin", "spin identity"),
        ("electron_count", "electron count"),
        ("nocc", "occupied partition"),
        ("frozen_core_orbitals", "frozen-core identity"),
        ("hamiltonian_id", "Hamiltonian identity"),
        ("equation_id", "equation identity"),
        ("integral_provider_id", "integral-provider identity"),
        ("approximation_settings", "approximation settings"),
    ):
        if getattr(source, field_name) != getattr(target, field_name):
            reasons.append(f"{label} changed")
    return reasons


_CLASSIFIED_TRANSPORT = object()


@dataclass(frozen=True)
class StateTransport:
    """A diagnosed transport decision with immutable target-from-source maps."""

    source: StateIdentity
    target: StateIdentity
    compatibility: TransportCompatibility
    occupied_map: np.ndarray
    virtual_map: np.ndarray
    diagnostics: OrbitalFrameDiagnostics
    messages: tuple[str, ...]
    policy: StateTransportPolicy
    identity: str = field(init=False)
    _classification_proof: InitVar[object] = None

    def __post_init__(self, _classification_proof: object) -> None:
        if _classification_proof is not _CLASSIFIED_TRANSPORT:
            raise TypeError("StateTransport decisions must be created by classify()")
        occupied = immutable(
            self.occupied_map, shape=(self.target.nocc, self.source.nocc)
        )
        virtual = immutable(
            self.virtual_map, shape=(self.target.nvir, self.source.nvir)
        )
        object.__setattr__(self, "occupied_map", occupied)
        object.__setattr__(self, "virtual_map", virtual)
        object.__setattr__(self, "messages", tuple(self.messages))
        object.__setattr__(
            self,
            "identity",
            canonical_hash(
                {
                    "source": self.source.identity,
                    "target": self.target.identity,
                    "compatibility": self.compatibility.value,
                    "occupied_map": _array_hash(occupied),
                    "virtual_map": _array_hash(virtual),
                    "diagnostics": asdict(self.diagnostics),
                    "policy": asdict(self.policy),
                }
            ),
        )

    @classmethod
    def classify(
        cls,
        request: StateTransportRequest,
        *,
        policy: StateTransportPolicy | None = None,
    ) -> StateTransport:
        if not isinstance(request, StateTransportRequest):
            raise TypeError("classify requires a StateTransportRequest")
        selected = StateTransportPolicy() if policy is None else policy
        if not isinstance(selected, StateTransportPolicy):
            raise TypeError("policy must be StateTransportPolicy")
        with np.errstate(over="ignore", invalid="ignore"):
            full_map = (
                request.target_coefficients.T
                @ request.cross_overlap
                @ request.source_coefficients
            )
        if not np.isfinite(full_map).all():
            raise ValueError("orbital-frame map is nonfinite")
        source_o = request.source.occupied_orbitals
        source_v = request.source.virtual_orbitals
        target_o = request.target.occupied_orbitals
        target_v = request.target.virtual_orbitals
        occupied = full_map[np.ix_(target_o, source_o)]
        virtual = full_map[np.ix_(target_v, source_v)]
        target_o_source_v = full_map[np.ix_(target_o, source_v)]
        target_v_source_o = full_map[np.ix_(target_v, source_o)]
        mixing = float(
            max(
                np.max(np.abs(target_o_source_v), initial=0.0),
                np.max(np.abs(target_v_source_o), initial=0.0),
            )
        )
        occupied_rank, occupied_minimum = _rank_and_minimum(
            occupied, selected.rank_tolerance
        )
        virtual_rank, virtual_minimum = _rank_and_minimum(
            virtual, selected.rank_tolerance
        )
        occupied_error = _unitarity_error(occupied)
        virtual_error = _unitarity_error(virtual)
        occupied_energy_error = _energy_frame_error(
            occupied,
            tuple(request.source.orbital_energies[index] for index in source_o),
            tuple(request.target.orbital_energies[index] for index in target_o),
        )
        virtual_energy_error = _energy_frame_error(
            virtual,
            tuple(request.source.orbital_energies[index] for index in source_v),
            tuple(request.target.orbital_energies[index] for index in target_v),
        )
        reference_energy_difference = abs(
            request.source.reference_energy - request.target.reference_energy
        )
        diagnostics = OrbitalFrameDiagnostics(
            occupied_rank=occupied_rank,
            virtual_rank=virtual_rank,
            occupied_min_singular=occupied_minimum,
            virtual_min_singular=virtual_minimum,
            occupied_unitarity_error=occupied_error,
            virtual_unitarity_error=virtual_error,
            occupied_energy_frame_error=occupied_energy_error,
            virtual_energy_frame_error=virtual_energy_error,
            reference_energy_difference=reference_energy_difference,
            occupied_virtual_mixing=mixing,
            source_virtual_nullity=request.source.nvir - virtual_rank,
            target_virtual_nullity=request.target.nvir - virtual_rank,
        )
        reasons = _endpoint_mismatches(request.source, request.target)
        same_reference = request.source.reference_id == request.target.reference_id
        if same_reference and request.source.identity != request.target.identity:
            reasons.append("one reference ID carries different transport metadata")
        if same_reference and (
            full_map.shape[0] != full_map.shape[1]
            or np.max(np.abs(full_map - np.eye(full_map.shape[0])), initial=0.0)
            > selected.exact_tolerance
        ):
            reasons.append(
                "one reference ID does not map to the identical orbital frame"
            )
        if mixing > selected.mixing_tolerance:
            reasons.append("occupied-virtual mixing changes the reference determinant")
        if occupied_rank != request.source.nocc:
            reasons.append("occupied subspace loses rank")
        exact_frame = (
            occupied_error is not None
            and virtual_error is not None
            and occupied_error <= selected.exact_tolerance
            and virtual_error <= selected.exact_tolerance
            and virtual_rank == request.source.nvir == request.target.nvir
        )
        if (
            not reasons
            and not same_reference
            and exact_frame
            and (
                occupied_energy_error is None
                or virtual_energy_error is None
                or occupied_energy_error > selected.exact_tolerance
                or virtual_energy_error > selected.exact_tolerance
                or reference_energy_difference > selected.exact_tolerance
            )
        ):
            reasons.append(
                "reference energy/operator changed across an exact frame map"
            )
        if reasons:
            compatibility = TransportCompatibility.incompatible
            messages = tuple(reasons)
        elif same_reference:
            compatibility = TransportCompatibility.identity
            messages = ("source and target are the same complete state identity",)
        elif exact_frame:
            compatibility = TransportCompatibility.exact_orbital_rotation
            messages = (
                "source and target differ only by block-unitary occupied/virtual frames",
            )
        elif (
            request.source.basis_id != request.target.basis_id
            or request.source.ao_representation != request.target.ao_representation
            or request.source.nmo != request.target.nmo
        ):
            compatibility = TransportCompatibility.projected_warm_start
            messages = (
                "basis/frame change is rank-compatible only as an approximate warm start",
            )
        else:
            compatibility = TransportCompatibility.incompatible
            messages = (
                "same-basis orbital map is not an exact block-unitary transformation",
            )
        return cls(
            request.source,
            request.target,
            compatibility,
            occupied,
            virtual,
            diagnostics,
            messages,
            selected,
            _classification_proof=_CLASSIFIED_TRANSPORT,
        )

    @property
    def exact(self) -> bool:
        return self.compatibility in (
            TransportCompatibility.identity,
            TransportCompatibility.exact_orbital_rotation,
        )

    def rotate_amplitudes(self, t1: np.ndarray, t2: np.ndarray) -> typing.Any:
        """Return an exact target-bound ``AmplitudeSnapshot``.

        The maps are ``<target|source>``.  Restricted amplitudes therefore
        transform covariantly on all occupied and virtual axes.  Projected and
        incompatible decisions raise before any output is published.
        """
        if not self.exact:
            raise ValueError(
                f"{self.compatibility.value} state transport cannot produce exact amplitudes"
            )
        from .gpu_state import AmplitudeSnapshot

        source = AmplitudeSnapshot(self.source.reference_id, t1, t2)
        if source.t1.shape != (self.source.nocc, self.source.nvir):
            raise ValueError("source amplitudes do not match the transport identity")
        singles = self.occupied_map @ source.t1 @ self.virtual_map.T
        doubles = np.einsum(
            "ki,ijab->kjab", self.occupied_map, source.t2, optimize=False
        )
        doubles = np.einsum("lj,kjab->klab", self.occupied_map, doubles, optimize=False)
        doubles = np.einsum("ca,klab->klcb", self.virtual_map, doubles, optimize=False)
        doubles = np.einsum("db,klcb->klcd", self.virtual_map, doubles, optimize=False)
        return AmplitudeSnapshot(
            self.target.reference_id,
            np.ascontiguousarray(singles),
            np.ascontiguousarray(doubles),
        )
