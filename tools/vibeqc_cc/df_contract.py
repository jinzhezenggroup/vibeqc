"""Shared DF-CCSD(T) scientific identity without dense validation dependencies."""

from __future__ import annotations

import math
import typing
from dataclasses import asdict, dataclass, field, fields, replace

from vibeqc.profiles import canonical_hash

from tools.vibeqc_posthf.df import MetricFactor
from tools.vibeqc_posthf.reference import ReferenceSnapshot


@dataclass(frozen=True)
class DFCCSDTMethodContract:
    """Immutable scientific identity for the first #157 DF-CCSD(T) variant."""

    reference_identity: str
    reference_hamiltonian_id: str
    correlation_hamiltonian_id: str
    correlation_snapshot_identity: str
    geometry_hash: str
    orbital_basis_hash: str
    auxiliary_basis_hash: str
    representation: str
    metric_convention: str
    metric_relative_threshold: float
    metric_rank: int
    metric_dimension: int
    metric_absolute_threshold: float
    metric_condition_number: float
    reference_mode: str = "conventional-rhf"
    correlation_mode: str = "density-fitting"
    fock_policy: str = "preserve-conventional-rhf"
    precision: str = "float64"
    frozen_orbitals: tuple[int, ...] = ()
    triples_variant: str = "standard-canonical"
    identity: str = field(init=False)

    def __post_init__(self) -> None:
        fixed = {
            "reference_mode": "conventional-rhf",
            "correlation_mode": "density-fitting",
            "fock_policy": "preserve-conventional-rhf",
            "precision": "float64",
            "triples_variant": "standard-canonical",
        }
        if any(getattr(self, key) != value for key, value in fixed.items()):
            raise ValueError("unsupported DF-CCSD(T) method convention")
        if self.reference_hamiltonian_id != "conventional-unscreened":
            raise ValueError("slice A requires a conventional unscreened RHF reference")
        if self.frozen_orbitals:
            raise ValueError("DF-CCSD(T) slice A is all-electron only")
        if (
            not 0 < self.metric_relative_threshold < 1
            or self.metric_rank < 1
            or self.metric_dimension < self.metric_rank
            or not math.isfinite(self.metric_absolute_threshold)
            or self.metric_absolute_threshold < 0
            or not math.isfinite(self.metric_condition_number)
            or self.metric_condition_number < 1
        ):
            raise ValueError("invalid DF metric contract")
        payload = {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if item.name != "identity"
        }
        object.__setattr__(self, "identity", canonical_hash(payload))

    def record(self) -> dict[str, typing.Any]:
        value = asdict(self)
        value["schema"] = "vibeqc.df-ccsd-t.method/1"
        return value


def correlation_df_reference(
    reference: ReferenceSnapshot,
    source: typing.Any,
    metric: MetricFactor,
) -> tuple[ReferenceSnapshot, DFCCSDTMethodContract]:
    """Relabel only correlation integral identity; preserve the RHF state/Fock."""

    if not isinstance(reference, ReferenceSnapshot) or not isinstance(
        metric, MetricFactor
    ):
        raise TypeError("validated ReferenceSnapshot and MetricFactor are required")
    if (
        reference.algorithm != "RHF"
        or reference.hamiltonian_id != "conventional-unscreened"
        or reference.precision != "float64"
        or reference.screening_tolerance != 0
        or reference.frozen_mask
    ):
        raise ValueError("DF-CCSD(T) slice A requires all-electron unscreened FP64 RHF")
    required = (
        "geometry_hash",
        "basis_hash",
        "auxiliary_hash",
        "representation",
        "nbf",
        "naux",
    )
    if any(not hasattr(source, name) for name in required):
        raise TypeError("DF source is missing scientific identity metadata")
    if (
        reference.geometry_hash != source.geometry_hash
        or reference.basis_hash != source.basis_hash
        or reference.representation != source.representation
        or reference.nmo != source.nbf
        or metric.geometry_hash != source.geometry_hash
        or metric.auxiliary_hash != source.auxiliary_hash
        or metric.inverse_square_root.shape != (source.naux, source.naux)
    ):
        raise ValueError("reference/source/metric identity mismatch")
    correlated = replace(reference, hamiltonian_id=metric.hamiltonian_id)
    contract = DFCCSDTMethodContract(
        reference_identity=reference.identity,
        reference_hamiltonian_id=reference.hamiltonian_id,
        correlation_hamiltonian_id=metric.hamiltonian_id,
        correlation_snapshot_identity=correlated.identity,
        geometry_hash=reference.geometry_hash,
        orbital_basis_hash=reference.basis_hash,
        auxiliary_basis_hash=metric.auxiliary_hash,
        representation=reference.representation,
        metric_convention=metric.convention,
        metric_relative_threshold=metric.relative_threshold,
        metric_rank=metric.rank,
        metric_dimension=source.naux,
        metric_absolute_threshold=metric.absolute_threshold,
        metric_condition_number=metric.condition_number,
    )
    # C, eps, F and E_HF are deliberately unchanged.  The new snapshot identity
    # only says that correlation integral blocks belong to the selected DF
    # Hamiltonian instead of the four-center Hamiltonian.
    return correlated, contract
