"""Typed execution schedules and deterministic admission for generated grid/XC paths.

The schedule changes lowering only.  Scientific identity is carried separately so
an autotuner cannot make a different functional, grid, screening rule, precision,
or density source look like the same candidate.
"""

from __future__ import annotations

import typing
from dataclasses import asdict, dataclass, replace

from vibeqc_compiler.common.gpu_profitability import GpuProfitability
from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.common.schedule import (
    ScheduleContract,
    ScheduleResources,
    ScheduleTopology,
)


@dataclass(frozen=True)
class GridXcExecutionSchedule:
    """Executable lowering choices currently supported by PreparedXCContractions."""

    name: str
    fusion: str
    point_tile: int | None = None
    resident_jets: bool = False
    matrix_accumulation: str = "symmetric"

    def __post_init__(self) -> None:
        if self.name not in ("device_fused", "host_unfused"):
            raise ValueError("unknown grid/XC schedule")
        expected_fusion = {
            "device_fused": "device_xc_vxc",
            "host_unfused": "host_xc_vxc",
        }[self.name]
        if self.fusion != expected_fusion:
            raise ValueError("grid/XC schedule name and fusion policy disagree")
        if self.point_tile is not None and (
            type(self.point_tile) is not int or self.point_tile < 1
        ):
            raise ValueError("grid/XC point tile must be a positive integer")
        if type(self.resident_jets) is not bool:
            raise TypeError("resident_jets must be boolean")
        if self.resident_jets != (self.name == "device_fused"):
            raise ValueError(
                "grid/XC jet residency is not executable for this schedule"
            )
        if self.matrix_accumulation != "symmetric":
            raise ValueError("only symmetric Vxc accumulation is executable")

    def resolved(self, point_tile: int) -> GridXcExecutionSchedule:
        if type(point_tile) is not int or point_tile < 1:
            raise ValueError("resolved point tile must be a positive integer")
        if self.point_tile is not None and self.point_tile != point_tile:
            raise ValueError("schedule point tile differs from prepared grid topology")
        return replace(self, point_tile=point_tile)

    def to_payload(self) -> dict[str, typing.Any]:
        return {
            "schema": "vibeqc.grid-xc-schedule.v1",
            **asdict(self),
        }

    @property
    def identity(self) -> str:
        return canonical_hash(self.to_payload())


DEVICE_FUSED = GridXcExecutionSchedule(
    name="device_fused",
    fusion="device_xc_vxc",
    resident_jets=True,
)
HOST_UNFUSED = GridXcExecutionSchedule(
    name="host_unfused",
    fusion="host_xc_vxc",
    resident_jets=False,
)


def grid_xc_schedule(value: typing.Any) -> GridXcExecutionSchedule:
    """Normalize a typed schedule, validated profile payload, or stable public name."""

    if isinstance(value, GridXcExecutionSchedule):
        return value
    if isinstance(value, dict):
        expected = {
            "schema",
            "name",
            "fusion",
            "point_tile",
            "resident_jets",
            "matrix_accumulation",
        }
        if (
            set(value) != expected
            or value.get("schema") != "vibeqc.grid-xc-schedule.v1"
        ):
            raise ValueError("invalid grid/XC schedule profile payload")
        return GridXcExecutionSchedule(
            name=value["name"],
            fusion=value["fusion"],
            point_tile=value["point_tile"],
            resident_jets=value["resident_jets"],
            matrix_accumulation=value["matrix_accumulation"],
        )
    if value == "device_fused":
        return DEVICE_FUSED
    if value == "host_unfused":
        return HOST_UNFUSED
    raise ValueError("grid/XC schedule must be device_fused or host_unfused")


@dataclass(frozen=True)
class GridXcScientificIdentity:
    """Mathematical/workload identity; deliberately excludes execution schedule."""

    architecture: str
    functional: str
    functional_identity: str
    ingredients: tuple[str, ...]
    jet_outputs: tuple[tuple[int, int, int], ...]
    grid_identity: str
    grid_model: str
    screening_identity: str | None
    precision: str
    spin: str
    observable: str
    density_route: str
    source_identity: str

    def __post_init__(self) -> None:
        if not self.architecture.startswith("sm_"):
            raise ValueError("grid/XC architecture must use an sm_* identity")
        if self.precision != "fp64":
            raise ValueError("DFT09 schedules currently preserve FP64 arithmetic")
        if self.spin not in ("polarized", "unpolarized"):
            raise ValueError("grid/XC spin identity is invalid")
        if self.observable not in ("energy", "potential", "response", "geometry"):
            raise ValueError("unknown grid/XC observable")
        if self.density_route not in ("density_matrix", "orbitals"):
            raise ValueError("grid/XC density route must be explicit")
        if not self.functional or not self.functional_identity:
            raise ValueError("functional identity is required")
        if not self.grid_identity or not self.grid_model:
            raise ValueError("grid identity/model is required")
        if self.screening_identity is not None and (
            not isinstance(self.screening_identity, str) or not self.screening_identity
        ):
            raise ValueError("screening identity must be a non-empty string or None")
        if not self.source_identity:
            raise ValueError("source identity is required")
        if not self.ingredients or not self.jet_outputs:
            raise ValueError("ingredient and jet-output identities are required")

    def to_payload(self) -> dict[str, typing.Any]:
        payload = asdict(self)
        payload["schema"] = "vibeqc.grid-xc-scientific.v1"
        return payload

    @property
    def identity(self) -> str:
        return canonical_hash(self.to_payload())


def molecular_grid_xc_workload(
    *,
    architecture: str,
    functional: typing.Any,
    atoms: typing.Any,
    grid_spec: typing.Any,
    charge: int,
    multiplicity: int,
    source_identity: str,
    screening_identity: str | None = None,
    observable: str = "potential",
    density_route: str = "density_matrix",
) -> GridXcScientificIdentity:
    """Build the canonical DFT09 workload without materializing molecular quadrature."""

    from vibeqc_compiler.dft.ao import jet_indices
    from vibeqc_compiler.dft.grid import molecular_grid_identity

    declared = tuple(functional.ingredients)
    if "tau" in declared:
        ingredients = ("rho", "gradient", "sigma", "tau")
        ao_order = 1
    elif "sigma" in declared:
        ingredients = ("rho", "gradient", "sigma")
        ao_order = 1
    else:
        ingredients = ("rho",)
        ao_order = 0
    return GridXcScientificIdentity(
        architecture=architecture,
        functional=functional.identifier,
        functional_identity=functional.identity,
        ingredients=ingredients,
        jet_outputs=jet_indices(ao_order),
        grid_identity=molecular_grid_identity(
            atoms,
            grid_spec,
            charge=charge,
            multiplicity=multiplicity,
        ),
        grid_model=canonical_hash(
            {"kind": "MolecularGrid", "model": asdict(grid_spec)}
        ),
        screening_identity=screening_identity,
        precision="fp64",
        spin=functional.spin,
        observable=observable,
        density_route=density_route,
        source_identity=source_identity,
    )


@dataclass(frozen=True)
class GridXcCandidateShape:
    """Measured/planned shape used by deterministic pre-compilation admission."""

    npoint: int
    tile_points: int
    nao: int
    max_active_ao: int
    spins: int
    jet_components: int
    device_workspace_bytes: int
    generated_source_bytes: int

    def __post_init__(self) -> None:
        integer_fields = (
            self.npoint,
            self.tile_points,
            self.nao,
            self.max_active_ao,
            self.spins,
            self.jet_components,
            self.device_workspace_bytes,
            self.generated_source_bytes,
        )
        if any(type(value) is not int or value < 0 for value in integer_fields):
            raise ValueError(
                "grid/XC candidate shape values must be nonnegative integers"
            )
        if (
            self.npoint < 1
            or self.tile_points < 1
            or self.nao < 1
            or self.max_active_ao < 1
            or self.spins not in (1, 2)
            or self.jet_components < 1
        ):
            raise ValueError("grid/XC candidate shape has an empty executable domain")
        if self.max_active_ao > self.nao:
            raise ValueError("active AO capacity exceeds the basis")
        if self.device_workspace_bytes < 1 or self.generated_source_bytes < 1:
            raise ValueError(
                "grid/XC admission requires measured workspace and generated-source sizes"
            )


@dataclass(frozen=True)
class GridXcCandidateLimits:
    """Hard deterministic bounds supplied by the target/resource owner."""

    device_bytes: int
    live_values: int
    source_bytes: int

    def __post_init__(self) -> None:
        if any(
            type(value) is not int or value < 1
            for value in (self.device_bytes, self.live_values, self.source_bytes)
        ):
            raise ValueError("grid/XC candidate limits must be positive integers")


@dataclass(frozen=True)
class GridXcCandidateAssessment:
    schedule_hash: str
    legal: bool
    reasons: tuple[str, ...]
    live_values: int
    device_workspace_bytes: int
    generated_source_bytes: int
    schedule_contract: ScheduleContract

    def to_payload(self) -> dict[str, typing.Any]:
        return {
            "schedule_hash": self.schedule_hash,
            "legal": self.legal,
            "reasons": list(self.reasons),
            "live_values": self.live_values,
            "device_workspace_bytes": self.device_workspace_bytes,
            "generated_source_bytes": self.generated_source_bytes,
            "schedule_contract": self.schedule_contract.to_payload(),
        }


def _live_values(schedule: GridXcExecutionSchedule, shape: GridXcCandidateShape) -> int:
    """Conservative per-tile live-range model, not a PTXAS register prediction."""

    points = min(shape.npoint, shape.tile_points)
    ao_jets = points * shape.max_active_ao * shape.jet_components
    density_panel = shape.spins * points * shape.max_active_ao
    features = shape.spins * points * (1 + 3 + 3 + 1)
    local_vxc = shape.spins * shape.max_active_ao * shape.max_active_ao
    if schedule.name == "device_fused":
        # AO jets, D*AO/features and local Vxc overlap on the device path.
        return ao_jets + density_panel + features + local_vxc
    # The staged path downloads AO/features before generated host XC/Vxc work;
    # charge the extra host-visible copy so reuse is not treated as free.
    return 2 * ao_jets + density_panel + 2 * features + local_vxc


def assess_grid_xc_schedule(
    schedule: GridXcExecutionSchedule,
    shape: GridXcCandidateShape,
    limits: GridXcCandidateLimits,
    *,
    device_xc_available: bool,
    observable: str,
    functional: str,
    scientific: GridXcScientificIdentity | None = None,
) -> GridXcCandidateAssessment:
    """Reject impossible/incompatible candidates before any timing comparison."""

    if scientific is not None:
        if not isinstance(scientific, GridXcScientificIdentity):
            raise TypeError("scientific identity must be GridXcScientificIdentity")
        if (
            scientific.functional != functional
            or scientific.observable != observable
            or shape.spins != (2 if scientific.spin == "polarized" else 1)
        ):
            raise ValueError(
                "scientific identity disagrees with admitted grid/XC workload"
            )
    resolved = schedule.resolved(shape.tile_points)
    reasons: list[str] = []
    if resolved.name == "device_fused":
        if not device_xc_available:
            reasons.append("native device XC capability is unavailable")
        if observable != "potential":
            reasons.append("device-fused schedule only supports potential output")
        if functional not in ("LDA_XC_PW", "PBE"):
            reasons.append("device-fused schedule only supports canonical LDA/PBE")
    live = _live_values(resolved, shape)
    if live > limits.live_values:
        reasons.append("estimated live-value bound exceeds target limit")
    if shape.device_workspace_bytes > limits.device_bytes:
        reasons.append("planned device workspace exceeds target limit")
    if shape.generated_source_bytes > limits.source_bytes:
        reasons.append("generated source/compile-size bound exceeds target limit")
    contract = ScheduleContract(
        consumer="dft.grid_xc",
        schedule_hash=resolved.identity,
        workload_hash=scientific.identity if scientific is not None else None,
        profile_key=(
            schedule_profile_key(scientific) if scientific is not None else None
        ),
        target_hash=(
            canonical_hash({"backend": "cuda", "architecture": scientific.architecture})
            if scientific is not None
            else None
        ),
        precision_schedule_hash=(
            canonical_hash({"kind": "fixed", "precision": scientific.precision})
            if scientific is not None
            else None
        ),
        fallback=resolved.name == "host_unfused",
        legal=not reasons,
        reasons=tuple(reasons),
        topology=ScheduleTopology(
            tiles=(shape.tile_points,),
            fusion=resolved.fusion,
            materialization=(
                "fused-vxc" if resolved.name == "device_fused" else "materialized-vxc"
            ),
            residency=(
                "device-resident-jets" if resolved.resident_jets else "host-staged-jets"
            ),
            reduction=resolved.matrix_accumulation,
            bucket="grid-points",
        ),
        resources=ScheduleResources(
            device_bytes=shape.device_workspace_bytes,
            workspace_bytes=shape.device_workspace_bytes,
            peak_live_values=live,
            source_bytes=shape.generated_source_bytes,
        ),
        profitability=GpuProfitability(
            peak_live_values=live,
            source_bytes=shape.generated_source_bytes,
        ),
        provenance=(
            ("domain_schedule", resolved.name),
            ("resource_scope", "grid-xc-admission"),
        ),
    )
    return GridXcCandidateAssessment(
        schedule_hash=resolved.identity,
        legal=not reasons,
        reasons=tuple(reasons),
        live_values=live,
        device_workspace_bytes=shape.device_workspace_bytes,
        generated_source_bytes=shape.generated_source_bytes,
        schedule_contract=contract,
    )


def schedule_profile_key(
    scientific: GridXcScientificIdentity | dict[str, typing.Any],
) -> str:
    """Canonical workload key used inside the existing #136 profile bundle."""

    payload = (
        scientific.to_payload()
        if isinstance(scientific, GridXcScientificIdentity)
        else scientific
    )
    return canonical_hash(payload)
