"""Original table-free atom-centered quadrature and equal-radius Becke weights.

Angular integration uses a Gauss-Legendre polar × periodic trapezoidal rule,
not a Lebedev table. Radial r=R*t/(1-t), t in (0,1), uses Gauss-Legendre nodes.
The recorded element radii set R; they do not add a heteronuclear partition
correction. No empirical angular/radius tables or runtime downloads are used.
"""

from __future__ import annotations

import json
import typing
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from vibeqc_compiler.common.arrays import immutable
from vibeqc_compiler.common.provenance import canonical_hash


def checked_int(
    value: typing.Any,
    name: typing.Any,
    low: typing.Any = 1,
    high: typing.Any = 2**31 - 1,
) -> typing.Any:
    """Reject booleans, truncation and overflow at every public size boundary."""
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return value


def owned_atoms(atoms: typing.Any) -> typing.Any:
    """Canonicalize all coordinate storage, including user-constructed Atoms."""
    # Reuse the public atom contract only when accepting molecular input.
    from vibeqc import Atom

    result = []
    for atom in atoms:
        a = Atom.from_value(atom)
        checked_int(a.atomic_number, "atomic number", high=118)
        xyz = immutable(a.position, shape=(3,))
        result.append(
            Atom(
                a.atomic_number,
                (float(xyz[0]), float(xyz[1]), float(xyz[2])),
            )
        )
    if not result:
        raise ValueError("a grid requires atoms")
    return tuple(result)


@dataclass(frozen=True)
class GridSpec:
    """Versioned, fully specified quadrature; all lengths are in Bohr.

    Version 1 is the historical reference contract: default radial radii are
    exactly one Bohr for every element. Version 2 is the resolved production
    contract: every element that is actually used must have an explicit radius.
    Overrides are explicit (atomic_number, radius) pairs, not an opaque accuracy level.
    Coincident centers within the declared distance use an equal pair split.
    Screening and pruning are disabled; changing either needs a new contract.
    """

    radial_points: int = 48
    angular_polar: int = 16
    angular_azimuth: int = 32
    element_radii: tuple[tuple[int, float], ...] = ()
    partition_iterations: int = 3
    coincident_tolerance: float = 1e-12
    version: int = 1
    radial_rule: str = "rational-legendre"
    angular_rule: str = "legendre-trapezoid"
    partition: str = "becke-equal-radius"
    pruning: str = "none"
    units: str = "Bohr"
    ordering: str = "atom-radial-polar-azimuth"

    def __post_init__(self) -> None:
        checked_int(self.version, "grid version", high=2)
        checked_int(self.radial_points, "radial points", high=512)
        checked_int(self.angular_polar, "polar points", high=256)
        checked_int(self.angular_azimuth, "azimuth points", low=3, high=1024)
        checked_int(self.partition_iterations, "partition iterations", high=5)
        if (
            self.radial_rule,
            self.angular_rule,
            self.partition,
            self.pruning,
            self.units,
            self.ordering,
        ) != (
            "rational-legendre",
            "legendre-trapezoid",
            "becke-equal-radius",
            "none",
            "Bohr",
            "atom-radial-polar-azimuth",
        ):
            raise ValueError("unsupported grid prescription/version")
        if not np.isfinite(self.coincident_tolerance) or self.coincident_tolerance < 0:
            raise ValueError("invalid coincident-center tolerance")
        radii = []
        for z, r in self.element_radii:
            checked_int(z, "radius atomic number", high=118)
            if not np.isfinite(r) or r <= 0:
                raise ValueError("element radii must be finite and positive")
            radii.append((z, float(r)))
        if len({z for z, _ in radii}) != len(radii):
            raise ValueError("duplicate element radius")
        object.__setattr__(self, "element_radii", tuple(sorted(radii)))


# Canonical production radii. These are the pinned covalent_radius_bohr values
# extracted from xTBloom GFN1 data. Keep source identity beside the values.
GRID_POLICY_RADII_SOURCE = (
    "external/xtbloom-d3/covalent_radii.json@"
    "92b32fada844a337204b84f2d961473bad5737240765eb8d0727a62827de5111"
)
GRID_POLICY_UPSTREAM_REVISION = "2cbdf1db8661ccbd5cb7d3d4bfc868a848cbbff3"
_PRODUCTION_RADII = (
    0.8062831465047213,
    1.1590320231005369,
    3.0235617993927044,
    2.368456742857618,
    1.9401188212769855,
    1.8897261246204402,
    1.7889407313073502,
    1.5873699446811698,
    1.6125662930094427,
    1.6881553379942602,
    3.5274887659581555,
    3.1495435410340673,
    2.8471873610947966,
    2.620420226140344,
    2.771598316109979,
    2.5700275294837986,
    2.494438484498981,
    2.4188494395141635,
    4.434557305775966,
    3.880237642553971,
    3.3511143276602473,
    3.0739544960492493,
    3.048758147720977,
    2.771598316109979,
    2.696009271125162,
    2.620420226140344,
    2.5196348328272538,
    2.494438484498981,
    2.5448311811555264,
    2.746401967781707,
    2.8219910127665244,
    2.746401967781707,
    2.8975800577513415,
    2.771598316109979,
    2.8723837094230693,
    2.9479727544078864,
    4.7621098340435095,
    4.2077901708215135,
    3.7038632042560633,
    3.5022924176298824,
    3.325917979331975,
    3.1243471927057946,
    2.8975800577513415,
    2.8471873610947966,
    2.8471873610947966,
    2.721205619453434,
    2.8975800577513415,
    3.0991508443775224,
    3.2251325860188853,
    3.1747398893623395,
    3.1747398893623395,
    3.0991508443775224,
    3.325917979331975,
    3.3007216310037024,
    5.26603680060896,
    4.434557305775966,
    4.0818084291801515,
    3.7038632042560633,
    3.9810230358670613,
    3.9558266875387886,
    3.9306303392105164,
    3.9054339908822433,
    3.8046485975691535,
    3.8298449458974257,
    3.8046485975691535,
    3.7794522492408804,
    3.754255900912608,
    3.754255900912608,
    3.7290595525843355,
    3.8550412942256984,
    3.6786668559277906,
    3.451899720973338,
    3.3007216310037024,
    3.0991508443775224,
    2.9731691027361595,
    2.922776406079614,
    2.796794664438252,
    2.8219910127665244,
    2.8471873610947966,
    3.325917979331975,
    3.27552528267543,
    3.27552528267543,
    3.4267033726450653,
    3.3007216310037024,
    3.4770960693016097,
    3.5778814626147004,
)


@dataclass(frozen=True)
class GridProfile:
    """Resolved production topology before element radii are attached."""

    name: str
    radial_points: int
    angular_polar: int
    angular_azimuth: int
    partition_iterations: int = 3
    pruning: str = "none"
    screening: str = "none"
    topology: str = "atom-radial-polar-azimuth"


@dataclass(frozen=True)
class GridPolicy:
    """Canonical production grid resolver shared by KS and future consumers."""

    accuracy: str = "standard"

    def profile(self, method: str, *, derivative_order: int = 0) -> GridProfile:
        if self.accuracy not in ("standard", "tight"):
            raise ValueError("grid accuracy must be 'standard' or 'tight'")
        if derivative_order not in (0, 1):
            raise NotImplementedError(
                "production grid derivatives support orders 0 and 1"
            )
        name = str(method).lower().replace("_", "-")
        if name in ("lda", "lda-rks", "lda-uks"):
            family = "lda"
        elif name in ("gga", "pbe", "pbe-rks", "pbe-uks"):
            family = "gga"
        else:
            raise NotImplementedError(
                "production grid policy is qualified only for LDA and GGA/PBE"
            )
        tight = self.accuracy == "tight" or derivative_order == 1
        if family == "lda":
            shape = (64, 20, 40) if tight else (54, 16, 32)
        else:
            # Keep the qualified standard angular workload at the legacy
            # 16x32 topology while using the 54-point radial rule that meets
            # the retained independent energy/force convergence gate.
            shape = (72, 24, 48) if tight else (54, 16, 32)
        return GridProfile(f"{family}-{'tight' if tight else 'standard'}-v2", *shape)

    @property
    def provenance(self) -> dict[str, typing.Any]:
        return {
            "policy_version": 2,
            "radii_source": GRID_POLICY_RADII_SOURCE,
            "upstream_revision": GRID_POLICY_UPSTREAM_REVISION,
            "supported_atomic_numbers": (1, len(_PRODUCTION_RADII)),
            "pruning": "none",
            "screening": "none",
            "partition": "becke-equal-radius",
            "topology": "atom-radial-polar-azimuth",
            "derivative_topology": "fixed",
        }

    def resolve(self, method: str, *, derivative_order: int = 0) -> GridSpec:
        profile = self.profile(method, derivative_order=derivative_order)
        return GridSpec(
            version=2,
            radial_points=profile.radial_points,
            angular_polar=profile.angular_polar,
            angular_azimuth=profile.angular_azimuth,
            partition_iterations=profile.partition_iterations,
            element_radii=tuple(enumerate(_PRODUCTION_RADII, start=1)),
        )


def grid_policy_provenance(spec: GridSpec) -> dict[str, typing.Any]:
    """Describe the exact origin of a resolved grid without over-claiming policy provenance."""
    if spec.version == 1:
        return {"policy_version": 1, "contract": "reference-grid-v1"}
    if spec.version != 2:
        raise ValueError("unsupported grid policy version")

    # Version identifies the representation, not the source of user data.
    # Only exact prescribed resolver outputs may claim the pinned policy source.
    for accuracy in ("standard", "tight"):
        policy = GridPolicy(accuracy)
        if any(spec == policy.resolve(method) for method in ("lda", "pbe")):
            return policy.provenance
    return {"policy_version": 2, "contract": "explicit-grid-v2"}


def partition_weights(
    points: typing.Any,
    centers: typing.Any,
    *,
    iterations: typing.Any = 3,
    coincident_tolerance: typing.Any = 1e-12,
) -> typing.Any:
    """Return normalized ownership [point,atom] with stable log products.

    Becke's p(x)=(3x-x^3)/2 is composed ``iterations`` times. Identical centers
    share pair ownership equally, so duplicate atom grids do not double the
    molecular measure. Spatial, center and physical-atom derivatives of these
    weights are not supplied by this value-only partition routine.
    """
    checked_int(iterations, "partition iterations", high=5)
    if not np.isfinite(coincident_tolerance) or coincident_tolerance < 0:
        raise ValueError("invalid coincident-center tolerance")
    points, centers = immutable(points), immutable(centers)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or centers.ndim != 2
        or centers.shape[1] != 3
        or not len(centers)
    ):
        raise ValueError("points/centers require shape (n,3) and at least one center")
    distance = np.empty((len(points), len(centers)))
    # Avoid a point × atom × xyz temporary; this workspace is charged by plans.
    for a, center in enumerate(centers):
        delta = points - center
        distance[:, a] = np.hypot(np.hypot(delta[:, 0], delta[:, 1]), delta[:, 2])
    logs = np.zeros_like(distance)
    with np.errstate(divide="ignore"):
        for a in range(len(centers)):
            for b in range(a):
                delta = centers[a] - centers[b]
                separation = float(np.hypot(np.hypot(delta[0], delta[1]), delta[2]))
                if separation <= coincident_tolerance:
                    mu = np.zeros(len(points))
                else:
                    mu = np.clip((distance[:, a] - distance[:, b]) / separation, -1, 1)
                for _ in range(iterations):
                    mu = 0.5 * mu * (3 - mu * mu)
                pair = np.clip(0.5 * (1 - mu), 0, 1)
                logs[:, a] += np.log(pair)
                logs[:, b] += np.log1p(-pair)
    logs -= np.max(logs, axis=1, keepdims=True)
    np.exp(logs, out=logs)
    logs /= np.sum(logs, axis=1, keepdims=True)
    return immutable(logs)


@dataclass(frozen=True, eq=False)
class GridTile:
    """Owned point/weight tile and global point offset; weights include r² dr dΩ."""

    begin: int
    points: np.ndarray
    weights: np.ndarray
    owners: tuple[int, ...]


@dataclass(frozen=True, eq=False)
class MolecularGrid:
    """Prepared radial/angular topology; complete molecular points are streamed.

    Geometry, atom order, charge/spin policy and every grid parameter enter the
    identity. Construction retains no molecular grid-by-AO tensor.
    """

    if TYPE_CHECKING:
        centers: ClassVar[np.ndarray]
        directions: ClassVar[np.ndarray]
        angular_weights: ClassVar[np.ndarray]
        radii: ClassVar[np.ndarray]
        radial_weights: ClassVar[np.ndarray]
        resolved_radii: ClassVar[tuple[float, ...]]
        npoint: ClassVar[int]
        identity: ClassVar[str]
        numeric_bytes: ClassVar[int]
        setup_scratch_bytes: ClassVar[int]

    atoms: tuple
    spec: GridSpec = GridSpec()
    charge: int = 0
    multiplicity: int = 1

    def __post_init__(self) -> None:
        atoms = owned_atoms(self.atoms)
        if not isinstance(self.spec, GridSpec):
            raise TypeError("expected GridSpec")
        checked_int(self.charge, "charge", low=-(2**31), high=2**31 - 1)
        checked_int(self.multiplicity, "multiplicity")
        object.__setattr__(self, "atoms", atoms)
        centers = immutable([a.position for a in atoms])
        radii = dict(self.spec.element_radii)
        if self.spec.version == 1:
            resolved = tuple(radii.get(a.atomic_number, 1.0) for a in atoms)
        else:
            missing = sorted(
                {a.atomic_number for a in atoms if a.atomic_number not in radii}
            )
            if missing:
                raise ValueError(
                    "production GridSpec v2 has no sourced radius for atomic number(s) "
                    + ", ".join(map(str, missing))
                )
            resolved = tuple(radii[a.atomic_number] for a in atoms)
        z, wz = np.polynomial.legendre.leggauss(self.spec.angular_polar)
        phi = np.arange(self.spec.angular_azimuth) * (
            2 * np.pi / self.spec.angular_azimuth
        )
        zz, pp = np.meshgrid(z, phi, indexing="ij")
        rr = np.sqrt(1 - zz * zz)
        angular = np.stack((rr * np.cos(pp), rr * np.sin(pp), zz), axis=-1).reshape(
            -1, 3
        )
        angular_weights = np.repeat(
            wz * (2 * np.pi / self.spec.angular_azimuth), len(phi)
        )
        t, wt = np.polynomial.legendre.leggauss(self.spec.radial_points)
        t, wt = (t + 1) * 0.5, wt * 0.5
        r = t / (1 - t)
        radial_weights = wt * r * r / (1 - t) ** 2
        for name, value in (
            ("centers", centers),
            ("directions", angular),
            ("angular_weights", angular_weights),
            ("radii", r),
            ("radial_weights", radial_weights),
        ):
            object.__setattr__(self, name, immutable(value))
        object.__setattr__(self, "resolved_radii", resolved)
        object.__setattr__(self, "npoint", len(atoms) * len(r) * len(angular))
        object.__setattr__(
            self,
            "identity",
            canonical_hash(
                {
                    "atoms": [asdict(a) for a in atoms],
                    "grid": asdict(self.spec),
                    "resolved_radii_bohr": resolved,
                    "charge": self.charge,
                    "multiplicity": self.multiplicity,
                    "charge_spin_policy": "independent-v1",
                }
            ),
        )
        object.__setattr__(
            self,
            "numeric_bytes",
            sum(
                getattr(self, n).nbytes
                for n in (
                    "centers",
                    "directions",
                    "angular_weights",
                    "radii",
                    "radial_weights",
                )
            )
            + 40 * len(atoms),
        )
        # NumPy's Legendre construction uses dense companion eigensolves.
        # Their numeric temporaries are setup-only and independent of grid
        # point count; conservatively charge them in prepared capacity bounds.
        object.__setattr__(
            self,
            "setup_scratch_bytes",
            64 * max(self.spec.radial_points, self.spec.angular_polar) ** 2
            + 128 * len(angular),
        )

    def _raw_tiles(self, tile_points: typing.Any = 256) -> typing.Any:
        """Shared atomic points/measures before molecular partitioning."""
        checked_int(tile_points, "tile points")
        na = len(self.directions)
        per_atom = len(self.radii) * na
        for begin in range(0, self.npoint, tile_points):
            indices = np.arange(begin, min(begin + tile_points, self.npoint))
            owner, local = np.divmod(indices, per_atom)
            radial, angular = np.divmod(local, na)
            scale = np.asarray(self.resolved_radii)[owner]
            points = (
                self.centers[owner]
                + self.directions[angular] * (self.radii[radial] * scale)[:, None]
            )
            raw = self.radial_weights[radial] * scale**3 * self.angular_weights[angular]
            yield GridTile(
                begin,
                immutable(points),
                immutable(raw),
                tuple(int(x) for x in owner),
            )

    def tiles(self, tile_points: typing.Any = 256) -> typing.Any:
        """Build moved points and fresh partition weights, including final tiles."""
        for raw in self._raw_tiles(tile_points):
            partition = partition_weights(
                raw.points,
                self.centers,
                iterations=self.spec.partition_iterations,
                coincident_tolerance=self.spec.coincident_tolerance,
            )
            weights = raw.weights * partition[np.arange(len(raw.owners)), raw.owners]
            yield GridTile(raw.begin, raw.points, immutable(weights), raw.owners)

    def explicit(self, *, max_points: typing.Any = 200_000) -> typing.Any:
        """Materialize a guarded small reference grid for independent exporters."""
        checked_int(max_points, "explicit grid limit")
        if self.npoint > max_points:
            raise ValueError("explicit grid exceeds the small-grid limit; use tiles")
        points, weights, owners = [], [], []
        for tile in self.tiles():
            points.append(tile.points)
            weights.append(tile.weights)
            owners.extend(tile.owners)
        return ExplicitGrid(
            np.concatenate(points),
            np.concatenate(weights),
            tuple(owners),
            {
                "grid_spec": asdict(self.spec),
                "grid_identity": self.identity,
                "atoms": [asdict(a) for a in self.atoms],
                "units": "Bohr",
            },
        )


@dataclass(frozen=True, eq=False)
class ExplicitGrid:
    """Portable exact point/weight data; it makes no grid-convergence claim."""

    points: np.ndarray
    weights: np.ndarray
    owners: tuple[int, ...]
    provenance: dict[str, object]

    if TYPE_CHECKING:
        _provenance_json: ClassVar[str]
        identity: ClassVar[str]

    def __post_init__(self) -> None:
        points, weights = immutable(self.points), immutable(self.weights)
        owners = tuple(self.owners)
        if (
            points.ndim != 2
            or points.shape[1] != 3
            or weights.shape != (len(points),)
            or len(owners) != len(points)
        ):
            raise ValueError("invalid explicit grid shapes")
        for owner in owners:
            checked_int(owner, "point owner", low=0)
        provenance = json.dumps(self.provenance, sort_keys=True, allow_nan=False)
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "owners", owners)
        # Preserve identity even if the caller later mutates its metadata dict.
        object.__setattr__(self, "_provenance_json", provenance)
        object.__setattr__(self, "identity", canonical_hash(self.record()))

    def record(self) -> typing.Any:
        return {
            "schema": "vibeqc.explicit-grid",
            "version": 1,
            "points_bohr": self.points.tolist(),
            "weights_bohr3": self.weights.tolist(),
            "owners": self.owners,
            "provenance": json.loads(self._provenance_json),
        }

    def write(self, path: typing.Any) -> None:
        """Write explicit data and a content hash; no external program is needed."""
        Path(path).write_text(
            json.dumps({**self.record(), "sha256": self.identity}, indent=2) + "\n"
        )

    @classmethod
    def read(cls, path: typing.Any) -> typing.Any:
        """Import only versioned, hash-verified Bohr point/weight arrays."""
        data = json.loads(Path(path).read_text())
        if data["schema"] != "vibeqc.explicit-grid" or data["version"] != 1:
            raise ValueError("unsupported explicit-grid schema")
        result = cls(
            data["points_bohr"],
            data["weights_bohr3"],
            tuple(data["owners"]),
            data["provenance"],
        )
        if result.identity != data["sha256"]:
            raise ValueError("explicit grid hash mismatch")
        return result
