"""Original table-free atom-centered quadrature and equal-radius Becke weights.

Angular integration uses a Gauss-Legendre polar × periodic trapezoidal rule,
not a Lebedev table. Radial r=R*t/(1-t), t in (0,1), uses Gauss-Legendre nodes.
The recorded element radii set R; they do not add a heteronuclear partition
correction. No empirical angular/radius tables or runtime downloads are used.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from vibeqc import Atom
from vibeqc.profiles import canonical_hash

from tools.vibeqc_posthf.reference import immutable


def checked_int(value, name, low=1, high=2**31 - 1):
    """Reject booleans, truncation and overflow at every public size boundary."""
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return value


def owned_atoms(atoms):
    """Canonicalize all coordinate storage, including user-constructed Atoms."""
    result = []
    for atom in atoms:
        a = Atom.from_value(atom)
        checked_int(a.atomic_number, "atomic number", high=118)
        xyz = immutable(a.position, shape=(3,))
        result.append(Atom(a.atomic_number, tuple(float(x) for x in xyz)))
    if not result:
        raise ValueError("a grid requires atoms")
    return tuple(result)


@dataclass(frozen=True)
class GridSpec:
    """Versioned, fully specified quadrature; all lengths are in Bohr.

    Default radial radii are exactly one Bohr for every element. Overrides are
    explicit (atomic_number, radius) pairs, not an opaque accuracy level.
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

    def __post_init__(self):
        checked_int(self.version, "grid version", high=1)
        checked_int(self.radial_points, "radial points", high=512)
        checked_int(self.angular_polar, "polar points", high=256)
        checked_int(self.angular_azimuth, "azimuth points", low=3, high=1024)
        checked_int(self.partition_iterations, "partition iterations", high=5)
        if (
            self.version,
            self.radial_rule,
            self.angular_rule,
            self.partition,
            self.pruning,
            self.units,
            self.ordering,
        ) != (
            1,
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


def partition_weights(points, centers, *, iterations=3, coincident_tolerance=1e-12):
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

    atoms: tuple
    spec: GridSpec = GridSpec()
    charge: int = 0
    multiplicity: int = 1

    def __post_init__(self):
        atoms = owned_atoms(self.atoms)
        if not isinstance(self.spec, GridSpec):
            raise TypeError("expected GridSpec")
        checked_int(self.charge, "charge", low=-(2**31), high=2**31 - 1)
        checked_int(self.multiplicity, "multiplicity")
        object.__setattr__(self, "atoms", atoms)
        centers = immutable([a.position for a in atoms])
        radii = dict(self.spec.element_radii)
        resolved = tuple(radii.get(a.atomic_number, 1.0) for a in atoms)
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

    def tiles(self, tile_points=256):
        """Build moved points and fresh partition weights, including final tiles."""
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
            partition = partition_weights(
                points,
                self.centers,
                iterations=self.spec.partition_iterations,
                coincident_tolerance=self.spec.coincident_tolerance,
            )
            weights = raw * partition[np.arange(len(indices)), owner]
            yield GridTile(
                begin,
                immutable(points),
                immutable(weights),
                tuple(int(x) for x in owner),
            )

    def explicit(self, *, max_points=200_000):
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
    provenance: dict

    def __post_init__(self):
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

    def record(self):
        return {
            "schema": "vibeqc.explicit-grid",
            "version": 1,
            "points_bohr": self.points.tolist(),
            "weights_bohr3": self.weights.tolist(),
            "owners": self.owners,
            "provenance": json.loads(self._provenance_json),
        }

    def write(self, path):
        """Write explicit data and a content hash; no external program is needed."""
        Path(path).write_text(
            json.dumps({**self.record(), "sha256": self.identity}, indent=2) + "\n"
        )

    @classmethod
    def read(cls, path):
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
