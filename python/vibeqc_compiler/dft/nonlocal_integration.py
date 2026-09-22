"""Bounded fixed-density CPU execution for VV10/rVV10 energy and AO potential."""

from __future__ import annotations

import typing
from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256

import numpy as np

from vibeqc_compiler.common.arrays import immutable
from vibeqc_compiler.common.nonlocal_correlation import NonlocalCorrelationSpec
from vibeqc_compiler.common.provenance import canonical_hash

from .ao import NativeAO, jet_indices
from .features import spin_densities
from .grid import ExplicitGrid, MolecularGrid, checked_int
from .nonlocal_reference import (
    nonlocal_energy_reference,
    nonlocal_explicit_geometry_derivatives_reference,
    nonlocal_feature_derivatives_reference,
)


@dataclass(frozen=True, eq=False)
class NonlocalGeometry:
    """Complete fixed-density VV10 geometric partials before grid chain rule."""

    centers: np.ndarray
    points: np.ndarray
    weights: np.ndarray
    identity: str
    basis_identity: str
    grid_identity: str
    quadrature_identity: str
    spec_identity: str
    density_identity: str
    coefficient: Fraction
    backend: str = "cpu-reference"
    host_workspace_bytes: int = 0
    device_workspace_bytes: int = 0
    pair_evaluations: int = 0
    provider_identity: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.coefficient, Fraction) or self.coefficient <= 0:
            raise ValueError(
                "nonlocal geometry coefficient requires a positive Fraction"
            )
        centers = immutable(self.centers)
        points = immutable(self.points)
        weights = immutable(self.weights)
        if centers.ndim != 2 or centers.shape[1:] != (3,):
            raise ValueError("nonlocal centers require [atom,xyz]")
        if points.ndim != 2 or points.shape[1:] != (3,):
            raise ValueError("nonlocal points require [point,xyz]")
        if weights.shape != (len(points),):
            raise ValueError("nonlocal weights require one value per point")
        if not all(np.isfinite(value).all() for value in (centers, points, weights)):
            raise ValueError("nonlocal geometry partials must be finite")
        object.__setattr__(self, "centers", centers)
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "weights", weights)

    def directional(
        self, *, centers: typing.Any, points: typing.Any, weights: typing.Any
    ) -> float:
        """Contract independent AO-center, grid-point and quadrature motions."""
        dc = immutable(centers, shape=self.centers.shape)
        dp = immutable(points, shape=self.points.shape)
        dw = immutable(weights, shape=self.weights.shape)
        value = float(
            np.sum(self.centers * dc)
            + np.sum(self.points * dp)
            + np.sum(self.weights * dw)
        )
        if not np.isfinite(value):
            raise ArithmeticError("nonfinite nonlocal directional gradient")
        return value

    def validate_replay(
        self,
        basis: typing.Any,
        grid: typing.Any,
        density: typing.Any,
        *,
        spec: NonlocalCorrelationSpec,
        coefficient: Fraction,
    ) -> ExplicitGrid:
        """Revalidate the current kernel, coefficient, basis, density and grid."""
        if not isinstance(spec, NonlocalCorrelationSpec):
            raise TypeError("nonlocal geometry replay requires NonlocalCorrelationSpec")
        if not isinstance(coefficient, Fraction) or coefficient <= 0:
            raise ValueError("nonlocal replay coefficient requires a positive Fraction")
        if spec.identity != self.spec_identity:
            raise ValueError("nonlocal geometry/specification identity mismatch")
        if coefficient != self.coefficient:
            raise ValueError("nonlocal geometry/coefficient identity mismatch")
        if not isinstance(basis, NativeAO):
            raise TypeError("nonlocal geometry replay requires NativeAO")
        if basis.identity != self.basis_identity:
            raise ValueError("nonlocal geometry/basis identity mismatch")
        separate = np.asarray(density).ndim == 3
        spin_density = spin_densities(density, basis.nao)
        if (
            _density_identity(spin_density, separate, basis.nao)
            != self.density_identity
        ):
            raise ValueError("nonlocal geometry/density identity mismatch")
        if not isinstance(grid, MolecularGrid):
            raise TypeError("nonlocal geometry replay requires MolecularGrid")
        if grid.identity != self.grid_identity:
            raise ValueError("nonlocal geometry/grid identity mismatch")
        explicit = grid.explicit(max_points=len(self.points))
        if explicit.identity != self.quadrature_identity:
            raise ValueError("nonlocal geometry/quadrature identity mismatch")
        return explicit


@dataclass(frozen=True, eq=False)
class NonlocalIntegral:
    """Fixed-density nonlocal energy and exact discrete AO derivative."""

    energy: float
    potential: np.ndarray
    identity: str
    basis_identity: str
    grid_identity: str
    spec_identity: str
    density_identity: str
    points: int
    tiles: int
    quadrature_identity: str = ""
    backend: str = "cpu-reference"
    host_workspace_bytes: int = 0
    device_workspace_bytes: int = 0
    pair_evaluations: int = 0
    provider_identity: str | None = None


def _density_identity(spin_density: np.ndarray, separate: bool, nao: int) -> str:
    return canonical_hash(
        {
            "spin_density_sha256": sha256(spin_density.tobytes()).hexdigest(),
            "layout": "separate" if separate else "total",
            "nao": nao,
        }
    )


def _ao_atoms(basis: NativeAO) -> np.ndarray:
    counts = [
        2 * shell.angular_momentum + 1
        if basis.representation == "real_spherical"
        else (shell.angular_momentum + 1) * (shell.angular_momentum + 2) // 2
        for shell in basis.shells
    ]
    atoms = np.repeat([shell.atom_index for shell in basis.shells], counts)
    if atoms.shape != (basis.nao,):
        raise ValueError("native AO ownership is inconsistent with the basis")
    return atoms


class FixedDensityNonlocalCorrelation:
    """Execute a small-grid VV10-family energy/potential with bounded memory.

    Pair evaluation is quadratic in work.  With no pair_provider this remains the
    small independent CPU oracle with a hard max_points gate; a runtime-owned
    pair_provider supplies the bounded native production lowerer.
    """

    def __init__(
        self,
        spec: typing.Any,
        *,
        coefficient: typing.Any = Fraction(1),
        max_points: typing.Any = 4096,
        pair_provider: typing.Any = None,
    ) -> None:
        if not isinstance(spec, NonlocalCorrelationSpec):
            raise TypeError("expected NonlocalCorrelationSpec")
        if not isinstance(coefficient, Fraction) or coefficient <= 0:
            raise ValueError("nonlocal coefficient requires a positive Fraction")
        checked_int(max_points, "nonlocal reference max points")
        self.spec = spec
        self.coefficient = coefficient
        self.max_points = max_points
        self.pair_provider = pair_provider

    def _validate_grid(
        self, basis: typing.Any, grid: typing.Any
    ) -> tuple[ExplicitGrid, str]:
        if not isinstance(basis, NativeAO):
            raise TypeError("expected NativeAO")
        if not isinstance(grid, (MolecularGrid, ExplicitGrid)):
            raise TypeError("expected MolecularGrid or ExplicitGrid")
        if isinstance(grid, MolecularGrid):
            if (
                grid.atoms != basis.atoms
                or grid.charge != basis.charge
                or grid.multiplicity != basis.multiplicity
            ):
                raise ValueError(
                    "stale molecular grid: atoms/charge/spin do not match basis"
                )
            limit = self.max_points if self.pair_provider is None else grid.npoint
            return grid.explicit(max_points=limit), grid.identity
        if self.pair_provider is None and len(grid.points) > self.max_points:
            raise ValueError(
                "nonlocal CPU reference exceeds its explicit max_points admission gate"
            )
        return grid, grid.identity

    @staticmethod
    def _total_features(jets: typing.Any, total_density: typing.Any) -> typing.Any:
        phi = jets[0]
        weighted = phi @ total_density
        rho = np.sum(phi * weighted, axis=1)
        gradient = np.stack(
            [2.0 * np.sum(derivative * weighted, axis=1) for derivative in jets[1:4]],
            axis=1,
        )
        return rho, gradient

    @staticmethod
    def _assemble_potential(
        jets: np.ndarray,
        weights: np.ndarray,
        density_gradient: np.ndarray,
        vrho: np.ndarray,
        vsigma: np.ndarray,
    ) -> np.ndarray:
        """Assemble the total-density AO potential from provider feature derivatives."""
        phi = jets[0]
        derivatives = jets[1:4]
        matrix = phi.T @ ((weights * vrho)[:, None] * phi)
        spatial = 2.0 * vsigma[:, None] * density_gradient
        panel = sum(spatial[:, k, None] * derivatives[k] for k in range(3))
        cross = phi.T @ (weights[:, None] * panel)
        matrix += cross + cross.T
        return 0.5 * (matrix + matrix.T)

    def integrate(
        self,
        basis: typing.Any,
        grid: typing.Any,
        density: typing.Any,
        *,
        tile_points: typing.Any = 256,
    ) -> typing.Any:
        """Return E_nlc and V_nlc with delta E = Tr(V delta D)."""
        checked_int(tile_points, "nonlocal AO tile points")
        grid, source_grid_identity = self._validate_grid(basis, grid)
        separate = np.asarray(density).ndim == 3
        spin_density = spin_densities(density, basis.nao)
        total_density = np.asarray(spin_density[0] + spin_density[1])
        points = np.asarray(grid.points, dtype=np.float64)
        weights = np.asarray(grid.weights, dtype=np.float64)
        ngrid = len(points)
        rho = np.empty(ngrid, dtype=np.float64)
        gradient = np.empty((ngrid, 3), dtype=np.float64)
        tiles = 0

        for begin in range(0, ngrid, tile_points):
            end = min(begin + tile_points, ngrid)
            jets = basis.evaluate(points[begin:end], 1)
            rho[begin:end], gradient[begin:end] = self._total_features(
                jets, total_density
            )
            tiles += 1
        native = None
        if self.pair_provider is None:
            coefficient = float(self.coefficient)
            energy = coefficient * nonlocal_energy_reference(
                points, weights, rho, gradient, self.spec, tile_size=tile_points
            )
            vrho, vsigma = nonlocal_feature_derivatives_reference(
                points, weights, rho, gradient, self.spec, tile_size=tile_points
            )
            vrho *= coefficient
            vsigma *= coefficient
        else:
            native = self.pair_provider.evaluate(
                points,
                weights,
                rho,
                gradient,
                self.spec,
                self.coefficient,
                tile_points=tile_points,
            )
            energy = native.energy
            vrho = np.asarray(native.vrho)
            vsigma = np.asarray(native.vsigma)

        potential = np.zeros((basis.nao, basis.nao), dtype=np.float64)
        for begin in range(0, ngrid, tile_points):
            end = min(begin + tile_points, ngrid)
            jets = basis.evaluate(points[begin:end], 1)
            potential += self._assemble_potential(
                jets,
                weights[begin:end],
                gradient[begin:end],
                vrho[begin:end],
                vsigma[begin:end],
            )
        potential = 0.5 * (potential + potential.T)
        published = np.stack((potential, potential)) if separate else potential
        density_identity = _density_identity(spin_density, separate, basis.nao)
        payload = {
            "schema": "vibeqc.fixed-density-nonlocal-reference/v1",
            "basis_identity": basis.identity,
            "grid_identity": source_grid_identity,
            "quadrature_identity": grid.identity,
            "spec_identity": self.spec.identity,
            "coefficient": str(self.coefficient),
            "density_identity": density_identity,
            "max_points": self.max_points,
        }
        backend = "cpu-reference"
        if native is not None:
            backend = f"native-{native.backend}"
            payload = {
                **payload,
                "schema": "vibeqc.fixed-density-nonlocal-native/v1",
                "provider": native.provider_identity,
                "pair_evaluation": native.identity,
                "backend": backend,
                "host_workspace_bytes": native.host_workspace_bytes,
                "device_workspace_bytes": native.device_workspace_bytes,
                "pair_evaluations": native.pair_evaluations,
            }
        return NonlocalIntegral(
            energy=float(energy),
            potential=immutable(published),
            identity=canonical_hash(payload),
            basis_identity=basis.identity,
            grid_identity=source_grid_identity,
            quadrature_identity=grid.identity,
            spec_identity=self.spec.identity,
            density_identity=density_identity,
            points=ngrid,
            tiles=tiles,
            backend=backend,
            host_workspace_bytes=0 if native is None else native.host_workspace_bytes,
            device_workspace_bytes=0
            if native is None
            else native.device_workspace_bytes,
            pair_evaluations=0 if native is None else native.pair_evaluations,
            provider_identity=None if native is None else native.provider_identity,
        )

    def geometry(
        self,
        basis: typing.Any,
        grid: typing.Any,
        density: typing.Any,
        *,
        tile_points: typing.Any = 256,
    ) -> NonlocalGeometry:
        """Return all fixed-density first geometric partials for #491C.

        The AO-center and grid-point feature chain uses second AO jets.  Explicit
        pair-distance and both quadrature-weight legs come from the same audited
        nonlocal definition.  Orbital response and other stationary sources are
        intentionally outside this primitive.
        """
        checked_int(tile_points, "nonlocal geometry tile points")
        grid, source_grid_identity = self._validate_grid(basis, grid)
        separate = np.asarray(density).ndim == 3
        spin_density = spin_densities(density, basis.nao)
        total_density = np.asarray(spin_density[0] + spin_density[1])
        points = np.asarray(grid.points, dtype=np.float64)
        weights = np.asarray(grid.weights, dtype=np.float64)
        ngrid = len(points)
        rho = np.empty(ngrid, dtype=np.float64)
        gradient = np.empty((ngrid, 3), dtype=np.float64)
        for begin in range(0, ngrid, tile_points):
            end = min(begin + tile_points, ngrid)
            jets = basis.evaluate(points[begin:end], 1)
            rho[begin:end], gradient[begin:end] = self._total_features(
                jets, total_density
            )

        native = None
        if self.pair_provider is None:
            vrho, vsigma = nonlocal_feature_derivatives_reference(
                points, weights, rho, gradient, self.spec, tile_size=tile_points
            )
            explicit_points, weight_partials = (
                nonlocal_explicit_geometry_derivatives_reference(
                    points, weights, rho, gradient, self.spec, tile_size=tile_points
                )
            )
            coefficient = float(self.coefficient)
            vrho *= coefficient
            vsigma *= coefficient
            point_partials = coefficient * explicit_points
            weight_partials = coefficient * weight_partials
        else:
            native = self.pair_provider.evaluate(
                points,
                weights,
                rho,
                gradient,
                self.spec,
                self.coefficient,
                tile_points=tile_points,
                geometry=True,
            )
            vrho = np.asarray(native.vrho)
            vsigma = np.asarray(native.vsigma)
            point_partials = np.array(native.point_derivative, copy=True)
            weight_partials = np.array(native.weight_derivative, copy=True)
        center_partials = np.zeros((basis.natom, 3), dtype=np.float64)
        ao_atoms = _ao_atoms(basis)
        lookup = {axis: i for i, axis in enumerate(jet_indices(2))}
        first = [lookup[tuple(int(i == k) for i in range(3))] for k in range(3)]

        for begin in range(0, ngrid, tile_points):
            end = min(begin + tile_points, ngrid)
            jets = basis.evaluate(points[begin:end], 2)
            phi = jets[0]
            derivatives = [jets[index] for index in first]
            weighted = phi @ total_density
            derivative_work = [value @ total_density for value in derivatives]
            local_gradient = gradient[begin:end]
            gradient_coeff = 2.0 * vsigma[begin:end, None] * local_gradient
            hessian = np.empty((end - begin, 3, 3), dtype=np.float64)
            for j in range(3):
                for k in range(3):
                    axis = [0, 0, 0]
                    axis[j] += 1
                    axis[k] += 1
                    second = jets[lookup[tuple(axis)]]
                    hessian[:, j, k] = 2.0 * (
                        np.sum(second * weighted, axis=1)
                        + np.sum(derivatives[j] * derivative_work[k], axis=1)
                    )
            feature_points = np.empty((end - begin, 3), dtype=np.float64)
            for k in range(3):
                feature_points[:, k] = weights[begin:end] * (
                    vrho[begin:end] * local_gradient[:, k]
                    + np.einsum("pj,pj->p", gradient_coeff, hessian[:, :, k])
                )
            point_partials[begin:end] += feature_points

            for atom in range(basis.natom):
                owned = ao_atoms == atom
                for k in range(3):
                    drho = -2.0 * np.sum(
                        derivatives[k][:, owned] * weighted[:, owned], axis=1
                    )
                    dgradient = np.empty((end - begin, 3), dtype=np.float64)
                    for j in range(3):
                        axis = [0, 0, 0]
                        axis[j] += 1
                        axis[k] += 1
                        second = jets[lookup[tuple(axis)]]
                        dgradient[:, j] = -2.0 * (
                            np.sum(second[:, owned] * weighted[:, owned], axis=1)
                            + np.sum(
                                derivatives[k][:, owned] * derivative_work[j][:, owned],
                                axis=1,
                            )
                        )
                    center_partials[atom, k] += float(
                        np.sum(
                            weights[begin:end]
                            * (
                                vrho[begin:end] * drho
                                + np.einsum("pj,pj->p", gradient_coeff, dgradient)
                            )
                        )
                    )

        if not all(
            np.isfinite(value).all()
            for value in (center_partials, point_partials, weight_partials)
        ):
            raise FloatingPointError("nonfinite nonlocal geometric pullback")
        density_identity = _density_identity(spin_density, separate, basis.nao)
        payload = {
            "schema": "vibeqc.fixed-density-nonlocal-geometry/v1",
            "basis_identity": basis.identity,
            "grid_identity": source_grid_identity,
            "quadrature_identity": grid.identity,
            "spec_identity": self.spec.identity,
            "coefficient": str(self.coefficient),
            "density_identity": density_identity,
            "sources": ("nonlocal_ao", "nonlocal_grid", "nonlocal_weight"),
        }
        backend = "cpu-reference"
        if native is not None:
            backend = f"native-{native.backend}"
            payload = {
                **payload,
                "schema": "vibeqc.fixed-density-nonlocal-geometry-native/v1",
                "provider": native.provider_identity,
                "pair_evaluation": native.identity,
                "backend": backend,
                "host_workspace_bytes": native.host_workspace_bytes,
                "device_workspace_bytes": native.device_workspace_bytes,
                "pair_evaluations": native.pair_evaluations,
            }
        return NonlocalGeometry(
            centers=immutable(center_partials),
            points=immutable(point_partials),
            weights=immutable(weight_partials),
            identity=canonical_hash(payload),
            basis_identity=basis.identity,
            grid_identity=source_grid_identity,
            quadrature_identity=grid.identity,
            spec_identity=self.spec.identity,
            density_identity=density_identity,
            coefficient=self.coefficient,
            backend=backend,
            host_workspace_bytes=0 if native is None else native.host_workspace_bytes,
            device_workspace_bytes=0
            if native is None
            else native.device_workspace_bytes,
            pair_evaluations=0 if native is None else native.pair_evaluations,
            provider_identity=None if native is None else native.provider_identity,
        )
