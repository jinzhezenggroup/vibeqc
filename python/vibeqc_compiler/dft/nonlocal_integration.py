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

from .ao import NativeAO
from .features import spin_densities
from .grid import ExplicitGrid, MolecularGrid, checked_int
from .nonlocal_reference import (
    assemble_nonlocal_potential_reference,
    nonlocal_energy_reference,
    nonlocal_feature_derivatives_reference,
)


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
    backend: str = "cpu-reference"


class FixedDensityNonlocalCorrelation:
    """Execute a small-grid VV10-family energy/potential with bounded memory.

    Pair evaluation is tiled in memory, but remains quadratic in work. max_points
    is a hard admission gate rather than a performance promise. Production
    large-grid execution belongs to issue 491 slice D.
    """

    def __init__(
        self,
        spec: typing.Any,
        *,
        coefficient: typing.Any = Fraction(1),
        max_points: typing.Any = 4096,
    ) -> None:
        if not isinstance(spec, NonlocalCorrelationSpec):
            raise TypeError("expected NonlocalCorrelationSpec")
        if not isinstance(coefficient, Fraction) or coefficient <= 0:
            raise ValueError("nonlocal coefficient requires a positive Fraction")
        checked_int(max_points, "nonlocal reference max points")
        self.spec = spec
        self.coefficient = coefficient
        self.max_points = max_points

    def _validate_grid(self, basis: typing.Any, grid: typing.Any) -> None:
        if not isinstance(basis, NativeAO):
            raise TypeError("expected NativeAO")
        if not isinstance(grid, (MolecularGrid, ExplicitGrid)):
            raise TypeError("expected MolecularGrid or ExplicitGrid")
        if isinstance(grid, MolecularGrid) and (
            grid.atoms != basis.atoms
            or grid.charge != basis.charge
            or grid.multiplicity != basis.multiplicity
        ):
            raise ValueError(
                "stale molecular grid: atoms/charge/spin do not match basis"
            )
        if len(grid.points) > self.max_points:
            raise ValueError(
                "nonlocal CPU reference exceeds its explicit max_points admission gate"
            )

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
        self._validate_grid(basis, grid)
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
        coefficient = float(self.coefficient)
        energy = coefficient * nonlocal_energy_reference(
            points,
            weights,
            rho,
            gradient,
            self.spec,
            tile_size=tile_points,
        )
        vrho, vsigma = nonlocal_feature_derivatives_reference(
            points,
            weights,
            rho,
            gradient,
            self.spec,
            tile_size=tile_points,
        )
        vrho *= coefficient
        vsigma *= coefficient

        potential = np.zeros((basis.nao, basis.nao), dtype=np.float64)
        for begin in range(0, ngrid, tile_points):
            end = min(begin + tile_points, ngrid)
            jets = basis.evaluate(points[begin:end], 1)
            potential += assemble_nonlocal_potential_reference(
                jets,
                weights[begin:end],
                gradient[begin:end],
                vrho[begin:end],
                vsigma[begin:end],
            )
        potential = 0.5 * (potential + potential.T)
        published = np.stack((potential, potential)) if separate else potential
        density_identity = canonical_hash(
            {
                "spin_density_sha256": sha256(spin_density.tobytes()).hexdigest(),
                "layout": "separate" if separate else "total",
                "nao": basis.nao,
            }
        )
        payload = {
            "schema": "vibeqc.fixed-density-nonlocal-reference/v1",
            "basis_identity": basis.identity,
            "grid_identity": grid.identity,
            "spec_identity": self.spec.identity,
            "coefficient": str(self.coefficient),
            "density_identity": density_identity,
            "max_points": self.max_points,
        }
        return NonlocalIntegral(
            energy=float(energy),
            potential=immutable(published),
            identity=canonical_hash(payload),
            basis_identity=basis.identity,
            grid_identity=grid.identity,
            spec_identity=self.spec.identity,
            density_identity=density_identity,
            points=ngrid,
            tiles=tiles,
        )
