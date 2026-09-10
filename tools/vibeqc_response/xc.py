"""Second-order semilocal XC feature kernels for CPKS response actions."""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
from time import perf_counter

import numpy as np
from vibeqc.profiles import canonical_hash
from vibeqc_compiler.dft import ExplicitGrid, MolecularGrid
from vibeqc_compiler.dft.features import spin_densities
from vibeqc_compiler.dft.grid import GridTile, checked_int
from vibeqc_compiler.xc.contractions import (
    ContractionProgram,
    density_feature_response,
)
from vibeqc_compiler.xc.spec import UnsupportedXC

from tools.vibeqc_posthf.reference import immutable

__all__ = ["FixedDensityXCDerivativeKernel", "density_feature_response"]


def _tiles(grid, tile_points):
    if isinstance(grid, MolecularGrid):
        yield from grid.tiles(tile_points)
    else:
        for begin in range(0, len(grid.points), tile_points):
            end = begin + tile_points
            yield GridTile(
                begin,
                grid.points[begin:end],
                grid.weights[begin:end],
                grid.owners[begin:end],
            )


class FixedDensityXCDerivativeKernel:
    """Analytic second-derivative kernel for one fixed basis/grid/density.

    The kernel evaluates the audited XC feature Hessian from #161 and contracts
    it with the exact first-order feature response.  It is deliberately bound
    to one reference density and fails closed on exact-exchange/RSH metadata or
    a nonzero tau derivative that has not been validated for CPKS.

    An optional borrowed ``prepared`` response owner selects bounded native
    CPU execution with the identical contract, basis and grid. Its resource
    plan defines the numeric-capacity scope of the reported peak statistic;
    closing the kernel does not transfer ownership of that prepared object.
    """

    def __init__(
        self, spec, basis, grid, reference_density, *, tile_points=256, prepared=None
    ):
        checked_int(tile_points, "XC response tile points")
        if spec.exact_exchange or spec.range_omega or spec.long_range_exchange:
            raise UnsupportedXC(
                "CPKS response currently supports semilocal LDA/GGA only; "
                "exact exchange/RSH derivatives are not available"
            )
        if isinstance(grid, MolecularGrid) and (
            grid.atoms != basis.atoms
            or grid.charge != basis.charge
            or grid.multiplicity != basis.multiplicity
        ):
            raise ValueError("stale molecular grid for XC response kernel")
        if not isinstance(grid, (MolecularGrid, ExplicitGrid)):
            raise TypeError("expected MolecularGrid or ExplicitGrid")
        self.spec = spec
        self.basis = basis
        self.grid = grid
        self.reference_density = immutable(reference_density)
        self.tile_points = tile_points
        self._contraction = ContractionProgram(spec, "response")
        if prepared is not None:
            from vibeqc_compiler.xc.prepared import PreparedXCContractions

            if not isinstance(prepared, PreparedXCContractions):
                raise TypeError("expected a prepared native XC contraction owner")
            if (
                prepared.program.contract.identity
                != self._contraction.contract.identity
                or prepared.basis.identity != basis.identity
                or prepared.grid.identity != grid.identity
            ):
                raise ValueError("prepared XC response contract/basis/grid mismatch")
            prepared._check()
        self._prepared = prepared
        self._program = self._contraction.program
        self.geometry_hash = canonical_hash([asdict(atom) for atom in basis.atoms])
        self.basis_hash = canonical_hash(
            {
                "shells": [asdict(shell) for shell in basis.shells],
                "representation": basis.representation,
            }
        )
        self.basis_identity = self.basis_hash
        self.grid_identity = grid.identity
        self.functional_identity = spec.identity
        self.identity = canonical_hash(
            {
                "backend": "fixed-density-xc-feature-hessian-v2",
                "contraction": self._contraction.contract.identity,
                "prepared": None if prepared is None else prepared.identity,
                "functional_identity": self.functional_identity,
                "basis_hash": self.basis_hash,
                "geometry_hash": self.geometry_hash,
                "grid_identity": self.grid_identity,
                "density_sha256": sha256(
                    np.ascontiguousarray(self.reference_density, dtype="<f8").tobytes()
                ).hexdigest(),
                "tile_points": tile_points,
            }
        )
        self.statistics = {
            "actions": 0,
            "tiles": 0,
            "seconds": 0.0,
            "peak_bytes": 0,
        }

    def apply_spin(self, delta_density):
        """Return functional-spin response before the restricted solver reduction.

        Polarized requests keep alpha/beta and cross-spin terms separately.
        Unpolarized requests have one total-density functional channel. The
        common contraction layer owns all XC differentiation and assembly.
        """
        started = perf_counter()
        density = spin_densities(self.reference_density, self.basis.nao)
        direction = spin_densities(delta_density, self.basis.nao)
        if self.spec.spin == "unpolarized" and (
            not np.array_equal(density[0], density[1])
            or not np.array_equal(direction[0], direction[1])
        ):
            raise UnsupportedXC(
                "unpolarized response requires equal spin matrices and directions"
            )
        nspin = 2 if self.spec.spin == "polarized" else 1
        if self._prepared is not None:
            result = self._prepared.execute(density, delta_density=direction)
            self.statistics["actions"] += 1
            self.statistics["tiles"] += self._prepared.statistics["tiles"]
            self.statistics["seconds"] += perf_counter() - started
            self.statistics["peak_bytes"] = self._prepared.resource_plan.peak_bytes[
                "host"
            ]
            return result["response"]
        response = np.zeros((nspin, self.basis.nao, self.basis.nao))
        for tile in _tiles(self.grid, self.tile_points):
            jets = self.basis.evaluate(tile.points, self._contraction.contract.ao_order)
            values = self._contraction.evaluate(
                jets,
                density,
                tile.weights,
                delta_density=direction,
            )
            response += values["response"]
            self.statistics["tiles"] += 1
        self.statistics["actions"] += 1
        self.statistics["seconds"] += perf_counter() - started
        # Preserve the legacy statistic's limited matrix-capacity scope.
        # A complete native execution plan reports its separate resource bound.
        self.statistics["peak_bytes"] = max(
            self.statistics["peak_bytes"],
            4 * self.basis.nao * self.basis.nao * 8,
        )
        return immutable(response)

    def apply(self, delta_density):
        """Restricted solver adapter: total-D directions split equally by spin."""
        return immutable(self.apply_spin(delta_density).mean(axis=0))

    def apply_transpose(self, delta_density):
        """Apply the symmetric semilocal kernel transpose."""
        return self.apply(delta_density)

    def validate_reference(self, reference):
        """Require the kernel to belong to the exact converged KS reference."""
        if reference.geometry_hash != self.geometry_hash:
            raise ValueError("XC kernel/reference geometry mismatch")
        if reference.basis_hash != self.basis_hash:
            raise ValueError("XC kernel/reference basis mismatch")
        if reference.functional_identity != self.functional_identity:
            raise ValueError("XC kernel/reference functional mismatch")
        if reference.grid_identity != self.grid_identity:
            raise ValueError("XC kernel/reference grid mismatch")
        expected = (reference.coefficients * reference.occupations) @ (
            reference.coefficients.T
        )
        expected_spins = spin_densities(expected, self.basis.nao)
        actual_spins = spin_densities(self.reference_density, self.basis.nao)
        error = float(np.max(np.abs(expected_spins - actual_spins)))
        if error > reference.validation_tolerance:
            raise ValueError(
                f"XC kernel/reference density mismatch: max difference {error:.3e}"
            )
        return self
