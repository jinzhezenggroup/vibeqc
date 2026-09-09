"""Second-order semilocal XC feature kernels for CPKS response actions."""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256

import numpy as np
from vibeqc.profiles import canonical_hash

from tools.vibeqc_dft import ExplicitGrid, MolecularGrid
from tools.vibeqc_dft.features import density_features, spin_densities
from tools.vibeqc_dft.grid import GridTile, checked_int
from tools.vibeqc_posthf.reference import immutable
from tools.vibeqc_xc.potential import assemble_potential
from tools.vibeqc_xc.program import build_program, pack_grid_features
from tools.vibeqc_xc.spec import UnsupportedXC


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


def density_feature_response(jets, density, delta_density):
    """Return first-order density-feature response for a supplied AO matrix.

    The response is linear in ``delta_density``.  It uses the same spatial AO
    jets and spin conventions as :func:`density_features`; no finite
    differences or numerical denominators are introduced.
    """
    jets = np.asarray(jets)
    if jets.ndim != 3 or jets.shape[0] != 4:
        raise ValueError("feature response requires first-order AO jets")
    d = spin_densities(density, jets.shape[2])
    dd = spin_densities(delta_density, jets.shape[2])
    value, derivatives = jets[0], jets[1:4]
    rho, gradient, tau = [], [], []
    sigma = []
    for spin in range(2):
        weighted = value @ d[spin]
        delta_weighted = value @ dd[spin]
        current_gradient = np.stack(
            [2 * np.sum(derivative * weighted, axis=1) for derivative in derivatives],
            axis=-1,
        )
        delta_rho = np.sum(value * delta_weighted, axis=1)
        delta_gradient = np.stack(
            [
                2 * np.sum(derivative * delta_weighted, axis=1)
                for derivative in derivatives
            ],
            axis=-1,
        )
        delta_tau = 0.5 * sum(
            np.sum((derivative @ dd[spin]) * derivative, axis=1)
            for derivative in derivatives
        )
        rho.append(delta_rho)
        gradient.append(delta_gradient)
        tau.append(delta_tau)
        sigma.append(current_gradient)
    sigma_aa = 2.0 * np.sum(sigma[0] * gradient[0], axis=1)
    sigma_ab = np.sum(sigma[0] * gradient[1] + sigma[1] * gradient[0], axis=1)
    sigma_bb = 2.0 * np.sum(sigma[1] * gradient[1], axis=1)
    return {
        "rho": immutable(np.asarray(rho)),
        "gradient": immutable(np.asarray(gradient)),
        "sigma": immutable(np.stack((sigma_aa, sigma_ab, sigma_bb))),
        "tau": immutable(np.asarray(tau)),
    }


class FixedDensityXCDerivativeKernel:
    """Analytic second-derivative kernel for one fixed basis/grid/density.

    The kernel evaluates the audited XC feature Hessian from #161 and contracts
    it with the exact first-order feature response.  It is deliberately bound
    to one reference density and fails closed on exact-exchange/RSH metadata or
    a nonzero tau derivative that has not been validated for CPKS.
    """

    def __init__(self, spec, basis, grid, reference_density, *, tile_points=256):
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
        self._program = build_program(spec, order=2)
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
                "backend": "fixed-density-xc-feature-hessian-v1",
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

    def apply(self, delta_density):
        """Return the AO response potential for one density response."""
        d = spin_densities(self.reference_density, self.basis.nao)
        dd = spin_densities(delta_density, self.basis.nao)
        if self.spec.spin == "unpolarized" and not np.array_equal(dd[0], dd[1]):
            raise UnsupportedXC(
                "unpolarized XC response requires equal spin-density responses"
            )
        del d
        response = np.zeros((self.basis.nao, self.basis.nao))
        for tile in _tiles(self.grid, self.tile_points):
            jets = self.basis.evaluate(tile.points, 1)
            features = density_features(jets, self.reference_density)
            delta_features = density_feature_response(
                jets, self.reference_density, delta_density
            )
            packed = pack_grid_features(self.spec, features)
            delta_packed = pack_grid_features(self.spec, delta_features)
            values = self._program.unpack(self._program.evaluate(packed))
            hessian = values["hessian"]
            tau_indices = [
                index
                for index, name in enumerate(self.spec.features)
                if name.startswith("tau")
            ]
            if tau_indices and (
                np.any(values["gradient"][tau_indices])
                or np.any(hessian[tau_indices, :])
                or np.any(hessian[:, tau_indices])
            ):
                raise UnsupportedXC(
                    "nonzero tau feature derivatives are not validated for CPKS"
                )
            delta_gradient = np.einsum(
                "ijp,jp->ip", hessian, delta_packed, optimize=False
            )
            density_gradient = features["gradient"]
            delta_density_gradient = delta_features["gradient"]
            if self.spec.spin == "unpolarized":
                density_gradient = density_gradient.sum(axis=0)
                delta_density_gradient = delta_density_gradient.sum(axis=0)
            # The response has two independent first-order terms:
            # dV = (dV/de)(de/dD) dD + (dV/d(grad D)) (d grad D/dD) dD.
            # The first term uses the response feature gradient with only the
            # reference sigma coefficients; the rho/tau coefficients are
            # supplied by the second term through the feature Hessian.
            sigma_indices = [
                index
                for index, name in enumerate(self.spec.features)
                if name.startswith("sigma")
            ]
            reference_sigma_only = np.zeros_like(values["gradient"])
            reference_sigma_only[sigma_indices] = values["gradient"][sigma_indices]
            tile_response = assemble_potential(
                self.spec,
                jets,
                delta_density_gradient,
                reference_sigma_only,
                tile.weights,
            )
            tile_response = tile_response + assemble_potential(
                self.spec,
                jets,
                density_gradient,
                delta_gradient,
                tile.weights,
            )
            # Restricted total-density variations split equally between the
            # two spin channels, so the separate alpha/beta potentials must be
            # averaged (unpolarized has one functional-spin channel).
            response += np.mean(tile_response, axis=0)
            self.statistics["tiles"] += 1
        self.statistics["actions"] += 1
        self.statistics["peak_bytes"] = max(
            self.statistics["peak_bytes"],
            4 * self.basis.nao * self.basis.nao * 8,
        )
        return immutable(response)

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
