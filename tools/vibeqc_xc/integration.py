"""Fixed-density CPU XC integration; no SCF or public method registration."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

import numpy as np
from vibeqc.profiles import canonical_hash

from tools.vibeqc_dft import ExplicitGrid, MolecularGrid, NativeAO
from tools.vibeqc_dft.features import density_features, spin_densities
from tools.vibeqc_dft.grid import GridTile, checked_int
from tools.vibeqc_posthf.reference import immutable

from .potential import assemble_potential
from .program import build_program, pack_grid_features
from .spec import FunctionalSpec, UnsupportedXC


@dataclass(frozen=True, eq=False)
class XCIntegral:
    """XC-only energy and derivative in the supplied density layout, in Hartree.

    ``potential`` has the same shape as D; ``electrons`` always has two spin
    entries. Both arrays own immutable storage. Electron counts are quadrature
    diagnostics, never a normalization target. Identity binds numerical inputs.
    """

    energy: float
    potential: np.ndarray
    electrons: np.ndarray
    identity: str
    basis_identity: str
    grid_identity: str
    functional_identity: str
    density_identity: str
    points: int
    tiles: int
    backend: str = "cpu"


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


class FixedDensityXC:
    """Reuse an audited first-derivative expression, not cached numerical grids.

    NativeAO evaluates bounded CPU jets; DFT01 contracts density features and
    DFT02 supplies the expression and potential coefficients. Every call reads
    the supplied immutable basis/grid and D anew. Domain errors propagate:
    neither zero weights nor grid tails authorize clipping unsupported inputs.
    """

    def __init__(self, spec):
        if not isinstance(spec, FunctionalSpec):
            raise TypeError("expected FunctionalSpec")
        if any((spec.exact_exchange, spec.range_omega, spec.long_range_exchange)):
            raise UnsupportedXC(
                "fixed-density LDA/GGA integration requires semilocal metadata"
            )
        self._program = build_program(spec, order=1)

    @property
    def spec(self):
        return self._program.spec

    def integrate(self, basis, grid, density, *, tile_points=256):
        """Return E_xc and V_xc with delta E = sum_s Tr(V_s delta D_s).

        Total [AO,AO] input means Da=Db=D/2. Separate [2,AO,AO] input
        preserves both spins. Unpolarized functionals require equal matrices;
        their derivative is with respect to total density. For separate equal
        matrices it is returned on both spin channels.

        Molecular grids must match basis atoms, charge and spin policy. An
        ExplicitGrid is deliberately a fixed laboratory-frame quadrature;
        use a new grid when molecular geometry/rules change. No identity cache
        can return an old energy, AO tile or potential.
        """
        checked_int(tile_points, "tile points")
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
        separate = np.asarray(density).ndim == 3
        d = spin_densities(density, basis.nao)
        if self.spec.spin == "unpolarized" and not np.array_equal(d[0], d[1]):
            raise UnsupportedXC("unpolarized integration requires equal spin matrices")
        energy = 0.0
        nspin = 2 if self.spec.spin == "polarized" else 1
        potential = np.zeros((nspin, basis.nao, basis.nao))
        electrons = np.zeros(2)
        points = tiles = 0
        for tile in _tiles(grid, tile_points):
            jets = basis.evaluate(tile.points, 1)
            features = density_features(jets, d)
            try:
                values = self._program.unpack(
                    self._program.evaluate(pack_grid_features(self.spec, features))
                )
            except UnsupportedXC as error:
                raise UnsupportedXC(
                    f"XC tile starting at point {tile.begin}: {error}"
                ) from error
            gradient = features["gradient"]
            if self.spec.spin == "unpolarized":
                gradient = gradient.sum(axis=0)
            energy += float(tile.weights @ values["energy_density"])
            potential += assemble_potential(
                self.spec, jets, gradient, values["gradient"], tile.weights
            )
            electrons += features["rho"] @ tile.weights
            points += len(tile.weights)
            tiles += 1
        if separate:
            if nspin == 1:
                potential = np.repeat(potential, 2, axis=0)
        else:
            potential = potential.mean(axis=0) if nspin == 2 else potential[0]
        if not np.isfinite(energy):
            raise ArithmeticError("nonfinite integrated XC energy")
        density_identity = canonical_hash(
            {
                "spin_density_sha256": sha256(d.tobytes()).hexdigest(),
                "layout": "separate" if separate else "total",
                "nao": basis.nao,
            }
        )
        identities = {
            "basis_identity": basis.identity,
            "grid_identity": grid.identity,
            "functional_identity": self.spec.identity,
            "density_identity": density_identity,
        }
        return XCIntegral(
            energy=energy,
            potential=immutable(potential),
            electrons=immutable(electrons),
            identity=canonical_hash({"contract": "fixed-density-xc-v1", **identities}),
            **identities,
            points=points,
            tiles=tiles,
        )
