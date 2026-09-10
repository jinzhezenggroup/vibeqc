"""Fixed-density CPU XC integration; no SCF or public method registration."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

import numpy as np

from vibeqc_compiler.common.arrays import immutable
from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.dft import ExplicitGrid, MolecularGrid, NativeAO
from vibeqc_compiler.dft.features import density_features, spin_densities
from vibeqc_compiler.dft.grid import GridTile, checked_int

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
    approximation_identity: str | None = None


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

    def integrate(self, basis, grid, density, *, tile_points=256, spatial=None):
        """Return E_xc and V_xc with delta E = sum_s Tr(V_s delta D_s).

        Total [AO,AO] input means Da=Db=D/2. Separate [2,AO,AO] input
        preserves both spins. Unpolarized functionals require equal matrices;
        their derivative is with respect to total density. For separate equal
        matrices it is returned on both spin channels.

        Molecular grids must match basis atoms, charge and spin policy. An
        ExplicitGrid is deliberately a fixed laboratory-frame quadrature;
        use a new grid when molecular geometry/rules change. No identity cache
        can return an old energy, AO tile or potential.

        ``spatial`` selects a prepared CPU local-dense candidate with this
        exact basis/quadrature. Its fixed mask defines the approximated energy
        and potential consistently; an AO cutoff is not an energy error bound.
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
        if spatial is not None:
            from vibeqc_compiler.dft.spatial_prepared import PreparedSpatialGrid

            if not isinstance(spatial, PreparedSpatialGrid):
                raise TypeError("expected PreparedSpatialGrid")
            if spatial.backend != "cpu":
                raise ValueError(
                    "CPU fixed-density integration requires a CPU spatial candidate"
                )
            if (
                spatial.basis.identity != basis.identity
                or spatial.source_grid.identity != grid.identity
            ):
                raise ValueError("stale spatial basis/quadrature")
        separate = np.asarray(density).ndim == 3
        d = spin_densities(density, basis.nao)
        if self.spec.spin == "unpolarized" and not np.array_equal(d[0], d[1]):
            raise UnsupportedXC("unpolarized integration requires equal spin matrices")
        energy = 0.0
        nspin = 2 if self.spec.spin == "polarized" else 1
        potential = np.zeros((nspin, basis.nao, basis.nao))
        electrons = np.zeros(2)
        points = tiles = 0

        def collocation():
            if spatial is not None:
                for tile in spatial.iter_features(d, include_jets=True):
                    yield tile, tile.ao_jets, tile.features, tile.ao_ids
            else:
                for tile in _tiles(grid, tile_points):
                    jets = basis.evaluate(tile.points, 1)
                    yield tile, jets, density_features(jets, d), None

        for tile, jets, features, ao_ids in collocation():
            if ao_ids is not None and len(ao_ids) == 0:
                # A structurally empty fixed mask defines a constant-zero
                # contribution for every D. No scalar vacuum derivative is
                # evaluated; nonempty masks retain the usual domain checks.
                points += len(tile.weights)
                tiles += 1
                continue
            try:
                values = self._program.unpack(
                    self._program.evaluate(pack_grid_features(self.spec, features))
                )
            except UnsupportedXC as error:
                begin = tile.begin if ao_ids is None else int(tile.point_ids[0])
                raise UnsupportedXC(
                    f"XC tile starting at point {begin}: {error}"
                ) from error
            gradient = features["gradient"]
            if self.spec.spin == "unpolarized":
                gradient = gradient.sum(axis=0)
            energy += float(tile.weights @ values["energy_density"])
            block = assemble_potential(
                self.spec, jets, gradient, values["gradient"], tile.weights
            )
            if ao_ids is None:
                potential += block
            else:
                # AO maps are sorted and unique. Retain every cross-local-AO
                # term, and scatter both symmetric legs exactly once.
                potential[:, ao_ids[:, None], ao_ids[None, :]] += block
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
        approximation = None if spatial is None else spatial.tasks.identity
        contract = {"contract": "fixed-density-xc-v1", **identities}
        if approximation is not None:
            contract["spatial_fixed_mask"] = approximation
        return XCIntegral(
            energy=energy,
            potential=immutable(potential),
            electrons=immutable(electrons),
            identity=canonical_hash(contract),
            **identities,
            points=points,
            tiles=tiles,
            approximation_identity=approximation,
        )
