"""Fixed-density semilocal mean-field consumer of native common J/K sources."""

from dataclasses import dataclass

import numpy as np
from vibeqc_compiler.common.arrays import immutable
from vibeqc_compiler.common.provenance import canonical_hash

from .fock import FockPlan


@dataclass(frozen=True, eq=False)
class MeanFieldEvaluation:
    """Full fixed-density energy and AO Fock including semilocal XC.

    The energy includes nuclear repulsion. This object does not imply SCF
    stationarity and contains no geometric XC gradient or complete force.
    """

    energy: float
    fock: np.ndarray
    xc_energy: float
    identity: str
    fock_identity: str
    xc_identity: str


class FixedDensityMeanField:
    """Combine the existing executable XC integrator with common native J.

    The supplied FockPlan declares exact or fitted Coulomb explicitly. The
    current semilocal consumer requires cJ=1 and absent K; future executable
    hybrid methods must supply their own audited exchange metadata. No HF
    J/K contraction is duplicated here and no complete DFT SCF is registered.
    """

    def __init__(self, fock, xc):
        from vibeqc_compiler.xc.integration import FixedDensityXC

        if not isinstance(fock, FockPlan) or not isinstance(xc, FixedDensityXC):
            raise TypeError("expected FockPlan and FixedDensityXC")
        spec = fock.spec
        if (
            not spec.coulomb.present
            or spec.coulomb.coefficient != 1
            or spec.exchange.present
        ):
            raise ValueError(
                "semilocal mean field requires unit Coulomb and absent exchange"
            )
        self._fock, self._xc = fock, xc

    def integrate(self, grid, density, *, tile_points=256):
        """Evaluate the same density and immutable basis in both consumers.

        XC validates molecular grid/basis compatibility and functional domain
        before native J executes. Both component identities enter the result.
        """
        if np.iscomplexobj(density):
            raise TypeError("mean-field densities must be real")
        snapshot = np.array(density, dtype=np.float64, order="C", copy=True)
        xc = self._xc.integrate(
            self._fock.basis, grid, snapshot, tile_points=tile_points
        )
        native = self._fock.evaluate(snapshot)
        fock = native.fock + xc.potential
        energy = native.energy + xc.energy
        if not np.isfinite(energy) or not np.isfinite(fock).all():
            raise ArithmeticError("nonfinite combined mean-field result")
        return MeanFieldEvaluation(
            energy,
            immutable(fock),
            xc.energy,
            canonical_hash(
                {
                    "schema": "vibeqc.fixed-density-mean-field/v1",
                    "fock": native.identity,
                    "xc": xc.identity,
                }
            ),
            native.identity,
            xc.identity,
        )
