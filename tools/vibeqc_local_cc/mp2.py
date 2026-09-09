"""Projected canonical-MP2 diagnostics in localized occupied/PNO pair spaces.

This is a representation and controlled truncation experiment, not a solver
for approximate local-MP2 or local-CC equations. Canonical amplitudes are
rotated exactly, preserving occupied Fock coupling. No diagonal denominators
are invented for the localized occupied orbitals.
"""

import time
from dataclasses import dataclass

import numpy as np

from tools.vibeqc_posthf.conventions import MOBlock, ovov_to_ijab
from tools.vibeqc_posthf.df import DFProvider
from tools.vibeqc_posthf.mp2 import restricted_mp2
from tools.vibeqc_posthf.providers import ConventionalProvider
from tools.vibeqc_posthf.reference import immutable

from .common import checked_budget, checked_reference, number
from .spaces import PairSpace, make_pair_space


@dataclass(frozen=True)
class LocalSpacePlan:
    """Conservative numeric-storage reservation composed with a bounded provider.

    The provider owns its source/snapshot/cache accounting. This layer reserves
    canonical MP2 temporaries, pair-local transforms, retained pair records and
    immutable publication copies. Python objects and library allocator overhead
    are excluded; no whole-molecule AO four-index tensor is requested.
    """

    budget_bytes: int
    local_numeric_bytes: int
    provider_budget_bytes: int
    pair_count: int
    maximum_pair_rank: int


def plan_local_spaces(snapshot, localization, domain, *, budget_bytes=128 << 20):
    """Preflight local storage before reading any provider integrals."""
    no, nv = checked_reference(snapshot)
    budget = checked_budget(budget_bytes)
    if (
        snapshot.identity != localization.reference_id
        or snapshot.identity != domain.reference_id
    ):
        raise ValueError("local-space planning rejects stale reference identities")
    if domain.columns.shape[0] != nv or localization.rotation.shape != (no, no):
        raise ValueError("local-space dimensions do not match the reference")
    u = localization.rotation
    if not np.allclose(
        localization.coefficients, snapshot.coefficients[:, :no] @ u, atol=1e-10, rtol=0
    ) or not np.allclose(
        localization.occupied_fock,
        u.T @ np.diag(snapshot.orbital_energies[:no]) @ u,
        atol=1e-10,
        rtol=0,
    ):
        raise ValueError(
            "occupied localization does not represent its parent reference"
        )
    pairs = no * (no + 1) // 2
    borrowed = sum(
        a.nbytes
        for a in (
            localization.rotation,
            localization.coefficients,
            localization.occupied_fock,
            localization.populations,
            domain.columns,
            domain.gram_eigenvalues,
        )
    )
    local = borrowed + 8 * (8 * no**2 * nv**2 + 6 * pairs * (nv**2 + nv) + 32 * nv**2)
    if local >= budget:
        raise MemoryError(
            f"local-space numeric storage alone needs {local} bytes; budget={budget}"
        )
    return LocalSpacePlan(budget, local, budget - local, pairs, domain.rank)


@dataclass(frozen=True, eq=False)
class PairMP2Result:
    """Unordered-pair restricted amplitudes/integrals in a declared PNO gauge."""

    space: PairSpace
    amplitudes: np.ndarray
    integrals: np.ndarray
    energy: float
    full_virtual_pair_energy: float

    def __post_init__(self):
        if not isinstance(self.space, PairSpace):
            raise TypeError("pair result requires a typed PNO space")
        shape = (self.space.rank, self.space.rank)
        for name in ("amplitudes", "integrals"):
            object.__setattr__(self, name, immutable(getattr(self, name), shape=shape))
        for value in (self.energy, self.full_virtual_pair_energy):
            if not np.isfinite(value):
                raise ValueError("pair energies must be finite")
        expected = self.multiplicity * float(
            np.sum(self.amplitudes * (2 * self.integrals - self.integrals.T))
        )
        if abs(expected - self.energy) > 1e-12:
            raise ValueError(
                "pair energy disagrees with its amplitude/index convention"
            )

    @property
    def multiplicity(self):
        """RHF sums all ordered occupied pairs; store i<=j and count i<j twice."""
        return 1 if self.space.pair[0] == self.space.pair[1] else 2


@dataclass(frozen=True, eq=False)
class LocalMP2Result:
    """Local-space evidence retaining rank branches and the canonical comparator."""

    reference_id: str
    hamiltonian_id: str
    localization_id: str
    pairs: tuple[PairMP2Result, ...]
    correlation_energy: float
    canonical_correlation_energy: float
    minimum_absolute_denominator: float
    plan: LocalSpacePlan
    provider_peak_bytes: int
    provider_seconds: float
    pair_transform_seconds: float
    total_seconds: float
    full_space_recovery: bool
    derivative_status: str = "unsupported"

    def __post_init__(self):
        pairs = tuple(self.pairs)
        if not pairs or any(
            not isinstance(p, PairMP2Result)
            or p.space.reference_id != self.reference_id
            or p.space.localization_id != self.localization_id
            for p in pairs
        ):
            raise ValueError("local pair results must share their parent state")
        if len({p.space.pair for p in pairs}) != len(pairs):
            raise ValueError("duplicate local occupied pair")
        if len(pairs) != self.plan.pair_count:
            raise ValueError("local result omits a required occupied pair")
        complete = all(p.space.rank == p.space.columns.shape[0] for p in pairs)
        if (
            type(self.full_space_recovery) is not bool
            or self.full_space_recovery != complete
        ):
            raise ValueError("full-space status differs from the retained pair ranks")
        for name in ("correlation_energy", "canonical_correlation_energy"):
            if not np.isfinite(getattr(self, name)):
                raise ValueError("local energy diagnostics must be finite")
        if abs(sum(p.energy for p in pairs) - self.correlation_energy) > 1e-12:
            raise ValueError("local total energy differs from its ordered-pair sum")
        if self.derivative_status != "unsupported":
            raise ValueError("local projector/localization response is not implemented")
        if (
            complete
            and abs(self.correlation_energy - self.canonical_correlation_energy) > 1e-10
        ):
            raise ValueError(
                "claimed full-space result failed the canonical energy gate"
            )
        object.__setattr__(self, "pairs", pairs)

    @property
    def observed_energy_difference(self):
        """Measured truncation error; pair-density losses are not error bounds."""
        return abs(self.correlation_energy - self.canonical_correlation_energy)

    @property
    def retained_numeric_bytes(self):
        return sum(
            a.nbytes
            for p in self.pairs
            for a in (
                p.space.columns,
                p.space.occupation_eigenvalues,
                p.amplitudes,
                p.integrals,
            )
        )


def build_local_mp2(
    snapshot,
    source,
    localization,
    domain,
    *,
    metric=None,
    occupation_threshold=1e-7,
    cluster_tolerance=1e-12,
    keep_full_space=False,
    denominator_threshold=1e-10,
    budget_bytes=128 << 20,
    axis_tile=2,
):
    """Build bounded local transforms from #147's same-Hamiltonian MP2 provider.

    A fresh provider receives the remaining budget after local storage is
    reserved, avoiding interference from another caller's cached blocks. The
    native source remains caller-owned. This first local-space consumer is CPU;
    it never advertises a hidden GPU-to-host execution path.
    """
    started = time.perf_counter()
    no, nv = checked_reference(snapshot)
    plan = plan_local_spaces(snapshot, localization, domain, budget_bytes=budget_bytes)
    ao_atoms = tuple(
        shell.atom_index
        for shell in source.shells
        for _ in range(
            (shell.angular_momentum + 1) * (shell.angular_momentum + 2) // 2
            if source.representation == "cartesian"
            else 2 * shell.angular_momentum + 1
        )
    )
    if ao_atoms != localization.ao_atoms:
        raise ValueError(
            "localization atomic partition differs from the integral source"
        )
    number(occupation_threshold, "PNO threshold")
    number(cluster_tolerance, "cluster tolerance", positive=True)
    if type(keep_full_space) is not bool:
        raise TypeError("keep_full_space requires an explicit boolean")
    # The provider checks source geometry/basis and, for DF, exact metric identity.
    provider = (
        ConventionalProvider(
            snapshot,
            source,
            budget_bytes=plan.provider_budget_bytes,
            axis_tile=axis_tile,
        )
        if metric is None
        else DFProvider(
            snapshot,
            source,
            metric,
            budget_bytes=plan.provider_budget_bytes,
            axis_tile=axis_tile,
        )
    )
    with provider:
        canonical = restricted_mp2(
            snapshot, provider, denominator_threshold=denominator_threshold
        )
        integrals = ovov_to_ijab(
            provider.get(MOBlock.from_spaces(snapshot, "ovov")).to_host()
        )
        provider_seconds = time.perf_counter() - started
        pair_started = time.perf_counter()
        pairs = []
        u = localization.rotation
        for i in range(no):
            for j in range(i, no):
                # Only one local pair is materialized at a time. These are exact
                # tensor rotations of canonical MP2, including occupied coupling.
                t = np.einsum(
                    "k,l,klab->ab",
                    u[:, i],
                    u[:, j],
                    canonical.amplitudes,
                    optimize=True,
                )
                g = np.einsum(
                    "k,l,klab->ab", u[:, i], u[:, j], integrals, optimize=True
                )
                space = make_pair_space(
                    snapshot,
                    localization,
                    domain,
                    (i, j),
                    t,
                    occupation_threshold=occupation_threshold,
                    cluster_tolerance=cluster_tolerance,
                    keep_full_space=keep_full_space,
                )
                q = space.columns
                projected_t, projected_g = q.T @ t @ q, q.T @ g @ q
                weight = 1 if i == j else 2
                energy = weight * float(
                    np.sum(projected_t * (2 * projected_g - projected_g.T))
                )
                full_pair = weight * float(np.sum(t * (2 * g - g.T)))
                pairs.append(
                    PairMP2Result(space, projected_t, projected_g, energy, full_pair)
                )
        transform_seconds = time.perf_counter() - pair_started
        full_pair_sum = sum(p.full_virtual_pair_energy for p in pairs)
        if abs(full_pair_sum - canonical.correlation_energy) > 1e-10:
            raise RuntimeError("occupied rotation failed full-space MP2 recovery")
        energy = sum(p.energy for p in pairs)
        full_space = domain.rank == nv and all(p.space.rank == nv for p in pairs)
        if full_space and abs(energy - canonical.correlation_energy) > 1e-10:
            raise RuntimeError("PNO gauge failed full-space MP2 recovery")
        peak = provider.statistics["peak_bytes"]
    return LocalMP2Result(
        snapshot.identity,
        snapshot.hamiltonian_id,
        localization.identity,
        tuple(pairs),
        energy,
        canonical.correlation_energy,
        canonical.minimum_absolute_denominator,
        plan,
        peak,
        provider_seconds,
        transform_seconds,
        time.perf_counter() - started,
        full_space,
    )


def recover_canonical_amplitudes(
    snapshot, localization, result, *, budget_bytes=128 << 20
):
    """Small diagnostic inverse transform; pair spaces may deliberately be truncated.

    Both orientations of i<j are reconstructed, with T_ji=T_ij^T. This function
    is optional and not used by the bounded pair construction/energy path.
    """
    no, nv = checked_reference(snapshot)
    if (
        snapshot.identity != result.reference_id
        or localization.identity != result.localization_id
    ):
        raise ValueError("cannot transport amplitudes across changed parent identities")
    required = 8 * (3 * no**2 * nv**2 + 4 * nv**2) + result.retained_numeric_bytes
    if required > checked_budget(budget_bytes):
        raise MemoryError(
            f"canonical amplitude recovery needs {required} numeric bytes"
        )
    recovered = np.zeros((no, no, nv, nv))
    for pair in result.pairs:
        i, j = pair.space.pair
        q = pair.space.columns
        t = q @ pair.amplitudes @ q.T
        for left, right, amplitudes in (
            ((i, j, t),) if i == j else ((i, j, t), (j, i, t.T))
        ):
            recovered += np.einsum(
                "k,l,ab->klab",
                localization.rotation[:, left],
                localization.rotation[:, right],
                amplitudes,
                optimize=True,
            )
    return immutable(recovered)
