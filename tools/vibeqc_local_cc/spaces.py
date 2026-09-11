"""Metric projected virtual spaces and spin-adapted pair-natural orbitals."""

from dataclasses import dataclass

import numpy as np

from tools.vibeqc_posthf.reference import immutable

from .common import (
    checked_budget,
    checked_reference,
    fingerprint,
    number,
    orthogonality,
)


def spectral_selection(values, threshold, *, cluster_tolerance=1e-12, retain_all=False):
    """Retain whole adjacent eigenvalue clusters when any member passes.

    Values are ascending. A threshold close to any eigenvalue is reported as
    a rank crossing; no smooth derivative is claimed there. The absolute
    cluster tolerance is explicit in the units of the supplied spectrum.
    """
    values = immutable(values)
    threshold = number(threshold, "spectral threshold")
    cluster_tolerance = number(cluster_tolerance, "cluster tolerance", positive=True)
    if values.ndim != 1 or np.any(np.diff(values) < 0):
        raise ValueError("spectral eigenvalues must be ascending")
    selected = np.ones(len(values), dtype=bool) if retain_all else values > threshold
    start = 0
    for stop in range(1, len(values) + 1):
        if stop == len(values) or values[stop] - values[stop - 1] > cluster_tolerance:
            selected[start:stop] = np.any(selected[start:stop])
            start = stop
    crossing = bool(np.any(np.abs(values - threshold) <= cluster_tolerance))
    return selected, crossing


@dataclass(frozen=True, eq=False)
class ProjectedVirtualSpace:
    """A metric-orthonormal projected AO domain expressed in canonical virtuals.

    The columns are a gauge for the subspace. Compare the projectors numerically
    across different eigensolver gauges; exact value hashes are audit identities,
    not a tolerance-based equivalence test.
    """

    reference_id: str
    columns: np.ndarray
    source_ao_indices: tuple[int, ...]
    gram_eigenvalues: np.ndarray
    absolute_cutoff: float
    rank_crossing: bool
    cluster_tolerance: float

    def __post_init__(self):
        q, values = immutable(self.columns), immutable(self.gram_eigenvalues)
        if q.ndim != 2 or not q.shape[0] or not q.shape[1] or orthogonality(q) > 1e-8:
            raise ValueError("invalid projected virtual metric basis")
        indices = tuple(self.source_ao_indices)
        if values.shape != (len(indices),) or np.any(np.diff(values) < 0):
            raise ValueError("invalid projected AO spectrum")
        if not self.reference_id or any(type(i) is not int or i < 0 for i in indices):
            raise ValueError("invalid projected AO identity")
        number(self.absolute_cutoff, "projected AO cutoff", positive=True)
        number(self.cluster_tolerance, "cluster tolerance", positive=True)
        selected, crossing = spectral_selection(
            values, self.absolute_cutoff, cluster_tolerance=self.cluster_tolerance
        )
        if (
            q.shape[1] != int(selected.sum())
            or type(self.rank_crossing) is not bool
            or self.rank_crossing != crossing
        ):
            raise ValueError("projected rank/branch differs from the declared spectrum")
        object.__setattr__(self, "columns", q)
        object.__setattr__(self, "gram_eigenvalues", values)
        object.__setattr__(self, "source_ao_indices", indices)

    @property
    def rank(self):
        return self.columns.shape[1]

    @property
    def projector(self):
        """Orthogonal virtual-coordinate projector, independent of column phases."""
        return immutable(self.columns @ self.columns.T)

    @property
    def identity(self):
        return fingerprint(
            {
                "reference": self.reference_id,
                "source_aos": self.source_ao_indices,
                "cutoff": self.absolute_cutoff,
                "cluster_tolerance": self.cluster_tolerance,
            },
            projector=self.projector,
        )


def projected_virtual_space(
    snapshot,
    ao_indices=None,
    *,
    relative_threshold=1e-10,
    absolute_threshold=1e-12,
    cluster_tolerance=1e-12,
    budget_bytes=128 << 20,
):
    """Project selected AOs with (1-Cocc Cocc^T S), then remove metric nullspace.

    In the validated canonical virtual basis this projection is Cvirt^T S E.
    Duplicate AOs and occupied-contaminated directions appear as rank loss,
    which is recorded rather than filled with invented virtual orbitals.
    """
    no, nv = checked_reference(snapshot)
    indices = tuple(range(snapshot.nmo)) if ao_indices is None else tuple(ao_indices)
    if not indices or any(
        type(i) is not int or not 0 <= i < snapshot.nmo for i in indices
    ):
        raise ValueError("invalid projected AO domain indices")
    relative = number(relative_threshold, "relative metric cutoff", positive=True)
    absolute = number(absolute_threshold, "absolute metric cutoff", positive=True)
    if relative >= 1:
        raise ValueError("relative metric cutoff must be below one")
    peak = 8 * (6 * len(indices) ** 2 + 8 * snapshot.nmo * len(indices) + 4 * nv**2)
    if peak > checked_budget(budget_bytes):
        raise MemoryError(f"projected virtual construction needs {peak} numeric bytes")
    cv = snapshot.coefficients[:, no:]
    projected = cv.T @ snapshot.overlap[:, indices]
    gram = projected.T @ projected
    values, vectors = np.linalg.eigh((gram + gram.T) / 2)
    cutoff = max(absolute, relative * float(values[-1]))
    # Rank removal is a metric conditioning rule, not a PNO occupation rule:
    # never retain numerical null vectors just to complete a cluster.
    selected, crossing = spectral_selection(
        values, cutoff, cluster_tolerance=cluster_tolerance
    )
    if np.any(values[selected] <= absolute):
        raise ValueError("projected metric cluster intersects its nullspace cutoff")
    if not np.any(selected):
        raise ValueError("projected AO domain has no supported virtual rank")
    q = projected @ (vectors[:, selected] / np.sqrt(values[selected]))
    if q.shape[1] > nv:
        raise ValueError("projected AO rank exceeds the virtual space")
    return ProjectedVirtualSpace(
        snapshot.identity, q, indices, values, cutoff, crossing, cluster_tolerance
    )


def pair_density(amplitudes, *, diagonal_pair):
    """Restricted PNO density D=(tildeT^T T+tildeT T^T)/(1+delta_ij).

    T is the unantisymmetrized restricted MP2 pair amplitude in one orthonormal
    virtual domain; tildeT=2T-T^T. Equivalently D=(2 S^2+6 A^T A)/(1+delta_ij)
    for its symmetric/antisymmetric parts. This form makes positivity explicit.
    """
    t = immutable(amplitudes)
    if t.ndim != 2 or t.shape[0] != t.shape[1] or type(diagonal_pair) is not bool:
        raise ValueError("pair density requires a square restricted amplitude")
    symmetric, antisymmetric = (t + t.T) / 2, (t - t.T) / 2
    return immutable(
        (2 * symmetric @ symmetric + 6 * antisymmetric.T @ antisymmetric)
        / (1 + diagonal_pair)
    )


@dataclass(frozen=True, eq=False)
class PairSpace:
    """One unordered occupied pair's PNO projector and explicit rank branch."""

    reference_id: str
    localization_id: str
    virtual_domain_id: str
    pair: tuple[int, int]
    columns: np.ndarray
    occupation_eigenvalues: np.ndarray
    retained_indices: tuple[int, ...]
    occupation_threshold: float
    cluster_tolerance: float
    rank_crossing: bool
    keep_full_space: bool

    def __post_init__(self):
        q, eigenvalues = immutable(self.columns), immutable(self.occupation_eigenvalues)
        pair, retained = tuple(self.pair), tuple(self.retained_indices)
        if (
            len(pair) != 2
            or any(type(i) is not int or i < 0 for i in pair)
            or pair[0] > pair[1]
        ):
            raise ValueError("pair labels require ordered nonnegative occupied indices")
        if (
            q.ndim != 2
            or not q.shape[0]
            or q.shape[1] != len(retained)
            or orthogonality(q) > 1e-8
        ):
            raise ValueError("invalid orthonormal PNO columns")
        if (
            eigenvalues.ndim != 1
            or not len(eigenvalues)
            or len(eigenvalues) > q.shape[0]
            or np.any(np.diff(eigenvalues) < 0)
            or (len(eigenvalues) and eigenvalues[0] < -1e-12)
        ):
            raise ValueError("invalid PNO density spectrum")
        if any(
            type(i) is not int or not 0 <= i < len(eigenvalues) for i in retained
        ) or len(set(retained)) != len(retained):
            raise ValueError("invalid retained PNO indices")
        for value in (self.reference_id, self.localization_id, self.virtual_domain_id):
            if not isinstance(value, str) or not value:
                raise ValueError("PNOs require complete parent identities")
        number(self.occupation_threshold, "PNO threshold")
        number(self.cluster_tolerance, "cluster tolerance", positive=True)
        if (
            type(self.rank_crossing) is not bool
            or type(self.keep_full_space) is not bool
        ):
            raise ValueError("PNO branch flags must be booleans")
        selected, crossing = spectral_selection(
            eigenvalues,
            self.occupation_threshold,
            cluster_tolerance=self.cluster_tolerance,
            retain_all=self.keep_full_space,
        )
        if (
            retained != tuple(int(i) for i in np.flatnonzero(selected))
            or self.rank_crossing != crossing
        ):
            raise ValueError("PNO rank/branch differs from the declared spectrum")
        for name, value in (
            ("columns", q),
            ("occupation_eigenvalues", eigenvalues),
            ("pair", pair),
            ("retained_indices", retained),
        ):
            object.__setattr__(self, name, value)

    @property
    def rank(self):
        return self.columns.shape[1]

    @property
    def projector(self):
        return immutable(self.columns @ self.columns.T)

    @property
    def discarded_weight(self):
        mask = np.ones(len(self.occupation_eigenvalues), dtype=bool)
        mask[list(self.retained_indices)] = False
        return float(np.sum(self.occupation_eigenvalues[mask]))

    @property
    def identity(self):
        """Projector identity excludes arbitrary PNO signs/column rotations."""
        return fingerprint(
            {
                "reference": self.reference_id,
                "localized": self.localization_id,
                "domain": self.virtual_domain_id,
                "pair": self.pair,
                "threshold": self.occupation_threshold,
                "cluster": self.cluster_tolerance,
                "retained": self.retained_indices,
                "full": self.keep_full_space,
            },
            projector=self.projector,
        )

    @property
    def gauge_identity(self):
        """Amplitude coordinates additionally depend on the chosen column gauge."""
        return fingerprint({"projector": self.identity}, columns=self.columns)

    def overlap(self, other):
        """Pair-to-pair virtual overlap needed by later coupled local equations."""
        if (
            not isinstance(other, PairSpace)
            or self.reference_id != other.reference_id
            or self.localization_id != other.localization_id
        ):
            raise ValueError(
                "pair overlap requires the same reference/localized occupied space"
            )
        return immutable(self.columns.T @ other.columns)


def make_pair_space(
    snapshot,
    localization,
    domain,
    pair,
    local_amplitudes,
    *,
    occupation_threshold=1e-7,
    cluster_tolerance=1e-12,
    keep_full_space=False,
):
    """Construct PNOs from canonical-MP2 amplitudes rotated to localized occupieds."""
    no, nv = checked_reference(snapshot)
    if (
        snapshot.identity != localization.reference_id
        or snapshot.identity != domain.reference_id
    ):
        raise ValueError("stale local-space parent reference")
    if domain.columns.shape[0] != nv or not 0 <= pair[0] <= pair[1] < no:
        raise ValueError("pair/domain dimensions do not match the reference")
    t = immutable(local_amplitudes, shape=(nv, nv))
    density = pair_density(
        domain.columns.T @ t @ domain.columns, diagonal_pair=pair[0] == pair[1]
    )
    eigenvalues, eigenvectors = np.linalg.eigh(density)
    selected, crossing = spectral_selection(
        eigenvalues,
        occupation_threshold,
        cluster_tolerance=cluster_tolerance,
        retain_all=keep_full_space,
    )
    columns = domain.columns @ eigenvectors[:, selected]
    return PairSpace(
        snapshot.identity,
        localization.identity,
        domain.identity,
        tuple(pair),
        columns,
        eigenvalues,
        tuple(int(i) for i in np.flatnonzero(selected)),
        occupation_threshold,
        cluster_tolerance,
        crossing,
        keep_full_space,
    )
