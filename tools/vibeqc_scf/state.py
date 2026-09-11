"""Portable scientific SCF state, with #138/#173 model and #147 array identities.

An iteration is not a converged reference. Fractional ensemble densities can
occur after explicit damping. Neither stationarity nor Aufbau populations
certify stability or the lowest electronic solution.
"""

from dataclasses import dataclass, field
from hashlib import sha256

import numpy as np
from vibeqc.accuracy import ResolvedModel
from vibeqc.profiles import canonical_hash

from tools.vibeqc_posthf.reference import immutable


def safeguard_policy():
    """Portable version-1 acceptance policy, shared by trace/replay metadata.

    The native implementation is checked against the independent replay in
    numerical tests. Changing these semantics requires a new policy version.
    """
    return {
        "version": 1,
        "backend": "cpu",
        "precision": "fp64",
        "energy_slack_Eh": 1e-9,
        "residual_rms_floor": 1e-12,
        "residual_relative_descent": 1e-4,
        "fractions": [1.0, 0.5, 0.25, 0.125],
        "maximum_failed_proposals": 3,
        "active_proposer_convergence": "traditional_changes_and_true_residual",
        "replay_scope": "one_snapshot_counterfactual_without_trajectory_failure_budget",
    }


def spin_counts(model):
    """Return electrons per matrix block and the maximum metric occupation."""
    if model.method == "rhf":
        return (model.electron_count,), 2.0
    unpaired = model.multiplicity - 1
    return (
        (model.electron_count + unpaired) // 2,
        (model.electron_count - unpaired) // 2,
    ), 1.0


def metric_root(overlap):
    """Positive symmetric root; rank reduction is not silently performed."""
    eigenvalues, vectors = np.linalg.eigh(overlap)
    if eigenvalues[0] < 1e-10:
        raise ValueError("singular_metric")
    return (vectors * np.sqrt(eigenvalues)) @ vectors.T


def validate_density(density, overlap, model, *, determinant=False):
    """Enforce symmetry, spin traces, PSD and upper occupations without repair."""
    counts, weight = spin_counts(model)
    n = len(overlap)
    p = immutable(density, shape=(len(counts), n, n))
    if not np.allclose(p, p.swapaxes(-1, -2), atol=1e-7, rtol=0):
        raise ValueError("symmetry")
    if np.max(np.abs(np.einsum("xpq,qp->x", p, overlap) - counts)) > 1e-7:
        raise ValueError("electron_count")
    root = metric_root(overlap)
    occupations = np.linalg.eigvalsh(root @ p @ root)
    if np.min(occupations) < -1e-7 or np.max(occupations) > weight + 1e-7:
        raise ValueError("occupation_bounds")
    if (
        determinant
        and np.max(np.minimum(abs(occupations), abs(occupations - weight))) > 1e-7
    ):
        raise ValueError("nonidempotent")
    return p


@dataclass(frozen=True, eq=False)
class ScfSnapshot:
    """Detached, immutable physical iteration and its traditional next density.

    Owner and generation prevent cross-item/solve reuse; model identity covers
    geometry, basis, charge, spin and Hamiltonian. The content hash additionally
    covers every matrix and iteration. Timings belong to decisions, not this
    scientific identity. All matrices use real AO coordinates, bohr and Hartree.
    """

    model: ResolvedModel
    owner: str
    generation: int
    iteration: int
    fock_builds: int
    energy: float
    density: np.ndarray
    fock: np.ndarray
    residual: np.ndarray
    overlap: np.ndarray
    baseline: np.ndarray
    identity: str = field(init=False)

    def __post_init__(self):
        if not isinstance(self.model, ResolvedModel):
            raise TypeError("snapshot requires a resolved model")
        if not isinstance(self.owner, str) or not self.owner:
            raise ValueError("snapshot owner must be nonempty")
        for name in ("generation", "iteration", "fock_builds"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"invalid {name}")
        if not np.isfinite(self.energy):
            raise ValueError("nonfinite energy")
        overlap = immutable(self.overlap)
        if (
            overlap.ndim != 2
            or overlap.shape[0] != overlap.shape[1]
            or not 0 < len(overlap) <= 12
        ):
            raise ValueError("snapshot supports square metrics through 12 AOs")
        if not np.allclose(overlap, overlap.T, atol=1e-12, rtol=0):
            raise ValueError("overlap symmetry")
        object.__setattr__(self, "overlap", overlap)
        for name, determinant in (("density", False), ("baseline", True)):
            object.__setattr__(
                self,
                name,
                validate_density(
                    getattr(self, name), overlap, self.model, determinant=determinant
                ),
            )
        for name in ("fock", "residual"):
            object.__setattr__(
                self, name, immutable(getattr(self, name), shape=self.density.shape)
            )
        if not np.allclose(self.fock, self.fock.swapaxes(-1, -2), atol=1e-10, rtol=0):
            raise ValueError("Fock symmetry")
        expected = (
            self.fock @ self.density @ overlap - overlap @ self.density @ self.fock
        )
        if not np.allclose(self.residual, expected, atol=1e-10, rtol=1e-10):
            raise ValueError("residual is not the physical AO commutator")
        object.__setattr__(self, "identity", canonical_hash(self.record()))

    @property
    def residual_rms(self):
        return float(np.sqrt(np.mean(self.residual**2)))

    def record(self):
        """JSON metadata/checksums; matrices are separately serialized as FP64."""
        return {
            "schema": "vibeqc.scf_snapshot",
            "schema_version": 1,
            "model": self.model.to_dict(),
            "owner": self.owner,
            "generation": self.generation,
            "iteration": self.iteration,
            "fock_builds": self.fock_builds,
            "energy": self.energy,
            "arrays": {
                name: {
                    "shape": list(getattr(self, name).shape),
                    "sha256": sha256(
                        getattr(self, name).astype("<f8").tobytes()
                    ).hexdigest(),
                }
                for name in ("density", "fock", "residual", "overlap", "baseline")
            },
        }

    def diagnostics(self):
        """Stationarity and occupations, with explicit unperformed state checks."""
        counts, weight = spin_counts(self.model)
        root = metric_root(self.overlap)
        x = np.linalg.inv(root)
        occupations = np.linalg.eigvalsh(root @ self.density @ root)
        eps, c = np.linalg.eigh(x @ self.fock @ x)
        projected = (
            (x @ c).swapaxes(-1, -2)
            @ self.overlap
            @ self.density
            @ self.overlap
            @ (x @ c)
        )
        leakage = []
        gaps = []
        for spin, electrons in enumerate(counts):
            occupied = int(electrons / weight)
            leakage.append(float(np.trace(projected[spin, occupied:, occupied:])))
            gaps.append(
                float(eps[spin, occupied] - eps[spin, occupied - 1])
                if 0 < occupied < len(root)
                else None
            )
        return {
            "residual_rms": self.residual_rms,
            "metric_occupations": occupations.tolist(),
            "gaps": gaps,
            "aufbau_virtual_population": leakage,
            "stability": "not_evaluated",
            "intended_state": "unverified",
            "reference_snapshot_eligible": False,
        }


@dataclass(frozen=True, eq=False)
class DensityProposal:
    """Candidate with exact parent identity and an explicit density domain.

    Invalid shapes/values are retained until validation so failed model output
    has a recorded rejection. Copying the buffer still prevents mutation races.
    """

    parent_id: str
    density: np.ndarray
    representation: str = "ensemble_density"
    repair_seconds: float = 0.0
    repair: str = "none"

    def __post_init__(self):
        raw = np.asarray(self.density)
        # The host proposal bridge is explicitly capped at two 12-AO blocks.
        # Reject a malformed model output before making another owned copy.
        if raw.ndim > 3 or raw.size > 2 * 12 * 12:
            raise ValueError("proposal exceeds the small-system output budget")
        if np.iscomplexobj(raw):
            raise ValueError("complex proposals are unsupported")
        raw = np.asarray(raw, dtype=np.float64, order="C")
        object.__setattr__(
            self,
            "density",
            np.frombuffer(raw.tobytes(), dtype=np.float64).reshape(raw.shape),
        )
        if self.representation not in (
            "ensemble_density",
            "determinant_density",
            "reset",
        ):
            raise ValueError("unknown_representation")
        if not isinstance(self.parent_id, str) or not self.parent_id:
            raise ValueError("proposal requires a parent identity")
        if not np.isfinite(self.repair_seconds) or self.repair_seconds < 0:
            raise ValueError("invalid repair timing")

    def validate(self, state):
        if self.parent_id != state.identity:
            raise ValueError("stale_state")
        if self.representation == "reset":
            return
        validate_density(
            self.density,
            state.overlap,
            state.model,
            determinant=self.representation == "determinant_density",
        )
