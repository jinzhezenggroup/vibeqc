"""Internal stationary KS derivative contracts for Issue #163.

This module validates method state and derivative ownership. It does not
assemble a complete molecular gradient or enable public DFT forces.
"""

from dataclasses import dataclass, field
from numbers import Real

import numpy as np
from vibeqc_compiler.common.arrays import immutable
from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.dft.ao import NativeAO
from vibeqc_compiler.dft.grid import ExplicitGrid
from vibeqc_compiler.xc.contractions import ContractionProgram, GeometryPartials
from vibeqc_compiler.xc.spec import FunctionalSpec

from .ks import resolve_ks_method

_METHODS = ("lda-rks", "pbe-rks", "lda-uks", "pbe-uks")
_ARRAY_TOLERANCE = 1e-8  # Match the absolute canonicality cap of the #162 handoff.
_RESIDUAL_TOLERANCE = 1e-8


@dataclass(frozen=True)
class StationaryKsIdentity:
    """Exact method, topology, provider and successful-solve identity."""

    method: str
    model_identity: str
    geometry_identity: str
    basis_identity: str
    overlap_identity: str
    grid_identity: str
    topology_identity: str
    functional_identity: str
    regularization_identity: str
    provider_identity: str
    owner: int
    solve_epoch: int
    density_generation: int
    fock_generation: int
    orbital_generation: int

    def __post_init__(self):
        if self.method not in _METHODS:
            raise ValueError("stationary derivatives support LDA/PBE RKS/UKS only")
        for name in (
            "model_identity",
            "geometry_identity",
            "basis_identity",
            "overlap_identity",
            "grid_identity",
            "topology_identity",
            "functional_identity",
            "regularization_identity",
            "provider_identity",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"stationary identity requires nonempty {name}")
        for name in (
            "owner",
            "solve_epoch",
            "density_generation",
            "fock_generation",
            "orbital_generation",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"stationary identity requires positive {name}")

    def to_payload(self):
        return {
            "schema": "vibeqc.stationary-ks-state/v1",
            **{name: getattr(self, name) for name in self.__dataclass_fields__},
        }


@dataclass(frozen=True, eq=False)
class StationaryKsState:
    """Detached KS arrays; only ``from_native`` supplies current-owner proof.

    Direct construction is useful for numerical diagnostics and negative tests.
    It does not authorize a stationary derivative, even with consistent arrays.
    """

    identity: StationaryKsIdentity
    density: np.ndarray
    fock: np.ndarray
    coefficients: np.ndarray
    orbital_energies: np.ndarray
    occupations: np.ndarray
    weighted_density: np.ndarray
    overlap: np.ndarray
    physical_residual: float
    successful: bool
    converged: bool
    physical: bool
    _source: object = field(default=None, repr=False)

    @classmethod
    def from_native(cls, batch, basis, grid=None, *, index=0):
        """Read the actual current #162 state and verify its AO/grid sources.

        CPU plans have no #162 handoff and fail explicitly. Native SCF keeps
        its own regularization identity; generated interior-only XC geometry
        remains unsupported for that model until the domains are reconciled.
        """
        from ._ks_snapshot import NativeKsSnapshot

        if not isinstance(basis, NativeAO) or (
            grid is not None and not isinstance(grid, ExplicitGrid)
        ):
            raise TypeError("stationary snapshot requires NativeAO and ExplicitGrid")
        source = NativeKsSnapshot(batch, index)
        try:
            state = cls(**source.decode(basis, grid))
            StationaryDerivativeContract(state.identity).validate(state)
            return state
        except Exception:
            source.close()
            raise

    @property
    def grid(self):
        """The exact native quadrature, retained independently of batch replay."""
        if self._source is None:
            raise ValueError("state has no native quadrature source")
        return self._source.grid

    def __post_init__(self):
        if not isinstance(self.identity, StationaryKsIdentity):
            raise TypeError("stationary state requires a typed identity")
        for name in ("successful", "converged", "physical"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"stationary {name} gate must be boolean")
        if isinstance(self.physical_residual, bool) or not isinstance(
            self.physical_residual, Real
        ):
            raise TypeError("physical residual must be one real scalar")
        if not np.isfinite(self.physical_residual):
            raise ValueError("physical residual must be finite")
        object.__setattr__(self, "physical_residual", float(self.physical_residual))
        for name in (
            "density",
            "fock",
            "coefficients",
            "orbital_energies",
            "occupations",
            "weighted_density",
            "overlap",
        ):
            object.__setattr__(self, name, immutable(getattr(self, name)))


@dataclass(frozen=True)
class StationaryDerivativeContract:
    """Validate a complete state before any partial gradient is consumed."""

    state_identity: StationaryKsIdentity
    topology_policy: str = "stable-explicit-grid-v1"
    sign: str = "gradient"
    identity: str = field(init=False)

    def __post_init__(self):
        if not isinstance(self.state_identity, StationaryKsIdentity):
            raise TypeError("stationary derivative requires a KS state identity")
        if self.topology_policy != "stable-explicit-grid-v1":
            raise NotImplementedError(
                "topology-changing DFT derivatives are unsupported"
            )
        if self.sign != "gradient":
            raise NotImplementedError("partial DFT forces are not a public capability")
        object.__setattr__(self, "identity", canonical_hash(self.to_payload()))

    @property
    def spin(self):
        method_ir, _ = resolve_ks_method(self.state_identity.method)
        return method_ir.spin

    @property
    def family(self):
        _, functional = resolve_ks_method(self.state_identity.method)
        return "lda" if functional.ingredients == ("rho",) else "gga"

    def to_payload(self):
        return {
            "schema": "vibeqc.stationary-dft-derivative/v1",
            "state": self.state_identity.to_payload(),
            "spin": self.spin,
            "family": self.family,
            "topology_policy": self.topology_policy,
            "sign": self.sign,
            "held_fixed": "density while evaluating explicit XC geometry partials",
            "force_capability": "unsupported",
        }

    def validate(self, state):
        """Require both numerical consistency and the live native #162 proof."""
        from ._ks_snapshot import NativeKsSnapshot

        self._validate_arrays(state)
        if type(state._source) is not NativeKsSnapshot:
            raise ValueError(
                "stationary derivatives require a current native #162 snapshot"
            )
        state._source.validate(state)
        return state

    def _validate_arrays(self, state):
        """Check array algebra only; this diagnostic never proves stationarity."""
        if not isinstance(state, StationaryKsState):
            raise TypeError("expected a stationary KS state")
        if state.identity != self.state_identity:
            raise ValueError("stationary KS identity mismatch")
        if not (state.successful and state.converged and state.physical):
            raise ValueError(
                "stationary derivative requires successful converged physical state"
            )
        if (
            not np.isfinite(state.physical_residual)
            or not 0 <= state.physical_residual <= _RESIDUAL_TOLERANCE
        ):
            raise ValueError("physical residual exceeds the stationary derivative gate")

        spins = 1 if self.spin == "unpolarized" else 2
        overlap = _matrix(state.overlap, "overlap")
        n = len(overlap)
        if not np.allclose(overlap, overlap.T, atol=_ARRAY_TOLERANCE, rtol=0):
            raise ValueError("overlap is not symmetric")
        if np.linalg.eigvalsh(overlap)[0] <= 0:
            raise ValueError("overlap is not positive definite")

        density = _spin_matrices(state.density, spins, n, "density")
        fock = _spin_matrices(state.fock, spins, n, "Fock")
        coefficients = _spin_matrices(state.coefficients, spins, n, "coefficients")
        weighted = _spin_matrices(state.weighted_density, spins, n, "weighted density")
        energies = _spin_vectors(state.orbital_energies, spins, n, "orbital energies")
        occupations = _spin_vectors(state.occupations, spins, n, "occupations")

        maximum = 2.0 if spins == 1 else 1.0
        if np.min(occupations) < 0 or np.max(occupations) > maximum:
            raise ValueError("occupations exceed the supported spin domain")
        eye = np.eye(n)
        for block in range(spins):
            c = coefficients[block]
            if not np.allclose(
                c.T @ overlap @ c, eye, atol=_ARRAY_TOLERANCE, rtol=1e-10
            ):
                raise ValueError("orbitals are not S-orthonormal")
            expected_density = (c * occupations[block]) @ c.T
            if not np.allclose(
                density[block], expected_density, atol=_ARRAY_TOLERANCE, rtol=1e-10
            ):
                raise ValueError("density reconstruction mismatch")
            expected_weighted = (c * (occupations[block] * energies[block])) @ c.T
            if not np.allclose(
                weighted[block], expected_weighted, atol=_ARRAY_TOLERANCE, rtol=1e-10
            ):
                raise ValueError("weighted-density reconstruction mismatch")
            residual = fock[block] @ c - (overlap @ c) * energies[block]
            scale = max(1.0, float(np.max(np.abs(fock[block] @ c))))
            if float(np.max(np.abs(residual))) > _ARRAY_TOLERANCE * scale:
                raise ValueError(
                    "Fock eigen residual exceeds the stationary derivative gate"
                )
        return state


@dataclass(frozen=True, eq=False)
class StableGridMotion:
    """Independent source motions on one asserted fixed topology branch."""

    topology_identity: str
    centers: np.ndarray
    points: np.ndarray
    weights: np.ndarray
    topology_changed: bool = False

    def __post_init__(self):
        if not isinstance(self.topology_identity, str) or not self.topology_identity:
            raise ValueError("motion requires a topology identity")
        if type(self.topology_changed) is not bool:
            raise ValueError("topology change flag must be boolean")


@dataclass(frozen=True)
class XcDirectionalComponents:
    """Explicit XC directional gradient split by its three source owners."""

    center: float
    point: float
    weight: float

    @property
    def total(self):
        total = self.center + self.point + self.weight
        if not np.isfinite(total):
            raise ArithmeticError("nonfinite total XC directional gradient")
        return total


@dataclass(frozen=True, eq=False)
class FixedDensityXcGeometry:
    """Generated fixed-density diagnostic partials, without stationarity proof."""

    state_identity: StationaryKsIdentity
    discrete_contract_identity: str
    basis_identity: str
    geometry_identity: str
    grid_identity: str
    topology_identity: str
    functional_identity: str
    regularization_identity: str
    density_generation: int
    partials: GeometryPartials
    force_capability: str = field(init=False, default="unsupported")

    def __post_init__(self):
        if not isinstance(self.state_identity, StationaryKsIdentity):
            raise TypeError("generated XC geometry requires a stationary identity")
        expected = {
            "basis_identity": self.state_identity.basis_identity,
            "geometry_identity": self.state_identity.geometry_identity,
            "grid_identity": self.state_identity.grid_identity,
            "topology_identity": self.state_identity.topology_identity,
            "functional_identity": self.state_identity.functional_identity,
            "regularization_identity": self.state_identity.regularization_identity,
            "density_generation": self.state_identity.density_generation,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"generated XC {name.replace('_', ' ')} mismatch")
        if (
            not isinstance(self.discrete_contract_identity, str)
            or not self.discrete_contract_identity
        ):
            raise ValueError("generated XC requires a discrete contract identity")
        if not isinstance(self.partials, GeometryPartials):
            raise TypeError("generated XC result requires GeometryPartials")
        centers = immutable(self.partials.centers)
        points = immutable(self.partials.points)
        weights = immutable(self.partials.weights)
        if centers.ndim != 2 or centers.shape[1:] != (3,):
            raise ValueError("generated XC centers require [atom,xyz]")
        if points.ndim != 2 or points.shape[1:] != (3,):
            raise ValueError("generated XC points require [point,xyz]")
        if weights.shape != (len(points),):
            raise ValueError("generated XC weights require one value per point")
        if not all(np.isfinite(value).all() for value in (centers, points, weights)):
            raise ValueError("generated XC partials must be finite")
        object.__setattr__(
            self,
            "partials",
            GeometryPartials(centers=centers, points=points, weights=weights),
        )

    def directional(self, motion):
        """Contract each generated source exactly once, preserving gradient sign."""
        if not isinstance(motion, StableGridMotion):
            raise TypeError("expected stable-grid motion")
        if motion.topology_identity != self.topology_identity:
            raise ValueError("motion topology identity mismatch")
        if motion.topology_changed:
            raise ValueError("topology change is not differentiable in slice A")
        values = []
        for direction, shape, name in (
            (motion.centers, self.partials.centers.shape, "center direction"),
            (motion.points, self.partials.points.shape, "point direction"),
            (motion.weights, self.partials.weights.shape, "weight direction"),
        ):
            array = np.asarray(direction)
            if np.iscomplexobj(array) or array.shape != shape:
                raise ValueError(f"{name} has incompatible shape")
            if not np.isfinite(array).all():
                raise ValueError(f"nonfinite {name}")
            values.append(array)
        centers, points, weights = values
        with np.errstate(over="ignore", invalid="ignore"):
            components = tuple(
                float(np.sum(partial * direction))
                for partial, direction in zip(
                    (
                        self.partials.centers,
                        self.partials.points,
                        self.partials.weights,
                    ),
                    (centers, points, weights),
                    strict=True,
                )
            )
        if not np.isfinite(components).all():
            raise ArithmeticError("nonfinite XC directional component")
        return XcDirectionalComponents(*components)


@dataclass(frozen=True, eq=False)
class GeneratedXcGeometry(FixedDensityXcGeometry):
    """Stationary XC partials whose native owner must remain current at use."""

    state: StationaryKsState

    def __post_init__(self):
        super().__post_init__()
        contract = StationaryDerivativeContract(self.state_identity)
        contract.validate(self.state)
        _, spec = resolve_ks_method(self.state_identity.method)
        if self.regularization_identity != xc_regularization_identity(spec):
            raise ValueError("stationary regularization identity mismatch")

    def directional(self, motion):
        """Recheck eligibility even when partials were computed before replay."""
        StationaryDerivativeContract(self.state_identity).validate(self.state)
        return super().directional(motion)


def native_ao_geometry_identity(basis):
    """Hash the nuclear geometry owned by one exact native AO basis."""
    if not isinstance(basis, NativeAO):
        raise TypeError("geometry identity requires NativeAO")
    return canonical_hash(
        {
            "schema": "vibeqc.stationary-geometry/v1",
            "atoms": [
                [atom.atomic_number, *map(float, atom.position)] for atom in basis.atoms
            ],
            "units": "Bohr",
        }
    )


def _native_ao_atoms(basis):
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


def xc_geometry_topology_identity(basis, grid):
    """Hash stable AO ownership and explicit grid membership, excluding motion."""
    if not isinstance(basis, NativeAO) or not isinstance(grid, ExplicitGrid):
        raise TypeError("XC topology identity requires NativeAO and ExplicitGrid")
    if any(owner >= basis.natom for owner in grid.owners):
        raise ValueError("grid owner is outside the molecular atom topology")
    return canonical_hash(
        {
            "schema": "vibeqc.stationary-xc-topology/v1",
            "natom": basis.natom,
            "nao": basis.nao,
            "ao_atoms": _native_ao_atoms(basis).tolist(),
            "grid_owners": list(grid.owners),
            "npoint": len(grid.points),
        }
    )


def xc_regularization_identity(functional):
    """Identify the generated functional's exact fixed numerical domain."""
    if not isinstance(functional, FunctionalSpec):
        raise TypeError("XC regularization identity requires a typed functional")
    payload = functional.to_payload()
    return canonical_hash(
        {
            "schema": "vibeqc.stationary-xc-regularization/v1",
            "functional": functional.identity,
            "version": functional.version,
            "domain": payload["domain"],
        }
    )


def bind_generated_xc_geometry(
    contract,
    state,
    functional,
    basis,
    grid,
):
    """Evaluate Issue #236 geometry pullbacks and bind them to method state."""
    if not isinstance(contract, StationaryDerivativeContract):
        raise TypeError("expected a stationary derivative contract")
    state = contract.validate(state)
    result = _fixed_density_xc_geometry(contract, state, functional, basis, grid)
    # Recheck after evaluation, including source identities and snapshot content.
    return GeneratedXcGeometry(
        **{
            name: getattr(result, name)
            for name, item in result.__dataclass_fields__.items()
            if item.init
        },
        state=state,
    )


def _fixed_density_xc_geometry(contract, state, functional, basis, grid):
    """Numerical diagnostic shared with the oracle tests; never authorize KS use.

    Only ``bind_generated_xc_geometry`` supplies the live-owner gate and returns
    a stationary result. This helper returns explicitly fixed-density partials.
    """
    contract._validate_arrays(state)
    if not isinstance(functional, FunctionalSpec):
        raise TypeError("expected a typed XC functional")
    if not isinstance(basis, NativeAO) or not isinstance(grid, ExplicitGrid):
        raise TypeError("stationary XC geometry requires NativeAO and ExplicitGrid")
    if functional.identity != state.identity.functional_identity:
        raise ValueError("stationary functional identity mismatch")
    if functional.spin != contract.spin:
        raise ValueError("stationary spin identity mismatch")
    family = "lda" if functional.ingredients == ("rho",) else "gga"
    if family != contract.family:
        raise ValueError("stationary functional family mismatch")
    _, expected_functional = resolve_ks_method(state.identity.method)
    if functional.identity != expected_functional.identity:
        raise ValueError(
            "stationary "
            f"{contract.family.upper()} requires canonical {expected_functional.identifier}"
        )
    if state.identity.regularization_identity != xc_regularization_identity(functional):
        raise ValueError("stationary regularization identity mismatch")
    for name, actual in (
        ("basis_identity", basis.identity),
        ("geometry_identity", native_ao_geometry_identity(basis)),
        ("grid_identity", grid.identity),
        ("topology_identity", xc_geometry_topology_identity(basis, grid)),
    ):
        if actual != getattr(state.identity, name):
            raise ValueError(f"stationary {name.replace('_', ' ')} mismatch")

    program = ContractionProgram(functional, "geometry")
    density = state.density[0] if contract.spin == "unpolarized" else state.density
    partials = program.evaluate(
        basis.evaluate(grid.points, program.contract.ao_order),
        density,
        grid.weights,
        ao_atoms=_native_ao_atoms(basis),
        natom=basis.natom,
    )["geometry"]
    return FixedDensityXcGeometry(
        state_identity=state.identity,
        discrete_contract_identity=program.contract.identity,
        basis_identity=basis.identity,
        geometry_identity=native_ao_geometry_identity(basis),
        grid_identity=grid.identity,
        topology_identity=xc_geometry_topology_identity(basis, grid),
        functional_identity=functional.identity,
        regularization_identity=xc_regularization_identity(functional),
        density_generation=state.identity.density_generation,
        partials=partials,
    )


def _matrix(value, name):
    array = immutable(value)
    if np.iscomplexobj(value) or array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"{name} must be one real square matrix")
    if not np.isfinite(array).all():
        raise ValueError(f"nonfinite {name}")
    return array


def _spin_matrices(value, spins, n, name):
    array = immutable(value)
    if np.iscomplexobj(value) or array.shape != (spins, n, n):
        raise ValueError(f"{name} must contain every spin matrix")
    if not np.isfinite(array).all():
        raise ValueError(f"nonfinite {name}")
    if name in ("density", "Fock", "weighted density") and not np.allclose(
        array, array.swapaxes(-1, -2), atol=_ARRAY_TOLERANCE, rtol=0
    ):
        raise ValueError(f"{name} is not symmetric")
    return array


def _spin_vectors(value, spins, n, name):
    array = immutable(value)
    if np.iscomplexobj(value) or array.shape != (spins, n):
        raise ValueError(f"{name} must contain every spin orbital")
    if not np.isfinite(array).all():
        raise ValueError(f"nonfinite {name}")
    return array
