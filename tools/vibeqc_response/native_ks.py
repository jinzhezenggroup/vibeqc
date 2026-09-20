"""Live native CPU RKS/UKS handoff into the shared CPKS operator and solver.

The batch and AO basis are borrowed; the integral source and native snapshot
lease are owned. SCF replay, failed replay, or closure revokes this response.
No SCF rerun, recanonicalization, or relabeling of an HF reference occurs here.
"""

from __future__ import annotations

import typing

import numpy as np
from vibeqc._dft_gradient import StationaryDerivativeContract, StationaryKsState
from vibeqc.ks import resolve_ks_method
from vibeqc.profiles import canonical_hash
from vibeqc_compiler.dft.features import density_features
from vibeqc_compiler.xc.potential import assemble_coefficients

from tools.vibeqc_posthf.reference import ReferenceSnapshot
from tools.vibeqc_posthf.sources import NativeSource

from .backends import NativeJKBackend
from .operators import CPKSResponseOperator, cpks_operator_identity
from .problem import ResponseUnsupported
from .uhf import UHFReferenceSnapshot, UKSResponseOperator, uks_operator_identity
from .xc import FixedDensityXCDerivativeKernel


class _NativeKSXCKernel(FixedDensityXCDerivativeKernel):
    """Bind the common tile/assembly path to the SCF-domain point differential."""

    def __init__(
        self,
        state: typing.Any,
        spec: typing.Any,
        basis: typing.Any,
        *,
        tile_points: int,
    ) -> None:
        self.state = state
        super().__init__(
            spec,
            basis,
            state.grid,
            state.density[0] if spec.spin == "unpolarized" else state.density,
            tile_points=tile_points,
        )
        self.identity = canonical_hash(
            {
                "kernel": self.identity,
                "point_model": state.identity.regularization_identity,
                "native_state": state.identity.to_payload(),
                "derivative": f"{spec.spin}-cartesian-directional-v1",
            }
        )

    def _response_tile(
        self,
        jets: typing.Any,
        density: typing.Any,
        direction: typing.Any,
        weights: typing.Any,
    ) -> typing.Any:
        ingredients = (
            ("rho", "gradient") if "sigma" in self.spec.ingredients else ("rho",)
        )
        features = density_features(jets, density, ingredients=ingredients)
        delta = density_features(jets, direction, ingredients=ingredients)
        zero = np.zeros((2, jets.shape[1], 3))
        if self.spec.spin == "polarized":
            coefficients = self.state._source.evaluate_uks_response_points(
                "sigma" in self.spec.ingredients,
                features["rho"],
                features.get("gradient", zero),
                delta["rho"],
                delta.get("gradient", zero),
            )
        else:
            coefficients = self.state._source.evaluate_rks_response_points(
                "sigma" in self.spec.ingredients,
                features["rho"].sum(axis=0),
                features.get("gradient", zero).sum(axis=0),
                delta["rho"].sum(axis=0),
                delta.get("gradient", zero).sum(axis=0),
            )
        if "sigma" not in self.spec.ingredients:
            coefficients.pop("gradient")
        return assemble_coefficients(jets, coefficients, weights)


class _NativeKSLease:
    """Shared live-state binding and owned/borrowed lifetime for CPU KS response."""

    _spin_blocks: int
    _response_identity: typing.ClassVar[typing.Any]

    @classmethod
    def from_native(
        cls,
        batch: typing.Any,
        basis: typing.Any,
        grid: typing.Any = None,
        *,
        index: int = 0,
        functional: typing.Any = None,
        tile_points: int = 256,
        axis_tile: int = 2,
        perturbation_labels: tuple[str, ...] = (),
    ) -> typing.Self:
        """Bind actual converged all-electron CPU KS orbitals without rerunning SCF."""
        state = StationaryKsState.from_native(batch, basis, grid, index=index)
        source = None
        try:
            spin_method = "rks" if cls._spin_blocks == 1 else "uks"
            if (
                state._source.backend != "cpu"
                or state.identity.method
                not in (f"lda-{spin_method}", f"pbe-{spin_method}")
                or state._source.hamiltonian != "all-electron"
            ):
                raise ResponseUnsupported(
                    f"native CPKS requires all-electron CPU {spin_method.upper()} LDA/PBE"
                )
            _, expected = resolve_ks_method(state.identity.method)
            spec = expected if functional is None else functional
            if spec.identity != state.identity.functional_identity:
                raise ValueError("native CPKS functional identity mismatch")
            source = NativeSource(
                basis.atoms,
                basis=basis.shells,
                charge=basis.charge,
                multiplicity=basis.multiplicity,
                representation=basis.representation,
            )
            overlap, hcore = source.one_electron()
            if not np.allclose(overlap, state.overlap, atol=1e-12, rtol=0):
                raise ValueError("native CPKS integral source overlap mismatch")
            common = {
                "overlap": state.overlap,
                "hcore": hcore,
                "reference_energy": state._source.energy(),
                "scf_residual": state.physical_residual,
                "geometry_hash": source.geometry_hash,
                "basis_hash": source.basis_hash,
                "generation_id": canonical_hash(state.identity.to_payload()),
                "representation": basis.representation,
                "hf_backend": f"native-cpu-{spin_method}",
                "functional_identity": spec.identity,
                "grid_identity": state.grid.identity,
            }
            if cls._spin_blocks == 1:
                reference = ReferenceSnapshot(
                    **common,
                    fock=state.fock[0],
                    coefficients=state.coefficients[0],
                    orbital_energies=state.orbital_energies[0],
                    occupations=state.occupations[0],
                    electron_count=source.electron_count,
                    algorithm="KS",
                )
            else:
                # Preserve the independently canonical alpha and beta frames;
                # the shared spin reference never reinterprets them as RHF.
                reference = UHFReferenceSnapshot(
                    **common,
                    algorithm="UKS",
                    **{
                        f"{name}_{spin}": getattr(state, name)[index]
                        for name in (
                            "fock",
                            "coefficients",
                            "orbital_energies",
                            "occupations",
                        )
                        for index, spin in enumerate(("alpha", "beta"))
                    },
                )
            backend = NativeJKBackend(source, axis_tile=axis_tile)
            kernel = _NativeKSXCKernel(state, spec, basis, tile_points=tile_points)
            problem = cls.build_problem(
                reference, backend, kernel, perturbation_labels=perturbation_labels
            )
            result = cls(problem, backend, kernel)
            result.state = state
            result._source = source
            result._reference_identity = reference.identity
            result._backend_identity = backend.identity
            result._contract = StationaryDerivativeContract(state.identity)
            result.validate_current()
            return result
        except Exception:
            state._source.close()
            if source is not None:
                source.close()
            raise

    def validate_current(self) -> None:
        """Reject revoked state even for a zero RHS or an already-converged guess."""
        self._contract.validate(self.state)
        self._source._check_open()
        if not self.xc_kernel.basis._handle:
            raise ValueError("native CPKS AO basis is closed")
        if (
            self.problem.reference.identity != self._reference_identity
            or self.backend.identity != self._backend_identity
            or self.backend.source is not self._source
            or self.xc_kernel.state is not self.state
            or self.xc_kernel.spec.identity != self.state.identity.functional_identity
            or self.xc_kernel.grid.identity != self.state.identity.grid_identity
            or self.xc_kernel.basis.identity != self.state.identity.basis_identity
            or self.problem.operator_identity
            != self._response_identity(self.backend, self.xc_kernel)
        ):
            raise ValueError("native CPKS state/provider/kernel identity mismatch")
        self.backend.validate_reference(self.problem.reference)
        self.xc_kernel.validate_reference(self.problem.reference)

    def close(self) -> None:
        """Revoke the owned response lease; leave the borrowed batch and AO open."""
        if hasattr(self, "state"):
            self.state._source.close()
        if hasattr(self, "_source"):
            self._source.close()

    def __enter__(self) -> typing.Self:
        self.validate_current()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


class NativeRKSResponse(_NativeKSLease, CPKSResponseOperator):
    """Owned CPU LDA/PBE RKS adapter for the shared ``solve``/``solve_many``.

    Construct with ``from_native`` after successful SCF. Keep the borrowed
    batch and NativeAO open; this object owns its snapshot lease and integrals.
    The exact native state/grid/provider identity gates every action and solve.
    AO/MO transforms and Krylov remain on the host.
    """

    _spin_blocks = 1
    _response_identity = staticmethod(cpks_operator_identity)

    def _base_action(
        self, vector: typing.Any, *, transpose: bool = False
    ) -> typing.Any:
        self.validate_current()
        result = super()._base_action(vector, transpose=transpose)
        self.validate_current()
        return result


class NativeUKSResponse(_NativeKSLease, UKSResponseOperator):
    """Live native CPU LDA/PBE UKS adapter with coupled alpha/beta response.

    Uses the same lease, point model, XC assembly, spin layout and Krylov
    implementation as the existing restricted/spin consumers. No spin average,
    SCF rerun or synthetic reference substitution is performed.
    """

    _spin_blocks = 2
    _response_identity = staticmethod(uks_operator_identity)

    def apply(self, vector: typing.Any) -> typing.Any:
        self.validate_current()
        result = super().apply(vector)
        self.validate_current()
        return result
