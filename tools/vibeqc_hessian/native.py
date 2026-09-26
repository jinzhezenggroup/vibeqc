"""Native RHF input state for the bounded analytic Hessian integration.

SCF and zero-order integral work belong to VibeQC's existing CPU source/export.
This tools endpoint is deliberately limited to 12 Cartesian AOs and four atoms;
no CUDA, DF, ECP, open-shell or production-size Hessian capability is inferred.
"""

import typing
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
from vibeqc_compiler.integral.weight_pullback import normalized_radial_primitives

from tools.vibeqc_posthf.export import export_rhf
from tools.vibeqc_posthf.reference import ReferenceSnapshot, immutable
from tools.vibeqc_posthf.sources import NativeSource
from tools.vibeqc_response.backends import NativeJKBackend


def _validate_source(source: typing.Any) -> None:
    if not isinstance(source, NativeSource):
        raise TypeError("analytic Hessian requires a VibeQC NativeSource")
    source._check_open()
    if not 1 <= source.nbf <= 12 or not 1 <= len(source.atoms) <= 4:
        raise ValueError("analytic Hessian is bounded to 12 AOs and four atoms")
    if (
        source.representation != "cartesian"
        or source.multiplicity != 1
        or source.electron_count % 2
        or source.auxiliary_shells
    ):
        raise ValueError("analytic Hessian requires direct Cartesian closed-shell RHF")
    if any(shell.angular_momentum > 3 for shell in source.shells):
        raise ValueError("analytic Hessian derivative providers support s/p/d/f shells")


@dataclass(frozen=True, eq=False)
class NativeRHFState:
    """Borrow a live immutable source and one owned, converged native snapshot.

    The caller owns source lifetime. No Hessian helper reruns SCF, invents a
    residual, or substitutes a reference-engine density. Host canonicalization
    belongs to the existing small native export bridge, not to a new SCF solver.
    """

    source: NativeSource
    reference: ReferenceSnapshot
    cache: Path = Path(".artifacts")

    def __post_init__(self) -> None:
        self.validate()
        object.__setattr__(self, "cache", Path(self.cache))

    @classmethod
    def from_source(
        cls,
        source: typing.Any,
        *,
        cache: typing.Any = None,
        tolerance: typing.Any = 1e-12,
        max_iterations: typing.Any = 400,
    ) -> typing.Any:
        _validate_source(source)
        reference, _ = export_rhf(
            source, backend="cpu", tolerance=tolerance, max_iterations=max_iterations
        )
        return cls(source, reference, Path(cache) if cache is not None else cls.cache)

    def validate(self) -> typing.Any:
        _validate_source(self.source)
        if not isinstance(self.reference, ReferenceSnapshot):
            raise TypeError("native Hessian requires a validated ReferenceSnapshot")
        if (
            self.reference.algorithm != "RHF"
            or self.reference.hf_backend != "native-cpu"
        ):
            raise ValueError("Hessian input must come from a native CPU RHF solve")
        NativeJKBackend(self.source).validate_reference(self.reference)
        if self.reference.electron_count != self.source.electron_count:
            raise ValueError("Hessian source/reference electron-count mismatch")
        if self.reference.nmo != self.source.nbf:
            raise ValueError("Hessian source/reference AO-dimension mismatch")
        gap = (
            self.reference.orbital_energies[self.reference.nocc]
            - self.reference.orbital_energies[self.reference.nocc - 1]
        )
        if gap <= 1e-8:
            raise ValueError("native Hessian requires a nonzero occupied/virtual gap")
        return self

    @property
    def nbf(self) -> typing.Any:
        return self.source.nbf

    @property
    def nmo(self) -> typing.Any:
        return self.reference.nmo

    @property
    def nocc(self) -> typing.Any:
        return self.reference.nocc

    @property
    def nat(self) -> typing.Any:
        return len(self.source.atoms)

    @property
    def C(self) -> typing.Any:
        return self.reference.coefficients

    @property
    def eps(self) -> typing.Any:
        return self.reference.orbital_energies

    @property
    def S0(self) -> typing.Any:
        return self.reference.overlap

    @property
    def P0(self) -> typing.Any:
        return immutable((self.C * self.reference.occupations) @ self.C.T)

    @property
    def coords(self) -> typing.Any:
        return np.array([atom.position for atom in self.source.atoms])

    @property
    def Z(self) -> typing.Any:
        return np.array([atom.atomic_number for atom in self.source.atoms])

    @property
    def occ(self) -> typing.Any:
        return np.arange(self.nocc)

    @property
    def virt(self) -> typing.Any:
        return np.arange(self.nocc, self.nmo)

    @property
    def offsets(self) -> typing.Any:
        return np.cumsum((0, *self.source.shell_sizes))

    @property
    def primitives(self) -> typing.Any:
        return tuple(
            normalized_radial_primitives(
                shell.angular_momentum,
                tuple((p.exponent, p.coefficient) for p in shell.primitives),
            )
            for shell in self.source.shells
        )

    @cached_property
    def first_order_inputs(self) -> typing.Any:
        """State-bound, immutable H1/S1; geometry/state changes require a new owner."""
        from .first_order import generated_first_order

        return tuple(immutable(value) for value in generated_first_order(self))
