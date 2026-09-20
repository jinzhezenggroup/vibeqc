"""Immutable current-density provenance and validated CPU D/C alternatives.

External orbital factors are checked once against the supplied D in row panels.
No SCF producer is trusted implicitly, and no eigendecomposition repairs D. This
fixed-input reference is not a native prepared SCF or GPU memory-budget API.
"""

from __future__ import annotations

import typing
from copy import copy
from dataclasses import asdict, dataclass
from hashlib import sha256

import numpy as np

from vibeqc_compiler.common.provenance import canonical_hash

from .features import _spin_orbitals, density_features, orbital_features, spin_densities
from .grid import checked_int


@dataclass(frozen=True)
class DensityStamp:
    """Content plus producer generations, including geometry through basis identity.

    Use NativeAO.identity (or an equally complete geometry/basis identity).
    Advance basis_generation after reconfiguration and density_generation for
    each new producer state, even when its numerical matrix happens to repeat.
    A stamp is provenance supplied by the caller, not proof that C describes D.
    """

    basis_identity: str
    basis_generation: int
    density_generation: int
    density_identity: str
    layout: str
    role: str

    def __post_init__(self) -> None:
        for name in ("basis_identity", "density_identity"):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)
            ):
                raise ValueError(f"{name} must be a SHA256 content identity")
        checked_int(self.basis_generation, "basis generation", low=0)
        checked_int(self.density_generation, "density generation", low=0)
        if self.layout not in ("total", "separate_spin"):
            raise ValueError("unsupported density occupation convention")
        if self.role not in ("state", "response"):
            raise ValueError("density role must be state or response")

    @property
    def identity(self) -> typing.Any:
        """Hash all state metadata; equal shapes never establish compatibility."""
        return canonical_hash({"schema": "vibeqc.density-source.v1", **asdict(self)})


@dataclass(frozen=True, init=False, eq=False)
class DensitySource:
    """Own the original normalized spin density and an optional checked C/f pair.

    Total D splits equally; C/f always uses separate per-spin occupations.
    Negative/indefinite D remains legal. ``with_orbitals`` returns a new source
    with either validated factors or an explicit D fallback reason. Published
    arrays have immutable backing storage and cannot be mutated by the caller.
    """

    density: np.ndarray
    stamp: DensityStamp
    coefficients: tuple[np.ndarray, np.ndarray] | None
    occupations: tuple[np.ndarray, np.ndarray] | None
    factor_identity: str | None
    fallback_reason: str | None
    validation_status: str
    validation_max_abs_error: float | None

    def __init__(
        self,
        density: typing.Any,
        *,
        basis_identity: typing.Any,
        basis_generation: typing.Any = 0,
        density_generation: typing.Any = 0,
        role: typing.Any = "state",
    ) -> None:
        raw = np.asarray(density)
        if raw.ndim not in (2, 3) or raw.shape[-1] == 0:
            raise ValueError("density must contain a nonempty square AO matrix")
        d = spin_densities(raw, raw.shape[-1])
        layout = "total" if raw.ndim == 2 else "separate_spin"
        density_identity = canonical_hash(
            {
                "shape": d.shape,
                "spin_density_sha256": sha256(d.tobytes()).hexdigest(),
                "layout": layout,
            }
        )
        stamp = DensityStamp(
            basis_identity,
            basis_generation,
            density_generation,
            density_identity,
            layout,
            role,
        )
        object.__setattr__(self, "density", d)
        object.__setattr__(self, "stamp", stamp)
        self._set_validation(None, None, None, "missing_orbitals", "density_only", None)

    def _set_validation(
        self,
        c: typing.Any,
        occ: typing.Any,
        identity: typing.Any,
        reason: typing.Any,
        status: typing.Any,
        error: typing.Any,
    ) -> None:
        # Called only while creating a detached result. Rejected candidates must
        # clear any previously accepted factor, not reuse it accidentally.
        for key, value in zip(
            (
                "coefficients",
                "occupations",
                "factor_identity",
                "fallback_reason",
                "validation_status",
                "validation_max_abs_error",
            ),
            (c, occ, identity, reason, status, error),
            strict=True,
        ):
            object.__setattr__(self, key, value)

    @property
    def source_kind(self) -> typing.Any:
        """Available CPU route; this does not assert a performance winner."""
        return "orbitals" if self.coefficients is not None else "density_matrix"

    def with_orbitals(
        self,
        coefficients: typing.Any,
        occupations: typing.Any,
        *,
        stamp: typing.Any,
        validation_rows: typing.Any = 64,
    ) -> typing.Any:
        """Check an external candidate once, retaining D on any validity failure.

        The candidate's stamp must identify this exact density and generation.
        All entries of C diag(f) C.T are checked against D with the established
        density tolerances (atol=1e-12, rtol=1e-10), in at most validation_rows
        by NAO panels. This is numerical compatibility within those tolerances,
        not an exact-arithmetic certificate. No full reconstructed D is retained.
        Validation is O(NAO^2*norb), performed on attachment, never tile replay.
        Invalid candidate values fall back; invalid API options raise.
        """
        checked_int(validation_rows, "factor validation rows")
        if not isinstance(stamp, DensityStamp):
            raise TypeError("orbital candidate requires a DensityStamp")
        result = copy(self)
        result._set_validation(None, None, None, "stale_orbitals", "rejected", None)
        if stamp != self.stamp:
            return result
        if self.stamp.role == "response":
            result._set_validation(
                None, None, None, "response_density", "rejected", None
            )
            return result
        try:
            c, occ = _spin_orbitals(coefficients, occupations, self.density.shape[1])
        except (ValueError, TypeError, OverflowError) as error:
            result._set_validation(
                None, None, None, f"invalid_orbitals: {error}", "rejected", None
            )
            return result
        maximum = 0.0
        for spin in range(2):
            with np.errstate(over="ignore", invalid="ignore"):
                factor = c[spin] * np.sqrt(occ[spin])
                for begin in range(0, len(factor), validation_rows):
                    end = begin + validation_rows
                    reconstructed = factor[begin:end] @ factor.T
                    target = self.density[spin, begin:end]
                    error = np.abs(reconstructed - target)
                    if not np.isfinite(error).all() or not np.allclose(
                        reconstructed, target, atol=1e-12, rtol=1e-10
                    ):
                        result._set_validation(
                            None, None, None, "incompatible_orbitals", "rejected", None
                        )
                        return result
                    maximum = max(maximum, float(error.max()))
        identity = canonical_hash(
            {
                "schema": "vibeqc.spin-orbitals.v1",
                "stamp": stamp.identity,
                "coefficients": [sha256(a.tobytes()).hexdigest() for a in c],
                "occupations": [sha256(a.tobytes()).hexdigest() for a in occ],
                "shapes": [a.shape for a in c],
                "occupation_convention": "per_spin",
            }
        )
        result._set_validation(c, occ, identity, None, "validated_external", maximum)
        return result

    def features(
        self,
        jets: typing.Any,
        *,
        stamp: typing.Any,
        route: typing.Any = "auto",
        ao_ids: typing.Any = None,
        ingredients: typing.Any = None,
    ) -> typing.Any:
        """Evaluate one CPU tile after an O(1) current-state stamp check.

        Supply the current consumer stamp; a stale source raises because even
        its D is stale. A rejected C candidate instead uses this source's D.
        ``route`` can force either density_matrix or orbitals for parity tests.
        For local jets pass their unique global AO IDs in exactly column order.
        Both routes restrict the same rows; all orbital columns are preserved.
        Geometry derivatives and native/prepared execution are not provided.
        """
        if stamp != self.stamp:
            raise ValueError("stale density source for the current consumer stamp")
        if route not in ("auto", "density_matrix", "orbitals"):
            raise ValueError("unsupported density feature route")
        selected = self.source_kind if route == "auto" else route
        coefficients = self.coefficients
        if selected == "orbitals" and coefficients is None:
            raise ValueError(f"orbital route unavailable: {self.fallback_reason}")
        nao = self.density.shape[1]
        if ao_ids is None:
            ids = np.arange(nao)
        else:
            ids = np.asarray(ao_ids)
            if ids.ndim != 1 or (
                ids.size
                and (
                    ids.dtype.kind not in "iu"
                    or np.any(ids < 0)
                    or np.any(ids >= nao)
                    or len(np.unique(ids)) != len(ids)
                )
            ):
                raise ValueError("local AO IDs must be unique in-range integers")
            ids = ids.astype(np.intp)
        jets = np.asarray(jets)
        if jets.ndim != 3 or jets.shape[2] != len(ids):
            raise ValueError("AO jet columns differ from the density source AO map")
        if selected == "density_matrix":
            # Gather the complete local block, not only its diagonal or shells.
            d = self.density[:, ids[:, None], ids]
            return density_features(jets, d, ingredients=ingredients)
        if coefficients is None:
            raise RuntimeError("orbital coefficients missing after route validation")
        return orbital_features(
            jets,
            tuple(c[ids] for c in coefficients),
            self.occupations,
            ingredients=ingredients,
        )
