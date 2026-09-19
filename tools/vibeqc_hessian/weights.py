"""Density weight folding for the analytic HF Hessian (issue #180).

The #178 second-derivative provider contracts a fixed external weight against
the second derivatives of the integrals, and applies no HF density formula of
its own. The Coulomb and exchange factors therefore have to be folded by the
caller, and they are folded here -- once, in code that is checked against the
energy it came from, rather than in prose that the assembly would have to
re-derive and could silently contradict.

The factor worth being careful about is the ``½`` in ``E = ... + ½ Tr[P G(P)]``.
It is already absorbed by the reindexing that puts the Coulomb and exchange
terms over a common integral (see ``docs/hessian.md``), so the folded weight
carries it and no further factor belongs in the contraction. Applying it twice
would halve both contributions -- an error that is small enough to look like
ordinary numerical disagreement rather than a mistake.
"""

import typing

import numpy as np

__all__ = [
    "two_electron_energy",
    "two_electron_weight",
    "weight_energy",
]


def _validated(matrix: typing.Any, *, name: str) -> typing.Any:
    """Return ``matrix`` as a finite square float64 array, or raise."""
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError(f"{name} must be a square matrix, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must be finite")
    return values


def two_electron_weight(density: typing.Any) -> typing.Any:
    """Return ``W_μνλσ = ½ P_μν P_λσ − ¼ P_μλ P_νσ``.

    This is the weight the #178 ``weighted_hessian`` consumer expects for the
    frozen-density two-electron energy skeleton. A Fock-derivative right-hand
    side instead contracts one density with a free AO pair; this scalar energy
    weight must not be substituted for that matrix-valued contraction.

    The result is indexed ``W[μ, ν, λ, σ]`` against the chemist-notation
    integral ``(μν|λσ)``.
    """
    density = _validated(density, name="density")
    coulomb = np.einsum("uv,ls->uvls", density, density)
    exchange = np.einsum("ul,vs->uvls", density, density)
    return 0.5 * coulomb - 0.25 * exchange


def weight_energy(weight: typing.Any, eri: typing.Any) -> typing.Any:
    """Contract an integral weight against chemist-notation ERIs ``(μν|λσ)``.

    This is the shape the #178 weighted consumer performs internally:
    ``Σ_{μνλσ} W_μνλσ (μν|λσ)``. Keeping it as a named function lets the
    folding be checked against the energy it is supposed to reproduce.
    """
    weight = np.asarray(weight, dtype=np.float64)
    eri = np.asarray(eri, dtype=np.float64)
    if weight.shape != eri.shape:
        raise ValueError(
            f"weight shape {weight.shape} does not match integral shape {eri.shape}"
        )
    return float(np.einsum("uvls,uvls->", weight, eri))


def two_electron_energy(density: typing.Any, eri: typing.Any) -> typing.Any:
    """Return ``½ Tr[P G(P)]`` straight from its definition.

    Computed independently of :func:`two_electron_weight` -- it builds ``G``
    explicitly and contracts it with ``P`` -- so that comparing the two is a
    real check of the folding rather than a restatement of it.
    """
    density = _validated(density, name="density")
    eri = np.asarray(eri, dtype=np.float64)
    coulomb = np.einsum("ls,uvls->uv", density, eri)
    exchange = np.einsum("ls,ulvs->uv", density, eri)
    fock = coulomb - 0.5 * exchange
    return float(0.5 * np.einsum("uv,uv->", density, fock))
