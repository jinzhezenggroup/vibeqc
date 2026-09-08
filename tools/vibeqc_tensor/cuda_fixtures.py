"""Controlled CC-like fragments for CUDA lowering, not a complete CC method."""

from dataclasses import dataclass

import numpy as np

from .ir import add, broadcast, divide, einsum, input_tensor, transpose
from .program import Program
from .types import Index, IndexSpace, TensorSpec


@dataclass(frozen=True)
class TensorFixture:
    """Fixed equation/populations and reproducible numerical state."""

    name: str
    program: Program
    inputs: dict[str, np.ndarray]
    seed: int
    nocc: int
    nvir: int


def cc_fixtures(nocc: int, nvir: int, *, seed: int = 146) -> tuple[TensorFixture, ...]:
    """Build a virtual residual and denominator update at explicit dimensions.

    Random supplied amplitudes/integrals have no undeclared symmetry. Orbital
    energies are separated from zero denominators; scaling all feeds retains
    this separation. References use the independent TensorIR interpreter.
    """
    occupied, virtual = (
        IndexSpace("occupied", "occupied", nocc),
        IndexSpace("virtual", "virtual", nvir),
    )
    axes = (
        Index("i", occupied),
        Index("j", occupied),
        Index("a", virtual),
        Index("b", virtual),
    )

    def tensor(name, indices):
        return input_tensor(
            name, TensorSpec(indices, role="parameter", representation="spin_orbital")
        )

    t, f = tensor("t", axes), tensor("f", (Index("a", virtual), Index("e", virtual)))
    eo, ev = tensor("eo", (axes[0],)), tensor("ev", (axes[2],))
    g = tensor("g", axes)
    denominator = add(
        broadcast(eo, axes, (0,)),
        broadcast(eo, axes, (1,)),
        broadcast(ev, axes, (2,)),
        broadcast(ev, axes, (3,)),
        coefficients=(1, 1, -1, -1),
    )
    term = einsum("ae,ijeb->ijab", f, t)
    rng = np.random.default_rng(seed)
    t_values = rng.normal(scale=0.02, size=(nocc, nocc, nvir, nvir))
    f_values = rng.normal(scale=0.1, size=(nvir, nvir))
    g_values = rng.normal(scale=0.02, size=t_values.shape)
    provenance = {
        "fixture_version": 1,
        "seed": seed,
        "nocc": nocc,
        "nvir": nvir,
        "scope": "fixed supplied tensors; no CC iterations, Hamiltonian or molecular calculation",
    }
    return (
        TensorFixture(
            "virtual_residual",
            Program(
                {
                    "residual": add(
                        term, transpose(term, (0, 1, 3, 2)), coefficients=(1, -1)
                    )
                },
                provenance=provenance,
            ),
            {"t": t_values, "f": f_values},
            seed,
            nocc,
            nvir,
        ),
        TensorFixture(
            "denominator_update",
            Program({"amplitudes": divide(g, denominator)}, provenance=provenance),
            {
                "g": g_values,
                "eo": np.linspace(-1.0, -0.5, nocc),
                "ev": np.linspace(0.2, 1.5, nvir),
            },
            seed,
            nocc,
            nvir,
        ),
    )
