"""Small reproducible tensor fragments with separately written loop oracles."""

from dataclasses import dataclass

import numpy as np

from .ir import add, broadcast, divide, einsum, input_tensor, transpose
from .oracles import (
    matrix_product,
    mp2_energy,
    restricted_pair_update,
    virtual_residual,
)
from .packing import PackedLayout
from .program import Program
from .types import Index, IndexSpace, Symmetry, TensorSpec


@dataclass(frozen=True)
class Example:
    """Self-contained numerical fixture; arrays use the explicit logical order."""

    name: str
    program: Program
    inputs: dict[str, np.ndarray]
    reference: np.ndarray
    packing: PackedLayout | None = None


def example_cases(seed: int = 145) -> tuple[Example, ...]:
    """Build four controlled FP64 cases, including noncontiguous and packed data."""
    rng = np.random.default_rng(seed)
    occupied = IndexSpace("occupied", "occupied", 2)
    virtual = IndexSpace("virtual", "virtual", 3)
    ao = IndexSpace("ao", "ao", 3)
    auxiliary = IndexSpace("auxiliary", "auxiliary", 4)
    ijab = (
        Index("i", occupied),
        Index("j", occupied),
        Index("a", virtual),
        Index("b", virtual),
    )
    provenance = {
        "example_version": 1,
        "seed": seed,
        "oracle": "independent explicit loops",
    }

    left = input_tensor(
        "left", TensorSpec((Index("p", ao), Index("P", auxiliary)), role="input")
    )
    right = input_tensor(
        "right", TensorSpec((Index("P", auxiliary), Index("q", ao)), role="input")
    )
    # Both views exercise the feed boundary without making the reference
    # depend on a contiguous copy or backend-specific physical strides.
    left_values = rng.normal(size=(3, 8))[:, ::2]
    right_values = rng.normal(size=(3, 4)).T
    matrix = Example(
        "matrix_product",
        Program({"value": einsum("pP,Pq->pq", left, right)}, provenance=provenance),
        {"left": left_values, "right": right_values},
        matrix_product(left_values, right_values),
    )

    antisymmetry = (Symmetry((1, 0, 2, 3), -1), Symmetry((0, 1, 3, 2), -1))
    spin_spec = TensorSpec(
        ijab, symmetries=antisymmetry, representation="spin_orbital", role="input"
    )
    packing = PackedLayout.from_spec(spin_spec)
    integrals = input_tensor("g", spin_spec)
    g = packing.unpack(rng.normal(scale=0.1, size=packing.size))
    eo = input_tensor(
        "eo", TensorSpec((ijab[0],), representation="spin_orbital", role="input")
    )
    ev = input_tensor(
        "ev", TensorSpec((ijab[2],), representation="spin_orbital", role="input")
    )
    occupied_values, virtual_values = np.array([-0.8, -0.5]), np.array([0.2, 0.4, 0.9])
    denominator = add(
        broadcast(eo, ijab, (0,)),
        broadcast(eo, ijab, (1,)),
        broadcast(ev, ijab, (2,)),
        broadcast(ev, ijab, (3,)),
        coefficients=(1, 1, -1, -1),
    )
    amplitudes = divide(integrals, denominator)
    mp2 = Example(
        "mp2_energy_fragment",
        Program(
            {"value": einsum("ijab,ijab->", integrals, amplitudes, coefficient="1/4")},
            provenance=provenance,
        ),
        {"g": g, "eo": occupied_values, "ev": virtual_values},
        mp2_energy(g, occupied_values, virtual_values),
        packing,
    )

    fock = input_tensor(
        "f",
        TensorSpec(
            (Index("a", virtual), Index("e", virtual)),
            representation="spin_orbital",
            role="input",
        ),
    )
    t2 = input_tensor(
        "t",
        TensorSpec(
            ijab,
            symmetries=antisymmetry,
            representation="spin_orbital",
            role="parameter",
            differentiable=True,
        ),
    )
    term = einsum("ae,ijeb->ijab", fock, t2)
    f, t = (
        rng.normal(scale=0.2, size=(3, 3)),
        packing.unpack(rng.normal(scale=0.1, size=packing.size)),
    )
    residual = Example(
        "cc_virtual_residual_fragment",
        Program(
            {"value": add(term, transpose(term, (0, 1, 3, 2)), coefficients=(1, -1))},
            provenance=provenance,
        ),
        {"f": f, "t": t},
        virtual_residual(f, t),
        packing,
    )

    trial = input_tensor(
        "trial",
        TensorSpec(
            ijab,
            representation="restricted_spatial",
            role="parameter",
            differentiable=True,
        ),
    )
    spatial_packing = PackedLayout.from_spec(
        TensorSpec(
            ijab,
            symmetries=(Symmetry((1, 0, 3, 2)),),
            representation="restricted_spatial",
            role="input",
        )
    )
    trial_values = rng.normal(scale=0.1, size=(2, 2, 3, 3))
    update = Example(
        "restricted_pair_update",
        Program(
            {
                "value": add(
                    trial, transpose(trial, (1, 0, 3, 2)), coefficients=("1/2", "1/2")
                )
            },
            provenance=provenance,
        ),
        {"trial": trial_values},
        restricted_pair_update(trial_values),
        spatial_packing,
    )
    return matrix, mp2, residual, update
