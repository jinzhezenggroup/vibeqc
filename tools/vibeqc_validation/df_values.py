"""Independent libcint fixtures for generated raw density-fitting values.

Normalization and the public Cartesian convention reuse the shared CG01/f-shell
adapters. References use libcint int2c2e/int3c2e directly, never a dummy shell or
the generated Gaussian moment algebra. Spherical fixture projection is an
explicit host check; native placement is validated separately at the DF source.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from itertools import product

import numpy as np
from vibeqc_compiler.integral.shell_spec import cartesian_components

from tools.generate_validation_references import pyscf_molecule

from .f_shell_numerics import _normalized_primitives
from .schema import canonical_hash

# The standalone CUDA fixture ABI is fixed little-endian FP64 with a checked
# 144-byte host record. It carries real positive-exponent basis functions only.
PRIMITIVE_DTYPE = np.dtype(
    [
        ("centers_count", "<u4"),
        ("angular", "<u4", (3, 3)),
        ("centers", "<f8", (3, 3)),
        ("exponents", "<f8", (3,)),
        ("weight", "<f8"),
    ],
    align=True,
)
assert PRIMITIVE_DTYPE.itemsize == 144


@dataclass
class DFValueFixture:
    """One complete contracted shell block and its independent public reference."""

    name: str
    inputs: dict
    records: np.ndarray
    primitive_count: int
    reference: np.ndarray
    spherical_reference: np.ndarray
    projections: tuple[np.ndarray, ...]

    @property
    def input_hash(self) -> str:
        return canonical_hash(
            {
                "inputs": self.inputs,
                "record_sha256": hashlib.sha256(self.records.tobytes()).hexdigest(),
            }
        )

    def contract(self, primitive_values: np.ndarray) -> np.ndarray:
        """Recover dense A/M order from consecutive primitive contractions."""
        return (
            primitive_values.reshape(-1, self.primitive_count)
            .sum(axis=1)
            .reshape(self.reference.shape)
        )

    def spherical(self, values: np.ndarray) -> np.ndarray:
        """Apply the independent public-basis projection on the host."""
        for axis, matrix in enumerate(self.projections):
            values = np.moveaxis(
                np.tensordot(matrix.T, values, axes=(1, axis)), 0, axis
            )
        return values


def make_df_value_fixture(
    angular: tuple[int, ...], *, variant="asymmetric", primitive_lengths=None
) -> DFValueFixture:
    """Generate a full Cartesian/spherical M or A shell block with signed contractions."""
    count = len(angular)
    if count not in (2, 3) or any(
        type(l) is not int or not 0 <= l <= 3 for l in angular
    ):
        raise ValueError("DF fixtures require two/three s/p/d/f shells")
    if variant not in ("asymmetric", "coincident"):
        raise ValueError("unknown DF geometry variant")
    coordinates = [[0.13, -0.31, 0.24], [-0.43, 0.27, 0.51], [0.68, -0.14, -0.22]][
        :count
    ]
    if variant == "coincident":
        coordinates = [coordinates[0]] * count
    lengths = tuple(primitive_lengths or (2 if i % 2 == 0 else 1 for i in range(count)))
    if len(lengths) != count or any(type(n) is not int or n < 1 for n in lengths):
        raise ValueError("positive contraction lengths are required")
    shells = []
    for i, (l, length) in enumerate(zip(angular, lengths)):
        shells.append(
            {
                "atom_index": i,
                "angular_momentum": l,
                "primitives": [
                    [0.57 + 0.23 * i + 0.71 * j, 0.83 if j == 0 else (-0.17 / j)]
                    for j in range(length)
                ],
            }
        )
    family = "metric" if count == 2 else "three_center"
    name = f"{family}/{'spdf'[angular[0]]}{''.join('spdf'[l] for l in angular[1:])}/{variant}"
    inputs = {
        "name": name,
        "atomic_numbers": [1] * count,
        "coordinates": coordinates,
        "shells": shells,
        "basis_representation": "cartesian",
        "charge": 0,
        "multiplicity": 1 + count % 2,
        "reference_version": 1,
        "units": {"coordinates": "bohr", "integrals": "atomic_units"},
    }
    mol, scales, _ = pyscf_molecule(inputs)
    operator = "int2c2e" if count == 2 else "int3c2e"
    reference = mol.intor_by_shell(operator, tuple(range(count)))
    locations = mol.ao_loc_nr()
    for axis in range(count):
        shape = [1] * count
        shape[axis] = reference.shape[axis]
        reference *= scales[locations[axis] : locations[axis + 1]].reshape(shape)
    sph_locations = mol.ao_loc_nr(cart=False)
    full_projection = mol.cart2sph_coeff() / scales[:, None]
    projections = tuple(
        full_projection[
            locations[i] : locations[i + 1], sph_locations[i] : sph_locations[i + 1]
        ]
        for i in range(count)
    )
    spherical_reference = mol.intor_by_shell(operator + "_sph", tuple(range(count)))

    components = list(product(*(cartesian_components(l) for l in angular)))
    radial_products = list(
        product(*(_normalized_primitives(shell) for shell in shells))
    )
    records = np.zeros((len(components), len(radial_products)), dtype=PRIMITIVE_DTYPE)
    records["centers_count"] = count
    slots = (0, 2) if count == 2 else (0, 1, 2)
    factors = np.ones(len(components))
    for source, slot in enumerate(slots):
        powers = np.array(
            [
                [component[source].count(axis) for axis in "xyz"]
                for component in components
            ],
            dtype=np.uint32,
        )
        records["angular"][:, :, slot, :] = powers[:, None, :]
        records["centers"][:, :, slot, :] = coordinates[source]
        records["exponents"][:, :, slot] = [
            primitives[source][0] for primitives in radial_products
        ]
        factors /= np.sqrt(
            [
                math.prod(math.prod(range(1, 2 * int(power), 2)) for power in row)
                for row in powers
            ]
        )
    radial_weights = np.array(
        [math.prod(c for _, c in primitives) for primitives in radial_products]
    )
    records["weight"] = factors[:, None] * radial_weights[None, :]
    return DFValueFixture(
        name,
        inputs,
        records.reshape(-1),
        len(radial_products),
        reference,
        spherical_reference,
        projections,
    )


def df_value_matrix() -> list[DFValueFixture]:
    """Cover all 16 metric and 64 three-center signatures in two geometry regimes."""
    return [
        make_df_value_fixture(angular, variant=variant)
        for count in (2, 3)
        for angular in product(range(4), repeat=count)
        for variant in ("asymmetric", "coincident")
    ]
