"""Independent libcint S/T/V blocks for all public Cartesian shell pairs."""

import math
from dataclasses import dataclass
from itertools import product

import numpy as np

from tools.generate_validation_references import pyscf_molecule
from tools.vibeqc_codegen.shell_spec import cartesian_components

from .f_shell_numerics import _normalized_primitives
from .schema import canonical_hash


@dataclass
class OneElectronValueFixture:
    """Separate normalized S/T/V matrices plus raw primitive inputs and weights."""

    inputs: dict
    records: np.ndarray
    weights: np.ndarray
    reference: np.ndarray
    spherical_reference: np.ndarray
    projections: tuple[np.ndarray, np.ndarray]

    @property
    def input_hash(self):
        return canonical_hash(self.inputs)

    def contract(self, values):
        """Contract each component without folding raw-output channels together."""
        values = values.reshape(*self.weights.shape, 3)
        result = (values * self.weights[:, :, None]).sum(axis=1)
        return result.T.reshape(self.reference.shape)

    def spherical(self, values):
        """Use libcint's independent normalized Cartesian-to-spherical map."""
        a, b = self.projections
        return np.array([a.T @ block @ b for block in values])


def make_one_electron_fixture(angular, *, variant="asymmetric", lengths=(2, 1)):
    """Cover charge signs, contraction normalization, AO order and Boys regimes."""
    if variant not in ("asymmetric", "coincident", "near", "intermediate", "distant"):
        raise ValueError("unknown one-electron fixture geometry")
    positions = [[0.2, -0.3, 0.1], [-0.4, 0.15, 0.5]]
    center = [0.17, -0.11, -0.4]
    if variant == "coincident":
        positions = [positions[0], positions[0]]
        center = positions[0]
    elif variant == "near":
        positions[1] = [0.2 + 1e-10, -0.3, 0.1]
        center = [0.2, -0.3, 0.1 + 1e-9]
    elif variant == "intermediate":
        center = [2.1, -1.3, 0.7]
    elif variant == "distant":
        center = [50.0, 30.0, -40.0]
    shells = [
        {
            "atom_index": i,
            "angular_momentum": l,
            "primitives": [
                [0.35 + 0.45 * i + 0.63 * j, 1.0 if j == 0 else -0.2 / j]
                for j in range(length)
            ],
        }
        for i, (l, length) in enumerate(zip(angular, lengths))
    ]
    inputs = {
        "name": "one_electron/" + "".join("spdf"[l] for l in angular) + "/" + variant,
        "atomic_numbers": [1, 1],
        "coordinates": positions,
        "shells": shells,
        "basis_representation": "cartesian",
        "charge": 0,
        "multiplicity": 1,
        "reference_version": 1,
        "units": {"coordinates": "bohr", "integrals": "atomic_units"},
        "operator_center": center,
        "operator_charge": 2.3,
    }
    mol, scales, _ = pyscf_molecule(inputs)
    with mol.with_rinv_origin(center):
        reference = np.array(
            [
                mol.intor_by_shell(name + "_cart", (0, 1)) * factor
                for name, factor in (
                    ("int1e_ovlp", 1),
                    ("int1e_kin", 1),
                    ("int1e_rinv", -2.3),
                )
            ]
        )
        spherical_reference = np.array(
            [
                mol.intor_by_shell(name + "_sph", (0, 1)) * factor
                for name, factor in (
                    ("int1e_ovlp", 1),
                    ("int1e_kin", 1),
                    ("int1e_rinv", -2.3),
                )
            ]
        )
    loc = mol.ao_loc_nr()
    reference *= scales[: loc[1]][None, :, None] * scales[loc[1] :][None, None, :]
    sloc = mol.ao_loc_nr(cart=False)
    full = mol.cart2sph_coeff() / scales[:, None]
    projections = tuple(
        full[loc[i] : loc[i + 1], sloc[i] : sloc[i + 1]] for i in range(2)
    )
    components = list(product(*(cartesian_components(l) for l in angular)))
    primitives = list(product(*(_normalized_primitives(shell) for shell in shells)))
    records = np.zeros((len(components), len(primitives), 14), dtype="<f8")
    weights = np.zeros(records.shape[:2])
    offsets = (0, 1, 4, 10)
    for k, pair in enumerate(components):
        factor = math.prod(
            math.prod(range(1, 2 * component.count(axis), 2)) ** -0.5
            for component in pair
            for axis in "xyz"
        )
        for p, radial in enumerate(primitives):
            records[k, p] = [
                radial[0][0],
                radial[1][0],
                *positions[0],
                *positions[1],
                *center,
                2.3,
                offsets[angular[0]] + cartesian_components(angular[0]).index(pair[0]),
                offsets[angular[1]] + cartesian_components(angular[1]).index(pair[1]),
            ]
            weights[k, p] = factor * radial[0][1] * radial[1][1]
    return OneElectronValueFixture(
        inputs,
        records.reshape(-1, 14),
        weights,
        reference,
        spherical_reference,
        projections,
    )


def one_electron_value_matrix():
    """Complete shell pairs in five geometries plus long signed contractions."""
    fixtures = [
        make_one_electron_fixture(angular, variant=variant)
        for angular in product(range(4), repeat=2)
        for variant in ("asymmetric", "coincident", "near", "intermediate", "distant")
    ]
    fixtures += [
        make_one_electron_fixture(angular, lengths=(9, 7))
        for angular in ((0, 0), (1, 2), (3, 3))
    ]
    return fixtures
