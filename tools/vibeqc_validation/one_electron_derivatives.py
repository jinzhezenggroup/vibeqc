"""Libcint raw derivative blocks using independent normalized AO conventions."""

from dataclasses import dataclass

import numpy as np

from tools.generate_validation_references import pyscf_molecule

from .one_electron_values import OneElectronValueFixture, one_electron_value_matrix


@dataclass
class OneElectronDerivativeFixture(OneElectronValueFixture):
    """Derivative channels are [operator S/T/V, center A/B/C, xyz, AO, AO]."""

    def contract(self, values):
        blocks = values.reshape(*self.weights.shape, 3, 3, 3)
        result = (blocks * self.weights[:, :, None, None, None]).sum(axis=1)
        return result.transpose(1, 2, 3, 0).reshape(self.reference.shape)

    def spherical(self, values):
        a, b = self.projections
        return np.einsum("ia,ocxij,jb->ocxab", a, values, b)


def one_electron_derivative_matrix():
    """All public pairs, difficult geometries and long normalized contractions."""
    fixtures = []
    for value in one_electron_value_matrix():
        mol, scales, _ = pyscf_molecule(value.inputs)
        charge = value.inputs["operator_charge"]
        references = []
        with mol.with_rinv_origin(value.inputs["operator_center"]):
            for representation in ("cart", "sph"):
                operators = []
                for name, factor in (("ovlp", 1), ("kin", 1), ("rinv", -charge)):
                    integral = f"int1e_ip{name}_{representation}"
                    first = -factor * mol.intor_by_shell(integral, (0, 1), comp=3)
                    second = -factor * mol.intor_by_shell(
                        integral, (1, 0), comp=3
                    ).transpose(0, 2, 1)
                    external = (
                        -first - second if name == "rinv" else np.zeros_like(first)
                    )
                    operators.append([first, second, external])
                references.append(np.array(operators))
        offset = mol.ao_loc_nr()[1]
        references[0] *= (
            scales[:offset][None, None, None, :, None]
            * scales[offset:][None, None, None, None, :]
        )
        fixtures.append(
            OneElectronDerivativeFixture(
                value.inputs,
                value.records,
                value.weights,
                *references,
                value.projections,
            )
        )
    return fixtures
