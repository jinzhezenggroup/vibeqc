"""Independent normalized libcint dM/dA blocks with mathematical center axes."""

import typing
from dataclasses import dataclass

import numpy as np

from tools.generate_validation_references import pyscf_molecule

from .df_values import DFValueFixture, make_df_value_fixture


@dataclass
class DFDerivativeFixture(DFValueFixture):
    """Raw layout is (center,xyz,AO...) and retains every mathematical center."""

    def contract(self, primitive_values: typing.Any) -> typing.Any:
        count = len(self.inputs["shells"])
        shape = self.reference.shape[2:]
        return (
            primitive_values.reshape(-1, self.primitive_count, count, 3)
            .sum(axis=1)
            .transpose(1, 2, 0)
            .reshape(count, 3, *shape)
        )

    def spherical(self, values: typing.Any) -> typing.Any:
        for axis, matrix in enumerate(self.projections):
            values = np.moveaxis(
                np.tensordot(matrix.T, values, axes=(1, axis + 2)), 0, axis + 2
            )
        return values


def make_df_derivative_fixture(
    angular: typing.Any,
    *,
    variant: typing.Any = "asymmetric",
    primitive_lengths: typing.Any = None,
) -> typing.Any:
    """Differentiate all real libcint Gaussian centers with independent signs."""
    value = make_df_value_fixture(
        angular, variant=variant, primitive_lengths=primitive_lengths
    )
    mol, scales, _ = pyscf_molecule(value.inputs)
    count = len(angular)
    references = []
    for representation in ("cart", "sph"):
        if count == 2:
            first = -mol.intor_by_shell("int2c2e_ip1_" + representation, (0, 1), comp=3)
            second = -mol.intor_by_shell(
                "int2c2e_ip1_" + representation, (1, 0), comp=3
            ).transpose(0, 2, 1)
            reference = np.array([first, second])
        else:
            first = -mol.intor_by_shell(
                "int3c2e_ip1_" + representation, (0, 1, 2), comp=3
            )
            second = -mol.intor_by_shell(
                "int3c2e_ip1_" + representation, (1, 0, 2), comp=3
            ).transpose(0, 2, 1, 3)
            third = -mol.intor_by_shell(
                "int3c2e_ip2_" + representation, (0, 1, 2), comp=3
            )
            reference = np.array([first, second, third])
        if representation == "cart":
            offsets = mol.ao_loc_nr()
            for axis in range(count):
                shape = [1] * (count + 2)
                shape[axis + 2] = reference.shape[axis + 2]
                reference *= scales[offsets[axis] : offsets[axis + 1]].reshape(shape)
        references.append(reference)
    return DFDerivativeFixture(
        value.name,
        value.inputs,
        value.records,
        value.primitive_count,
        *references,
        value.projections,
    )
