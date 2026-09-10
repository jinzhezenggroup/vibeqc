"""Small independent bindings and Libcint oracles for second-order diagnostics.

This module belongs to the optional reference tools. Native generated execution
does not import NumPy quadrature, SciPy or PySCF to evaluate its primitives.
"""

from math import pi, prod, sqrt

import numpy as np
from vibeqc_compiler.integral.shell_spec import cartesian_components

from .weighted_eri import primitive_variables


def second_primitive_variables(
    kernel, exponents, centers, weights=None, direction=None
):
    """Bind bounded moderate-argument fixtures with independent GL64 moments."""
    count = len(kernel.integral.operator.centers)
    centers = np.asarray(centers, dtype=float)
    exponents = np.asarray(exponents, dtype=float)
    if (
        centers.shape != (count, 3)
        or exponents.shape != (len(kernel.integral.signature.shells),)
        or not np.isfinite(centers).all()
        or not np.isfinite(exponents).all()
        or np.any(exponents <= 0)
    ):
        raise ValueError(
            "second derivative fixture requires finite centers and positive exponents"
        )
    if len(exponents) == 4:
        values = primitive_variables(exponents, centers, kernel.boys_count - 1)
    else:
        values = dict(zip(("alpha", "beta"), exponents, strict=True))
        values.update(
            {
                f"{'abc'[i]}_{axis}": centers[i, a]
                for i in range(count)
                for a, axis in enumerate("xyz")
            }
        )
        if kernel.boys_argument is not None:
            argument = kernel.graph.evaluate(kernel.boys_argument, values)
            if not 0 <= argument <= 100:
                raise ValueError(
                    "second derivative fixture quadrature requires T <= 100"
                )
            nodes, quadrature = np.polynomial.legendre.leggauss(64)
            nodes = (nodes + 1) / 2
            values.update(
                {
                    f"boys_{n}": np.dot(
                        quadrature, nodes ** (2 * n) * np.exp(-argument * nodes**2)
                    )
                    / 2
                    for n in range(kernel.boys_count)
                }
            )
    raw = kernel.integral.contractions[0].weights is None
    if raw:
        if weights is not None:
            raise ValueError("raw second derivative fixtures do not accept weights")
        weights = np.ones(kernel.integral.signature.component_count)
    else:
        weights = np.asarray(weights, dtype=float)
        if (
            weights.shape != (kernel.integral.signature.component_count,)
            or not np.isfinite(weights).all()
        ):
            raise ValueError(
                "second derivative fixture weights require the full flat shell shape"
            )
    values.update(
        {f"component_weight_{i}": weights[i] for i in kernel.component_indices}
    )
    if kernel.integral.contractions[0].output == "weighted_hvp":
        direction = np.asarray(direction, dtype=float)
        if (
            direction.shape != (len(kernel.recovery.centers), 3)
            or not np.isfinite(direction).all()
        ):
            raise ValueError(
                "second derivative fixture requires a finite fixed shell direction"
            )
        values.update(
            {
                f"direction_{i}_{axis}": direction[i, a]
                for i in range(len(direction))
                for a, axis in enumerate("xyz")
            }
        )
    elif direction is not None:
        raise ValueError("a Hessian fixture does not consume an HVP direction")
    return values


def evaluate_second_primitive(kernel, exponents, centers, weights=None, direction=None):
    """Interpret only the selected coordinate outputs of a small fixture."""
    values = second_primitive_variables(kernel, exponents, centers, weights, direction)
    return np.array([kernel.graph.evaluate(root, values) for root in kernel.outputs])


def _gaussian_squared_norm(exponent, component):
    """Closed Gaussian moments remove Libcint's radial normalization exactly."""
    return (pi / (2 * exponent)) ** 1.5 * prod(
        prod(range(1, 2 * component.count(axis), 2))
        / (4 * exponent) ** component.count(axis)
        for axis in "xyz"
    )


def libcint_one_electron_hessian(family, angular, exponents, centers, charge=1.0):
    """Independent AA/AB/BB analytic blocks; recover the external nuclear index.

    Both electronic-coordinate derivatives change sign under a Gaussian-center
    displacement, giving a positive second-derivative product. Attraction's
    physical minus charge is separate. No generated first/second DAG is used.
    """
    from pyscf import gto

    mol = gto.M(
        atom=[("ghost-H", centers[0]), ("ghost-He", centers[1])],
        basis={
            name: [[angular_momentum, [e, 1]]]
            for name, angular_momentum, e in zip(
                ("H", "He"), angular, exponents, strict=True
            )
        },
        unit="Bohr",
        cart=True,
        verbose=0,
    )
    components = tuple(
        cartesian_components(angular_momentum) for angular_momentum in angular
    )
    norms = [
        _gaussian_squared_norm(e, c)
        for e, cs in zip(exponents, components, strict=True)
        for c in cs
    ]
    scales = np.sqrt(mol.intor("int1e_ovlp_cart").diagonal() / norms)
    operator, factor = {
        "overlap": ("ovlp", 1),
        "kinetic": ("kin", 1),
        "nuclear_attraction": ("rinv", -charge),
    }[family]
    shape = tuple(map(len, components))
    count = 3 if family == "nuclear_attraction" else 2
    origin = centers[2] if count == 3 else (0, 0, 0)
    with mol.with_rinv_origin(origin):
        aa = factor * mol.intor_by_shell(
            f"int1e_ipip{operator}_cart", (0, 1), comp=9
        ).reshape(3, 3, *shape)
        ab = factor * mol.intor_by_shell(
            f"int1e_ip{operator}ip_cart", (0, 1), comp=9
        ).reshape(3, 3, *shape)
        bb = factor * mol.intor_by_shell(
            f"int1e_ipip{operator}_cart", (1, 0), comp=9
        ).reshape(3, 3, *shape[::-1]).transpose(0, 1, 3, 2)
    result = np.empty((count, 3, count, 3, *shape))
    result[0, :, 0], result[0, :, 1] = aa, ab
    result[1, :, 0], result[1, :, 1] = ab.transpose(1, 0, 2, 3), bb
    if count == 3:
        result[:2, :, 2] = -result[:2, :, :2].sum(axis=2)
        result[2] = -result[:2].sum(axis=0)
    n = shape[0]
    return result / (scales[:n, None] * scales[None, n:])


def libcint_eri_hessian(angular, exponents, centers):
    """Independent analytic ERI Hessian in original center and AO-slot order.

    Libcint supplies same-center, same-pair and cross-pair blocks. Exact ERI
    shell permutations move each requested center pair to the corresponding
    bra slots; the AO axes are then restored before normalization. All four
    diagonal blocks are computed independently of translation recovery.
    This intentionally small raw oracle must not be used for molecular AO^4
    response storage.
    """
    from pyscf import gto

    names = ("H", "He", "Li", "Be")
    mol = gto.M(
        atom=[
            (f"ghost-{name}", position)
            for name, position in zip(names, centers, strict=True)
        ],
        basis={
            name: [[angular_momentum, [e, 1]]]
            for name, angular_momentum, e in zip(names, angular, exponents, strict=True)
        },
        unit="Bohr",
        cart=True,
        verbose=0,
    )
    components = tuple(
        cartesian_components(angular_momentum) for angular_momentum in angular
    )
    shape = tuple(map(len, components))
    norms = [
        _gaussian_squared_norm(e, c)
        for e, cs in zip(exponents, components, strict=True)
        for c in cs
    ]
    scales = np.split(
        np.sqrt(mol.intor("int1e_ovlp_cart").diagonal() / norms), np.cumsum(shape)[:-1]
    )
    factor = (
        scales[0][:, None, None, None]
        * scales[1][None, :, None, None]
        * scales[2][None, None, :, None]
        * scales[3][None, None, None, :]
    )
    result = np.empty((4, 3, 4, 3, *shape))
    for first in range(4):
        for second in range(first, 4):
            other_pair = tuple(i for i in range(4) if i // 2 != first // 2)
            if first == second:
                operator, order = "ipip1", (first, first ^ 1, *other_pair)
            elif first // 2 == second // 2:
                operator, order = "ipvip1", (first, second, *other_pair)
            else:
                operator, order = "ip1ip2", (first, first ^ 1, second, second ^ 1)
            block = mol.intor_by_shell(f"int2e_{operator}_cart", order, comp=9)
            block = block.reshape(3, 3, *(shape[i] for i in order))
            block = (
                block.transpose(0, 1, *(2 + order.index(i) for i in range(4))) / factor
            )
            result[first, :, second] = block
            result[second, :, first] = block.transpose(1, 0, 2, 3, 4, 5)
    return result


def contracted_public_first_gradient(family, inputs, centers, weights):
    """Independent Libcint first gradients for contracted Cartesian/spherical FD.

    The caller holds the public cotangent fixed while displacing mathematical
    centers. Nuclear attraction moves its explicit external center separately
    from the two Gaussian centers. No generated derivative graph is involved.
    """
    from tools.generate_validation_references import pyscf_molecule, quartet_data

    mol, scales, _ = pyscf_molecule(
        {
            **inputs,
            "coordinates": np.asarray(centers)[: len(inputs["coordinates"])].tolist(),
        }
    )
    if family == "eri":
        gradient = np.asarray(quartet_data(mol, scales)["gradient"])
        return np.sum(gradient * weights, axis=(2, 3, 4, 5))
    operator, factor = {
        "overlap": ("ovlp", 1),
        "kinetic": ("kin", 1),
        "nuclear_attraction": ("rinv", -inputs["operator_charge"]),
    }[family]
    origin = centers[2] if family == "nuclear_attraction" else (0, 0, 0)
    with mol.with_rinv_origin(origin):
        first = -factor * mol.intor_by_shell(f"int1e_ip{operator}", (0, 1), comp=3)
        second = -factor * mol.intor_by_shell(
            f"int1e_ip{operator}", (1, 0), comp=3
        ).transpose(0, 2, 1)
    gradient = np.array(
        [first, second] + ([-first - second] if family == "nuclear_attraction" else [])
    )
    offset = mol.ao_loc_nr()[1]
    gradient *= (
        scales[:offset][None, None, :, None] * scales[offset:][None, None, None, :]
    )
    return np.sum(gradient * weights, axis=(2, 3))


def normalized_cartesian_rotation(angular, rotation):
    """Independent polynomial rotation in the unit-Cartesian shell convention.

    Expand each rotated coordinate monomial directly, then apply the ratio of
    closed even Gaussian moments. This validation transform is independent of
    derivative DAGs and supports arbitrary orthogonal three-dimensional R.
    """
    rotation = np.asarray(rotation, dtype=float)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError("Cartesian rotation requires a finite 3 by 3 matrix")
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=2e-14)
    labels = cartesian_components(angular)
    powers = tuple(
        tuple(component.count(axis) for axis in "xyz") for component in labels
    )
    norms = [prod(prod(range(1, 2 * n, 2)) for n in power) for power in powers]
    result = np.zeros((len(labels), len(labels)))
    for row, label in enumerate(labels):
        polynomial = {(0, 0, 0): 1.0}
        for axis in label:
            updated = {}
            for power, coefficient in polynomial.items():
                for column in range(3):
                    key = tuple(n + int(i == column) for i, n in enumerate(power))
                    updated[key] = (
                        updated.get(key, 0)
                        + coefficient * rotation["xyz".index(axis), column]
                    )
            polynomial = updated
        for column, power in enumerate(powers):
            result[row, column] = polynomial.get(power, 0) * sqrt(
                norms[column] / norms[row]
            )
    return result


def libcint_primitive_gradient(family, angular, exponents, centers, charge=1.0):
    """Unnormalized primitive first derivatives for independent directional FD."""
    from pyscf import gto

    names = ("H", "He", "Li", "Be")[: len(angular)]
    mol = gto.M(
        atom=[(f"ghost-{name}", position) for name, position in zip(names, centers)],
        basis={
            name: [[angular_momentum, [e, 1]]]
            for name, angular_momentum, e in zip(names, angular, exponents, strict=True)
        },
        unit="Bohr",
        cart=True,
        verbose=0,
    )
    labels = tuple(
        cartesian_components(angular_momentum) for angular_momentum in angular
    )
    shape = tuple(map(len, labels))
    norms = [
        _gaussian_squared_norm(e, component)
        for e, components in zip(exponents, labels, strict=True)
        for component in components
    ]
    scales = np.split(
        np.sqrt(mol.intor("int1e_ovlp_cart").diagonal() / norms), np.cumsum(shape)[:-1]
    )
    normalization = np.ones(shape)
    for slot, scale in enumerate(scales):
        dimensions = [1] * len(shape)
        dimensions[slot] = len(scale)
        normalization *= scale.reshape(dimensions)
    if family == "eri":
        blocks = []
        for order in ((0, 1, 2, 3), (1, 0, 2, 3), (2, 3, 0, 1), (3, 2, 0, 1)):
            raw = -mol.intor_by_shell("int2e_ip1_cart", order, comp=3)
            blocks.append(raw.transpose(0, *(1 + order.index(i) for i in range(4))))
    else:
        operator, factor = {
            "overlap": ("ovlp", 1),
            "kinetic": ("kin", 1),
            "nuclear_attraction": ("rinv", -charge),
        }[family]
        origin = centers[2] if family == "nuclear_attraction" else (0, 0, 0)
        with mol.with_rinv_origin(origin):
            first = -factor * mol.intor_by_shell(
                f"int1e_ip{operator}_cart", (0, 1), comp=3
            )
            second = -factor * mol.intor_by_shell(
                f"int1e_ip{operator}_cart", (1, 0), comp=3
            ).transpose(0, 2, 1)
        blocks = [first, second] + (
            [-first - second] if family == "nuclear_attraction" else []
        )
    return np.asarray(blocks) / normalization
