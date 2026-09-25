"""Independent CPU finite differences for stationary explicit XC geometry.

This validation helper rebuilds AO collocation on displaced inputs and only
evaluates the scalar discrete XC energy. It never calls the generated geometry
pullback under test and is not a production molecular-gradient implementation.
"""

import typing
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from vibeqc._dft_gradient import StableGridMotion
from vibeqc_compiler.dft import NativeAO
from vibeqc_compiler.xc.contractions import (
    ContractionProgram,
    ExternalPointContraction,
)
from vibeqc_compiler.xc.spec import FunctionalSpec


def h2_overlap(basis: typing.Any) -> typing.Any:
    """Analytic s-Gaussian metric for the two-AO H2 diagnostic fixture only.

    Integrate the actual normalized packed primitives; no SCF state or
    stationarity is manufactured by this independent overlap check.
    """
    if (
        basis.nao != 2
        or basis.natom != 2
        or any(s.angular_momentum for s in basis.shells)
    ):
        raise ValueError("H2 overlap oracle requires two s AOs")
    primitives = basis.packed[6 : 6 + 2 * basis.nprimitive].reshape(-1, 2)
    records = basis.packed[6 + 2 * basis.nprimitive :].reshape(2, 16)
    centers = basis.packed[:6].reshape(2, 3)
    result = np.empty((2, 2))
    for i, left in enumerate(records):
        for j, right in enumerate(records):
            distance = np.sum((centers[int(left[0])] - centers[int(right[0])]) ** 2)
            result[i, j] = (
                sum(
                    ca
                    * cb
                    * (np.pi / (a + b)) ** 1.5
                    * np.exp(-a * b / (a + b) * distance)
                    for a, ca in primitives[int(left[1]) : int(left[1] + left[2])]
                    for b, cb in primitives[int(right[1]) : int(right[1] + right[2])]
                )
                * left[7]
                * right[7]
            )
    return result


@dataclass(frozen=True)
class DirectionalFiniteDifference:
    """Multistep central differences and their most stable adjacent region."""

    steps: tuple[float, ...]
    estimates: tuple[float, ...]
    stable_pair: tuple[int, int]

    @property
    def stable_estimate(self) -> typing.Any:
        i, j = self.stable_pair
        return 0.5 * (self.estimates[i] + self.estimates[j])

    @property
    def spread(self) -> typing.Any:
        i, j = self.stable_pair
        return abs(self.estimates[i] - self.estimates[j])


def finite_difference_xc_directional(
    functional: typing.Any,
    basis_arguments: typing.Any,
    points: typing.Any,
    weights: typing.Any,
    density: typing.Any,
    motion: typing.Any,
    *,
    steps: typing.Any = (1e-3, 3e-4, 1e-4),
    point_energy: typing.Any = None,
) -> typing.Any:
    """Re-evaluate scalar XC energy on independently displaced inputs.

    An optional point-energy callback may supply an independently audited
    per-point scalar model; this still never calls the geometry pullback.
    """
    if not isinstance(functional, FunctionalSpec):
        raise TypeError("oracle requires a typed XC functional")
    if point_energy is not None and not callable(point_energy):
        raise TypeError("point-energy oracle must be callable")
    if not isinstance(motion, StableGridMotion):
        raise TypeError("oracle requires stable-grid motion")
    if motion.topology_changed:
        raise ValueError("oracle cannot cross a topology change")
    if not isinstance(basis_arguments, dict) or "atoms" not in basis_arguments:
        raise ValueError("oracle requires reconstructable basis arguments")

    points = _direction_domain(points, motion.points, "point")
    weights = _direction_domain(weights, motion.weights, "weight")
    atoms = tuple(basis_arguments["atoms"])
    centers = np.asarray(motion.centers)
    if (
        np.iscomplexobj(centers)
        or centers.shape != (len(atoms), 3)
        or not np.isfinite(centers).all()
    ):
        raise ValueError("oracle center direction has incompatible domain")
    density = np.asarray(density)
    if np.iscomplexobj(density) or not np.isfinite(density).all():
        raise ValueError("oracle density must be finite and real")

    steps = tuple(float(step) for step in steps)
    if (
        len(steps) < 3
        or any(not np.isfinite(step) or step <= 0 for step in steps)
        or any(a <= b for a, b in pairwise(steps))
    ):
        raise ValueError("oracle requires at least three decreasing positive steps")

    energy = (
        ContractionProgram(functional, "energy")
        if point_energy is None
        else ExternalPointContraction(functional, "energy")
    )
    estimates = []
    for step in steps:
        values = []
        for sign in (1.0, -1.0):
            moved_atoms = [
                (atom, np.asarray(position) + sign * step * delta)
                for (atom, position), delta in zip(atoms, centers, strict=True)
            ]
            with NativeAO(**{**basis_arguments, "atoms": moved_atoms}) as basis:
                jets = basis.evaluate(
                    points + sign * step * np.asarray(motion.points),
                    energy.contract.ao_order,
                )
            displaced_weights = weights + sign * step * np.asarray(motion.weights)
            if point_energy is None:
                scalar = energy.evaluate(jets, density, displaced_weights)["energy"]
            else:
                features = energy.features(jets, density)
                per_point = np.asarray(point_energy(features))
                if (
                    np.iscomplexobj(per_point)
                    or per_point.shape != displaced_weights.shape
                    or not np.isfinite(per_point).all()
                ):
                    raise ValueError("point-energy oracle returned an invalid domain")
                scalar = float(np.dot(displaced_weights, per_point))
            values.append(scalar)
        estimates.append((values[0] - values[1]) / (2 * step))

    differences = np.abs(np.diff(estimates))
    first = int(np.argmin(differences))
    return DirectionalFiniteDifference(
        steps=steps,
        estimates=tuple(float(value) for value in estimates),
        stable_pair=(first, first + 1),
    )


def _direction_domain(
    value: typing.Any, direction: typing.Any, name: typing.Any
) -> typing.Any:
    array = np.asarray(value)
    delta = np.asarray(direction)
    if (
        np.iscomplexobj(array)
        or np.iscomplexobj(delta)
        or array.shape != delta.shape
        or not np.isfinite(array).all()
        or not np.isfinite(delta).all()
    ):
        raise ValueError(f"oracle {name} direction has incompatible domain")
    return array
