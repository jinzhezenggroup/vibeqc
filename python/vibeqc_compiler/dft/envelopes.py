"""Conservative region bounds for the existing normalized public AO jets.

These envelopes decide a fixed collocation mask before AO evaluation. They
bound individual AO derivatives, not density, XC energy or molecular forces.
Wide regions may produce loose bounds. Arithmetic overflow always retains an
AO rather than claiming it is negligible.
"""

from __future__ import annotations

import math

import numpy as np

from vibeqc_compiler.common.arrays import immutable

from .ao import NativeAO, jet_indices


def derivative_domain(derivatives):
    """Canonical nonempty subset of ordinary spatial derivatives through three."""
    values = tuple(tuple(d) for d in derivatives)
    allowed = jet_indices(3)
    if not values or any(
        len(d) != 3 or any(type(k) is not int or k < 0 for k in d) or d not in allowed
        for d in values
    ):
        raise ValueError("AO derivative domain must contain supported multi-indices")
    if len(set(values)) != len(values):
        raise ValueError("duplicate AO derivative multi-index")
    return tuple(d for d in allowed if d in values)


def _up(value):
    return math.nextafter(value, math.inf)


def _add(a, b):
    if a == 0:
        return b
    if b == 0:
        return a
    return _up(a + b)


def _multiply(a, b):
    if a == 0 or b == 0:
        return 0.0
    return _up(float(a) * float(b))


def _axis_bound(power, derivative, alpha, lower, upper):
    """Bound |d^d(x^l exp(-a*x*x))| without polynomial cancellation.

    Differentiation maps P to P'-2axP. Propagating absolute coefficients
    bounds every term, including derivatives at AO nodes. Bounding the
    polynomial and Gaussian separately is conservative on any interval.
    Positive arithmetic rounds upward; the Gaussian exponent rounds toward
    zero. A small libm guard and a positive underflow floor avoid screening on
    the last representable exponential. Nonfinite bounds propagate as infinity.
    """
    alpha, lower, upper = float(alpha), float(lower), float(upper)
    coefficients = [0.0] * (power + derivative + 1)
    coefficients[power] = 1.0
    for degree in range(derivative):
        following = [0.0] * len(coefficients)
        for k in range(power + degree + 1):
            if k:
                following[k - 1] = _add(following[k - 1], _multiply(k, coefficients[k]))
            following[k + 1] = _add(
                following[k + 1], _multiply(_multiply(2.0, alpha), coefficients[k])
            )
        coefficients = following
    maximum = max(abs(lower), abs(upper))
    minimum = 0.0 if lower <= 0 <= upper else min(abs(lower), abs(upper))
    polynomial, monomial = 0.0, 1.0
    for coefficient in coefficients:
        polynomial = _add(polynomial, _multiply(coefficient, monomial))
        monomial = _multiply(monomial, maximum)
    if not math.isfinite(polynomial):
        return math.inf
    # Lower bounds on positive products give an upper bound on exp(-a*r^2).
    square = max(0.0, math.nextafter(minimum * minimum, -math.inf))
    exponent = max(0.0, math.nextafter(alpha * square, -math.inf))
    gaussian = _up(math.exp(-exponent))
    gaussian = _multiply(gaussian, 1.0 + 32.0 * np.finfo(float).eps)
    return _multiply(polynomial, gaussian)


def ao_region_envelopes(basis, bounds, derivatives=((0, 0, 0),)):
    """Return [requested jet, public AO] absolute bounds on an axis-aligned box.

    ``bounds`` is [lower/upper, xyz] in Bohr. Basis records carry the native
    contraction normalization and sparse Cartesian-to-spherical coefficients;
    their absolute values are included once. No AO is sampled to discover its
    bound. The working storage is O(NAO * requested jets), independent of the
    number of grid points in the region. Infinite entries mean unscreenable.
    """
    if not isinstance(basis, NativeAO):
        raise TypeError("AO envelopes require an existing NativeAO basis")
    bounds = immutable(bounds, shape=(2, 3))
    if np.any(bounds[0] > bounds[1]):
        raise ValueError("region lower bound exceeds upper bound")
    derivatives = derivative_domain(derivatives)
    centers = basis.packed[: 3 * basis.natom].reshape(-1, 3)
    offset = 3 * basis.natom
    primitives = basis.packed[offset : offset + 2 * basis.nprimitive].reshape(-1, 2)
    records = basis.packed[offset + 2 * basis.nprimitive :].reshape(-1, 16)
    result = np.zeros((len(derivatives), basis.nao))
    for ao, record in enumerate(records):
        atom, begin, count, terms = map(int, record[:4])
        # Overflowed coordinate differences are deliberately unscreenable.
        with np.errstate(over="ignore"):
            lower = np.nextafter(bounds[0] - centers[atom], -np.inf)
            upper = np.nextafter(bounds[1] - centers[atom], np.inf)
        for jet, derivative in enumerate(derivatives):
            total = 0.0
            for alpha, coefficient in primitives[begin : begin + count]:
                for term in range(terms):
                    powers = tuple(map(int, record[4 + 4 * term : 7 + 4 * term]))
                    bound = _multiply(abs(coefficient), abs(record[7 + 4 * term]))
                    for axis in range(3):
                        bound = _multiply(
                            bound,
                            _axis_bound(
                                powers[axis],
                                derivative[axis],
                                alpha,
                                lower[axis],
                                upper[axis],
                            ),
                        )
                    total = _add(total, bound)
            result[jet, ao] = total
    # Unlike scientific output arrays, diagnostic envelopes may contain +inf.
    # Freeze their storage without routing them through the finite-only helper.
    return np.frombuffer(result.tobytes(), dtype=np.float64).reshape(result.shape)
