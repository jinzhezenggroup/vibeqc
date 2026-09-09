"""Explicit signed symmetry-orbit layouts for small independent amplitudes.

The map stores x[dense_coordinate] = sign * packed[orbit]. Antisymmetric
fixed points are structural zeros. Orbit multiplicities are the metric for
dense inner products; the exact pack/unpack transposes used by AD are provided
here and must not be replaced by ordinary pack/unpack. This bounded reference
enumerator is not a large-system packing planner.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from .types import TensorSpec, checked_size, spec_from_payload, spec_to_payload


@dataclass(frozen=True, init=False)
class PackedLayout:
    """Validated independent-coordinate mapping and dense inner-product weights."""

    spec: TensorSpec
    dense_to_packed: tuple[int, ...]
    signs: tuple[int, ...]
    representatives: tuple[int, ...]
    weights: tuple[int, ...]

    def __init__(self, spec: TensorSpec, *, max_elements: int = 1_000_000):
        """Build signed orbits using declared generators, including zero orbits."""
        checked_size(max_elements, "packing element budget")
        if spec.size > max_elements:
            raise ValueError("packing reference element budget exceeded")
        coordinates = list(product(*(range(n) for n in spec.shape)))
        locations = {coordinate: i for i, coordinate in enumerate(coordinates)}
        mapping, signs = [-1] * spec.size, [0] * spec.size
        seen, representatives, weights = set(), [], []
        for first in coordinates:
            if first in seen:
                continue
            orbit, pending, forced_zero = {first: 1}, [first], False
            while pending:
                coordinate = pending.pop()
                for symmetry in spec.symmetries:
                    target = tuple(coordinate[i] for i in symmetry.permutation)
                    sign = orbit[coordinate] * symmetry.sign
                    if target in orbit:
                        forced_zero |= orbit[target] != sign
                    else:
                        orbit[target] = sign
                        pending.append(target)
            seen.update(orbit)
            if not forced_zero:
                packed_index = len(representatives)
                representatives.append(locations[first])
                weights.append(len(orbit))
                for coordinate, sign in orbit.items():
                    flat = locations[coordinate]
                    mapping[flat], signs[flat] = packed_index, sign
        object.__setattr__(self, "spec", spec)
        object.__setattr__(self, "dense_to_packed", tuple(mapping))
        object.__setattr__(self, "signs", tuple(signs))
        object.__setattr__(self, "representatives", tuple(representatives))
        object.__setattr__(self, "weights", tuple(weights))

    @classmethod
    def from_spec(
        cls, spec: TensorSpec, *, max_elements: int = 1_000_000
    ) -> PackedLayout:
        """Construct a checked map; physical strides are not part of packing."""
        return cls(spec, max_elements=max_elements)

    @property
    def size(self) -> int:
        return len(self.representatives)

    def _packed(self, values) -> np.ndarray:
        values = np.asarray(values)
        if values.shape != (self.size,) or values.dtype != np.dtype(self.spec.dtype):
            raise ValueError("packed array shape/real dtype does not match its layout")
        if not np.isfinite(values).all():
            raise ValueError("packed amplitudes must be finite")
        return values

    def _dense(self, values) -> np.ndarray:
        values = np.asarray(values)
        if values.shape != self.spec.shape or values.dtype != np.dtype(self.spec.dtype):
            raise ValueError("dense array shape/real dtype does not match its layout")
        if not np.isfinite(values).all():
            raise ValueError("dense values must be finite")
        return values

    def unpack(self, values) -> np.ndarray:
        """Expand independent amplitudes without imposing extra antisymmetries."""
        values = self._packed(values)
        result = np.zeros(self.spec.size, dtype=self.spec.dtype)
        for i, (packed, sign) in enumerate(zip(self.dense_to_packed, self.signs)):
            if sign:
                result[i] = sign * values[packed]
        return result.reshape(self.spec.shape)

    def pack(self, dense, *, atol: float = 1e-11, rtol: float = 1e-10) -> np.ndarray:
        """Pack an already symmetric tensor, rejecting lossy projection."""
        if not all(np.isfinite(x) and x >= 0 for x in (atol, rtol)):
            raise ValueError("packing tolerances must be finite and nonnegative")
        dense = np.asarray(dense)
        if dense.shape != self.spec.shape or dense.dtype != np.dtype(self.spec.dtype):
            raise ValueError("dense array shape/real dtype does not match its layout")
        packed = dense.reshape(-1)[list(self.representatives)].copy()
        if not np.isfinite(dense).all() or not np.allclose(
            dense, self.unpack(packed), atol=atol, rtol=rtol
        ):
            raise ValueError("dense tensor violates the declared packed symmetry")
        return packed

    def unpack_transpose(self, dense_bar) -> np.ndarray:
        """Adjoint of ``unpack`` under the dense and weighted packed metrics.

        The adjoint is the signed orbit sum divided by the orbit weight:

        ``<dense_bar, unpack(x)>_dense = <unpack_transpose(dense_bar), x>_packed``.

        This is deliberately not ordinary :meth:`pack`; orbit multiplicities
        are part of the packed inner product and must be removed here.
        """
        dense_bar = self._dense(dense_bar)
        flat = dense_bar.reshape(-1)
        result = np.zeros(self.size, dtype=self.spec.dtype)
        for flat_index, (packed_index, sign) in enumerate(
            zip(self.dense_to_packed, self.signs)
        ):
            if sign:
                result[packed_index] += sign * flat[flat_index]
        result /= np.asarray(self.weights, dtype=self.spec.dtype)
        return result

    def pack_transpose(self, packed_bar) -> np.ndarray:
        """Adjoint of ``pack`` under the weighted packed and dense metrics.

        ``<pack(y), x>_packed = <y, pack_transpose(x)>_dense``.  The map
        scatters ``weight * x`` to each orbit representative; it is not
        ordinary :meth:`unpack`.
        """
        packed_bar = self._packed(packed_bar)
        result = np.zeros(self.spec.size, dtype=self.spec.dtype)
        flat = result.reshape(-1)
        weights = np.asarray(self.weights, dtype=self.spec.dtype)
        for index, representative in enumerate(self.representatives):
            flat[representative] = weights[index] * packed_bar[index]
        return result.reshape(self.spec.shape)

    def inner_product(self, left, right) -> float:
        """Use orbit weights to equal sum(unpack(left) * unpack(right))."""
        left, right = self._packed(left), self._packed(right)
        return float(
            np.sum(np.asarray(self.weights, dtype=self.spec.dtype) * left * right)
        )

    def to_payload(self) -> dict:
        """Save the complete mapping/metric for replay and later adjoint consumers."""
        return {
            "schema": "vibeqc.tensor.packing",
            "schema_version": 1,
            "spec": spec_to_payload(self.spec),
            "dense_to_packed": list(self.dense_to_packed),
            "signs": list(self.signs),
            "representatives": list(self.representatives),
            "weights": list(self.weights),
        }

    @classmethod
    def from_payload(
        cls, payload: dict, *, max_elements: int = 1_000_000
    ) -> PackedLayout:
        """Recompute the map to reject corrupted signs, representatives, or weights."""
        if (
            not isinstance(payload, dict)
            or payload.get("schema") != "vibeqc.tensor.packing"
            or type(payload.get("schema_version")) is not int
            or payload["schema_version"] != 1
        ):
            raise ValueError("unsupported tensor packing schema/version")
        result = cls.from_spec(
            spec_from_payload(payload["spec"]), max_elements=max_elements
        )
        for field in ("dense_to_packed", "signs", "representatives", "weights"):
            if any(type(value) is not int for value in payload.get(field, ())):
                raise ValueError("packed mappings and weights must contain integers")
        if result.to_payload() != payload:
            raise ValueError(
                "packed symmetry map or inner-product weights are inconsistent"
            )
        return result
