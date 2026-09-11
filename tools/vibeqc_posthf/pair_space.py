"""Analytical symmetric AO-pair coordinates without a dense orbit table."""

from dataclasses import dataclass
from math import isqrt, sqrt

import numpy as np
from vibeqc_compiler.integral.shell_signature import checked_index


@dataclass(frozen=True)
class PairSpace:
    """Orthonormal real symmetric pairs, lower triangle in row order.

    A packed density is ``sqrt(m) * D[mu,nu]`` with multiplicity m=1 on
    the diagonal and m=2 elsewhere. The Coulomb matrix in these coordinates
    is ``sqrt(m_i*m_j) * (mu nu|rho sigma)``. Consequently ordinary packed
    dot products preserve the full matrix Frobenius product, including both
    off-diagonal entries. Pair indexing requires no O(NAO**4) lookup table.
    """

    nbf: int

    def __post_init__(self):
        if type(self.nbf) is not int or self.nbf < 1:
            raise ValueError("positive integer AO dimension required")
        checked_index(self.size, "symmetric AO pair count")

    @property
    def size(self):
        return self.nbf * (self.nbf + 1) // 2

    @property
    def convention(self):
        return "real-symmetric-lower-row-svec-v1"

    def index(self, mu, nu):
        """Map either ordering of a real symmetric AO pair to its coordinate."""
        if any(type(i) is not int or not 0 <= i < self.nbf for i in (mu, nu)):
            raise ValueError("AO pair index outside the basis")
        mu, nu = max(mu, nu), min(mu, nu)
        return mu * (mu + 1) // 2 + nu

    def pair(self, index):
        """Invert triangular indexing exactly, including large integer indices."""
        if type(index) is not int or not 0 <= index < self.size:
            raise ValueError("packed pair index outside the basis")
        mu = (isqrt(8 * index + 1) - 1) // 2
        return mu, index - mu * (mu + 1) // 2

    def scale(self, index):
        mu, nu = self.pair(index)
        return 1.0 if mu == nu else sqrt(2.0)

    def pack(self, matrix):
        """Pack a finite symmetric FP64 matrix; reject a lossy projection."""
        matrix = np.asarray(matrix)
        if (
            matrix.dtype != np.float64
            or matrix.shape != (self.nbf, self.nbf)
            or not np.isfinite(matrix).all()
            or not np.array_equal(matrix, matrix.T)
        ):
            raise ValueError("finite exactly symmetric FP64 AO matrix required")
        out = np.empty(self.size)
        for mu in range(self.nbf):
            begin = mu * (mu + 1) // 2
            out[begin : begin + mu] = sqrt(2.0) * matrix[mu, :mu]
            out[begin + mu] = matrix[mu, mu]
        return out

    def unpack(self, packed):
        """Recover both AO matrix triangles with the inverse svec factors."""
        packed = np.asarray(packed)
        if (
            packed.dtype != np.float64
            or packed.shape != (self.size,)
            or not np.isfinite(packed).all()
        ):
            raise ValueError("finite FP64 packed pair vector required")
        out = np.empty((self.nbf, self.nbf))
        for mu in range(self.nbf):
            begin = mu * (mu + 1) // 2
            out[mu, :mu] = packed[begin : begin + mu] / sqrt(2.0)
            out[:mu, mu] = out[mu, :mu]
            out[mu, mu] = packed[begin + mu]
        return out
