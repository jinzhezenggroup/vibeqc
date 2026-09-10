"""Bounded Coulomb pair columns from the existing values-only raw source."""

import time
from math import sqrt

import numpy as np
from vibeqc.profiles import canonical_hash

from .pair_space import PairSpace
from .sources import CPU_SOURCE_SCRATCH, NativeSource


class CoulombColumns:
    """Borrow immutable raw source state; never retain a four-index AO tensor.

    Source closure invalidates this view. Diagonal entries and column pieces
    are computed by the independent native contracted evaluator. This is an
    unscreened full-Coulomb PSD operator; range/DF/complex-pair factorizations
    require separately specified providers. Timings expose CPU source work
    even when a later factorization or consumer runs on a GPU.
    """

    def __init__(self, source):
        if not isinstance(source, NativeSource):
            raise TypeError("the existing native raw integral source is required")
        if "four_center_eri" not in source.supported_operators:
            raise ValueError("source does not supply the target full Coulomb operator")
        source._check_open()
        self.source = source
        self.space = PairSpace(source.nbf)
        self._source_identity = source.identity
        self.identity = canonical_hash(
            {
                "source": source.identity,
                "operator": "full-unscreened-coulomb",
                "pairs": self.space.convention,
                "precision": "float64",
            }
        )
        self.numeric_bytes = source.numeric_bytes + CPU_SOURCE_SCRATCH
        self.statistics = {"tiles": 0, "elements": 0, "seconds": 0.0}

    def check(self):
        """Reject stale source lifetime or replaced scientific metadata."""
        self.source._check_open()
        if self.source.identity != self._source_identity:
            raise ValueError("raw source identity changed; rebuild factorization")

    def _range(self, begin, count):
        self.check()
        if (
            type(begin) is not int
            or type(count) is not int
            or min(begin, count) < 0
            or begin > self.space.size
            or count > self.space.size - begin
        ):
            raise ValueError("invalid pair column range")

    def _read(self, begin, count):
        started = time.perf_counter()
        value = self.source._read("four_center_eri", begin, count)
        self.statistics["seconds"] += time.perf_counter() - started
        self.statistics["tiles"] += 1
        self.statistics["elements"] += value.size
        return value

    def diagonal(self, begin, count):
        """Return only the requested diagonal, including a final partial tile."""
        self._range(begin, count)
        out = np.empty(count)
        for local, pair in enumerate(range(begin, begin + count)):
            mu, nu = self.space.pair(pair)
            out[local] = (1 if mu == nu else 2) * self._read(
                (mu, nu, mu, nu), (1, 1, 1, 1)
            ).item()
        return out

    def column(self, pivot, begin, count):
        """Read contiguous triangular-row pieces with explicit pair weights."""
        self._range(begin, count)
        rho, sigma = self.space.pair(pivot)
        pivot_scale = self.space.scale(pivot)
        out = np.empty(count)
        offset = 0
        while offset < count:
            mu, nu = self.space.pair(begin + offset)
            width = min(mu - nu + 1, count - offset)
            block = self._read((mu, nu, rho, sigma), (1, width, 1, 1)).ravel()
            out[offset : offset + width] = block * (sqrt(2.0) * pivot_scale)
            if nu + width == mu + 1:
                out[offset + width - 1] = block[-1] * pivot_scale
            offset += width
        return out
