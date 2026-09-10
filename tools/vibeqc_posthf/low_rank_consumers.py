"""Bounded consumers of one captured Cholesky approximation and exact reference.

J/K and MO blocks are reconstructed from the same factor prefix. Approximate
integrals retain a distinct Hamiltonian identity even when the orbitals were
obtained from an exact conventional reference. No CC/MP2 result is implicitly
authorized to substitute these blocks for its declared Hamiltonian.
"""

import json
import threading
import time
from dataclasses import dataclass
from math import prod

import numpy as np
from vibeqc_compiler.common.arrays import immutable
from vibeqc_compiler.common.resources import (
    ResourceCandidate,
    ResourceEstimate,
    ResourceIdentity,
    ResourceRequest,
    byte_product,
    plan_resources,
)

from .low_rank import IncrementalCholesky
from .providers import BlockResult


@dataclass(frozen=True)
class LowRankJk:
    """Raw J and K for an explicitly captured approximate Coulomb tensor.

    A single density gives J[D] and K[D]. Two spin densities give J[Da+Db]
    and separate Ka/Kb. There are no occupation/exchange prefactors inside
    these matrices; the RHF/UHF method must apply its own factors exactly once.
    """

    coulomb: np.ndarray
    exchange: np.ndarray
    hamiltonian_id: str
    diagnostics: dict


class LowRankProvider:
    """Borrow factors and optionally an exact reference; reject stale generations.

    No transformed-block cache is retained. Requested MO outputs are bounded
    individually and factor rows are streamed through one physical AO matrix.
    Resource planning composes all factor/source storage with current consumer
    workspace and publication. Caller-retained earlier outputs are excluded
    after return; applications that retain several must reserve them separately.
    Recreate this view after refinement and rebuild dependent solver residuals.
    """

    def __init__(self, factor, snapshot=None, *, budget=None):
        if not isinstance(factor, IncrementalCholesky):
            raise TypeError("an incremental Coulomb factorization is required")
        self._factor = factor
        self._snapshot = snapshot
        self._identity = factor.identity
        self._budget = factor.resource_plan.budget if budget is None else budget
        self._lock = threading.RLock()
        self._closed = False
        if snapshot is not None:
            source = getattr(factor._columns, "source", None)
            # Equal AO topology does not establish an equal electronic state.
            # ReferenceSnapshot has closed-shell 2/0 occupations; the raw-source
            # identity also includes charge and multiplicity, so both must agree.
            if (
                source is None
                or snapshot.nmo != factor.space.nbf
                or snapshot.electron_count != source.electron_count
                or source.multiplicity != 1
                or snapshot.geometry_hash != source.geometry_hash
                or snapshot.basis_hash != source.basis_hash
                or snapshot.representation != source.representation
                or snapshot.hamiltonian_id != "conventional-unscreened"
                or snapshot.screening_tolerance != 0
            ):
                raise ValueError("exact reference and target raw Coulomb source differ")

    @property
    def factor(self):
        return self._factor

    @property
    def snapshot(self):
        return self._snapshot

    def _check(self):
        if self._closed:
            raise RuntimeError("low-rank consumer is closed")
        if self.factor.identity != self._identity:
            raise ValueError(
                "factor generation changed; rebuild consumer and solver history"
            )

    @property
    def hamiltonian_id(self):
        self._check()
        return self.factor.hamiltonian_id

    def _plan(self, observable, elements):
        """Compose the single shared factor owner with one consumer operation."""
        request = ResourceRequest(
            "low_rank_consumer",
            ResourceIdentity(
                "posthf",
                "cholesky_contraction",
                "cuda"
                if observable == "J_K" and hasattr(self.factor, "_native_jk")
                else "cpu",
                "fp64",
                json.dumps({"nbf": self.factor.space.nbf, "rank": self.factor.rank}),
                (observable,),
                "streamed_physical_factor_v1",
            ),
            (
                ResourceCandidate(
                    "one_factor_at_a_time",
                    "streamed",
                    (
                        ResourceEstimate(
                            "consumer_numeric_peak",
                            byte_product(8, elements),
                            "pageable",
                            0,
                            1,
                        ),
                    ),
                ),
            ),
            (
                "BLAS runtime overhead",
                "caller-retained earlier outputs",
                "Python object metadata",
            ),
        )
        plan = plan_resources(
            (*self.factor.resource_plan.requests, request), self._budget
        )
        plan.require_feasible()
        return plan

    def _physical_factors(self):
        for rank in range(self.factor.rank):
            yield self.factor.space.unpack(
                self.factor.factor_tile(rank, 1, identity=self._identity)[0]
            )

    def jk(self, density):
        """Contract finite real symmetric densities without constructing AO**4."""
        with self._lock, self.factor._lock:
            self._check()
            density = np.asarray(density)
            n = self.factor.space.nbf
            if density.dtype != np.float64 or density.shape not in ((n, n), (2, n, n)):
                raise ValueError("one or two FP64 AO density matrices required")
            if not np.isfinite(density).all() or not np.array_equal(
                density, density.swapaxes(-1, -2)
            ):
                raise ValueError("finite symmetric AO density matrices required")
            spins = 1 if density.ndim == 2 else 2
            # Inputs, total density, physical factor, ordered matmul temporaries,
            # accumulators and irreversible detached publication can coexist.
            plan = self._plan("J_K", (12 + 8 * spins) * n * n)
            if hasattr(self.factor, "_native_jk"):
                return self.factor._native_jk(density, plan)
            started = time.perf_counter()
            blocks = density.reshape(spins, n, n).copy()
            total = blocks.sum(axis=0)
            coulomb = np.zeros((n, n))
            exchange = np.zeros_like(blocks)
            for physical in self._physical_factors():
                coulomb += physical * np.sum(physical * total)
                for spin in range(spins):
                    exchange[spin] += physical @ blocks[spin] @ physical
            # Exact symmetry is part of this public density contract. Remove
            # only multiplication-order rounding, not asymmetric input data.
            coulomb = (coulomb + coulomb.T) * 0.5
            exchange = (exchange + exchange.swapaxes(-1, -2)) * 0.5
            result = LowRankJk(
                immutable(coulomb),
                immutable(exchange[0] if spins == 1 else exchange),
                self.hamiltonian_id,
                {
                    "rank": self.factor.rank,
                    "spins": spins,
                    "wall_seconds": time.perf_counter() - started,
                    "resource_plan": plan.to_dict(),
                    "derivatives": "unsupported",
                },
            )
            return result

    def get(self, block):
        """Return chemists' MOBlock with distinct reference/approximation identities."""
        with self._lock, self.factor._lock:
            self._check()
            if self.snapshot is None:
                raise ValueError(
                    "MO blocks require an explicit validated exact reference"
                )
            block.validate(self.snapshot)
            n = self.factor.space.nbf
            sizes = block.shape
            output = prod(sizes)
            coefficients = n * sum(sizes)
            # Both transformed factor pairs, each cyclic matrix stage, current
            # outer product, accumulated output and detached copies are charged.
            pair_outputs = sizes[0] * sizes[1] + sizes[2] * sizes[3]
            elements = (
                self.snapshot.numeric_bytes // 8
                + 4 * coefficients
                + 4 * n * n
                + 4 * pair_outputs
                + 3 * output
            )
            plan = self._plan("MO_block", elements)
            started = time.perf_counter()
            value = np.zeros(sizes)
            if output:
                panels = tuple(
                    np.ascontiguousarray(self.snapshot.coefficients[:, slot])
                    for slot in block.slots
                )
                for physical in self._physical_factors():
                    left = panels[0].T @ physical @ panels[1]
                    right = panels[2].T @ physical @ panels[3]
                    value += left[:, :, None, None] * right[None, None, :, :]
            return BlockResult(
                block,
                immutable(value),
                self.snapshot.identity,
                self.hamiltonian_id,
                {
                    "rank": self.factor.rank,
                    "reference_hamiltonian_id": self.snapshot.hamiltonian_id,
                    "correlation_hamiltonian_id": self.hamiltonian_id,
                    "wall_seconds": time.perf_counter() - started,
                    "resource_plan": plan.to_dict(),
                    "method_compatibility": "requires an explicit approximate-correlation adapter",
                },
            )

    def close(self):
        with self._lock:
            self._closed = True

    def __enter__(self):
        self._check()
        return self

    def __exit__(self, *_):
        self.close()
