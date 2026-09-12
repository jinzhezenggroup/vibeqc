"""Bounded internal MP2 energy consumer of CG10 and CG08/CG09.

The molecular loop is a reference/development driver, not a public native
prepared method. CUDA executes each complete tile equation natively; integral
tiles and the final scalar fold currently pass through the host. These limits
must remain visible until the coordinated native connection layer is delivered.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from itertools import product

import numpy as np

from tools.vibeqc_posthf.conventions import MOBlock
from tools.vibeqc_posthf.providers import ConventionalProvider
from tools.vibeqc_posthf.reference import ReferenceSnapshot
from tools.vibeqc_tensor import execute
from tools.vibeqc_tensor.types import checked_size

from .equations import cpu_capacity, energy_program


@dataclass(frozen=True)
class EnergyResult:
    energy: float
    correlation_energy: float
    opposite_spin: float
    same_spin: float
    minimum_absolute_denominator: float
    reference_id: str
    hamiltonian_id: str
    tile_count: int
    numeric_capacity_bytes: int
    equation_hashes: tuple[str, ...]
    energy_backend: str
    integral_backend: str
    mo_host_staging: bool
    scalar_fold_backend: str = "cpu-compensated-sum"


def denominator_check(snapshot, threshold):
    """Check extrema of the separable denominator before any integral reads."""
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("denominator threshold must be finite and positive")
    e, no = snapshot.orbital_energies, snapshot.nocc
    hi = int(np.argmax(e[:no]))
    lo = no + int(np.argmin(e[no:]))
    # Match the equation's arithmetic order, including overflow of an occupied
    # pair even when cancellation against virtual energies would be finite.
    with np.errstate(over="raise", invalid="raise"):
        try:
            largest = e[hi] + e[hi] - e[lo] - e[lo]
            smallest = 2 * np.min(e[:no]) - np.max(e[no:]) - np.max(e[no:])
        except FloatingPointError as exc:
            raise ValueError("nonfinite MP2 denominator extrema") from exc
    if not np.isfinite(largest) or not np.isfinite(smallest):
        raise ValueError("nonfinite MP2 denominator extrema")
    if largest >= 0:
        raise ValueError(
            "MP2 requires occupied energies strictly below virtual energies"
        )
    if -largest <= threshold:
        raise ValueError(
            f"near-zero MP2 denominator at global ijab={(hi, hi, lo, lo)}: "
            f"{largest}; no regularization applied"
        )
    return float(-largest)


class PreparedMP2Energy:
    """Own a private conventional provider; borrow an immutable reference/source.

    Each ordered occupied/virtual rectangle requests direct and exchange
    integrals separately and clears only this private provider. This bounds
    storage but repeats AO-to-MO source passes, so it is a correctness baseline,
    not a performance claim. The caller owns and closes the source separately.
    No dense AO tensor or complete molecular T2 is requested by this consumer.
    """

    def __init__(
        self,
        snapshot,
        source,
        *,
        occupied_tile=1,
        virtual_tile=2,
        axis_tile=2,
        budget_bytes=256 << 20,
        denominator_threshold=1e-10,
        energy_backend="cpu",
        integral_backend="cpu",
        transform_artifact=None,
        compiler=None,
        cache=None,
        device_id=0,
    ):
        if not isinstance(snapshot, ReferenceSnapshot):
            raise TypeError("MP2 requires a validated CG10 ReferenceSnapshot")
        for name, value in (
            ("occupied_tile", occupied_tile),
            ("virtual_tile", virtual_tile),
            ("budget_bytes", budget_bytes),
        ):
            checked_size(value, name)
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if energy_backend not in ("cpu", "cuda"):
            raise ValueError("unknown MP2 energy backend; no fallback is allowed")
        if energy_backend == "cuda" and (compiler is None or cache is None):
            raise ValueError("CUDA MP2 requires an explicit compiler and cache")
        if type(device_id) is not int or device_id < 0:
            raise ValueError("invalid visible CUDA ordinal")
        self._minimum = denominator_check(snapshot, denominator_threshold)
        self._snapshot, self._source = snapshot, source
        self._source_identity = source.identity
        self._lock = threading.RLock()
        self._provider = ConventionalProvider(
            snapshot,
            source,
            budget_bytes=budget_bytes,
            axis_tile=axis_tile,
            backend=integral_backend,
            cuda_artifact=transform_artifact,
            device_id=device_id,
        )
        self._ot, self._vt = occupied_tile, virtual_tile
        self._energy_backend, self._integral_backend = energy_backend, integral_backend
        self._compiler, self._cache, self._device = compiler, cache, device_id
        self._budget = budget_bytes
        self._programs, self._plans = {}, {}
        self._capacity = 0
        self._state, self._last_result = "prepared", None
        # At most 16 full/final shape combinations. Query the shared planners
        # without reading integrals or allocating device/runtime state.
        no, nv = snapshot.nocc, snapshot.nmo - snapshot.nocc
        o_sizes = {min(no, occupied_tile), no % occupied_tile or min(no, occupied_tile)}
        v_sizes = {min(nv, virtual_tile), nv % virtual_tile or min(nv, virtual_tile)}
        for shape in product(o_sizes, o_sizes, v_sizes, v_sizes):
            ni, nj, na, nb = shape
            block = MOBlock(
                (
                    tuple(range(ni)),
                    tuple(range(no, no + na)),
                    tuple(range(nj)),
                    tuple(range(no, no + nb)),
                )
            )
            exchange = MOBlock(
                (block.slots[0], block.slots[3], block.slots[2], block.slots[1])
            )
            provider_peak = max(
                self._provider.plan(b).peak_bytes for b in (block, exchange)
            )
            program = self._programs[shape] = energy_program(shape)
            # Detached direct/exchange feeds plus possible contiguous copies.
            # Provider outputs are cleared before each next request. Count
            # this scratch even when the interpreter can consume their views.
            feeds = 32 * math.prod(shape) + 8 * sum(shape) + 64
            if energy_backend == "cuda":
                from tools.vibeqc_tensor.cuda_plan import plan_cuda

                remaining = budget_bytes - provider_peak - feeds
                if remaining <= 0:
                    raise MemoryError("MP2 budget cannot fit provider and tile feeds")
                try:
                    plan = plan_cuda(program, compiler.target, max_bytes=remaining)
                except ValueError as exc:
                    raise MemoryError(f"MP2 tensor plan infeasible: {exc}") from exc
                self._plans[shape] = plan
                tensor_peak = plan.peak_bytes
            else:
                tensor_peak = cpu_capacity(program)
            needed = checked_size(
                provider_peak + feeds + tensor_peak, "MP2 numeric capacity"
            )
            self._capacity = max(self._capacity, needed)
        if self._capacity > budget_bytes:
            raise MemoryError(
                f"MP2 needs {self._capacity} numeric bytes, budget is {budget_bytes}"
            )

    @property
    def state(self):
        return self._state

    @property
    def last_result(self):
        return self._last_result

    @property
    def numeric_capacity_bytes(self):
        return self._capacity

    def _blocks(self):
        no, nm = self._snapshot.nocc, self._snapshot.nmo
        for i, j, a, b in product(
            range(0, no, self._ot),
            range(0, no, self._ot),
            range(no, nm, self._vt),
            range(no, nm, self._vt),
        ):
            yield MOBlock(
                (
                    tuple(range(i, min(i + self._ot, no))),
                    tuple(range(a, min(a + self._vt, nm))),
                    tuple(range(j, min(j + self._ot, no))),
                    tuple(range(b, min(b + self._vt, nm))),
                )
            )

    def _read(self, block):
        result = self._provider.get(block)
        if (
            result.reference_id != self._snapshot.identity
            or result.hamiltonian_id != self._snapshot.hamiltonian_id
        ):
            raise ValueError("MP2 block reference/Hamiltonian mismatch")
        values = result.to_host()
        self._provider.clear()
        return values

    def execute(self, *, properties=("energy",)):
        """Recompute with fresh private blocks; errors never publish partial energy."""
        with self._lock:
            if self._state == "closed":
                raise RuntimeError("MP2 plan is closed")
            self._last_result = None
            self._state = "running"
            runtime = None
            try:
                if tuple(properties) != ("energy",):
                    raise NotImplementedError(
                        "MP2 supports energy only; forces/amplitudes are unavailable"
                    )
                if self._source.identity != self._source_identity:
                    raise ValueError(
                        "MP2 source changed; prepare a fresh reference/provider"
                    )
                self._source._check_open()
                totals, corrections = [0.0, 0.0], [0.0, 0.0]
                count, active_shape = 0, None
                hashes = set()
                eps = self._snapshot.orbital_energies
                for block in self._blocks():
                    i, a, j, b = block.slots
                    shape = (len(i), len(j), len(a), len(b))
                    # A previous shape's tensor arena must not overlap the
                    # new shape's provider workspace: each preflight composes
                    # capacities for one shape only.
                    if runtime is not None and active_shape != shape:
                        runtime.close()
                        runtime = None
                    # Release previous tile feeds before another provider allocation.
                    g = self._read(block).transpose(0, 2, 1, 3)
                    x = self._read(MOBlock((i, b, j, a))).transpose(0, 2, 3, 1)
                    feeds = {"g": g, "x": x}
                    feeds.update(
                        {
                            "e" + k: eps[list(slot)]
                            for k, slot in zip("ijab", (i, j, a, b))
                        }
                    )
                    program = self._programs[shape]
                    hashes.add(program.logical_hash)
                    if self._energy_backend == "cuda":
                        from tools.vibeqc_tensor.cuda_execute import (
                            PreparedCuda,
                            compile_cuda,
                        )

                        if active_shape != shape:
                            plan = self._plans[shape]
                            artifact = compile_cuda(plan, self._compiler, self._cache)
                            runtime = PreparedCuda(plan, artifact, device=self._device)
                            active_shape = shape
                        values = runtime.execute(feeds).outputs
                    else:
                        values = execute(
                            program, feeds, max_bytes=cpu_capacity(program)
                        ).outputs
                    for k, name in enumerate(("opposite_spin", "same_spin")):
                        value = float(values[name])
                        if not math.isfinite(value):
                            raise ValueError("nonfinite MP2 tile energy")
                        # Constant-memory Kahan fold; do not retain one scalar per tile.
                        adjusted = value - corrections[k]
                        updated = totals[k] + adjusted
                        corrections[k] = (updated - totals[k]) - adjusted
                        totals[k] = updated
                    count += 1
                    del feeds, g, x, values
                correlation = math.fsum(totals)
                energy = self._snapshot.reference_energy + correlation
                if not all(math.isfinite(v) for v in (*totals, correlation, energy)):
                    raise ValueError("nonfinite MP2 accumulated energy")
                result = EnergyResult(
                    energy,
                    correlation,
                    *totals,
                    self._minimum,
                    self._snapshot.identity,
                    self._snapshot.hamiltonian_id,
                    count,
                    self._capacity,
                    tuple(sorted(hashes)),
                    "native-cuda-tile"
                    if self._energy_backend == "cuda"
                    else "numpy-cpu-interpreter",
                    self._integral_backend,
                    self._energy_backend == "cuda" or self._integral_backend == "cuda",
                )
            except BaseException:
                self._state = "failed"
                raise
            finally:
                self._provider.clear()
                if runtime is not None:
                    runtime.close()
            self._last_result = result
            self._state = "ready"
            return result

    def close(self):
        with self._lock:
            self._provider.close()
            self._last_result = None
            self._state = "closed"

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
