"""A1 mathematics and bounded consumer checks; fixtures are independent PySCF.

Dense arrays occur only in the test source/oracle. No native/GPU claim follows
from these tests; the actual native source and device gates are separate.
"""

from dataclasses import replace
from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest

from tools.vibeqc_codegen.cuda_target import cuda_target_info
from tools.vibeqc_mp2 import PreparedMP2Energy
from tools.vibeqc_mp2.energy import denominator_check
from tools.vibeqc_mp2.equations import energy_program
from tools.vibeqc_posthf.fixtures import fixture_snapshot, load_fixture
from tools.vibeqc_tensor import Program, execute
from tools.vibeqc_tensor.cuda_emit import emit_cuda
from tools.vibeqc_tensor.cuda_plan import plan_cuda


class FixtureSource:
    """Small dense AO test oracle, adapted to the unmodified CG10 provider."""

    backend = "pyscf-fixture-test-only"

    def __init__(self, snapshot, ao):
        self.nbf = snapshot.nmo
        self.geometry_hash = snapshot.geometry_hash
        self.basis_hash = snapshot.basis_hash
        self.representation = snapshot.representation
        self.shell_sizes = (self.nbf,)
        self.numeric_bytes = ao.nbytes
        self.identity = "fixture-ao"
        self.ao = ao
        self.reads = 0
        self.closed = False

    def _check_open(self):
        if self.closed:
            raise RuntimeError("fixture source is closed")

    def requests(self, operator, *, axis_tile, budget_bytes):
        assert operator == "four_center_eri"
        for starts in product(range(0, self.nbf, axis_tile), repeat=4):
            yield starts, tuple(min(axis_tile, self.nbf - b) for b in starts)

    def tile(self, request):
        self.reads += 1
        starts, sizes = request
        return np.ascontiguousarray(
            self.ao[tuple(slice(b, b + n) for b, n in zip(starts, sizes))]
        )

    def global_offsets(self, request):
        return request[0]


def fixture(name="water"):
    metadata, arrays = load_fixture(name)
    snapshot = fixture_snapshot(metadata, arrays)
    return snapshot, FixtureSource(snapshot, arrays["ao"]), metadata, arrays


def spin_components(snapshot, mo):
    """Explicit spin deltas and 1/4 prefactor, split by occupied spin labels.

    Independently derives OS/SS, without calling the restricted energy program
    or bridge, and uses the unchanged PySCF MO tensor at identical orbitals.
    """
    no, nm = snapshot.nocc, snapshot.nmo
    eps = snapshot.orbital_energies
    result = [0.0, 0.0]
    for i, j, a, b in product(range(no), range(no), range(no, nm), range(no, nm)):
        denominator = eps[i] + eps[j] - eps[a] - eps[b]
        for si, sj, sa, sb in product(range(2), repeat=4):
            g = mo[i, a, j, b] if si == sa and sj == sb else 0.0
            x = mo[i, b, j, a] if si == sb and sj == sa else 0.0
            result[si == sj] += 0.25 * (g - x) ** 2 / denominator
    return result


@pytest.mark.parametrize("name", ["h2", "water", "lih", "f_heh"])
@pytest.mark.parametrize("tiles", [(1, 2), (2, 3)])
def test_independent_molecular_components_and_final_tiles(name, tiles):
    s, source, meta, arrays = fixture(name)
    with PreparedMP2Energy(
        s, source, occupied_tile=tiles[0], virtual_tile=tiles[1], axis_tile=4
    ) as p:
        r = p.execute()
        os, ss = spin_components(s, arrays["conventional_mo"])
        np.testing.assert_allclose(
            [r.opposite_spin, r.same_spin], [os, ss], atol=1e-11, rtol=1e-10
        )
        expected = meta["records"]["conventional"]["correlation_energy"]
        assert abs(r.correlation_energy - expected) <= 1e-9
        assert abs(r.energy - (s.reference_energy + expected)) <= 1e-9
        assert r.reference_id == s.identity
        assert r.numeric_capacity_bytes <= 256 << 20
        assert r.energy_backend == "numpy-cpu-interpreter"
        assert not r.mo_host_staging
        assert not hasattr(r, "amplitudes") and not hasattr(r, "forces")
        if name == "h2":
            assert abs(r.same_spin) < 1e-14
        if name == "water":
            assert abs(r.same_spin) > 1e-5
        assert p.state == "ready" and p.last_result is r
    assert p.state == "closed" and p.last_result is None
    assert not source.closed


def test_rectangular_equations_exchange_prefactors_and_replay():
    rng = np.random.default_rng(193)
    shape = (2, 1, 3, 2)
    feeds = {
        "g": rng.normal(size=shape),
        "x": rng.normal(size=shape),
        "ei": np.array([-1.0, -0.8]),
        "ej": np.array([-0.6]),
        "ea": np.array([0.2, 0.4, 0.7]),
        "eb": np.array([0.3, 0.9]),
    }
    expected = np.zeros(2)
    for i, j, a, b in np.ndindex(shape):
        g, x = feeds["g"][i, j, a, b], feeds["x"][i, j, a, b]
        d = feeds["ei"][i] + feeds["ej"][j] - feeds["ea"][a] - feeds["eb"][b]
        expected += [g * g / d, g * (g - x) / d]
    p = energy_program(shape)
    r = execute(p, feeds).outputs
    np.testing.assert_allclose(
        [r["opposite_spin"], r["same_spin"]], expected, atol=1e-12
    )
    assert p.logical_hash == energy_program(shape).logical_hash
    replay = Program.from_payload(p.to_payload())
    assert replay.logical_hash == p.logical_hash
    # Swapping occupied pairs and virtual pairs together preserves each spin
    # contribution, including different axis lengths and asymmetric tensors.
    permuted = {
        "g": feeds["g"].transpose(1, 0, 3, 2),
        "x": feeds["x"].transpose(1, 0, 3, 2),
        "ei": feeds["ej"],
        "ej": feeds["ei"],
        "ea": feeds["eb"],
        "eb": feeds["ea"],
    }
    rr = execute(energy_program((1, 2, 2, 3)), permuted).outputs
    for key in r:
        assert abs(float(r[key]) - float(rr[key])) < 1e-12
    wrong = execute(p, {**feeds, "x": np.zeros(shape)}).outputs
    assert abs(float(wrong["same_spin"]) - expected[1]) > 0.01
    assert abs(sum(expected) - 2 * expected[0]) > 0.01


def test_budget_before_integrals_exact_boundary_and_tile_storage():
    s, source, *_ = fixture()
    p = PreparedMP2Energy(s, source, axis_tile=4)
    required = p.numeric_capacity_bytes
    with pytest.raises(MemoryError, match="MP2 needs"):
        PreparedMP2Energy(s, source, axis_tile=4, budget_bytes=required - 1)
    assert source.reads == 0
    with PreparedMP2Energy(s, source, axis_tile=4, budget_bytes=required) as exact:
        shapes = [block.shape for block in exact._blocks()]
        assert max(np.prod(shape) for shape in shapes) <= 4
        assert (
            sum(np.prod(shape) for shape in shapes) == s.nocc**2 * (s.nmo - s.nocc) ** 2
        )
        exact.execute()


def test_force_rejection_invalidation_failure_and_neighbors():
    s, source, *_ = fixture("h2")
    p, neighbor = PreparedMP2Energy(s, source), PreparedMP2Energy(s, source)
    with pytest.raises(NotImplementedError, match="energy only"):
        p.execute(properties=("energy", "forces"))
    assert source.reads == 0 and p.last_result is None and p.state == "failed"
    before = p.execute()
    assert p.execute().energy == before.energy
    source.identity = "changed-geometry-generation"
    with pytest.raises(ValueError, match="source changed"):
        p.execute()
    assert p.last_result is None and p.state == "failed"
    source.identity = "fixture-ao"
    assert neighbor.execute().energy == before.energy
    p.close()
    with pytest.raises(RuntimeError, match="closed"):
        p.execute()
    source.closed = True
    with pytest.raises(RuntimeError, match="closed"):
        neighbor.execute()
    neighbor.close()


def test_nonfinite_blocks_fail_without_partial_result():
    s, source, *_ = fixture("h2")
    with PreparedMP2Energy(s, source) as p:
        p.execute()
        source.ao[:] = np.nan
        with pytest.raises(ValueError, match="finite"):
            p.execute()
        assert p.state == "failed" and p.last_result is None


def test_finite_integrals_with_overflow_do_not_publish_energy():
    shape = (1, 1, 1, 1)
    feeds = {
        "g": np.full(shape, 1e200),
        "x": np.zeros(shape),
        "ei": np.array([-1.0]),
        "ej": np.array([-1.0]),
        "ea": np.array([1.0]),
        "eb": np.array([1.0]),
    }
    with pytest.raises(ValueError, match="non-finite"):
        execute(energy_program(shape), feeds)


def test_denominator_and_invalid_reference_preflight():
    s, source, *_ = fixture("h2")
    threshold = denominator_check(s, 1e-10)
    with pytest.raises(ValueError, match="near-zero.*global ijab"):
        PreparedMP2Energy(s, source, denominator_threshold=threshold)
    for bad in (0, -1, np.nan, np.inf):
        with pytest.raises(ValueError, match="threshold"):
            PreparedMP2Energy(s, source, denominator_threshold=bad)
    for change in (
        {"algorithm": "UHF"},
        {"frozen_mask": (0,)},
        {"converged": False},
        {"coefficients": s.coefficients.astype(complex)},
        {"scf_residual": 1.0},
    ):
        with pytest.raises(ValueError):
            replace(s, **change)
    with pytest.raises(TypeError, match="ReferenceSnapshot"):
        PreparedMP2Energy(SimpleNamespace(), source)
    eps = np.zeros(s.nmo)
    zero = replace(s, orbital_energies=eps, fock=np.zeros_like(s.fock))
    with pytest.raises(ValueError, match="strictly below"):
        PreparedMP2Energy(zero, source)
    narrow_eps = np.array([-1e-12, 1e-12])
    c, overlap = s.coefficients, s.overlap
    narrow = replace(
        s, orbital_energies=narrow_eps, fock=overlap @ (c * narrow_eps) @ c.T @ overlap
    )
    with pytest.raises(ValueError, match="near-zero"):
        PreparedMP2Energy(narrow, source)
    with pytest.raises(ValueError, match="nonfinite.*denominator"):
        denominator_check(
            SimpleNamespace(nocc=1, orbital_energies=np.array([-1e308, 1e308])), 1e-10
        )
    assert source.reads == 0


@pytest.mark.parametrize(
    "kw",
    [
        {"virtual_tile": 0},
        {"occupied_tile": True},
        {"budget_bytes": 2**63},
        {"energy_backend": "auto"},
        {"energy_backend": "cuda"},
        {"device_id": -1},
    ],
)
def test_invalid_configuration(kw):
    s, source, *_ = fixture("h2")
    with pytest.raises(ValueError):
        PreparedMP2Energy(s, source, **kw)
    assert source.reads == 0


def test_cuda_source_and_plan_are_not_device_validation():
    p = energy_program((2, 1, 3, 2))
    plan = plan_cuda(p, cuda_target_info("sm_80"), max_bytes=1 << 20)
    assert plan.peak_bytes <= 1 << 20
    assert plan.provider_bytes == 0  # elementwise/reductions need no cuBLAS
    code = emit_cuda(plan)
    assert "tensor_run" in code
    prefixed = emit_cuda(plan, symbol_prefix="mp2_sm80_t2_")
    assert "namespace mp2_sm80_t2_generated {" in prefixed
    assert 'extern "C" int mp2_sm80_t2_tensor_run(' in prefixed
    assert 'extern "C" int tensor_run(' not in prefixed
    assert "mp2_sm80_t2_read_" in prefixed
    with pytest.raises(ValueError, match="symbol_prefix"):
        emit_cuda(plan, symbol_prefix="mp2-sm80-")
    assert plan.program.logical_hash == p.logical_hash
