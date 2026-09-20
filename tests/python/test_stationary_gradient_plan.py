"""Independent algebra gates for #163 B2.1; no molecular-force claim."""

import os
import subprocess
import sys
import typing
from dataclasses import replace
from fractions import Fraction
from itertools import product

import numpy as np
import pytest
from vibeqc_compiler.common.cuda_target import cuda_target_info
from vibeqc_compiler.method import MethodSpec, UnsupportedMethod, resolve_method
from vibeqc_compiler.method.stationary_gradient import (
    SCF_POINT_MODEL,
    StationaryGradientPlan,
    StationaryMeanField,
)
from vibeqc_compiler.tensor import Program, execute
from vibeqc_compiler.tensor.cuda_emit import emit_cuda
from vibeqc_compiler.tensor.cuda_plan import TensorSchedule, plan_cuda


def plan(spin: typing.Any = "unpolarized", method: typing.Any = "PBE") -> typing.Any:
    return StationaryGradientPlan(
        resolve_method(method, spin=spin), StationaryMeanField(SCF_POINT_MODEL)
    )


def fixture(source: typing.Any, spins: typing.Any) -> typing.Any:
    """All ordered 3-AO pair/quartet tuples, including off-diagonal entries."""
    rng = np.random.default_rng(163)
    d = rng.normal(size=(spins, 3, 3))
    d += d.transpose(0, 2, 1)
    w = rng.normal(size=(spins, 3, 3))
    w += w.transpose(0, 2, 1)
    tuples = list(product(range(3), repeat=4 if source == "coulomb" else 2))
    left = np.array([[row[a, b] for a, b, *_ in tuples] for row in d])
    feeds = {"density_left": left}
    if source == "coulomb":
        feeds["density_right"] = np.array(
            [[row[c, e] for _, _, c, e in tuples] for row in d]
        )
    if source == "overlap_pulay":
        feeds = {
            "weighted_density": np.array([[row[a, b] for a, b in tuples] for row in w])
        }
    feeds["integral_derivatives"] = rng.normal(size=(len(tuples), 6))
    integrals = rng.normal(size=len(tuples))
    return feeds, integrals


def independent_weights(source: typing.Any, feeds: typing.Any) -> typing.Any:
    """Separate scalar loops, not generated primal or another TensorIR lowering."""
    left = feeds["weighted_density" if source == "overlap_pulay" else "density_left"]
    result = []
    for t in range(left.shape[1]):
        value = sum(float(row[t]) for row in left)
        if source == "coulomb":
            value *= 0.5 * sum(float(row[t]) for row in feeds["density_right"])
        elif source == "overlap_pulay":
            value = -value
        result.append(value)
    return np.array(result)


@pytest.mark.parametrize("spin", ["unpolarized", "polarized"])
def test_ecp_stationary_plan_has_complete_sources_and_generated_weights(
    spin: typing.Any,
) -> None:
    ae = plan(spin)
    ecp = StationaryGradientPlan(
        ae.method,
        StationaryMeanField(SCF_POINT_MODEL, hamiltonian="scalar-semilocal-ecp"),
    )
    assert ecp.identity != ae.identity
    assert len(ecp.sources) == 9
    assert ecp.sources[0].primitive == "kinetic_effective_charge_attraction"
    assert ecp.sources[-1].primitive == "effective_charge_nuclear_repulsion"
    for source in ("ecp_local", "ecp_nonlocal"):
        with pytest.raises(ValueError, match="integral-gradient"):
            ae.integral_block(source, terms=1)
        feeds, integrals = fixture(source, ecp.spin_blocks)
        block = ecp.integral_block(source, terms=len(integrals), coordinates=6)
        expected = independent_weights(source, feeds) @ feeds["integral_derivatives"]
        np.testing.assert_allclose(
            execute(block.contraction, feeds).outputs["gradient"], expected, atol=2e-13
        )
        with pytest.raises(ValueError, match="coverage"):
            ecp.reduction_program(
                atoms=2, sources=[s for s in ecp.source_names if s != source]
            )
    with pytest.raises(NotImplementedError):
        ecp.require_native_endpoint("cuda")


@pytest.mark.parametrize("source", ["one_electron", "coulomb", "overlap_pulay"])
@pytest.mark.parametrize("spin", ["unpolarized", "polarized"])
def test_generated_weights_and_all_coordinate_components_have_independent_oracle(
    source: typing.Any, spin: typing.Any
) -> None:
    p = plan(spin)
    feeds, integrals = fixture(source, p.spin_blocks)
    block = p.integral_block(source, terms=len(integrals), coordinates=6)
    weights = independent_weights(source, feeds)
    actual = execute(block.weights, feeds).outputs["weights"]
    np.testing.assert_allclose(actual, weights, atol=2e-14, rtol=2e-14)
    expected = np.array(
        [
            sum(
                float(weights[t]) * float(feeds["integral_derivatives"][t, q])
                for t in range(len(weights))
            )
            for q in range(6)
        ]
    )
    gradient = execute(block.contraction, feeds).outputs["gradient"]
    np.testing.assert_allclose(gradient, expected, atol=2e-13, rtol=2e-13)
    # Independent displaced source energy; not the generated primal as oracle.
    # Three steps also catch direction shape/sign and an accidental extra factor.
    for step in (1e-3, 2e-4, 4e-5):
        fd = []
        for q in range(6):
            plus = sum(
                float(v) * float(x)
                for v, x in zip(
                    weights, integrals + step * feeds["integral_derivatives"][:, q]
                )
            )
            minus = sum(
                float(v) * float(x)
                for v, x in zip(
                    weights, integrals - step * feeds["integral_derivatives"][:, q]
                )
            )
            fd.append((plus - minus) / (2 * step))
        np.testing.assert_allclose(gradient, fd, atol=4e-9, rtol=4e-10)
    live = {n.attrs["name"] for n in block.contraction.live_nodes if n.op == "input"}
    assert "integrals" not in live and "bar_energy" not in live
    assert all(len(n.spec.indices) <= 2 for n in block.contraction.live_nodes)
    # Output replay uses the same typed DAG, not a separate equation interpreter.
    replay = Program.loads(block.contraction.dumps())
    np.testing.assert_array_equal(execute(replay, feeds).outputs["gradient"], gradient)


@pytest.mark.parametrize(
    "spin,expected_factor",
    [("unpolarized", Fraction(-1, 16)), ("polarized", Fraction(-1, 8))],
)
def test_exact_exchange_weights_are_same_spin_and_use_methodir_fraction(
    spin: typing.Any, expected_factor: typing.Any
) -> None:
    """Independent scalar K oracle: 1/2*cK and no alpha/beta cross terms."""
    p = plan(spin, method="PBE0")
    rng = np.random.default_rng(165)
    tuples = list(product(range(3), repeat=4))
    density = rng.normal(size=(p.spin_blocks, 3, 3))
    density += density.transpose(0, 2, 1)
    left = np.array([[row[a, c] for a, b, c, d in tuples] for row in density])
    right = np.array([[row[b, d] for a, b, c, d in tuples] for row in density])
    feeds = {
        "density_left": left,
        "density_right": right,
        "integral_derivatives": rng.normal(size=(len(tuples), 5)),
    }
    block = p.integral_block("exact_exchange", terms=len(tuples), coordinates=5)
    actual = execute(block.weights, feeds).outputs["weights"]
    expected = np.array(
        [
            float(expected_factor)
            * sum(float(left[s, t]) * float(right[s, t]) for s in range(p.spin_blocks))
            for t in range(len(tuples))
        ]
    )
    np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=2e-14)
    np.testing.assert_allclose(
        execute(block.contraction, feeds).outputs["gradient"],
        expected @ feeds["integral_derivatives"],
        atol=2e-13,
        rtol=2e-13,
    )
    if p.spin_blocks == 2:
        cross_spin = float(expected_factor) * (left[0] * right[1] + left[1] * right[0])
        assert not np.allclose(actual, expected + cross_spin)


def test_exact_exchange_fraction_and_zero_exchange_recover_expected_plans() -> None:
    pbe = plan(method="PBE")
    pbe0 = plan(method="PBE0")
    assert "exact_exchange" not in pbe.source_names
    assert "exact_exchange" in pbe0.source_names

    custom = MethodSpec(
        "half-hybrid",
        (("GGA_X_PBE", Fraction(1, 2)), ("GGA_C_PBE", Fraction(1))),
        exact_exchange=Fraction(1, 2),
    )
    hybrid = plan(method=custom)
    assert hybrid.exchange.fock_coefficient("unpolarized") == Fraction(-1, 4)
    assert hybrid.exchange.fock_coefficient("polarized") == Fraction(-1, 2)

    semilocal_only = MethodSpec(
        "same-semilo-no-k",
        (("GGA_X_PBE", Fraction(1, 2)), ("GGA_C_PBE", Fraction(1))),
    )
    assert "exact_exchange" not in plan(method=semilocal_only).source_names


@pytest.mark.parametrize("spin", ["unpolarized", "polarized"])
def test_tau_semilocal_method_reuses_stationary_source_inventory(
    spin: typing.Any,
) -> None:
    """r2SCAN adds an ingredient, not a second stationary source equation stack."""
    pbe = plan(spin, "PBE")
    r2scan = plan(spin, "R2SCAN")
    assert r2scan.source_names == pbe.source_names
    assert r2scan.spin_blocks == pbe.spin_blocks
    assert r2scan.identity != pbe.identity
    assert "tau" in r2scan.method.requirements["ingredients"]
    for source in ("one_electron", "coulomb", "overlap_pulay"):
        block = r2scan.integral_block(source, terms=3)
        assert block.source == source
        assert block.plan_identity == r2scan.identity


@pytest.mark.parametrize(
    "source,coefficient",
    [
        ("exchange_short_range", Fraction(19, 100)),
        ("exchange_long_range", Fraction(65, 100)),
    ],
)
@pytest.mark.parametrize("spin", ["unpolarized", "polarized"])
def test_rsh_exchange_gradient_uses_methodir_coefficient_and_same_spin_density(
    source: typing.Any, coefficient: typing.Any, spin: typing.Any
) -> None:
    p = plan(spin, "CAM-B3LYP")
    primitive = p.range_exchange_primitive(source)
    assert primitive.coefficient == coefficient
    assert primitive.omega == Fraction(33, 100)
    assert primitive.operator.replace("-", "_") in source
    assert source in p.source_names

    rng = np.random.default_rng(167)
    terms, coordinates = 19, 6
    left = rng.normal(size=(p.spin_blocks, terms))
    right = rng.normal(size=(p.spin_blocks, terms))
    derivatives = rng.normal(size=(terms, coordinates))
    integrals = rng.normal(size=terms)
    feeds = {
        "density_left": left,
        "density_right": right,
        "integral_derivatives": derivatives,
    }
    factor = -float(coefficient) * (0.5 if p.spin_blocks == 2 else 0.25)
    expected_weights = factor * np.sum(left * right, axis=0)

    block = p.integral_block(source, terms=terms, coordinates=coordinates)
    np.testing.assert_allclose(
        execute(block.weights, feeds).outputs["weights"],
        expected_weights,
        atol=2e-14,
        rtol=2e-14,
    )
    np.testing.assert_allclose(
        execute(block.contraction, feeds).outputs["gradient"],
        expected_weights @ derivatives,
        atol=2e-13,
        rtol=2e-13,
    )
    expected_energy = float(expected_weights @ integrals)
    assert execute(block.objective, {**feeds, "integrals": integrals}).outputs[
        "energy"
    ] == pytest.approx(expected_energy, abs=2e-13)

    # Three displaced-integral steps independently check the analytic derivative.
    for step in (1e-3, 2e-4, 4e-5):
        finite = np.array(
            [
                (
                    float(expected_weights @ (integrals + step * derivatives[:, q]))
                    - float(expected_weights @ (integrals - step * derivatives[:, q]))
                )
                / (2 * step)
                for q in range(coordinates)
            ]
        )
        np.testing.assert_allclose(
            execute(block.contraction, feeds).outputs["gradient"],
            finite,
            atol=4e-9,
            rtol=4e-10,
        )

    if p.spin_blocks == 2:
        # Summing spins before forming exchange would introduce forbidden
        # alpha-beta cross terms.
        wrong = factor * left.sum(axis=0) * right.sum(axis=0)
        assert not np.allclose(expected_weights, wrong)


def test_rsh_gradient_inventory_and_identity_bind_operator_and_omega() -> None:
    p = plan(method="CAM-B3LYP")
    assert p.source_names == (
        "one_electron",
        "coulomb",
        "exchange_short_range",
        "exchange_long_range",
        "xc_ao",
        "xc_grid",
        "xc_weight",
        "overlap_pulay",
        "nuclear",
    )
    assert tuple(source.primitive for source in p.range_exchange_sources) == (
        "short-range-exchange",
        "long-range-exchange",
    )
    assert all(
        "nuclear-gradient" in primitive.derivative_capabilities
        for primitive in p.range_exchange_primitives
    )

    from vibeqc_compiler.method.spec import METHOD_CATALOG

    changed_spec = replace(
        METHOD_CATALOG["CAM-B3LYP"],
        identifier="CAM-B3LYP-omega-test",
        range_omega=Fraction(2, 5),
    )
    changed = plan(method=changed_spec)
    assert changed.identity != p.identity
    for source in ("exchange_short_range", "exchange_long_range"):
        assert changed.range_exchange_primitive(source).omega == Fraction(2, 5)
        assert (
            changed.integral_block(source, terms=3).identity
            != p.integral_block(source, terms=3).identity
        )
    with pytest.raises(ValueError, match="range-exchange"):
        p.range_exchange_primitive("exchange_full_range")


def test_rsh_reduction_requires_both_exchange_components() -> None:
    p = plan(method="CAM-B3LYP")
    components = {
        name: np.full((2, 3), index + 1.0) for index, name in enumerate(p.source_names)
    }
    expected = sum(components.values(), np.zeros((2, 3)))
    np.testing.assert_array_equal(
        p.reduce_diagnostic(components, atoms=2),
        expected,
    )
    for source in ("exchange_short_range", "exchange_long_range"):
        with pytest.raises(ValueError, match="coverage"):
            p.reduce_diagnostic(
                {name: value for name, value in components.items() if name != source},
                atoms=2,
            )


def test_uks_coulomb_includes_cross_spin_and_recovers_total_density_rks() -> None:
    feeds, integrals = fixture("coulomb", 2)
    uks = plan("polarized").integral_block(
        "coulomb", terms=len(integrals), coordinates=6
    )
    rks = plan().integral_block("coulomb", terms=len(integrals), coordinates=6)
    total = {
        name: value.sum(axis=0, keepdims=True) if name.startswith("density_") else value
        for name, value in feeds.items()
    }
    u = execute(uks.contraction, feeds).outputs["gradient"]
    np.testing.assert_allclose(
        u, execute(rks.contraction, total).outputs["gradient"], atol=2e-13, rtol=2e-13
    )
    wrong = 0.5 * np.sum(feeds["density_left"] * feeds["density_right"], axis=0)
    assert not np.allclose(wrong @ feeds["integral_derivatives"], u)


def test_tiled_ordered_quartets_sum_to_the_unsplit_result() -> None:
    p = plan("polarized")
    feeds, integrals = fixture("coulomb", 2)
    expected = execute(
        p.integral_block("coulomb", terms=len(integrals), coordinates=6).contraction,
        feeds,
    ).outputs["gradient"]
    actual = np.zeros(6)
    for start in range(0, len(integrals), 7):
        stop = min(start + 7, len(integrals))
        tile = {
            name: value[start:stop]
            if name == "integral_derivatives"
            else value[:, start:stop]
            for name, value in feeds.items()
        }
        block = p.integral_block("coulomb", terms=stop - start, coordinates=6)
        actual += execute(block.contraction, tile).outputs["gradient"]
    np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=2e-13)


def test_complete_reduction_requires_all_sources_once_and_preserves_inputs() -> None:
    p = plan()
    components = {
        name: np.arange(6, dtype=float).reshape(2, 3) + i + 1
        for i, name in enumerate(p.source_names)
    }
    expected = np.zeros((2, 3))
    for value in components.values():
        expected += value
    result = p.reduce_diagnostic(components, atoms=2)
    np.testing.assert_array_equal(result, expected)
    result[:] = 999
    assert all(not np.any(value == 999) for value in components.values())
    for omitted in p.source_names:
        with pytest.raises(ValueError, match="coverage"):
            p.reduce_diagnostic(
                {k: v for k, v in components.items() if k != omitted}, atoms=2
            )
        # A deliberate zeroed or sign-reversed source must fail the numeric oracle.
        for scale in (0, -1):
            bad = {**components, omitted: scale * components[omitted]}
            assert not np.allclose(p.reduce_diagnostic(bad, atoms=2), expected)
    with pytest.raises(ValueError, match="duplicate"):
        p.reduction_program(atoms=2, sources=(*p.source_names, "nuclear"))
    with pytest.raises(ValueError, match="coverage"):
        p.reduce_diagnostic({**components, "extra": np.zeros((2, 3))}, atoms=2)
    normal = p.reduction_program(atoms=2)
    reordered = p.reduction_program(atoms=2, sources=reversed(p.source_names))
    assert normal.logical_hash == reordered.logical_hash


def test_plan_identity_uses_semantics_not_names_or_live_solve_epochs() -> None:
    p = plan()
    renamed = replace(p, method=replace(p.method, identifier="an-equivalent-alias"))
    assert p.identity == renamed.identity
    assert (
        p.integral_block("coulomb", terms=3).identity
        == renamed.integral_block("coulomb", terms=3).identity
    )
    custom = MethodSpec(
        "half-x", (("GGA_X_PBE", Fraction(1, 2)), ("GGA_C_PBE", Fraction(1)))
    )
    changed = plan(method=custom)
    assert p.identity != changed.identity
    assert p.identity != plan("polarized").identity
    assert (
        p.identity != replace(p, mean_field=StationaryMeanField("interior-v1")).identity
    )
    assert "owner" not in p.to_payload() and "solve_epoch" not in p.to_payload()
    assert (
        p.integral_block("coulomb", terms=3).identity
        != p.integral_block("coulomb", terms=4).identity
    )
    lda = plan(method="LDA_XC_PW")
    assert lda.source_names == p.source_names
    # Common envelope contractions need no LDA/PBE-specific scientific branch.
    assert (
        lda.integral_block("one_electron", terms=3).contraction.logical_hash
        == p.integral_block("one_electron", terms=3).contraction.logical_hash
    )


@pytest.mark.parametrize(
    "change",
    [
        {"hamiltonian": "ecp"},
        {"coulomb": "df"},
        {"occupations": "fractional"},
        {"dtype": "float32"},
        {"topology_policy": "frozen-physical-grid"},
        {"point_model": "unknown"},
    ],
)
def test_unsupported_envelope_is_not_silently_substituted(
    change: typing.Any,
) -> None:
    with pytest.raises(UnsupportedMethod):
        StationaryMeanField(**{"point_model": SCF_POINT_MODEL, **change})


def test_global_hybrid_plan_adds_exact_exchange_without_granting_public_forces() -> (
    None
):
    hybrid = plan(method="PBE0")
    assert hybrid.source_names == (
        "one_electron",
        "coulomb",
        "exact_exchange",
        "xc_ao",
        "xc_grid",
        "xc_weight",
        "overlap_pulay",
        "nuclear",
    )
    assert hybrid.exchange.coefficient == Fraction(1, 4)
    for backend in ("cpu", "cuda"):
        with pytest.raises(NotImplementedError, match="qualification"):
            hybrid.require_native_endpoint(backend)
    with pytest.raises(ValueError, match="backend"):
        hybrid.require_native_endpoint("silently-use-pyscf")


def test_budget_shape_dtype_and_nonfinite_fail_before_publishing() -> None:
    p = plan()
    for kwargs in (
        {"terms": True},
        {"terms": 0},
        {"terms": 3, "coordinates": 0},
        {"terms": 3, "max_elements": True},
        {"terms": 10**10},
    ):
        with pytest.raises(ValueError):
            p.integral_block("coulomb", **kwargs)
    with pytest.raises(ValueError, match="primitive"):
        p.integral_block("xc_grid", terms=3)
    components = {name: np.ones((1, 3)) for name in p.source_names}
    for bad in (
        np.ones((1, 3), dtype=np.float32),
        np.ones((3,)),
        np.full((1, 3), np.nan),
    ):
        with pytest.raises(ValueError):
            p.reduce_diagnostic({**components, "nuclear": bad}, atoms=1)
    with pytest.raises(ValueError, match="budget"):
        p.reduce_diagnostic(components, atoms=1, max_bytes=1)


def test_same_tensor_graph_has_deterministic_cuda_source_and_separate_schedule_identity() -> (
    None
):
    p = plan("polarized")
    programs = [
        p.integral_block(source, terms=5).contraction
        for source in ("one_electron", "coulomb", "overlap_pulay")
    ]
    rsh = plan("polarized", "CAM-B3LYP")
    programs.extend(
        rsh.integral_block(source, terms=5).contraction
        for source in ("exchange_short_range", "exchange_long_range")
    )
    programs.append(p.reduction_program(atoms=2))
    hybrid = plan("polarized", method="PBE0")
    programs.append(hybrid.integral_block("exact_exchange", terms=5).contraction)
    programs.append(hybrid.reduction_program(atoms=2))
    target = cuda_target_info("sm_80")
    for program in programs:
        schedule = TensorSchedule(direct_gemm=False)
        lowered = plan_cuda(program, target, schedule=schedule)
        source = emit_cuda(lowered)
        assert source == emit_cuda(plan_cuda(program, target, schedule=schedule))
        assert "__global__" in source
        other = plan_cuda(program, target, schedule=replace(schedule, threads=64))
        assert lowered.identity != other.identity
        assert (
            lowered.program.logical_hash
            == other.program.logical_hash
            == program.logical_hash
        )
    # Source emission is not a real-device numerical or public-force gate.
    with pytest.raises(NotImplementedError):
        p.require_native_endpoint("cuda")


def test_missing_xc_derivative_rule_rejects_plan(monkeypatch: typing.Any) -> None:
    from vibeqc_compiler.method import SemilocalXCPrimitive

    monkeypatch.setattr(
        SemilocalXCPrimitive,
        "derivative_capabilities",
        property(lambda self: ("energy-density",)),
    )
    with pytest.raises(UnsupportedMethod, match="derivative is unavailable"):
        plan()


def test_missing_exact_exchange_derivative_rule_rejects_hybrid_plan(
    monkeypatch: typing.Any,
) -> None:
    from vibeqc_compiler.method import ExactExchangePrimitive

    monkeypatch.setattr(
        ExactExchangePrimitive,
        "derivative_capabilities",
        property(lambda self: ("energy", "fock")),
    )
    with pytest.raises(UnsupportedMethod, match="exchange ERI derivative"):
        plan(method="PBE0")


def test_source_generation_does_not_import_public_runtime_or_reference_frameworks() -> (
    None
):
    script = """
import importlib.abc
import sys
class BlockRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'vibeqc', 'pyscf', 'cupy', 'torch'}:
            raise AssertionError('compiler imported ' + fullname)
sys.meta_path.insert(0, BlockRuntime())
from vibeqc_compiler.method import StationaryGradientPlan, StationaryMeanField, resolve_method
from vibeqc_compiler.common.cuda_target import cuda_target_info
from vibeqc_compiler.tensor.cuda_emit import emit_cuda
from vibeqc_compiler.tensor.cuda_plan import plan_cuda
p = StationaryGradientPlan(resolve_method('PBE'), StationaryMeanField('interior-v1'))
for source in ('one_electron', 'coulomb', 'overlap_pulay'):
    block = p.integral_block(source, terms=2)
    assert emit_cuda(plan_cuda(block.contraction, cuda_target_info('sm_80')))
rsh = StationaryGradientPlan(resolve_method('CAM-B3LYP'), StationaryMeanField('interior-v1'))
for source in ('exchange_short_range', 'exchange_long_range'):
    block = rsh.integral_block(source, terms=2)
    assert emit_cuda(plan_cuda(block.contraction, cuda_target_info('sm_80')))
assert p.reduction_program(atoms=1).logical_hash
"""
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    subprocess.run(
        [sys.executable, "-c", script], env=environment, check=True, timeout=30
    )
