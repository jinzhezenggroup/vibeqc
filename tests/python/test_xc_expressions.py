"""Independent energy, derivative, boundary and grid-chain-rule XC gates."""

import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from vibeqc.profiles import file_hash
from vibeqc_compiler.dft.features import density_features
from vibeqc_compiler.integral.cuda import CudaEmitter
from vibeqc_compiler.integral.expr import AlgebraForm, Graph, Node
from vibeqc_compiler.xc import (
    FunctionalSpec,
    UnsupportedXC,
    build_program,
    functional,
    pack_grid_features,
    validate_features,
)
from vibeqc_compiler.xc.capabilities import query_capability
from vibeqc_compiler.xc.cuda import plan_tiles
from vibeqc_compiler.xc.cuda_emit import XCSchedule, emit_cuda
from vibeqc_compiler.xc.fixtures import load_fixture
from vibeqc_compiler.xc.potential import potential_coefficients
from vibeqc_compiler.xc.reference import exchange_reference
from vibeqc_compiler.xc.spec import CATALOG

from tools.vibeqc_validation.schema import block_error


def check(actual, expected, *, atol=1e-11, rtol=1e-10):
    result = block_error(actual, expected, atol=atol, rtol=rtol)
    assert result["passed"], result


@pytest.mark.parametrize("name", CATALOG)
@pytest.mark.parametrize("spin", ["polarized", "unpolarized"])
@pytest.mark.parametrize("domain", ["typical", "boundary"])
def test_each_feature_derivative_against_pinned_independent_oracles(name, spin, domain):
    metadata, features, expected, raw = load_fixture(name, spin=spin, domain=domain)
    program = build_program(functional(name, spin=spin))
    actual = program.evaluate(features)
    tolerance = metadata[f"{domain}_tolerance"]
    # Keep the existing FP64 block gate, including small-element handling.
    check(actual, expected, **tolerance)
    assert raw.shape == actual.shape
    assert np.all(
        actual[
            [
                i
                for i, output in enumerate(program.outputs)
                if any(j >= (5 if spin == "polarized" else 2) for j in output)
            ]
        ]
        == 0
    )


def test_licenses_sources_and_reference_generator_hashes():
    root = Path(__file__).resolve().parents[2]
    source = root / "external/libxc-7.0.0"
    manifest = json.loads((source / "manifest.json").read_text())
    for name, item in manifest["files"].items():
        assert file_hash(source / name) == item["sha256"]
    metadata = json.loads((root / "tests/data/xc/libxc.json").read_text())
    assert metadata["script_sha256"] == file_hash(
        root / "tools/generate_xc_references.py"
    )


@pytest.mark.parametrize(
    "operation,point", [("log", 0.3), ("log1p", 1e-15), ("expm1", 1e-15)]
)
def test_stable_unary_rebuilds_second_derivatives_and_cuda(operation, point):
    graph = Graph()
    x = graph.variable("x")
    root = graph.stable_unary(operation, x)
    first = graph.differentiate(root, x)
    second = graph.differentiate(first, x)
    expected = (
        getattr(np, operation)(point),
        1 / point
        if operation == "log"
        else 1 / (1 + point)
        if operation == "log1p"
        else np.exp(point),
        -1 / point**2
        if operation == "log"
        else -1 / (1 + point) ** 2
        if operation == "log1p"
        else np.exp(point),
    )
    for form in AlgebraForm:
        target, roots = graph.apply_algebra_form((root, first, second), form)
        target, roots = target.lower_small_integer_powers(roots)
        check([target.evaluate(r, {"x": point}) for r in roots], expected, atol=1e-25)
        emitter = CudaEmitter(target, {"x": "x"})
        emitter.emit(roots)
        assert operation + "(" in "\n".join(emitter.lines)
    with pytest.raises(ValueError):
        graph.stable_unary("sin", x)
    bad = graph._intern(Node("unsupported", (x.identifier,)))
    with pytest.raises(ValueError):
        graph.differentiate(bad, x)
    with pytest.raises(ValueError):
        graph.evaluate(bad, {"x": 1})
    with pytest.raises(ValueError):
        CudaEmitter(graph, {}).emit((bad,))


@pytest.mark.parametrize("name", ["LDA_XC_PW", "PBE"])
@pytest.mark.parametrize("spin", ["polarized", "unpolarized"])
def test_optimization_order_and_independently_derived_hessian_symmetry(name, spin):
    _, x, _, _ = load_fixture(name, spin=spin)
    spec = functional(name, spin=spin)
    outputs = (
        (),
        *((i,) for i in range(len(spec.features))),
        *((i, j) for i in range(len(spec.features)) for j in range(len(spec.features))),
    )
    programs = [
        build_program(spec, outputs=outputs, optimization=mode)
        for mode in ("none", "before", "after")
    ]
    results = [p.evaluate(x) for p in programs]
    for value in results[1:]:
        check(value, results[0])
    n = len(spec.features)
    h = results[0][1 + n :].reshape(n, n, x.shape[1])
    check(h, h.transpose(1, 0, 2))


@pytest.mark.parametrize("name", CATALOG)
def test_directional_energy_and_gradient_finite_differences_multiple_steps(name):
    _, x, _, _ = load_fixture(name)
    x = x[:, 12:17]
    program = build_program(functional(name))
    result = program.unpack(program.evaluate(x))
    rng = np.random.default_rng(27)
    direction = rng.normal(size=x.shape) * np.maximum(np.abs(x), 0.01) * 0.03
    first = np.einsum("ip,ip->p", result["gradient"], direction)
    second = np.einsum("ijp,jp->ip", result["hessian"], direction)
    for step in (1e-3, 3e-4, 1e-4):
        plus = program.unpack(program.evaluate(x + step * direction))
        minus = program.unpack(program.evaluate(x - step * direction))
        check(
            (plus["energy_density"] - minus["energy_density"]) / (2 * step),
            first,
            atol=2e-8,
            rtol=2e-6,
        )
        check(
            (plus["gradient"] - minus["gradient"]) / (2 * step),
            second,
            atol=2e-8,
            rtol=2e-6,
        )


@pytest.mark.parametrize("name", CATALOG)
def test_spin_exchange_and_unpolarized_chain_rule(name):
    _, x, _, _ = load_fixture(name)
    program = build_program(functional(name))
    permutation = [1, 0, 4, 3, 2, 6, 5]
    a = program.unpack(program.evaluate(x))
    b = program.unpack(program.evaluate(x[permutation]))
    check(b["energy_density"], a["energy_density"])
    check(b["gradient"], a["gradient"][permutation])
    check(b["hessian"], a["hessian"][permutation][:, permutation])
    _, u, _, _ = load_fixture(name, spin="unpolarized")
    transform = np.zeros((7, 3))
    transform[:2, 0] = 0.5
    transform[2:5, 1] = 0.25
    transform[5:, 2] = 0.5
    restricted = build_program(functional(name, spin="unpolarized"))
    expected = restricted.unpack(restricted.evaluate(u))
    full = program.unpack(program.evaluate(transform @ u))
    check(full["energy_density"], expected["energy_density"])
    check(transform.T @ full["gradient"], expected["gradient"])
    check(
        np.einsum("ia,ijp,jb->abp", transform, full["hessian"], transform),
        expected["hessian"],
    )


def test_domain_contract_zero_sigma_vacuum_polarization_extremes_and_no_clipping():
    spec = functional("PBE")
    x = np.array([[0.3, 0.2, 0, 0, 0, 0.2, 0.1]]).T
    assert np.isfinite(build_program(spec).evaluate(x)).all()
    zero = np.zeros((7, 2))
    check(build_program(spec, order=0).evaluate(zero), np.zeros((1, 2)))
    with pytest.raises(UnsupportedXC, match="vacuum"):
        build_program(spec).evaluate(zero)
    for point in (
        [1, 0, 0, 0, 0, 0, 0],
        [1e-300, 1e-300, 0, 0, 0, 0, 0],
        [1e300, 1, 0, 0, 0, 0, 0],
        [0.3, 0.2, 1, 2, 1, 0, 0],
        [0.3, 0.2, 1e200, 0, 1, 0, 0],
        [0.3, 0.2, -1e-20, 0, 0, 0, 0],
    ):
        with pytest.raises(UnsupportedXC):
            build_program(spec).evaluate(np.array(point)[:, None])
    # Evaluate both sides of each support boundary. There is no smoothing
    # branch whose input derivative could accidentally be treated as identity.
    for n in (1e-12 * 1.001, 1e12 * 0.999):
        point = np.array([0.4 * n, 0.6 * n, 0, 0, 0, 0, 0])[:, None]
        original = point.copy()
        assert np.isfinite(build_program(spec).evaluate(point)).all()
        np.testing.assert_array_equal(point, original)
    for fraction, supported in ((1.01e-10, True), (0.99e-10, False)):
        point = np.array([fraction, 1 - fraction, 0, 0, 0, 0, 0])[:, None]
        if supported:
            assert np.isfinite(build_program(spec).evaluate(point)).all()
        else:
            with pytest.raises(UnsupportedXC, match="spin fraction"):
                validate_features(spec, point)


def test_composition_metadata_pruning_and_capability_stages():
    pbe = functional("PBE")
    hybrid = FunctionalSpec(
        "test-PBE0",
        (("GGA_X_PBE", Fraction(3, 4)), ("GGA_C_PBE", Fraction(1))),
        exact_exchange=Fraction(1, 4),
    )
    _, x, _, _ = load_fixture("PBE")
    local = build_program(hybrid, order=0).evaluate(x)
    check(
        local,
        0.75 * build_program(functional("GGA_X_PBE"), order=0).evaluate(x)
        + build_program(functional("GGA_C_PBE"), order=0).evaluate(x),
    )
    assert hybrid.identity != pbe.identity
    program = build_program(pbe, outputs=((0,), (2, 3)))
    full = build_program(pbe)
    check(
        program.evaluate(x),
        full.evaluate(x)[[full.outputs.index(v) for v in program.outputs]],
    )
    assert len(program.graph.topological_order(program.roots)) < len(
        full.graph.topological_order(full.roots)
    )
    assert program.unpack(program.evaluate(x)) == {}
    cap = query_capability(program)
    assert cap["represented"] and cap["emitted"]
    assert not any(
        cap[k]
        for k in (
            "compiled",
            "validated",
            "promoted",
            "public_dft",
            "exact_exchange_evaluated",
            "hartree_evaluated",
            "nuclear_energy_evaluated",
        )
    )
    for kwargs in (
        {"order": 3},
        {"outputs": ((7,),)},
        {"outputs": ()},
        {"outputs": ((0,), (0,))},
    ):
        with pytest.raises(UnsupportedXC):
            build_program(pbe, **kwargs)
    with pytest.raises(UnsupportedXC):
        functional("unknown")
    with pytest.raises(UnsupportedXC):
        FunctionalSpec("bad", (("LDA_X", 0.5),))


def test_matrix_potential_factors_from_density_variations_including_tau():
    rng = np.random.default_rng(16)
    jets = rng.normal(size=(4, 9, 3))
    d = np.stack((np.eye(3), 0.7 * np.eye(3)))
    direction = rng.normal(size=d.shape)
    direction += direction.transpose(0, 2, 1)
    weights = rng.uniform(0.2, 1, 9)
    spec = functional("PBE")
    features = density_features(jets, d)
    x = pack_grid_features(spec, features)
    program = build_program(spec)
    v = program.unpack(program.evaluate(x))["gradient"]
    # A separate linear tau term exercises the one-half convention even though
    # this inventory's semilocal expressions are independent of tau.
    v[5:] += np.array([0.3, -0.2])[:, None]
    coefficients = potential_coefficients(spec, features["gradient"], v)
    phi = jets[0]
    grad = jets[1:4].transpose(1, 0, 2)
    potential = np.einsum("p,sp,pm,pn->smn", weights, coefficients["rho"], phi, phi)
    potential += np.einsum(
        "p,spk,pkm,pn->smn", weights, coefficients["gradient"], grad, phi
    )
    potential += np.einsum(
        "p,spk,pm,pkn->smn", weights, coefficients["gradient"], phi, grad
    )
    potential += np.einsum(
        "p,sp,pkm,pkn->smn", weights, coefficients["tau"], grad, grad
    )

    def energy(density):
        values = pack_grid_features(spec, density_features(jets, density))
        return np.dot(
            weights, program.evaluate(values)[0] + 0.3 * values[5] - 0.2 * values[6]
        )

    expected = np.sum(potential * direction)
    for step in (1e-4, 3e-5, 1e-5):
        check(
            (energy(d + step * direction) - energy(d - step * direction)) / (2 * step),
            expected,
            atol=1e-7,
            rtol=1e-8,
        )
    with pytest.raises(UnsupportedXC):
        pack_grid_features(functional("PBE", spin="unpolarized"), features)


def test_cuda_source_determinism_output_groups_and_numeric_budget():
    program = build_program(functional("PBE"))
    sources = []
    for variant, groups in (("baseline", 36), ("fused", 1), ("split", 5)):
        schedule = XCSchedule(variant)
        source, contract, _ = emit_cuda(program, schedule)
        assert emit_cuda(program, schedule)[0] == source
        assert len(contract["groups"]) == groups
        assert "log1p(" in source and "expm1(" in source
        sources.append(source)
    assert len(set(sources)) == 3
    plan = plan_tiles(program, tile_points=7)
    assert plan.device_bytes == 7 * (7 + 36) * 8 + 256
    assert plan.allocation_bytes <= plan.budget_bytes
    with pytest.raises(ValueError, match="budget"):
        plan_tiles(program, budget_bytes=plan.allocation_bytes - 1, tile_points=7)
    with pytest.raises(ValueError):
        plan_tiles(program, tile_points=True)


@pytest.mark.parametrize("spin", [False, True])
@pytest.mark.parametrize("gga", [False, True])
def test_closed_form_exchange_oracle_on_typical_domain(spin, gga):
    name = "GGA_X_PBE" if gga else "LDA_X"
    _, x, _, _ = load_fixture(name, spin="polarized" if spin else "unpolarized")
    e, v, h = exchange_reference(x, spin=spin, gga=gga)
    p = build_program(functional(name, spin="polarized" if spin else "unpolarized"))
    actual = p.unpack(p.evaluate(x))
    for key, expected in (("energy_density", e), ("gradient", v), ("hessian", h)):
        check(actual[key], expected)


def test_functional_spec_rejects_mutable_or_unidentified_composition():
    with pytest.raises(UnsupportedXC):
        FunctionalSpec(1, (("LDA_X", Fraction(1)),))
    with pytest.raises(UnsupportedXC):
        FunctionalSpec("mutable", (["LDA_X", Fraction(1)],))


def test_capability_cannot_relabel_cpu_or_failed_blocks_as_cuda_validation(tmp_path):
    from types import SimpleNamespace

    from vibeqc_compiler.xc.cuda import XCArtifact

    from tools.vibeqc_validation.schema import new_evidence, outcome

    program = build_program(functional("LDA_X"))
    _, contract, _ = emit_cuda(program)
    path = tmp_path / "test-artifact"
    path.write_bytes(b"test artifact metadata; never loaded or executed")
    binary_hash = file_hash(path)
    artifact = XCArtifact(
        SimpleNamespace(library=path, metadata={"binary_sha256": binary_hash}), contract
    )
    report = new_evidence(
        tier="cpu", subject="unit-test", inputs_hash=program.expression_hash
    )
    report.update(
        identity=contract["identity"], binary_sha256=binary_hash, backend_selected="cpu"
    )
    report["stages"]["compilation"] = outcome("pass")
    report["stages"]["numerical"] = outcome("pass")
    report["block_errors"] = {"test": {"passed": True, "max_scaled_error": 0}}
    assert not query_capability(program, artifact=artifact, evidence=report)[
        "validated"
    ]
    report.update(backend_selected="cuda", device={"kind": "unit-test fixture"})
    report["hardware"] = outcome("pass")
    report["block_errors"]["test"] = {"passed": False, "max_scaled_error": 2}
    assert not query_capability(program, artifact=artifact, evidence=report)[
        "validated"
    ]
