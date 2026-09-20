"""Higher-angular CUDA relaxation against a separate native derivative oracle."""

import numpy as np
import test_hessian_relaxation_cuda as _shared
from vibeqc import Primitive, Shell
from vibeqc_compiler.integral.first_gradient import (
    FirstGradientTerm,
    FirstGradientWeight,
)
from vibeqc_compiler.integral.first_gradient_execute import (
    FirstGradientAccumulator,
    compile_first_gradient,
)
from vibeqc_compiler.integral.weighted_eri import build_weighted_eri_ir

from tools.vibeqc_hessian import NativeRHFState, rhf_hvp
from tools.vibeqc_posthf.sources import NativeSource
from tools.vibeqc_validation.hessian_fixtures import fixture_inputs

pytestmark = _shared.pytestmark
compiler = _shared.compiler


def test_asymmetric_pd_weighted_gradient_matches_native_derivatives(compiler, tmp_path):
    inputs = fixture_inputs("h2")
    inputs["basis"] += (
        Shell(0, 1, (Primitive(0.8, 1.0),)),
        Shell(1, 2, (Primitive(0.6, 1.0),)),
    )
    with NativeSource(**inputs) as source:
        state = NativeRHFState.from_source(source, cache=tmp_path / "state")
        slots = (2, 3, 0, 2)
        angular = tuple(source.shells[s].angular_momentum for s in slots)
        assert angular == (1, 2, 0, 1)
        ir = build_weighted_eri_ir(angular)
        weights = np.random.default_rng(590).normal(size=(2, source.nbf, source.nbf))
        term = FirstGradientTerm
        weight = FirstGradientWeight
        terms = (
            term((weight(0, (0, 1)), weight(1, (2, 3))), 0.7),
            term((weight(1, (0, 2)), weight(0, (1, 3))), -0.2),
        )
        artifacts = [
            compile_first_gradient(
                ir,
                compiler,
                tmp_path / "cuda",
                component_indices=tuple(
                    range(i, min(i + 5, ir.signature.component_count))
                ),
                terms=terms,
            )
            for i in range(0, ir.signature.component_count, 5)
        ]
        atoms = tuple(source.shells[s].atom_index for s in slots)
        offsets = tuple(int(state.offsets[s]) for s in slots)
        with FirstGradientAccumulator(
            artifacts[0],
            nbf=source.nbf,
            natoms=state.nat,
            weight_slots=2,
            capacity=2,
            budget_bytes=4 << 20,
        ) as owner:
            owner.reset(weights)
            for artifact in artifacts:
                owner.append_shell(
                    artifact,
                    tuple(state.primitives[s] for s in slots),
                    state.coords[list(atoms)],
                    offsets=offsets,
                    atoms=atoms,
                )
            actual = owner.finish()
            assert owner.statistics["weight_uploads"] == 1
        axes = [np.arange(o, o + source.shell_sizes[s]) for o, s in zip(offsets, slots)]
        # The CPU derivative oracle has its own native recurrence. It does not
        # call the generated scalar DAG, emitter or CUDA contraction under test.
        raw = source.integral_derivatives()["eri"]
        block = raw[np.ix_(np.arange(3 * state.nat), *axes)]
        a, b, c, d = axes
        expected = 0.7 * np.einsum(
            "xijkl,ij,kl->x", block, weights[0][np.ix_(a, b)], weights[1][np.ix_(c, d)]
        ) - 0.2 * np.einsum(
            "xijkl,ik,jl->x", block, weights[1][np.ix_(a, c)], weights[0][np.ix_(b, d)]
        )
        assert np.max(np.abs(expected)) > 1e-6
        np.testing.assert_allclose(
            actual, expected.reshape(state.nat, 3), atol=3e-10, rtol=3e-10
        )
        np.testing.assert_allclose(actual.sum(axis=0), 0, atol=3e-10)


def test_water_p_shell_complete_hvp_has_no_cpu_relaxation_substitution(
    compiler, tmp_path, monkeypatch
):
    with NativeSource(**fixture_inputs("water")) as source:
        state = NativeRHFState.from_source(source, cache=tmp_path / "state")
        vector = np.random.default_rng(591).normal(size=(state.nat, 3))
        vector /= np.linalg.norm(vector)
        expected = rhf_hvp(state, vector)

        def forbidden(*args, **kwargs):
            raise AssertionError("CUDA relaxation entered CPU substitution")

        monkeypatch.setattr(
            "tools.vibeqc_hessian.hvp.generated_rhf_relaxation_contraction", forbidden
        )
        actual = rhf_hvp(
            state, vector, relaxation_backend="cuda", relaxation_compiler=compiler
        )
        np.testing.assert_allclose(
            actual.relaxation, expected.relaxation, atol=3e-10, rtol=3e-10
        )
        np.testing.assert_allclose(actual.value, expected.value, atol=1e-9, rtol=4e-10)
