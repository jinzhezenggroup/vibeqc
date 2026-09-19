"""Allocated-device numerical, record, ABI and lifetime directional gates."""

import os
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from vibeqc import Primitive, Shell
from vibeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
from vibeqc_compiler.common.cuda_target import cuda_target_info
from vibeqc_compiler.integral.first_directional import DirectionalMatrixTerm
from vibeqc_compiler.integral.first_directional_execute import (
    DirectionalFirstAccumulator,
    compile_directional_first,
)
from vibeqc_compiler.integral.one_electron_derivatives import (
    build_one_electron_derivative_ir,
)
from vibeqc_compiler.integral.weight_pullback import normalized_radial_primitives
from vibeqc_compiler.integral.weighted_eri import build_weighted_eri_ir

from tools.vibeqc_posthf.sources import NativeSource

pytestmark = pytest.mark.skipif(
    os.environ.get("VIBEQC_RESPONSE_CUDA_TEST") != "1",
    reason="explicit real-GPU qualification",
)


@pytest.fixture(scope="module")
def compiler():
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm"
    nvcc = shutil.which("nvcc")
    assert nvcc, "selected CUDA qualification needs nvcc on PATH"
    return CudaCompilerAdapter(
        Path(nvcc), cuda_target_info(os.environ.get("VIBEQC_TEST_CUDA_ARCH", "sm_120"))
    )


@pytest.fixture(scope="module")
def artifact(compiler):
    return compile_directional_first(
        build_one_electron_derivative_ir("overlap", (0, 0)),
        compiler,
        Path(".artifacts/directional-unit"),
        component_indices=(0,),
        terms=(DirectionalMatrixTerm(0, (0, 1)),),
    )


def test_signed_f_shell_eri_scatter_matches_independent_native_derivatives(compiler):
    xyz = np.array([[0.13, -0.24, 0.37], [-0.41, 0.22, 0.91], [0.72, 0.34, -0.31]])
    exponents = (0.7, 0.8, 1.1)
    shells = (
        Shell(0, 3, (Primitive(exponents[0], 1.0),)),
        Shell(1, 0, (Primitive(exponents[1], 1.0),)),
        Shell(2, 0, (Primitive(exponents[2], 1.0),)),
    )
    with NativeSource([(2, xyz[0]), (1, xyz[1]), (1, xyz[2])], basis=shells) as source:
        d = np.random.default_rng(180).normal(size=(source.nbf, source.nbf))
        v = np.random.default_rng(181).normal(size=(3, 3))
        independent = source.integral_derivatives()["eri"]
        selected = (0, 4, 9)
        terms = (
            DirectionalMatrixTerm(0, (0, 1), (2, 3)),
            DirectionalMatrixTerm(0, (0, 2), (1, 3), -0.5),
        )
        a = compile_directional_first(
            build_weighted_eri_ir((3, 0, 0, 0)),
            compiler,
            ".artifacts/directional-unit",
            component_indices=selected,
            terms=terms,
        )
        slots = (0, 1, 2, 1)
        primitives = tuple(
            normalized_radial_primitives(
                shells[s].angular_momentum,
                tuple((p.exponent, p.coefficient) for p in shells[s].primitives),
            )
            for s in slots
        )
        expected = np.zeros((1, source.nbf, source.nbf))
        for i in selected:
            value = float(v.ravel() @ independent[:, i, 10, 11, 10])
            expected[0, i, 10] += value * d[11, 10]
            expected[0, i, 11] -= 0.5 * value * d[10, 10]
        with DirectionalFirstAccumulator(
            a, nbf=source.nbf, natoms=3, outputs=1, capacity=1
        ) as owner:
            owner.reset(d, v)
            owner.append_shell(
                a, primitives, xyz[list(slots)], offsets=(0, 10, 11, 10), atoms=slots
            )
            actual = owner.finish()
            np.testing.assert_allclose(actual, expected, atol=3e-10, rtol=3e-10)
            owner.append_shell(
                a, primitives, xyz[list(slots)], offsets=(0, 10, 11, 10), atoms=slots
            )
            np.testing.assert_allclose(
                owner.finish(), 2 * expected, atol=3e-10, rtol=3e-10
            )
            assert owner.statistics["primitive_records"] == 2
            owner.reset(-0.3 * d, v)
            owner.append_shell(
                a, primitives, xyz[list(slots)], offsets=(0, 10, 11, 10), atoms=slots
            )
            np.testing.assert_allclose(
                owner.finish(), -0.3 * expected, atol=3e-10, rtol=3e-10
            )


def test_empty_partial_late_failure_and_clean_replay(artifact):
    primitives = (((0.7, 1.0),), ((0.8, 1.0),))
    centers = np.array([[0.1, 0.2, 0.3], [0.4, 0.3, 1.0]])
    weights = np.eye(2)
    v = np.array([[0.2, 0.3, 0.1], [0.1, 0.1, -0.2]])
    with DirectionalFirstAccumulator(
        artifact, nbf=2, natoms=2, outputs=1, capacity=1
    ) as owner:
        with pytest.raises(RuntimeError, match="reset"):
            owner.finish()
        owner.reset(weights, v)
        np.testing.assert_array_equal(owner.finish(), np.zeros((1, 2, 2)))
        owner.append_shell(artifact, primitives, centers, offsets=(0, 1), atoms=(0, 1))
        expected = owner.finish()
        with pytest.raises(ValueError):
            owner.append_shell(
                artifact,
                (((0.7, 1.0), (-1.0, 1.0)), primitives[1]),
                centers,
                offsets=(0, 1),
                atoms=(0, 1),
            )
        with pytest.raises(RuntimeError, match="reset"):
            owner.finish()
        owner.reset(weights, v)
        owner.append_shell(artifact, primitives, centers, offsets=(0, 1), atoms=(0, 1))
        np.testing.assert_allclose(owner.finish(), expected, atol=1e-13, rtol=1e-13)
        with pytest.raises(ValueError):
            expected.setflags(write=True)
    owner.close()
    with pytest.raises(RuntimeError, match="closed"):
        owner.reset(weights, v)


@pytest.mark.parametrize("failure", ["mapping", "target", "runtime", "terms"])
def test_bad_mapping_or_cached_program_rejected_and_resettable(artifact, failure):
    with DirectionalFirstAccumulator(artifact, nbf=2, natoms=2, outputs=1) as owner:
        owner.reset(np.eye(2), np.ones((2, 3)))
        changed = artifact
        offsets = (0, 1)
        if failure == "mapping":
            offsets = (0, 2)
        elif failure == "target":
            changed = replace(artifact, target=(8, 0))
        elif failure == "runtime":
            changed = replace(artifact, runtime_identity="a" * 64)
        else:
            changed = replace(
                artifact, terms=(DirectionalMatrixTerm(0, (0, 1), coefficient=2),)
            )
        with pytest.raises(ValueError):
            owner.append_shell(
                changed,
                (((0.7, 1.0),),) * 2,
                np.zeros((2, 3)),
                offsets=offsets,
                atoms=(0, 1),
            )
        with pytest.raises(RuntimeError, match="reset"):
            owner.finish()
        owner.reset(np.eye(2), np.ones((2, 3)))
        np.testing.assert_array_equal(owner.finish(), np.zeros((1, 2, 2)))


def test_impossible_budget_and_nonfinite_input_rejected(artifact):
    with pytest.raises(MemoryError, match="before allocation"):
        DirectionalFirstAccumulator(artifact, nbf=2, natoms=2, budget_bytes=1)
    with DirectionalFirstAccumulator(artifact, nbf=2, natoms=2, outputs=1) as owner:
        with pytest.raises(ValueError):
            owner.reset(np.eye(2), np.full((2, 3), np.inf))
        owner.reset(np.eye(2), np.zeros((2, 3)))
        np.testing.assert_array_equal(owner.finish(), np.zeros((1, 2, 2)))


def test_numerical_overflow_poisoning_and_replay(artifact):
    centers = np.array([[0.0, 0.0, 0.0], [0.1, 0.2, 1.0]])
    with DirectionalFirstAccumulator(artifact, nbf=2, natoms=2, outputs=1) as owner:
        owner.reset(np.eye(2), np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1e200]]))
        with pytest.raises(FloatingPointError):
            owner.append_shell(
                artifact,
                (((0.7, 1e200),), ((0.8, 1.0),)),
                centers,
                offsets=(0, 1),
                atoms=(0, 1),
            )
        with pytest.raises(RuntimeError, match="reset"):
            owner.finish()
        owner.reset(np.eye(2), np.zeros((2, 3)))
        np.testing.assert_array_equal(owner.finish(), np.zeros((1, 2, 2)))
