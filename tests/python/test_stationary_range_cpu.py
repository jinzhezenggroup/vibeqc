"""Stationary CPU bridge from MethodIR range exchange to the #249 provider."""

import shutil
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from vibeqc._stationary_cpu import _PrimitiveExecutor
from vibeqc._stationary_range_cpu import RangeExchangePrimitiveExecutor
from vibeqc_compiler.common.cpp_adapter import CppCompilerAdapter
from vibeqc_compiler.integral.weight_pullback import normalized_radial_primitives
from vibeqc_compiler.method.spec import RangeSeparatedExchangePrimitive


@pytest.mark.skipif(
    shutil.which("c++") is None, reason="native C++ compiler unavailable"
)
def test_sr_lr_bridge_closes_to_full_range_for_ordered_quartet(tmp_path) -> None:
    centers = np.array([[0.0, 0.0, 0.0], [0.4, -0.2, 1.1]])
    primitives = np.array(
        [
            normalized_radial_primitives(0, ((0.8, 1.0),))[0],
            normalized_radial_primitives(0, ((1.2, 1.0),))[0],
        ]
    )
    rows = np.zeros((2, 16))
    rows[:, 0] = (0, 1)
    rows[:, 1] = (0, 1)
    rows[:, 2] = 1
    rows[:, 3] = 1
    rows[:, 7] = 1
    basis = SimpleNamespace(
        natom=2,
        nao=2,
        nprimitive=2,
        shells=(
            SimpleNamespace(angular_momentum=0),
            SimpleNamespace(angular_momentum=0),
        ),
        packed=np.r_[centers.ravel(), primitives.ravel(), rows.ravel()],
    )
    compiler = CppCompilerAdapter(Path(shutil.which("c++")))
    full = _PrimitiveExecutor(basis, tmp_path, 8, compiler)
    ranged = RangeExchangePrimitiveExecutor(basis, tmp_path, 8, compiler)
    indices, weight = (0, 1, 0, 1), 0.31
    owners, g_full = full.integral("four_center_eri", indices, weight)
    sr = RangeSeparatedExchangePrimitive(Fraction(1), Fraction(1, 2), "short-range")
    lr = RangeSeparatedExchangePrimitive(Fraction(1), Fraction(1, 2), "long-range")
    sr_owners, g_sr = ranged.integral(sr, indices, weight)
    lr_owners, g_lr = ranged.integral(lr, indices, weight)
    work = ranged.compilation_work
    ranged.close()

    assert owners == sr_owners == lr_owners
    np.testing.assert_allclose(g_sr + g_lr, g_full, atol=1e-14, rtol=1e-13)
    np.testing.assert_allclose(g_sr.sum(axis=0), 0, atol=1e-14)
    np.testing.assert_allclose(g_lr.sum(axis=0), 0, atol=1e-14)
    assert full.records == 1
    assert work["range_exchange_primitive_records"] == 2
    assert work["range_exchange_compiled_plans"] == 2
