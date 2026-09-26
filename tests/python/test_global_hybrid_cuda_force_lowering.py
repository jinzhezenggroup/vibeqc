"""Composed CUDA forces preserve same-spin exchange and scientific source order."""

import ctypes
import shutil
import subprocess
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from vibeqc.ks import cuda_global_hybrid_force_eligible
from vibeqc_compiler.method import resolve_method
from vibeqc_compiler.method.stationary_cuda import (
    emit_stationary_weight_cuda,
    stationary_runtime_sources,
)
from vibeqc_compiler.method.stationary_gradient import (
    SCF_POINT_MODEL,
    StationaryGradientPlan,
    StationaryMeanField,
)


@pytest.mark.parametrize("name", ("PBE0", "B3LYP", "M06-2X", "MN15"))
@pytest.mark.parametrize("spin", ("unpolarized", "polarized"))
def test_generated_exchange_weight_matches_independent_density_contraction(
    tmp_path: Path,
    name: str,
    spin: str,
) -> None:
    """Compile the actual emitted expression and compare arbitrary AO quartets.

    Nonzero unequal spin densities expose accidental alpha/beta cross terms;
    unequal AO entries expose exchanging the Coulomb and exchange pairings.
    """
    compiler = shutil.which("c++")
    assert compiler is not None
    method = resolve_method(name, spin=spin)
    plan = StationaryGradientPlan(method, StationaryMeanField(SCF_POINT_MODEL))
    assert stationary_runtime_sources(plan) == plan.source_names
    source = "#include <cstdint>\n#include <cstddef>\n#include <cmath>\nusing std::isfinite;\n#define __device__\n"
    source += emit_stationary_weight_cuda(plan)
    source += """
extern "C" double weight(const double* density, const int64_t* ao) {
  return vibeqc_stationary_cuda::stationary_weight_exact_exchange(density, nullptr, 3, ao);
}
"""
    cpp, library = tmp_path / "weight.cpp", tmp_path / "weight.so"
    cpp.write_text(source)
    subprocess.run(
        [compiler, "-std=c++17", "-shared", "-fPIC", str(cpp), "-o", str(library)],
        check=True,
        timeout=30,
    )
    function = ctypes.CDLL(str(library)).weight
    function.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int64),
    ]
    function.restype = ctypes.c_double
    rng = np.random.default_rng(374)
    density = rng.normal(size=(plan.spin_blocks, 3, 3))
    density = np.ascontiguousarray(density + density.transpose(0, 2, 1))
    coefficient = -float(method.full_range_exact_exchange) / (
        4 if spin == "unpolarized" else 2
    )
    for quartet in np.ndindex((3,) * 4):
        a, b, c, d = quartet
        actual = function(
            density.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            (ctypes.c_int64 * 4)(*quartet),
        )
        expected = coefficient * np.sum(density[:, a, c] * density[:, b, d])
        assert actual == pytest.approx(expected, abs=2e-15, rel=2e-15)


def test_force_coverage_follows_primitives_and_not_method_labels() -> None:
    method = resolve_method("PBE0")
    alias = replace(method, identifier="custom-global-hybrid")
    assert cuda_global_hybrid_force_eligible(alias)
    # A new fraction is represented without changing derivative source code.
    changed = replace(
        alias,
        primitives=tuple(
            replace(node, coefficient=Fraction(3, 7))
            if node.kind == "full-range-exchange"
            else node
            for node in alias.primitives
        ),
    )
    assert cuda_global_hybrid_force_eligible(changed)
    for name in ("PBE", "CAM-B3LYP", "WB97M-V"):
        assert not cuda_global_hybrid_force_eligible(resolve_method(name))
