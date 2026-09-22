"""Boundary qualification for compiler-generated r2SCAN production code."""

from __future__ import annotations

import runpy
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
CPP = shutil.which("c++")

# Independent compiled Libxc 7.0.0 values from the #1028 qualification probe.
REFERENCE = (
    (
        (0.073, 0.0, 0.021315999999999998, 0.0, 0.0, 0.05109999999999999, 0.0),
        (
            -0.03267288239243613,
            -0.5905186926667251,
            12.78273217828023,
            -0.07732912932848196,
            0.07560965137214307,
            0.03780482568607153,
            0.04700651202712578,
            -0.020311812271276885,
        ),
    ),
    (
        (0.0073, 0.0, 0.00021316, 0.0, 0.0, 0.00511, 0.0),
        (
            -0.0014132531993828632,
            -0.25095400954486075,
            -2.5087209282090828,
            -1.2375950557366102,
            0.979904289463213,
            0.4899521447316065,
            0.07855975356000833,
            -0.022931692425350425,
        ),
    ),
    (
        (0.00073, 0.0, 2.1315999999999996e-06, 0.0, 0.0, 0.000511, 0.0),
        (
            -5.974988984373865e-05,
            -0.11840278794271093,
            -3.6854566480993625,
            0.5523913350949331,
            3.003930663581292,
            1.501965331790646,
            0.005595854566402719,
            -0.004881234106355745,
        ),
    ),
    (
        (7.3e-05, 0.0, 2.1316e-08, 0.0, 0.0, 5.1099999999999995e-05, 0.0),
        (
            -2.5637583731920647e-06,
            -0.055116348672106845,
            -0.5631602714694769,
            11.109323744143314,
            8.69087344225384,
            4.34543672112692,
            0.00029873019968308147,
            -0.0005169812760035973,
        ),
    ),
    (
        (
            9.190155506097421e-06,
            0.0,
            3.37835832905011e-10,
            0.0,
            0.0,
            6.433108854268195e-06,
            0.0,
        ),
        (
            -1.4372142317377134e-07,
            -0.026008158606713943,
            0.04094251399167861,
            55.254777114683854,
            29.050547991206304,
            14.525273995603152,
            2.6615594943264455e-05,
            -7.477735560731988e-05,
        ),
    ),
)


@pytest.mark.skipif(CPP is None, reason="C++ compiler unavailable")
def test_generated_r2scan_matches_zero_minority_spin_libxc(tmp_path: Path) -> None:
    namespace = runpy.run_path(str(ROOT / "tools/generate_xc_cpu.py"))
    body = namespace["emit_r2scan_polarized"]()
    header = tmp_path / "r2scan.hpp"
    header.write_text(
        "#include <cmath>\nnamespace vibeqc::dft::generated {\n"
        + body
        + "\n}  // namespace vibeqc::dft::generated\n"
    )

    calls = []
    for values, _ in REFERENCE:
        arguments = ", ".join(repr(value) for value in values)
        calls.append(
            "{ auto v = vibeqc::dft::generated::r2scan_polarized("
            + arguments
            + '); std::printf("%.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g\\n", '
            + "v.energy_density, v.feature_derivative[0], v.feature_derivative[1], "
            + "v.feature_derivative[2], v.feature_derivative[3], v.feature_derivative[4], "
            + "v.feature_derivative[5], v.feature_derivative[6]); }"
        )

    source = tmp_path / "probe.cpp"
    source.write_text(
        '#include <cstdio>\n#include "r2scan.hpp"\nint main() {\n'
        + "\n".join(calls)
        + "\n}\n"
    )
    executable = tmp_path / "probe"
    subprocess.run(
        [CPP, "-std=c++17", "-O2", str(source), "-o", str(executable)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    result = subprocess.run(
        [str(executable)], check=True, capture_output=True, text=True, timeout=30
    )
    actual = np.asarray(
        [
            [float(value) for value in line.split()]
            for line in result.stdout.splitlines()
        ]
    )
    expected = np.asarray([values for _, values in REFERENCE])
    np.testing.assert_allclose(actual, expected, rtol=5e-12, atol=1e-12)
