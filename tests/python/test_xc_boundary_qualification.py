"""Automatic physical-boundary qualification gates for generated XC."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from vibeqc_compiler.xc.boundary import (
    BOUNDARY_SEMANTICS,
    bulk_feature_names,
    semilocal_boundary_probes,
)

ROOT = Path(__file__).resolve().parents[2]
R2SCAN_REFERENCE = ROOT / "tests/data/xc/boundary/r2scan-zero-minority.json"


@pytest.mark.parametrize("family", ("lda", "gga", "mgga"))
@pytest.mark.parametrize("spin", ("polarized", "unpolarized"))
def test_standard_boundary_suite_matches_bulk_feature_layout(
    family: str, spin: str
) -> None:
    names = bulk_feature_names(family, spin)
    probes = semilocal_boundary_probes(names, spin=spin)
    assert probes
    assert all(probe.feature_names == names for probe in probes)
    assert all(probe.spin == spin for probe in probes)
    assert len({probe.label for probe in probes}) == len(probes)
    assert all(np.isfinite(probe.values).all() for probe in probes)


def test_zero_minority_probe_is_the_issue_1028_physical_endpoint() -> None:
    names = (
        "rho_a",
        "rho_b",
        "sigma_aa",
        "sigma_ab",
        "sigma_bb",
        "tau_a",
        "tau_b",
    )
    probe = next(
        item
        for item in semilocal_boundary_probes(names, spin="polarized")
        if item.label == "zero-minority"
    )
    rho = 0.073
    assert probe.values == pytest.approx(
        (rho, 0.0, 4.0 * rho * rho, 0.0, 0.0, 0.7 * rho, 0.0)
    )


def _r2scan_production_values(features: np.ndarray, tmp_path: Path) -> np.ndarray:
    """Compile the AOT entry point, including its MGGA boundary work wrapper.

    Bare derivative roots do not model density/kinetic floors or empty-spin
    handling. The CLI also installs compiler package stubs, so run it in a child
    process rather than corrupting imports for later tests.
    """
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    header = tmp_path / "xc.hpp"
    subprocess.run(
        [
            sys.executable,
            "-I",
            str(ROOT / "tools/generate_xc_cpu.py"),
            "--output",
            str(header),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    rows = ",\n".join(
        "{" + ", ".join(repr(float(v)) for v in row) + "}" for row in features
    )
    source = tmp_path / "probe.cpp"
    source.write_text(
        '#include <cstdio>\n#include "xc.hpp"\nint main() {\n'
        + "const double points[][7] = {"
        + rows
        + "};\n"
        + "for (const auto& p : points) {\n"
        + "const auto v = vibeqc::dft::generated::r2scan_polarized("
        + "p[0], p[1], p[2], p[3], p[4], p[5], p[6]);\n"
        + 'std::printf("%.17g", v.energy_density);\n'
        + 'for (double d : v.feature_derivative) std::printf(" %.17g", d);\n'
        + 'std::puts("");\n}\n}\n'
    )
    executable = tmp_path / "probe"
    subprocess.run(
        [compiler, "-std=c++17", "-O2", str(source), "-o", str(executable)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    result = subprocess.run(
        [str(executable)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return np.asarray(
        [[float(v) for v in row.split()] for row in result.stdout.splitlines()]
    )


def test_r2scan_zero_minority_boundary_status_is_machine_readable(
    tmp_path: Path,
) -> None:
    reference = json.loads(R2SCAN_REFERENCE.read_text(encoding="utf-8"))
    assert reference["schema"] == "vibeqc.xc-boundary-reference/v1"
    assert reference["boundary_semantics"] == BOUNDARY_SEMANTICS
    assert reference["oracle"]["libxc"] == "7.0.0"
    assert reference["functional"] == "R2SCAN"
    assert reference["spin"] == "polarized"

    features = np.asarray([p["features"] for p in reference["points"]])
    expected = np.asarray([p["expected"] for p in reference["points"]])
    # A spin permutation is an independent symmetry of the physical contract.
    features = np.concatenate((features, features[:, [1, 0, 4, 3, 2, 6, 5]]))
    expected = np.concatenate((expected, expected[:, [0, 2, 1, 5, 4, 3, 7, 6]]))
    actual = _r2scan_production_values(features, tmp_path)
    passed = (
        actual.shape == expected.shape
        and np.isfinite(actual).all()
        and np.allclose(
            actual, expected, rtol=reference["rtol"], atol=reference["atol"]
        )
    )

    status = "pass" if passed else "fail"
    assert status == reference["expected_current_status"], (
        "boundary qualification changed; update the machine-readable status only "
        "after reviewing the independent Libxc evidence"
    )


def test_retained_r2scan_oracle_matches_independent_libxc() -> None:
    """Validate the historical fixture through the independent oracle seam."""
    libxc = pytest.importorskip("pyscf.dft.libxc")
    if libxc.__version__ != "7.0.0":
        pytest.skip("the retained oracle requires Libxc 7.0.0")
    from tools.generate_libxc_boundary_reference import _configure, evaluate_reference

    library = libxc._itrf
    _configure(library)
    reference = json.loads(R2SCAN_REFERENCE.read_text())
    names = bulk_feature_names("mgga", "polarized")
    for point in reference["points"]:
        values = point["features"]
        full = (*values[:5], 0.0, 0.0, *values[5:])
        observed = np.zeros(10)
        for name, identifier in (("MGGA_X_R2SCAN", 497), ("MGGA_C_R2SCAN", 498)):
            result = evaluate_reference(
                library,
                {"name": name, "id": identifier, "family": "mgga"},
                "polarized",
                names,
                full,
            )
            assert result["oracle_finite"]
            observed += np.asarray(result["expected"])
        np.testing.assert_allclose(
            observed[[0, 1, 2, 3, 4, 5, 8, 9]],
            point["expected"],
            rtol=reference["rtol"],
            atol=reference["atol"],
        )


def test_boundary_qualification_does_not_replace_compiler_packages() -> None:
    # Import the test module as pytest collection does, in a clean interpreter.
    # Loading the generator module here used to replace package exports with stubs.
    code = (
        "import runpy, sys; import vibeqc_compiler.xc as before; "
        "runpy.run_path(sys.argv[1]); import vibeqc_compiler.xc as after; "
        "assert before is after; assert callable(after.functional)"
    )
    subprocess.run(
        [sys.executable, "-c", code, str(Path(__file__).resolve())],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
