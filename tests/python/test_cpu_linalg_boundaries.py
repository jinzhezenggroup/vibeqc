"""Compile the actual scalar provider and check independent boundary results."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def boundary_binary(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    output = tmp_path_factory.mktemp("cpu-linalg") / "boundaries"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O2",
            "-I" + str(ROOT / "src"),
            str(ROOT / "tests/native/cpu_linalg_boundary_cases.cpp"),
            str(ROOT / "src/tensor/cpu_linalg.cpp"),
            "-o",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=90,
    )
    return output


@pytest.mark.parametrize(
    "mode",
    [
        "beta_zero",
        "empty_inner",
        "alpha_zero",
        "gemm_extent",
        "cholesky_extent",
        "large_spectrum",
        "tiny_spectrum",
    ],
)
def test_cpu_linalg_boundary(boundary_binary: Path, mode: str) -> None:
    run = subprocess.run(
        [str(boundary_binary), mode],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert run.returncode == 0, (mode, run.stdout, run.stderr)


@pytest.fixture(scope="module")
def probe_binary(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    output = tmp_path_factory.mktemp("cpu-probe") / "probe"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O2",
            "-I" + str(ROOT / "src"),
            str(ROOT / "benchmarks/cpu_linalg_probe.cpp"),
            str(ROOT / "src/tensor/cpu_linalg.cpp"),
            "-o",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=90,
    )
    return output


@pytest.mark.parametrize("provider", ["auto", "scalar"])
def test_probe_multithread_contract(probe_binary: Path, provider: str) -> None:
    run = subprocess.run(
        [str(probe_binary), "8", "1", provider, "4"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    assert '"provider_threads":4' in run.stdout


def test_probe_reports_unavailable_provider(probe_binary: Path) -> None:
    run = subprocess.run(
        [str(probe_binary), "8", "1", "openblas", "1"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert run.returncode == 1
    assert "unavailable" in run.stderr
