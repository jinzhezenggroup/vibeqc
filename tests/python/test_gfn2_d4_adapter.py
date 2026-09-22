"""Compile the actual GFN2 D4 adapter and exercise its outer storage contract."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def adapter(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if sys.platform != "linux":
        pytest.skip("ELF section-GC fixture requires Linux")
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    output = tmp_path_factory.mktemp("gfn2-d4-adapter") / "probe"
    runtime = ROOT / "src/xtb/gfn2_runtime"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/generate_method_parameters.py"),
            "--source",
            str(ROOT / "python/vibeqc_compiler/method/method_parameters.json"),
            "--cpp-output",
            str(output.parent / "generated_method_parameters.hpp"),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O1",
            "-ffunction-sections",
            "-fdata-sections",
            f"-I{ROOT / 'src'}",
            f"-I{output.parent}",
            f"-I{runtime}",
            f"-I{runtime / 'src'}",
            f"-I{runtime / 'include'}",
            str(ROOT / "tests/native/test_gfn2_d4_adapter.cpp"),
            str(runtime / "src/model/gfn2/d4.cpp"),
            "-Wl,--gc-sections",
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
    "case",
    [
        "alias_positions",
        "alias_cache",
        "nonfinite_output",
        "late_gradient_failure",
        "late_energy_failure",
        "hotloop_shared_parity",
        "per_system_cache_replay",
        "per_system_failure_atomic",
        "success",
    ],
)
def test_adapter_storage_and_publication(adapter: Path, case: str) -> None:
    result = subprocess.run(
        [str(adapter), case], capture_output=True, text=True, timeout=30, check=False
    )
    assert result.returncode == 0, result.stderr
