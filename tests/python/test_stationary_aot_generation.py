"""Stationary AOT builds lower the shared primitive inventory only once."""

import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from tools import generate_stationary_force_aot as generator

ROOT = Path(__file__).resolve().parents[2]


def test_all_shards_preserve_sources_and_unchanged_output_times(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sources = tuple(
        ((), f"// primitive shard {index}\n")
        for index in range(generator.QUALIFIED_SPD_AOT_SHARDS)
    )
    lower = Mock(return_value=sources)
    monkeypatch.setattr(generator, "derivative_cuda_sources", lower)
    output = tmp_path / "generated"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_stationary_force_aot.py",
            "--output",
            str(output),
            "--component-domain",
            "spd",
            "--all-shards",
        ],
    )
    generator.main()
    lower.assert_called_once_with(generator.QUALIFIED_SPD_COMPONENTS)
    expected = {
        f"vibeqc_stationary_spd_primitive_{index}.cu": source
        for index, (_, source) in enumerate(sources)
    }
    assert {path.name: path.read_text() for path in output.iterdir()} == expected
    times = {path.name: path.stat().st_mtime_ns for path in output.iterdir()}
    generator.main()
    assert {path.name: path.stat().st_mtime_ns for path in output.iterdir()} == times


@pytest.mark.parametrize(
    "arguments",
    [
        ["--all-shards"],
        ["--all-shards", "--component-domain", "spd", "--shard-index", "0"],
        ["--all-shards", "--component-domain", "spd", "--functional", "0"],
        ["--all-shards", "--component-domain", "spd", "--spin", "unpolarized"],
        ["--component-domain", "spd", "--shard-index", "-1"],
        ["--component-domain", "spd", "--shard-index", "23"],
    ],
)
def test_invalid_shard_arguments_fail_before_lowering(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, arguments: list[str]
) -> None:
    lower = Mock(side_effect=AssertionError("invalid arguments reached lowering"))
    monkeypatch.setattr(generator, "derivative_cuda_sources", lower)
    output = tmp_path / "generated"
    monkeypatch.setattr(sys, "argv", ["generate", "--output", str(output), *arguments])
    with pytest.raises(SystemExit) as error:
        generator.main()
    assert error.value.code == 2
    lower.assert_not_called()
    assert not output.exists()


@pytest.mark.parametrize("backend", ["Ninja", "Unix Makefiles"])
def test_cmake_build_lowers_each_primitive_request_once(
    tmp_path: Path, backend: str
) -> None:
    """Exercise the real multi-output build rule with only CUDA emission stubbed."""
    cmake = shutil.which("cmake")
    builder = "ninja" if backend == "Ninja" else "make"
    if cmake is None or shutil.which(builder) is None:
        pytest.skip(f"CMake and {builder} are required for build graph evaluation")

    calls = tmp_path / "lowering.txt"
    stub = tmp_path / "generate.py"
    stub.write_text(
        "import sys\n"
        "from pathlib import Path\n"
        f"sys.path[:0] = [{str(ROOT)!r}, {str(ROOT / 'python')!r}]\n"
        "from tools import generate_stationary_force_aot as generator\n"
        "from vibeqc_compiler.integral import first_derivative_schedule as schedule\n"
        "def emit(requests, *, symbol):\n"
        f"    with Path({str(calls)!r}).open('a') as log:\n"
        "        log.write(f'{symbol} {len(requests)}\\n')\n"
        "    return f'// {symbol} {len(requests)}\\n'\n"
        "schedule.emit_first_derivative_cuda = emit\n"
        "generator.main()\n"
    )
    workflow = (ROOT / "cmake/VibeQCCuda.cmake").read_text()
    start = workflow.index("    set(_vibeqc_stationary_spd_primitive_sources)")
    stop = workflow.index("    add_library(vibeqc_stationary_spd_primitives", start)
    registration = workflow[start:stop].replace(
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_stationary_force_aot.py",
        stub.as_posix(),
    )
    output = tmp_path / "generated"
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.24)\n"
        "project(stationary_codegen NONE)\n"
        f'set(Python3_EXECUTABLE "{Path(sys.executable).as_posix()}")\n'
        f'set(CMAKE_CURRENT_SOURCE_DIR "{ROOT.as_posix()}")\n'
        f'set(PROJECT_SOURCE_DIR "{ROOT.as_posix()}")\n'
        f'set(VIBEQC_STATIONARY_AOT_DIRECTORY "{output.as_posix()}")\n'
        f'include("{ROOT.as_posix()}/cmake/VibeQCGenerated.cmake")\n'
        + registration
        + "add_custom_target(all_primitives ALL "
        "DEPENDS ${_vibeqc_stationary_spd_primitive_sources})\n"
    )
    build = tmp_path / "build"
    subprocess.run(
        [cmake, "-S", str(tmp_path), "-B", str(build), "-G", backend],
        check=True,
        capture_output=True,
        timeout=30,
    )

    def build_sources() -> None:
        subprocess.run(
            [cmake, "--build", str(build), "--parallel", "8"],
            check=True,
            capture_output=True,
            timeout=30,
        )

    build_sources()
    expected = [f"first_derivative_shard_{index} 16" for index in range(22)]
    expected.append("first_derivative_shard_22 10")
    assert calls.read_text().splitlines() == expected
    assert len(list(output.glob("*.cu"))) == 23
    build_sources()
    assert calls.read_text().splitlines() == expected

    # A missing secondary output must retrigger the one shared generation rule.
    missing = output / "vibeqc_stationary_spd_primitive_11.cu"
    missing.unlink()
    build_sources()
    assert missing.exists()
    assert calls.read_text().splitlines() == expected * 2
