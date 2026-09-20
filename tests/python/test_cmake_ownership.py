"""Structural build tests for modular CMake ownership."""

import json
import subprocess
import sys
import typing
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _read(relative: str) -> str:
    return (ROOT / relative).read_text()


def test_generated_commands_are_centralized() -> None:
    """Project declarations should use the shared generated-source helper."""
    helper = _read("cmake/VibeQCGenerated.cmake")
    assert helper.count("add_custom_command(") == 1
    for relative in (
        "CMakeLists.txt",
        "cmake/VibeQCGeneratedSources.cmake",
        "cmake/VibeQCCuda.cmake",
    ):
        assert "add_custom_command(" not in _read(relative)


def test_generated_command_preserves_list_valued_arguments(
    tmp_path: typing.Any,
) -> None:
    """A multi-SM generator request must arrive as one semicolon-delimited value."""
    generator = tmp_path / "record_args.py"
    generator.write_text(
        "import json, pathlib, sys\n"
        "pathlib.Path(sys.argv[1]).write_text(json.dumps(sys.argv[2:]))\n"
    )
    output = tmp_path / "arguments.json"
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.24)\n"
        "project(GeneratorArguments LANGUAGES NONE)\n"
        f'include("{ROOT / "cmake/VibeQCGenerated.cmake"}")\n'
        f'set(Python3_EXECUTABLE "{sys.executable}")\n'
        'set(architectures "90;120")\n'
        "vibeqc_register_generated_sources(\n"
        "  NAME generate\n"
        f'  GENERATOR "{generator}"\n'
        f'  OUTPUTS "{output}"\n'
        f'  ARGS "{output}" --architectures "${{architectures}}" "with spaces")\n'
    )
    build = tmp_path / "build"
    subprocess.run(["cmake", "-S", str(tmp_path), "-B", str(build)], check=True)
    subprocess.run(["cmake", "--build", str(build), "--target", "generate"], check=True)
    assert json.loads(output.read_text()) == [
        "--architectures",
        "90;120",
        "with spaces",
    ]


def test_production_sources_are_explicit_and_component_owned() -> None:
    """Do not trade a monolithic list for implicit recursive source globbing."""
    sources = _read("cmake/VibeQCSources.cmake")
    assert "file(GLOB" not in sources
    for owner in (
        "vibeqc_add_runtime_sources",
        "vibeqc_add_integrals_scf_sources",
        "vibeqc_add_dft_sources",
        "vibeqc_add_posthf_cc_sources",
    ):
        assert f"function({owner}" in sources


def test_top_level_is_composition_only_for_sources_and_codegen() -> None:
    cmake = _read("CMakeLists.txt")
    for module in (
        "VibeQCGeneratedSources.cmake",
        "VibeQCSources.cmake",
        "VibeQCCuda.cmake",
    ):
        assert module in cmake
    assert "vibeqc_add_component_sources(vibeqc)" in cmake
    assert "vibeqc_register_host_generated_sources(vibeqc)" in cmake
    assert "vibeqc_register_cuda_generated_sources(vibeqc)" in cmake
