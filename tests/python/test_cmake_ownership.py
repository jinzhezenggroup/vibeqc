"""Structural build tests for modular CMake ownership."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _read(relative: str) -> str:
    return (ROOT / relative).read_text()


def test_generated_commands_are_centralized():
    """Project declarations should use the shared generated-source helper."""
    helper = _read("cmake/VibeQCGenerated.cmake")
    assert helper.count("add_custom_command(") == 1
    for relative in (
        "CMakeLists.txt",
        "cmake/VibeQCGeneratedSources.cmake",
        "cmake/VibeQCCuda.cmake",
    ):
        assert "add_custom_command(" not in _read(relative)


def test_production_sources_are_explicit_and_component_owned():
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


def test_top_level_is_composition_only_for_sources_and_codegen():
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
