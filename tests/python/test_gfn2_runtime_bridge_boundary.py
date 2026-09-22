from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_public_xtb_method_has_no_xtbloom_runtime_dependency() -> None:
    source = (ROOT / "src/methods/xtb_method.cpp").read_text()
    assert "xtbloom" not in source.lower()
    assert '"runtime/gfn2_cpu_execution.hpp"' not in source
    assert '"runtime/gfn2_cuda_execution.hpp"' not in source
    assert '"methods/gfn2_runtime_bridge.hpp"' in source


def test_native_bridge_header_does_not_leak_vendor_abi() -> None:
    header = (ROOT / "src/methods/gfn2_runtime_bridge.hpp").read_text()
    assert "xtbloom" not in header.lower()
    assert "gfn2_cpu_execution" not in header
    assert "gfn2_cuda_execution" not in header


def test_method_layer_confines_vendor_abi_to_bridge_implementation() -> None:
    methods = ROOT / "src/methods"
    for path in methods.glob("*"):
        if not path.is_file() or path.name == "gfn2_runtime_bridge.cpp":
            continue
        assert "xtbloom" not in path.read_text().lower(), path


def test_vendor_runtime_dependency_is_confined_to_bridge_implementation() -> None:
    source = (ROOT / "src/methods/gfn2_runtime_bridge.cpp").read_text()
    assert '"runtime/gfn2_cpu_execution.hpp"' in source
    assert "xtbloom::detail::execute_restricted_gfn2_cpu" in source
    assert "xtbloom::detail::Gfn2CpuExecutionCache" in source


def test_bridge_is_part_of_main_library_sources() -> None:
    cmake = (ROOT / "cmake/VibeQCSources.cmake").read_text()
    assert "src/methods/gfn2_runtime_bridge.cpp" in cmake
