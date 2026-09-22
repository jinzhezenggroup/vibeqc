import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_public_xtb_method_uses_only_the_native_bridge() -> None:
    source = (ROOT / "src/methods/xtb_method.cpp").read_text()
    assert "xtbloom" not in source.lower()
    assert '"runtime/gfn2_cpu_execution.hpp"' not in source
    assert '"runtime/gfn2_cuda_execution.hpp"' not in source
    assert '"methods/gfn2_runtime_bridge.hpp"' in source


def test_native_bridge_header_does_not_leak_execution_descriptors() -> None:
    header = (ROOT / "src/methods/gfn2_runtime_bridge.hpp").read_text()
    assert "xtbloom" not in header.lower()
    assert "vibeqc_xtb_" not in header
    assert "gfn2_cpu_execution" not in header
    assert "gfn2_cuda_execution" not in header


def test_method_layer_has_no_vendor_abi() -> None:
    methods = ROOT / "src/methods"
    for path in methods.glob("*"):
        if not path.is_file():
            continue
        assert "xtbloom" not in path.read_text().lower(), path


def test_native_execution_is_confined_to_bridge_implementation() -> None:
    source = (ROOT / "src/methods/gfn2_runtime_bridge.cpp").read_text()
    assert '"runtime/gfn2_cpu_execution.hpp"' in source
    assert "vibeqc::xtb::detail::execute_restricted_gfn2_cpu" in source
    assert "vibeqc::xtb::detail::Gfn2CpuExecutionCache" in source


def test_retired_runtime_and_external_api_cannot_reenter_production() -> None:
    assert not (ROOT / "src/xtb/gfn2_runtime").exists()
    native = ROOT / "src/xtb/native"
    assert not (native / "include/xtbloom").exists()
    for path in native.rglob("*"):
        if path.suffix not in (".cpp", ".hpp", ".cu", ".cuh", ".h", ".c"):
            continue
        # Copyright and upstream evidence remain; executable identity does not.
        code = re.sub(r"/\*.*?\*/|//[^\n]*", "", path.read_text(), flags=re.DOTALL)
        assert not re.search(r"\b(?:xtbloom_|XTBLOOM_|xtbloom::)", code), path
    descriptors = (native / "src/runtime/types.hpp").read_text()
    assert 'extern "C"' not in descriptors
    assert "#define VIBEQC_XTB_API " not in descriptors
    for unused in (
        "context_options_t",
        "request_info_t",
        "workspace_query_t",
        "dlpack_view_t",
    ):
        assert unused not in descriptors
    for retired in (
        "request.hpp",
        "result_owner.hpp",
        "result_owner_cuda.cu",
        "dlpack_layout.hpp",
    ):
        assert not (native / "src/runtime" / retired).exists()
    assert "struct Context {" not in (native / "src/runtime/backend.hpp").read_text()


def test_bridge_is_part_of_main_library_sources() -> None:
    cmake = (ROOT / "cmake/VibeQCSources.cmake").read_text()
    assert "src/methods/gfn2_runtime_bridge.cpp" in cmake


def test_retired_feature_owners_are_absent() -> None:
    native = ROOT / "src/xtb/native/src"
    for owner in (
        "alpb",
        "lattice",
        "periodic_topology",
        "periodic_integrals",
        "periodic_ewald",
        "periodic_multipole",
    ):
        assert not (native / f"model/gfn2/{owner}.cpp").exists()
        assert not (native / f"model/gfn2/{owner}.hpp").exists()
    source = (native / "runtime/gfn2_cuda_execution.cu").read_text()
    for retired in (
        "RequestExecutionGraphOwner",
        "ActiveRequest",
        "Gfn2CudaExecutionIdentity",
        "enqueue_restricted_gfn2_cuda",
        "prepare_topology_only",
        "prepare_host",
    ):
        assert retired not in source
    # SCC graph capture and its bounded fallback still serve the real endpoint.
    assert "Gfn2SccLoopCudaGraphOwner" in source


def test_retired_descriptors_fail_before_staging(tmp_path: Path) -> None:
    """Exercise rejection using deliberately unreadable retired input buffers."""
    import shutil
    import subprocess

    import pytest

    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler is required for the private admission contract")
    source = tmp_path / "admission.cpp"
    source.write_text(r"""
#include "runtime/molecular_request.hpp"
#include <cassert>
int main() {
  vibeqc_xtb_batch_t batch{};
  vibeqc_xtb_compute_options_t options{};
  batch.batch_size = 1;
  options.model = VIBEQC_XTB_MODEL_GFN2_XTB;
  options.flags = VIBEQC_XTB_COMPUTE_ENERGY | VIBEQC_XTB_COMPUTE_FORCES;
  options.scc_start_mode = VIBEQC_XTB_SCC_START_FRESH;
  std::string error;
  const auto validate = [&] {
    return vibeqc::xtb::detail::validate_molecular_request(batch, options, error);
  };
  assert(validate() == VIBEQC_XTB_STATUS_SUCCESS);
  const auto original = batch;
  const vibeqc_xtb_const_buffer_t poison{reinterpret_cast<void*>(1), 8,
                                       VIBEQC_XTB_MEMORY_HOST, 0};
  for (auto member : {&vibeqc_xtb_batch_t::point_charge_offsets,
                      &vibeqc_xtb_batch_t::point_charge_positions,
                      &vibeqc_xtb_batch_t::point_charge_values,
                      &vibeqc_xtb_batch_t::point_charge_gammas,
                      &vibeqc_xtb_batch_t::charge_response_offsets,
                      &vibeqc_xtb_batch_t::cell_matrices,
                      &vibeqc_xtb_batch_t::periodic_axes,
                      &vibeqc_xtb_batch_t::atomic_potential_shifts,
                      &vibeqc_xtb_batch_t::charge_response_matrix,
                      &vibeqc_xtb_batch_t::interaction_descriptors,
                      &vibeqc_xtb_batch_t::interaction_payload}) {
    batch.*member = poison;
    assert(validate() == VIBEQC_XTB_STATUS_NOT_SUPPORTED);
    batch = original;
  }
  for (auto member : {&vibeqc_xtb_batch_t::total_point_charges,
                      &vibeqc_xtb_batch_t::total_charge_response_elements,
                      &vibeqc_xtb_batch_t::total_interactions}) {
    batch.*member = 1;
    assert(validate() == VIBEQC_XTB_STATUS_NOT_SUPPORTED);
    batch = original;
  }
  batch.batch_size = 2;
  assert(validate() == VIBEQC_XTB_STATUS_NOT_SUPPORTED);
  batch = original;
  options.scc_start_mode = VIBEQC_XTB_SCC_START_WARM;
  assert(validate() == VIBEQC_XTB_STATUS_NOT_SUPPORTED);
  options.scc_start_mode = VIBEQC_XTB_SCC_START_FRESH;
  options.flags |= VIBEQC_XTB_COMPUTE_STRAIN_DERIVATIVES;
  assert(validate() == VIBEQC_XTB_STATUS_NOT_SUPPORTED);
}
""")
    executable = tmp_path / "admission"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-I",
            str(ROOT / "src/xtb/native/src"),
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
    )
    subprocess.run([str(executable)], check=True)
