"""Source and resource gates for the ppps resident-bra emitter."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

from tools.vibeqc_codegen import emit_ppps_resident_bra_rys3_cuda


def _cuda_source() -> str:
    """Return the resident source with a tiny compile-only Boys stub."""

    boys_stub = """
template <unsigned MaximumOrder>
__device__ __forceinline__ void boys_values(
    double argument, double* values) {
  for (unsigned order = 0; order <= MaximumOrder; ++order) {
    values[order] = 1.0 /
        (2.0 * static_cast<double>(order) + 1.0 + argument);
  }
}
"""
    return boys_stub + emit_ppps_resident_bra_rys3_cuda()


def test_ppps_resident_bra_source_shape_is_complete():
    """Keep the 1110 mapping and force invariants visible in generated CUDA."""

    source = emit_ppps_resident_bra_rys3_cuda()
    assert "struct GeneratedPppsResidentTask" in source
    assert "std::uint32_t bra_pair;" in source
    assert "std::uint32_t ket_begin;" in source
    assert "std::uint32_t ket_count;" in source
    assert "kGeneratedPppsResidentBlockThreads = 256U" in source
    assert "resident_bra_pairs" in source
    assert "resident_bra_pairs[primitive]" in source
    assert "generated_ppps_rys3_roots" in source
    assert source.count("for (unsigned root_index = 0U; root_index < 3U;") == 1
    assert "double component_weight_0 = 0.0;" in source
    assert "double component_weight_26 = 0.0;" in source
    assert "double component_weights[kGeneratedPppsComponentCount]" not in source
    assert "generated_ppps_resident_fill_weights" not in source
    assert "generated_ppps_resident_bra_force_rhf_kernel" in source
    assert "generated_ppps_resident_bra_force_uhf_kernel" in source
    assert "-force_0 - force_3 - force_6" in source
    assert "local_ket += kGeneratedPppsResidentBlockThreads" in source
    assert "resident.ket_count > kGeneratedPppsResidentBlockThreads" not in source
    assert source.index("if (resident.ket_count == 0U) return;") < source.index(
        "primitive_pair_offsets[resident.bra_pair]"
    )
    assert "generated_ppps_shell_class_force_rhf_persistent_kernel" not in source
    assert "task_offset" not in source
    assert "task_head" not in source


def test_ppps_resident_bra_sm120_resource_probe_when_nvcc_is_configured(
    tmp_path: Path,
):
    """Compile both RHF/UHF entries and record the accepted sm_120 footprint."""

    nvcc = os.environ.get("VIBEQC_NVCC")
    if nvcc is None:
        pytest.skip("set VIBEQC_NVCC to run the generated CUDA compile gate")
    architecture = os.environ.get("VIBEQC_CUDA_ARCH", "sm_90")
    source_path = tmp_path / "generated_ppps_resident_rys3.cu"
    cubin_path = tmp_path / "generated_ppps_resident_rys3.cubin"
    source_path.write_text(_cuda_source(), encoding="utf-8")
    result = subprocess.run(
        [
            nvcc,
            "-std=c++17",
            f"-arch={architecture}",
            "-cubin",
            "-Xptxas=-v",
            str(source_path),
            "-o",
            str(cubin_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=240,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    if os.environ.get("VIBEQC_NVCC_VERBOSE"):
        print(output)
    if architecture != "sm_120":
        return
    records = re.findall(
        r"(\d+) bytes stack frame, (\d+) bytes spill stores, "
        r"(\d+) bytes spill loads",
        output,
    )
    assert records, output
    entries = tuple(tuple(map(int, record)) for record in records)
    assert max(entry[0] for entry in entries) <= 512
    assert max(entry[1] for entry in entries) <= 256
    assert max(entry[2] for entry in entries) <= 512
    assert "generated_ppps_resident_bra_force_rhf_kernel" in output
    assert "generated_ppps_resident_bra_force_uhf_kernel" in output
