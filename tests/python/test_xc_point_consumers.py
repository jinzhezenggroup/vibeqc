"""Execute the emitted bounded point selector without loading CUDA or the runtime."""

import shutil
import subprocess
from pathlib import Path

import pytest
from vibeqc_compiler.dft.ao_cuda import emit_native_xc_point_dispatch

from tools.generate_xc_split_hybrid_registry import emit_registry


def test_admitted_point_consumers(tmp_path: Path) -> None:
    """Admitted legacy and split keys resolve; unsupported keys must fail.

    Stub launchers record template arguments, so this tests the emitted host
    dispatch itself without duplicating its conditional implementation in Python.
    Generated MGGA codes share a launcher keyed by ingredient shape; the
    functional code remains separate layout data rather than a template key.
    Spin, AO precision and point counts are deliberately absent from the AOT key:
    they remain validated data/layout arguments rather than extra code variants.
    """
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    (tmp_path / "generated_split_hybrid_registry.cuh").write_text(
        emit_registry(), encoding="utf-8"
    )
    source = tmp_path / "dispatch.cpp"
    source.write_text(
        "#include <cstdint>\n#include <stdexcept>\n#include <limits>\n"
        '#include "generated_split_hybrid_registry.cuh"\n'
        "namespace generated = vibeqc::dft::generated;\n"
        "using CudaXcPointLauncher = void (*)();\n"
        "unsigned selected_functional; bool selected_response;\n"
        "template <unsigned F, bool R> void launch_points() {\n"
        "  selected_functional = F; selected_response = R;\n}\n"
        "template <unsigned Mask> void launch_split_hybrid_points() {\n"
        "  selected_functional = Mask; selected_response = false;\n}\n"
        + emit_native_xc_point_dispatch()
        + r"""
int main() {
  CudaXcPointLauncher entries[7]{};
  unsigned count = 0;
  for (unsigned f = 0; f < 6; ++f) {
    for (unsigned r = 0; r < 2; ++r) {
      const bool admitted = r ? f < 2 : f < 5;
      try {
        auto launch = resolve_point_launcher(f, r);
        if (!admitted || !launch) return 1;
        launch();
        if (selected_functional != f || selected_response != bool(r)) return 2;
        for (unsigned i = 0; i < count; ++i)
          if (entries[i] == launch) return 3;
        entries[count++] = launch;
      } catch (const std::invalid_argument&) {
        if (admitted) return 4;
      }
    }
  }
  if (count != 7) return 5;
  const std::uint32_t generated_codes[]{generated::kM062XFunctionalCode,
                                        generated::kMN15FunctionalCode};
  for (const auto code : generated_codes) {
    auto launch = resolve_point_launcher(code, false);
    if (launch != &launch_split_hybrid_points<5>) return 7;
    launch();
    if (selected_functional != 5 || selected_response) return 8;
    try {
      resolve_point_launcher(code, true);
      return 9;
    } catch (const std::invalid_argument&) {}
  }
  try {
    resolve_point_launcher(std::numeric_limits<std::uint32_t>::max(), false);
    return 6;
  } catch (const std::invalid_argument&) {}
}
"""
    )
    binary = tmp_path / "dispatch"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-I",
            str(tmp_path),
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
    )
    subprocess.run([str(binary)], check=True)
