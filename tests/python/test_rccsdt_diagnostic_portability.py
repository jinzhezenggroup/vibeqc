"""Public uint64 byte diagnostics need not share the platform size_t typedef."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("diagnostic_type", ("unsigned long", "unsigned long long"))
def test_triples_capacity_maximum_accepts_distinct_unsigned_types(
    tmp_path: Path, diagnostic_type: str
) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    source = (ROOT / "src/methods/rccsdt_method.cpp").read_text()
    statement = re.search(
        r"diagnostic\.numeric_capacity_bytes\s*=\s*std::max.*?;", source, re.DOTALL
    )
    assert statement is not None
    harness = f"""
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
std::size_t checked_add(std::size_t a, std::size_t b) {{
  if (b > std::numeric_limits<std::size_t>::max() - a) throw std::length_error("overflow");
  return a+b;
}}
int main() {{
  struct {{ {diagnostic_type} numeric_capacity_bytes; }} diagnostic{{}};
  for (std::uint64_t old : {{0ULL, 0x100000010ULL}})
    for (std::size_t retained : {{std::size_t(0), std::size_t(0x100000020ULL)}}) {{
      const std::size_t triples_workspace_bytes=31;
      diagnostic.numeric_capacity_bytes=old;
      {statement.group(0)}
      const auto expected=old > retained+31 ? old : retained+31;
      if (diagnostic.numeric_capacity_bytes!=expected) return 1;
    }}
}}
"""
    path, executable = tmp_path / "capacity.cpp", tmp_path / "capacity"
    path.write_text(harness)
    subprocess.run(
        [compiler, "-std=c++20", "-Wall", "-Werror", str(path), "-o", str(executable)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    subprocess.run([str(executable)], check=True, timeout=10)
