"""Public uint64 byte diagnostics need not share the platform size_t typedef."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("diagnostic_type", ("unsigned long", "unsigned long long"))
def test_df_source_device_capacity_accepts_distinct_unsigned_types(
    tmp_path: Path, diagnostic_type: str
) -> None:
    """Compile the DF owner assignment with both LP64 uint64_t conventions."""
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    source = (ROOT / "src/methods/rccsd_method.cpp").read_text()
    statement = re.search(
        r"metrics\.owned_device_bytes\s*=\s*std::max.*?;", source, re.DOTALL
    )
    assert statement is not None
    path, executable = tmp_path / "df_capacity.cpp", tmp_path / "df_capacity"
    path.write_text(f"""
#include <algorithm>
#include <cstddef>
#include <cstdint>
int main() {{
  struct {{ {diagnostic_type} owned_device_bytes; }} metrics{{}};
  struct {{ std::size_t device_capacity_bytes; }} fitted{{}};
  for (std::uint64_t old : {{0ULL, 0x100000030ULL}})
    for (std::size_t next : {{std::size_t(0), std::size_t(0x100000050ULL)}}) {{
      metrics.owned_device_bytes=old;
      fitted.device_capacity_bytes=next;
      {statement.group(0)}
      if (metrics.owned_device_bytes != (old > next ? old : next)) return 1;
    }}
}}
""")
    subprocess.run(
        [compiler, "-std=c++20", "-Wall", "-Werror", str(path), "-o", str(executable)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    subprocess.run([str(executable)], check=True, timeout=10)


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
  auto update_capacity = [&](std::size_t retained, std::size_t triples_workspace_bytes,
                             std::size_t warm_reservation) {{
    struct {{ std::size_t external_reservation_bytes; }} state{{warm_reservation}};
    {statement.group(0)}
  }};
  for (std::uint64_t old : {{0ULL, 0x100000010ULL}})
    for (std::size_t retained : {{std::size_t(0), std::size_t(0x100000020ULL)}})
      for (std::size_t warm : {{std::size_t(0), std::size_t(0x100000030ULL)}}) {{
        diagnostic.numeric_capacity_bytes=old;
        update_capacity(retained, 31, warm);
        const auto phase_capacity=retained+31+warm;
        const auto expected=old > phase_capacity ? old : phase_capacity;
        if (diagnostic.numeric_capacity_bytes!=expected) return 1;
      }}
  const auto limit=std::numeric_limits<std::size_t>::max();
  for (bool overflow_in_warm_add : {{false, true}}) {{
    diagnostic.numeric_capacity_bytes=17;
    try {{
      update_capacity(overflow_in_warm_add ? limit-31 : limit, 31,
                      overflow_in_warm_add ? 1 : 0);
      return 2;
    }} catch (const std::length_error&) {{
      if (diagnostic.numeric_capacity_bytes!=17) return 3;
    }}
  }}
}}
"""
    path, executable = tmp_path / "capacity.cpp", tmp_path / "capacity"
    path.write_text(harness)
    compiled = subprocess.run(
        [compiler, "-std=c++20", "-Wall", "-Werror", str(path), "-o", str(executable)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert compiled.returncode == 0, compiled.stderr
    subprocess.run([str(executable)], check=True, timeout=10)
