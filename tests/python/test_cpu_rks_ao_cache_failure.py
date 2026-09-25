"""Execute the RKS cache admission region with explicit allocation failures."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def cache_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a host C++ compiler, not a GPU")
    source = (ROOT / "src/dft/rks.cpp").read_text(encoding="utf-8")
    start = source.index("  constexpr std::size_t kCpuRksAoCacheMaximumBytes")
    stop = source.index("  std::shared_ptr<const OccupiedDensityFactor> factor;", start)
    region = source[start:stop]
    directory = tmp_path_factory.mktemp("rks-ao-cache-failures")
    program = (
        r"""
#include <cstddef>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
int mode, preparations=0, queries=0;
namespace dft {
struct RksAoCache { int value=17; };
std::size_t rks_ao_cache_bytes(int, int, unsigned) {
  ++queries;
  if (mode==4) throw std::invalid_argument("size");
  return (64ULL + (mode==5 ? 1 : 0))*1024*1024;
}
RksAoCache prepare_rks_ao_cache(int, int, unsigned) {
  ++preparations;
  if (mode==1) throw std::bad_alloc();
  if (mode==2) throw std::bad_array_new_length();
  if (mode==3) throw std::runtime_error("collocation");
  return {};
}
}
bool select() {
  const bool incremental_xc = mode==6;
  const struct { bool cached_direct; } evaluate_xc{mode!=7};
  const int basis=0, grid=0;
  const struct { unsigned ao_order; } ks{1};
"""
        + region
        + r"""
  if (ao_cache && ao_cache->value != 17) throw std::logic_error("changed cache");
  return ao_cache.has_value();
}
int main(int argc, char** argv) {
  if (argc!=2) return 9;
  mode=std::stoi(argv[1]);
  try {
    const bool cached=select();
    if (mode==3 || mode==4) return 1;
    if (cached != (mode==0)) return 2;
    if (preparations != (mode<=2 ? 1 : 0)) return 3;
    if (queries != (mode==6 || mode==7 ? 0 : 1)) return 4;
  } catch (const std::bad_alloc&) { return 5; }
    catch (const std::invalid_argument& e) {
      if (mode!=4 || std::string(e.what())!="size") return 6;
    } catch (const std::runtime_error& e) {
      if (mode!=3 || std::string(e.what())!="collocation") return 7;
    }
}
"""
    )
    path, executable = directory / "probe.cpp", directory / "probe"
    path.write_text(program, encoding="utf-8")
    subprocess.run(
        [compiler, "-std=c++17", "-O2", str(path), "-o", str(executable)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return executable


@pytest.mark.parametrize("mode", range(8))
def test_optional_cache_failure_keeps_streaming_available(
    cache_probe: Path, mode: int
) -> None:
    subprocess.run([str(cache_probe), str(mode)], check=True, timeout=5)
