"""An exhausted automatic envelope is infeasible, not a new zero/default request."""

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def budget_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    directory = tmp_path_factory.mktemp("empty-df-budget")
    source = directory / "probe.cpp"
    source.write_text(
        '#include "scf/df_preparation_budget.hpp"\n'
        "#include <cstdlib>\n#include <iostream>\n"
        "int main(int argc, char** argv) {\n"
        "  if (argc != 4) return 2;\n"
        "  using namespace vibeqc::scf;\n"
        "  const auto free = std::strtoull(argv[1], nullptr, 10);\n"
        "  const bool force = std::strtoul(argv[2], nullptr, 10);\n"
        "  const auto requested = std::strtoull(argv[3], nullptr, 10);\n"
        "  const auto r = resolve_df_budget({24, 60, 5, 2, 6, force},\n"
        "      {free, 8ULL << 30, true}, requested);\n"
        '  std::cout << r.feasible << " " << r.total_bytes << " "\n'
        '            << r.value_bytes << " " << r.response_bytes;\n'
        "}\n"
    )
    executable = directory / "probe"
    root = Path(__file__).resolve().parents[2]
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I" + str(root / "src"),
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
    )
    return executable


@pytest.mark.parametrize("free", [0, 1])
@pytest.mark.parametrize("force", [False, True])
def test_exhausted_automatic_envelope(
    budget_probe: Path, free: int, force: bool
) -> None:
    result = subprocess.check_output(
        [str(budget_probe), str(free), str(int(force)), "0"], text=True
    )
    assert [int(value) for value in result.split()] == [0, 0, 0, 0]


@pytest.mark.parametrize("force", [False, True])
def test_explicit_cap_is_not_changed_by_probe(budget_probe: Path, force: bool) -> None:
    result = subprocess.check_output(
        [str(budget_probe), "0", str(int(force)), "17"], text=True
    )
    feasible, total, value, response = [int(part) for part in result.split()]
    assert feasible == 1 and total == value + response == 17
    assert value > 0
    assert (response > 0) == force


def test_roomy_automatic_budget_retains_source_backed_device_value_floor(
    tmp_path: Path,
) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    source = tmp_path / "automatic.cpp"
    source.write_text(r"""
#include "scf/df_preparation_budget.hpp"
#include <cassert>
int main() {
  using namespace vibeqc::scf;
  constexpr std::size_t mib = 1024U * 1024U;
  const DfBudgetWorkload work{64, 64, 8, 1, 8, true};
  const DfResourceEnvelope roomy{8ULL << 30, 8ULL << 30, true};
  const auto automatic = resolve_df_budget(work, roomy, 0);
  const auto resident_floor = df_resident_value_admission_floor(work);
  assert(automatic.value_bytes >= resident_floor);
  const auto explicit_cap = resolve_df_budget(work, roomy, 12345);
  assert(explicit_cap.total_bytes == 12345);
  const DfResourceEnvelope tight{256 * mib, 8ULL << 30, true};
  const auto limited = resolve_df_budget(work, tight, 0);
  assert(limited.total_bytes <= automatic.total_bytes);
}
""")
    root = Path(__file__).resolve().parents[2]
    executable = tmp_path / "automatic"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-Wall",
            "-Wextra",
            "-Werror",
            f"-I{root / 'src'}",
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
    )
    subprocess.run([str(executable)], check=True)


def test_resolved_subbudget_preserves_origin_and_cannot_reopen_auto(
    tmp_path: Path,
) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    source = tmp_path / "subbudget.cpp"
    source.write_text(r"""
#include "scf/df_preparation_budget.hpp"
#include <cassert>
int main() {
  using namespace vibeqc::scf;
  for (bool force : {false, true}) {
    const DfBudgetWorkload work{64,96,12,1,6,force};
    for (std::size_t request : {std::size_t{0},std::size_t{10000}}) {
      const auto parent=resolve_df_budget(work,{8ULL<<30,16ULL<<30,true},request);
      const auto child=resolve_df_subbudget(work,parent,100);
      assert(child.feasible && child.total_bytes==parent.total_bytes-100);
      assert(child.value_bytes+child.response_bytes==child.total_bytes);
      assert(child.requested_bytes==parent.requested_bytes);
      assert(child.reserved_headroom_bytes==parent.reserved_headroom_bytes);
      assert(child.live_resource && child.observed_free_bytes==parent.observed_free_bytes);
      for (auto retained : {parent.total_bytes, parent.total_bytes+1}) {
        const auto empty=resolve_df_subbudget(work,parent,retained);
        assert(!empty.feasible && !empty.total_bytes && !empty.value_bytes && !empty.response_bytes);
      }
    }
    const auto empty=resolve_df_budget(work,{0,16ULL<<30,true},0);
    assert(!resolve_df_subbudget(work,empty,0).feasible);
  }
}
""")
    root = Path(__file__).resolve().parents[2]
    executable = tmp_path / "subbudget"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-Wall",
            "-Wextra",
            "-Werror",
            f"-I{root / 'src'}",
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
    )
    subprocess.run([str(executable)], check=True)
    production = (root / "src/scf/fock_prepared.cpp").read_text()
    assert (
        "resolve_df_subbudget(df_workload, resolved_df, diagnostic.device_bytes)"
        in production
    )
