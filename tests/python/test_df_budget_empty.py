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
