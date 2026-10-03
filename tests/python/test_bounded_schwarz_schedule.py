"""Compare the actual compressed block domain with exhaustive physical pairs."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_compressed_schwarz_domain_preserves_every_geometric_pair(
    tmp_path: Path,
) -> None:
    """Exercise ties, empty systems/rows, tails, zero gates and IEEE extremes."""
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    source = (ROOT / "src/scf/cuda/direct_queue_index.cuh").read_text()
    start = source.index("template <class Offset>")
    end = source.index("\n}", start) + 2
    decoder = source[start:end].replace("__device__", "")
    program = tmp_path / "schedule.cpp"
    program.write_text(
        '#include "scf/direct_block_schedule.hpp"\n'
        "#include <set>\n#include <random>\n#include <iostream>\n"
        + decoder
        + r"""
using namespace generativeqc::scf::detail;

void check(const std::vector<std::int64_t>& offsets,
           const std::vector<double>& bounds, double screening) {
  const auto schedule = make_bounded_direct_schedule(offsets, bounds, screening);
  std::vector<std::int64_t> block_offsets{0};
  std::set<std::uint64_t> expected, actual;
  for (std::size_t system = 0; system + 1 < offsets.size(); ++system) {
    const auto begin = static_cast<std::size_t>(offsets[system]);
    const auto end = static_cast<std::size_t>(offsets[system + 1]);
    block_offsets.push_back(block_offsets.back() +
        (end - begin + kBoundedDirectShellPairBlockSize - 1) / kBoundedDirectShellPairBlockSize);
    for (std::size_t first = begin; first < end; ++first) {
      const auto ordered = schedule.pair_order[first];
      if (ordered < begin || ordered >= end) throw std::runtime_error("cross-system order");
      if (first > begin) {
        const auto prior = schedule.pair_order[first - 1];
        if (bounds[prior] < bounds[ordered] ||
            (bounds[prior] == bounds[ordered] && prior > ordered))
          throw std::runtime_error("unstable descending order");
      }
      for (std::size_t second = begin; second <= first; ++second)
        if (!(bounds[first] * bounds[second] < screening))
          expected.insert((static_cast<std::uint64_t>(first) << 32U) | second);
    }
  }
  if (schedule.block_prefix.size() != block_offsets.back() + 1)
    throw std::runtime_error("row storage not linear in pair blocks");
  const auto rows = schedule.block_prefix.size() - 1;
  for (std::uint64_t ordinal = 0; ordinal < schedule.block_prefix.back(); ++ordinal) {
    const auto row = bounded_direct_block_row(schedule.block_prefix.data(), rows, ordinal);
    if (!(schedule.block_prefix[row] <= ordinal && ordinal < schedule.block_prefix[row + 1]))
      throw std::runtime_error("incorrect empty-row decoding");
    const auto system = bounded_direct_block_row(block_offsets.data(), block_offsets.size() - 1, row);
    const auto expected_system = std::upper_bound(block_offsets.begin(), block_offsets.end(), row)
                                 - block_offsets.begin() - 1;
    if (system != static_cast<std::size_t>(expected_system))
      throw std::runtime_error("incorrect empty-system decoding");
    const auto local_row = row - block_offsets[system];
    const auto column = ordinal - schedule.block_prefix[row];
    if (column > local_row) throw std::runtime_error("nontriangular prefix");
    const auto first_begin = offsets[system] + local_row * kBoundedDirectShellPairBlockSize;
    const auto second_begin = offsets[system] + column * kBoundedDirectShellPairBlockSize;
    const auto first_end = std::min<std::size_t>(offsets[system + 1], first_begin + kBoundedDirectShellPairBlockSize);
    const auto second_end = std::min<std::size_t>(offsets[system + 1], second_begin + kBoundedDirectShellPairBlockSize);
    for (auto first_index = first_begin; first_index < first_end; ++first_index)
      for (auto second_index = second_begin; second_index < second_end; ++second_index) {
        if (local_row == column && second_index > first_index) continue;
        const auto first = schedule.pair_order[first_index];
        const auto second = schedule.pair_order[second_index];
        if (bounds[first] * bounds[second] < screening) continue;
        const auto key = (static_cast<std::uint64_t>(std::max(first, second)) << 32U)
                         | std::min(first, second);
        if (!actual.insert(key).second) throw std::runtime_error("duplicate physical pair");
      }
  }
  if (actual != expected) throw std::runtime_error("changed geometry domain");
}

int main() {
  std::mt19937_64 random(20261003);
  std::size_t cases = 0;
  for (std::size_t trial = 0; trial < 80; ++trial) {
    std::vector<std::int64_t> offsets{0};
    std::vector<double> bounds;
    for (std::size_t system = 0; system < 5; ++system) {
      const std::size_t count = system == 2 ? 0 : random() % 70;
      for (std::size_t index = 0; index < count; ++index)
        bounds.push_back(std::pow(10.0, -static_cast<double>(random() % 28)));
      offsets.push_back(bounds.size());
    }
    for (double screening : {0.0, 1e-14, 1e-12, 0.1, 1.0, 2.0}) {
      check(offsets, bounds, screening);
      ++cases;
    }
  }
  check({0, 0, 0}, {}, 0.0);
  for (std::size_t count : {31U, 32U, 33U, 64U, 65U, 129U}) {
    std::vector<double> bounds(count);
    for (std::size_t index = 0; index < count; ++index)
      bounds[index] = index % 2 ? 0.0 : std::numeric_limits<double>::max();
    check({0, static_cast<std::int64_t>(count)}, bounds, 1e-12);
    check({0, static_cast<std::int64_t>(count)}, bounds, 0.0);
  }
  check({0, 4}, {1.0, std::nextafter(1.0, 0.0), std::nextafter(1.0, 2.0), 1e-200}, 1.0);
  for (double invalid : {-1.0, std::numeric_limits<double>::infinity(),
                          std::numeric_limits<double>::quiet_NaN()}) {
    bool rejected = false;
    try { (void)make_bounded_direct_schedule({0, 1}, {invalid}, 0.0); }
    catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected) throw std::runtime_error("invalid bound accepted");
  }
  std::cout << cases << " randomized geometry domains match\n";
}
"""
    )
    object_file = tmp_path / "schedule.o"
    executable = tmp_path / "schedule"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O2",
            "-fno-fast-math",
            "-I",
            str(ROOT / "src"),
            "-c",
            str(program),
            "-o",
            str(object_file),
        ],
        check=True,
        timeout=60,
    )
    subprocess.run(
        [compiler, str(object_file), "-o", str(executable)], check=True, timeout=60
    )
    result = subprocess.run(
        [str(executable)], check=True, capture_output=True, text=True, timeout=30
    )
    assert "480 randomized geometry domains match" in result.stdout


def test_actual_candidate_page_partition_preserves_tails(tmp_path: Path) -> None:
    """Compile the production claim/page arithmetic, including empty tail pages."""
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    source = (ROOT / "src/scf/cuda/direct_bounded_fallback.cu").read_text()
    start = source.index("  constexpr auto full_block_candidates")
    preparation = source[start : source.index("\n\n  while (true)", start)]
    claim = next(
        line
        for line in source.splitlines()
        if "const std::size_t packed_block_quartet" in line
    )
    start = source.index("    const std::size_t page_begin =")
    page = source[start : source.index("    for (std::size_t candidate_begin", start)]
    program = tmp_path / "pages.cpp"
    program.write_text(
        '#include "scf/direct_block_domain.hpp"\n'
        '#include "scf/direct_task_layout.hpp"\n'
        "#include <algorithm>\n#include <cassert>\n#include <vector>\n"
        "namespace detail = generativeqc::scf::detail;\n"
        "using std::min;\nint main() {\n"
        "for (bool indexed : {false, true}) {\n"
        "std::vector<std::size_t> counts{0, 1, 17, 32, 63, 64, 65, 255, 256, 257, 528, 1024};\n"
        "std::uint64_t prefix = 0;\n"
        "detail::BoundedDirectBlockDomain block_domain{indexed ? &prefix : nullptr, counts.size(), counts.size()};\n"
        "struct {std::size_t total_shell_pair_block_quartets;} batch{counts.size()};\n"
        + preparation
        + "\nstd::vector<std::vector<unsigned>> visits;\n"
        "for (auto count : counts) visits.emplace_back(count, 0);\n"
        "for (std::uint64_t block_quartet = 0; block_quartet < total; ++block_quartet) {\n"
        + claim
        + "\nassert(packed_block_quartet < counts.size());\n"
        "const auto candidate_count = counts[packed_block_quartet];\n"
        + page
        + "\nfor (auto candidate = page_begin; candidate < page_end; ++candidate)\n"
        "  ++visits[packed_block_quartet][candidate];\n"
        "}\nfor (const auto& row : visits) for (auto count : row) assert(count == 1);\n"
        "}\n}\n"
    )
    executable = tmp_path / "pages"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-I",
            str(ROOT / "src"),
            str(program),
            "-o",
            str(executable),
        ],
        check=True,
        timeout=30,
    )
    subprocess.run([str(executable)], check=True, timeout=10)


def test_schedule_policy_requires_explicit_opt_in(tmp_path: Path) -> None:
    """Compile the real selector so an absent override cannot promote the route."""
    compiler = shutil.which("c++")
    cache = shutil.which("ccache")
    if compiler is None or cache is None:
        pytest.skip("host C++ compiler and ccache required")
    source = (ROOT / "src/scf/cuda/rhf_policy.cpp").read_text()
    functions = []
    for declaration in (
        "bool selected(",
        "bool bounded_schwarz_schedule_requested(",
    ):
        start = source.index(declaration)
        functions.append(source[start : source.index("\n}", start) + 2])
    program = tmp_path / "policy.cpp"
    program.write_text(
        "#include <cassert>\n#include <cstdlib>\n#include <cstring>\n"
        + "\n".join(functions)
        + r"""
int main() {
  constexpr auto variable = "GENERATIVEQC_BOUNDED_SCHWARZ_SCHEDULE";
  unsetenv(variable);
  assert(!bounded_schwarz_schedule_requested());
  const char* values[] = {"", "0", "none", "auto", "invalid", "1", "indexed"};
  for (const auto* value : values) {
    setenv(variable, value, 1);
    const bool expected = std::strcmp(value, "1") == 0 || std::strcmp(value, "indexed") == 0;
    assert(bounded_schwarz_schedule_requested() == expected);
  }
}
"""
    )
    executable = tmp_path / "policy"
    subprocess.run(
        [cache, compiler, "-std=c++20", str(program), "-o", str(executable)],
        check=True,
        timeout=30,
    )
    subprocess.run([str(executable)], check=True, timeout=10)
