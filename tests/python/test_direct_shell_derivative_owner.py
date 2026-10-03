"""Guard the prepared Direct shell derivative owner used by DFT stationary forces."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _source(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_generated_exchange_owner_retains_bounded_force_state() -> None:
    header = _source("src/scf/cuda/direct_coulomb.hpp")
    source = _source("src/scf/cuda/direct_coulomb.cpp")
    consumer = _source("src/scf/cuda/direct_jk_kernels.cu")
    for token in (
        "force_capability",
        "bounded_pair_order",
        "shell_pair_block_bounds",
        "force_cursor",
        "bounded_value_capability",
        "bounded_value_overflow",
        "execute_generated_full_range_energy_derivatives",
        "execute_generated_rsh_energy_derivatives",
    ):
        assert token in header
        assert token in source
    assert "launch_bounded_shell_energy_derivative(" in source
    assert "launch_bounded_shell_rsh_derivatives(" in source
    rsh_begin = source.index("cudaError_t execute_generated_rsh_energy_derivatives(")
    rsh_end = source.index("cudaError_t enqueue_generated_coulomb(", rsh_begin)
    rsh_body = source[rsh_begin:rsh_end]
    assert rsh_body.count("launch_bounded_shell_rsh_derivatives(") == 1
    assert rsh_body.count("launch_bounded_shell_range_exchange_derivative(") == 1
    assert "if (omega == 0.3)" in rsh_body
    assert "for (unsigned source" not in rsh_body
    assert "direct_bounded_fallback.hpp" not in source
    assert "launch_bounded_direct_shell_quartet_kernel_scaled(" in consumer
    assert "DirectScreeningPurpose::Force" in consumer


def test_fused_rsh_scratch_budget_matches_owner_allocation() -> None:
    owner = _source("src/scf/cuda/direct_coulomb.cpp")
    capacity = _source("src/scf/cuda/direct_jk.cpp")
    assert "charge(product(atoms, 9), sizeof(double))" in owner
    assert "plan->force = doubles(product(atoms, 9))" in owner
    assert "add(atoms, 9 * sizeof(double))" in capacity


def test_full_range_derivative_shares_one_queue_and_download() -> None:
    """Work reduction is source reuse, not removal of an observable component."""
    source = _source("src/scf/cuda/direct_coulomb.cpp")
    body = source.split(
        "cudaError_t execute_generated_full_range_energy_derivatives(", 1
    )[1].split("cudaError_t execute_generated_rsh_energy_derivatives(", 1)[0]
    assert "std::vector<double> result(2U * coordinates)" in body
    assert body.count("launch_bounded_shell_energy_derivative(") == 1
    assert body.count("cudaMemcpyAsync(") == 1
    assert "for (unsigned source" not in body
    dispatcher = _source("src/scf/cuda/direct_bounded_fallback.cu")
    assert "DirectRangeOperator::FullSources" in dispatcher
    assert (
        "contract_bounded_direct_force_subtile_scaled<Unrestricted, true>" in dispatcher
    )
    contraction = _source("src/scf/cuda/direct_force_quartet.cuh")
    zero_weight_branch = contraction.split(
        "if (coefficient == 0.0 && exchange_weight == 0.0) {", 1
    )[1].split("}", 1)[0]
    assert "return;" in zero_weight_branch
    assert "contracted_eri" not in zero_weight_branch
    assert "SeparateSources ? 2U : 1U" in contraction


def test_retained_direct_plan_prepares_shell_derivative_lease() -> None:
    source = _source("src/scf/cuda/direct_jk.cpp")
    assert "prepare_generated_exchange(" in source
    assert "derivative_order != 0" in source
    assert "const bool bounded_through_f = through_f;" in source
    assert "execute_cuda_direct_shell_full_range_derivatives_device(" in source
    assert "direct_jk_generated_exchange_value_available(*plan, spec)" in source


def test_generated_exchange_value_eligibility_is_request_owned(tmp_path: Path) -> None:
    """Compile the actual dispatch predicates; no CUDA runtime or ERI stand-in executes."""
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a C++ compiler")
    header = _source("src/scf/cuda/direct_jk_plan.hpp")
    helper_start = header.index("inline bool direct_jk_bounded_value_enabled(")
    start = header.index("inline bool direct_jk_generated_exchange_value_available(")
    end = header.index("\n}", start) + 2
    predicate = header[helper_start:end]
    policy_field = next(
        line.strip()
        for line in header.splitlines()
        if "bool bounded_value_opt_in" in line
    )
    start = header.index("struct DirectJkValueDispatch")
    end = header.index("/** Own one exact public-AO provider", start)
    dispatch = header[start:end]
    harness = (
        r"""
#include <cassert>
#include "scf/fock_build.hpp"
namespace generativeqc::scf {
// The GPU storage owner is not constructed. Only presence/capability metadata
// are supplied to the unchanged production predicates extracted below.
struct SharedValueCapability { bool value_capability{}; };
struct GeneratedExchangeValueCapability {
  SharedValueCapability* shared{};
  bool bounded_value_capability{};
};
struct CudaDirectJkPlan {
  GeneratedExchangeValueCapability* generated_exchange{};
  unsigned derivative_order{};
  POLICY_FIELD
};
""".replace("POLICY_FIELD", policy_field)
        + dispatch
        + predicate
        + r"""
}
int main() {
  using namespace generativeqc::scf;
  assert(!CudaDirectJkPlan{}.bounded_value_opt_in);
  for (bool value_capability : {false, true})
    for (bool bounded_value_capability : {false, true})
    for (bool bounded_opt_in : {false, true})
      for (bool available : {false, true})
      for (bool shared_available : {false, true})
        for (unsigned order : {0U, 1U})
        for (bool want_j : {false, true})
          for (bool want_k : {false, true})
            for (auto radial : {FockOperator::FullRange, FockOperator::ShortRange,
                                FockOperator::LongRange})
              for (bool mixed_j : {false, true}) {
                SharedValueCapability shared{value_capability};
                GeneratedExchangeValueCapability exchange{
                    shared_available ? &shared : nullptr, bounded_value_capability};
                CudaDirectJkPlan plan{available ? &exchange : nullptr, 0U, bounded_opt_in};
                FockBuildSpec spec;
                spec.derivative_order = order;
                spec.coulomb.present = want_j;
                spec.exchange.present = want_k;
                spec.exchange.op = radial;
                spec.exchange.omega = radial == FockOperator::FullRange ? 0.0 : 0.37;
                const bool expected_bounded =
                    available && shared_available && bounded_value_capability && bounded_opt_in;
                assert(direct_jk_bounded_value_enabled(plan) == expected_bounded);
                const bool expected_shell =
                    available && shared_available && (value_capability || expected_bounded);
                assert(direct_jk_generated_full_range_value_available(plan) == expected_shell);
                const bool expected = order == 0 && want_k &&
                    (radial == FockOperator::FullRange ? expected_shell : expected_bounded);
                const bool selected = direct_jk_generated_exchange_value_available(plan, spec);
                assert(selected == expected);
                const auto route = direct_jk_value_dispatch(expected_shell, selected, want_j,
                                                            want_k, mixed_j);
                assert(route.generated_exchange == (expected && !mixed_j));
                assert(route.generic_exchange == (want_k && !route.generated_exchange));
                assert(route.generated_coulomb == (want_j && expected_shell && !mixed_j));
                assert(route.generic_coulomb == (want_j && !route.generated_coulomb));
              }
}
"""
    )
    source = tmp_path / "exchange_value_policy.cpp"
    executable = tmp_path / "exchange_value_policy"
    source.write_text(harness)
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(ROOT / "src"),
            "-I",
            str(ROOT / "include"),
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    subprocess.run([str(executable)], check=True, timeout=10)


def test_through_f_values_keep_canonical_and_bounded_sources() -> None:
    direct = _source("src/scf/cuda/direct_jk.cpp")
    owner = _source("src/scf/cuda/direct_coulomb.cpp")
    assert "const bool bounded_through_f = through_f;" in direct
    assert "shell_values_cover_request" in direct
    assert (
        "plan->canonical_pairs && !mixed_j &&\n        !shell_values_cover_request"
        in direct
    )
    assert "enqueue_generated_coulomb(*plan->generated_exchange" in direct
    assert "launch_bounded_shell_fock_source(" in owner
    assert "launch_bounded_shell_range_exchange_source(" in owner
    assert "p.force_cursor, range," in owner
    assert "p.force_cursor, false, true" in owner
    # The diagnostic expression can wrap after '=' when another fallback label
    # is added. Check its policy independently of clang-format's line wrapping.
    normalized = " ".join(direct.split())
    schedule = normalized[
        normalized.index("info.schedule = plan->generated_exchange") :
    ]
    schedule = schedule[: schedule.index("} else if (plan->generated_coulomb)")]
    assert "direct_jk_bounded_value_enabled(*plan)" in schedule
    assert "bounded_value_capability" not in schedule


def test_rsh_value_join_reuses_one_shell_density_preparation() -> None:
    owner = _source("src/scf/cuda/direct_coulomb.cpp")
    begin = owner.index("cudaError_t enqueue_generated_rsh_values(")
    end = owner.index("\n}\n\n}  // namespace generativeqc::scf::cuda_execution", begin)
    body = owner[begin:end]
    assert body.count("prepare_generated_exchange_density(") == 1
    assert "enqueue_generated_coulomb_prepared(" in body
    assert body.count("enqueue_generated_exchange_prepared(") == 2

    facade = _source("src/scf/cuda_fock_execution.cpp")
    assert "enqueue_cuda_direct_rsh_values_device(" in facade

    ks = _source("src/dft/cuda_ks.cpp")
    assert ks.count("enqueue_prepared_cuda_rsh_values(") == 2
    assert "jk_status != GENERATIVEQC_STATUS_NOT_IMPLEMENTED" in ks
    assert "has_range_correction && !fused_rsh_values" in ks


def test_rsh_value_join_preserves_value_policy_and_buffer_contract(
    tmp_path: Path,
) -> None:
    """Compile production span checks and cover every fused writable buffer."""
    direct = _source("src/scf/cuda/direct_jk.cpp")
    begin = direct.index("generativeqc_status enqueue_cuda_direct_rsh_values_device(")
    body = direct[begin : direct.index("\n}\n", begin)]
    assert "!direct_jk_bounded_value_enabled(*plan)" in body
    assert body.index("GENERATIVEQC_STATUS_NOT_IMPLEMENTED") < body.index(
        "return direct_jk_guard("
    )
    assert "correction = direct_jk_strategy(" in body
    check_start = body.index("const auto bytes = direct_jk_product(")
    check_end = body.index("direct_jk_check(cudaMemsetAsync(primary_error")
    checks = body[check_start:check_end]
    assert "for (unsigned i = 0; i < 5; ++i)" in checks
    assert "const double* inputs[]{density, beta};" in checks
    assert (
        "direct_jk_require_disjoint(primary_error, sizeof(int), range_error" in checks
    )
    for status in ("primary_error", "range_error"):
        assert f"direct_jk_require_disjoint(input, bytes, {status}" in checks
        assert f"direct_jk_require_disjoint(outputs[i], bytes, {status}" in checks

    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a C++ compiler")
    start = direct.index("void direct_jk_require_disjoint(")
    helper = direct[start : direct.index("\n}", start) + 2]
    harness = (
        r"""
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
void direct_jk_require(bool condition, const char* message) {
  if (!condition) throw std::invalid_argument(message);
}
"""
        + helper
        + r"""
int main() {
  const auto p = [](std::uintptr_t x) { return reinterpret_cast<const void*>(x); };
  direct_jk_require_disjoint(p(128), 64, p(192), 64);
  direct_jk_require_disjoint(p(192), 64, p(128), 64);
  direct_jk_require_disjoint(nullptr, 64, p(128), 64);
  const auto rejected = [&](std::uintptr_t a, std::size_t na,
                            std::uintptr_t b, std::size_t nb) {
    try { direct_jk_require_disjoint(p(a), na, p(b), nb); }
    catch (const std::invalid_argument&) { return true; }
    return false;
  };
  assert(rejected(128, 64, 128, 64));
  assert(rejected(128, 64, 160, 64));
  assert(rejected(160, 64, 128, 64));
  assert(rejected(128, 64, 144, sizeof(int)));
  assert(rejected(144, sizeof(int), 128, 64));
  assert(rejected(std::numeric_limits<std::uintptr_t>::max() - 7, 8, 128, 64));
  assert(rejected(128, 64, std::numeric_limits<std::uintptr_t>::max() - 7, 8));
}
"""
    )
    source = tmp_path / "rsh_writable_spans.cpp"
    executable = tmp_path / "rsh_writable_spans"
    source.write_text(harness)
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-Wall",
            "-Wextra",
            "-Werror",
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    subprocess.run([str(executable)], check=True, timeout=10)


def test_canonical_screening_fixture_preserves_default_and_opt_in_coverage() -> None:
    """Keep the screened oracle distinct while testing both retained value routes."""
    source = _source("tests/native/test_cuda_fock_provider.cpp")

    def body(name: str) -> str:
        begin = source.index(f"void {name}() {{")
        return source[begin : source.index("\n}\n", begin)]

    screened = body("canonical_screened_values")
    override = "plan->bounded_value_opt_in = false;"
    assert screened.count(override) == 1
    assert screened.index(override) < screened.index("direct_device(")
    assert "!direct_jk_generated_full_range_value_available(*plan)" in screened
    assert "!plan->generated_exchange->shared->value_capability" in screened
    assert "screened_cartesian_public_eri(" in screened
    assert "{0.08, 0.9}" in screened
    assert "work[0] == admitted" in screened
    assert (
        "work[1] == admitted * (operation == FockOperator::FullRange ? 1U : 2U)"
        in screened
    )

    default = body("canonical_value_provider")
    assert default.index("direct_device(") < default.index(override)
    assert "plan->generated_exchange->bounded_value_capability &&" in default
    assert "for (bool bounded_opt_in : {false, true})" in default
    assert "plan->bounded_value_opt_in = bounded_opt_in;" in default
    assert "!plan->bounded_value_opt_in" in default
    assert "plan->bounded_value_opt_in = bounded;" in default
    assert "work[0] == (canonical_route ? quartets : 0U)" in default
    assert "work[1] == radial_passes * quartets" in default
    for fixture in (screened, default):
        assert "FockSpin::Restricted, FockSpin::Unrestricted" in fixture
        for operation in ("FullRange", "ShortRange", "LongRange"):
            assert f"FockOperator::{operation}" in fixture

    entry = source[source.index('std::string(argv[1]) == "--canonical-values-only"') :]
    entry = entry[: entry.index("return 0;")]
    assert "canonical_screened_values();" in entry
    assert "canonical_value_provider();" in entry
    assert "std::abs(actual[i] - expected[i]) < 3e-12" in source


def test_canonical_outer_dispatch_preserves_mixed_source(tmp_path: Path) -> None:
    """Compile the real outer gate, including the competing through-f owner."""
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a C++ compiler")
    source = _source("src/scf/cuda/direct_jk.cpp")
    begin = source.index("if ((spec.coulomb.present || spec.exchange.present)")
    end = source.index(") {", begin)
    condition = source[begin + len("if (") : end]
    harness = r"""
#include <cassert>
struct Term { bool present; };
struct Spec { Term coulomb, exchange; };
struct Plan { bool canonical_pairs; };
bool canonical_route(const Plan* plan, Spec spec, bool mixed_j,
                     bool shell_values_cover_request) {
  return CONDITION;
}
int main() {
  for (unsigned mask = 0; mask < 32; ++mask) {
    const bool canonical = mask & 1U, mixed = mask & 2U;
    const bool shell = mask & 4U, j = mask & 8U, k = mask & 16U;
    const Plan plan{canonical};
    assert(canonical_route(&plan, {{j}, {k}}, mixed, shell) ==
           (canonical && !mixed && !shell && (j || k)));
  }
}
""".replace("CONDITION", condition)
    path = tmp_path / "canonical_value_policy.cpp"
    executable = tmp_path / "canonical_value_policy"
    path.write_text(harness)
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-Wall",
            "-Wextra",
            "-Werror",
            str(path),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    subprocess.run([str(executable)], check=True, timeout=10)


def test_prepared_rsh_uses_shell_sr_lr_scheduler() -> None:
    source = _source("src/scf/cuda_fock_execution.cpp")
    begin = source.index("execute_prepared_cuda_direct_rsh_energy_derivatives_device(")
    end = source.index(
        "execute_prepared_cuda_direct_shell_full_range_derivatives_device(", begin
    )
    body = source[begin:end]
    assert "execute_cuda_direct_shell_rsh_energy_derivatives_device(" in body
    assert "p.exchange.coefficient + c.exchange.coefficient" in body
    assert "c.exchange.omega" in body
    assert "unit_long_range" not in body
    assert "full_range[coordinates + coordinate]" not in body


def test_generic_stationary_cuda_adopts_prepared_full_range_shell_source() -> None:
    methods = _source("src/methods/dft_method.cpp")
    api = _source("src/api/c_api_ks_snapshot.cpp")
    snapshot = _source("python/generativeqc/_ks_snapshot.py")
    stationary = _source("python/generativeqc/_stationary_cuda.py")

    begin = methods.index("cuda_full_range_integral_derivatives(")
    end = methods.index("generativeqc_status cuda_integral_gradient(", begin)
    body = methods[begin:end]
    assert "resident_final_density(" in body
    assert "execute_prepared_cuda_direct_shell_full_range_derivatives_device(" in body
    assert "range_strategy_" in body

    assert "dft_cuda_full_range_integral_derivatives(" in api
    assert "generativeqc_ks_snapshot_cuda_full_range_derivatives_v1(" in api
    assert "def cuda_full_range_derivatives(" in snapshot
    assert "_native.STATUS_NOT_IMPLEMENTED" in snapshot

    assert "full_range_derivative_route=(" in stationary
    assert '"prepared-direct-shell"' in stationary
    assert "if native_shell_full_range" in stationary
    assert "records -= ao_quartet_primitive_records" in stationary
    assert "if native_shell_full_range" in stationary
    assert "task_sources =" in stationary


def test_prepared_one_electron_force_borrows_direct_shell_metadata() -> None:
    direct = _source("src/scf/cuda/direct_jk.cpp")
    generated = _source("src/scf/cuda/direct_coulomb.cpp")
    bridge = _source("src/scf/cuda/one_electron_gradient_bridge.cu")
    method = _source("src/methods/dft_method.cpp")

    assert "F(atomic_numbers)" in direct
    assert "F(shell_ao_offsets)" in generated
    assert "execute_prepared_cuda_stationary_one_electron_pair(" in bridge
    assert "generated_owner ? exchange->shared->batch : source->batch" in bridge
    assert (
        "exchange && exchange->force_capability && exchange->shared && exchange->force"
        in bridge
    )
    assert "constexpr unsigned schedule = 1" in bridge
    assert (
        "auto* output = generated_owner ? exchange->force : source->derivative"
        in bridge
    )
    assert 'trace_counter("host_to_device_bytes", 0)' in bridge
    assert "execute_prepared_cuda_stationary_one_electron_pair(" in method


def test_generic_stationary_cuda_reuses_complete_prepared_integral_sources() -> None:
    methods = _source("src/methods/dft_method.cpp")
    api = _source("src/api/c_api_ks_snapshot.cpp")
    snapshot = _source("python/generativeqc/_ks_snapshot.py")
    stationary = _source("python/generativeqc/_stationary_cuda.py")

    begin = methods.index("generativeqc_status cuda_integral_gradient(")
    end = methods.index("Result execute(bool compute_forces)", begin)
    body = methods[begin:end]
    assert "const bool range_exchange = execution_plan_.range_exchange;" in body
    assert "candidate.reserve((range_exchange ? 5 : 4) * nc)" in body
    assert "execute_prepared_cuda_stationary_one_electron_pair(" in body
    assert "execute_prepared_cuda_direct_rsh_energy_derivatives_device(" in body
    assert "execute_prepared_cuda_direct_shell_full_range_derivatives_device(" in body
    assert "SemilocalFamily::" not in body

    assert "snapshot->token.identity.model.range_correction ? 5U : 4U" in api
    assert "def cuda_integral_derivatives(" in snapshot
    assert '"generativeqc_ks_snapshot_cuda_integral_gradient_v1"' in snapshot

    assert "stationary_integral_derivative_route=(" in stationary
    assert '"prepared-native-complete"' in stationary
    assert "sources.reset_geometry(spec.coincident_tolerance)" in stationary
    assert "ao.set_density_device(" in stationary
    assert "if native_complete_integrals" in stationary
    assert "records -= ao_integral_primitive_records" in stationary
    assert '"one_electron_device_peak_bytes"' in stationary
    assert "additional_device_peak_bound=peak" in stationary
