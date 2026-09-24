"""Keep routine feedback bounded without truncating full qualification jobs."""

import re
import typing
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "filename, job, routine_minutes",
    [("ci.yml", "python", 30), ("cumetal-cuda.yml", "cuda-tests", 30)],
)
def test_full_qualification_has_a_separate_finite_budget(
    filename: typing.Any, job: typing.Any, routine_minutes: typing.Any
) -> None:
    path = Path(__file__).resolve().parents[2] / ".github/workflows" / filename
    section = path.read_text().split(f"\n  {job}:\n", 1)[1]
    match = re.search(
        r"timeout-minutes: \$\{\{ \(github.event_name == 'schedule' \|\| "
        r"github.event_name == 'workflow_dispatch'\) && (\d+) \|\| (\d+) \}\}",
        section,
    )
    assert match, "full qualification must not inherit the routine timeout"
    full, routine = map(int, match.groups())
    assert routine == routine_minutes
    assert 60 <= full <= 120
    if filename == "ci.yml":
        assert '!= "schedule"' in section and '!= "workflow_dispatch"' in section


def test_python_ci_shards_the_known_long_tail_without_invalidating_ccache() -> None:
    path = Path(__file__).resolve().parents[2] / ".github/workflows/ci.yml"
    section = (
        path.read_text()
        .split("\n  python:\n", 1)[1]
        .split("\n  upload-coverage:\n", 1)[0]
    )
    assert "shard: [core, runtime-heavy, posthf, compiler-heavy, ecp-forces]" in section
    assert "runtime-heavy)" in section
    assert "dist_mode=loadfile" in section
    assert "name: python (${{ matrix.shard }})" in section
    assert "coverage-report-python-${{ matrix.shard }}" in section
    assert "benchmark-debug-${{ github.run_id }}-${{ matrix.shard }}" in section
    cache_line = next(
        line for line in section.splitlines() if "key: ccache-python-" in line
    )
    assert "'tests/**'" not in cache_line
    assert "-DVIBEQC_BUILD_TESTS=OFF" in section
    for path_name in (
        "test_cc_complete_gradient.py",
        "test_ecp_heavy.py",
        "test_codegen.py",
        "test_second_derivatives_inputs.py",
        "test_ecp_public_cpu.py",
        "test_ecp_spd_cartesian_cpu.py",
        "test_ecp_spd_spherical_cpu.py",
        "test_response_native_rks.py",
        "test_response_native_uks.py",
        "test_mp2_public.py",
        "test_ccsd_t_orbital_response.py",
        "test_hessian_block.py",
        "test_grid_policy_convergence.py",
        "test_cc_triples_response_cuda.py",
        "test_codegen_schedules.py",
        "test_dft_batch.py",
        "test_ecp_multicenter.py",
        "test_cc_triples_lambda_response.py",
        "test_cc_lambda_response.py",
        "test_implicit_response.py",
        "test_df_rys.py",
        "test_hessian_numerical.py",
        "test_one_electron_cpu_codegen.py",
        "test_cc_solver.py",
        "test_cc_lambda_solver.py",
    ):
        assert path_name in section


def test_routine_python_ci_defers_qualification_scale_megatests() -> None:
    path = Path(__file__).resolve().parents[2] / ".github/workflows/ci.yml"
    section = (
        path.read_text()
        .split("\n  python:\n", 1)[1]
        .split("\n  upload-coverage:\n", 1)[0]
    )
    for nodeid in (
        "tests/python/test_ccsd_t_complete_gradient.py::test_complete_ccsdt_gradient_matches_pinned_pyscf[nh3]",
        "tests/python/test_ecp_spd_cartesian_cpu.py::test_spd_force_analytic_and_reconverged_fd[pbe-rks]",
        "tests/python/test_ecp_spd_spherical_cpu.py::test_spd_force_analytic_and_reconverged_fd[pbe-rks]",
        "tests/python/test_ecp_spd_cartesian_cpu.py::test_spd_force_analytic_and_reconverged_fd[lda-uks]",
        "tests/python/test_ecp_spd_spherical_cpu.py::test_spd_force_analytic_and_reconverged_fd[lda-uks]",
        "tests/python/test_wb97mv_complete.py::test_public_wb97mv_force_matches_reconverged_energy_differences[grid_shape0-water-rks]",
    ):
        assert nodeid in section
