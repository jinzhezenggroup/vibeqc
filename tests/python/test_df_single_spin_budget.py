"""The single-system budget path must preserve the requested spin policy."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_single_item_budget_does_not_treat_uhf_singlet_as_rhf(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    root = Path(__file__).resolve().parents[2]
    source = (root / "src/scf/rhf.cpp").read_text()
    begin = source.index("DfResolvedBudget resolve_df_budget_for_system(")
    end = source.index("\n}\n", begin) + 3
    resolver = source[begin:end]
    begin = source.index(
        "data.resolved_budget =",
        source.index("DensityFittingScfData prepare_density_fitting_data("),
    )
    end = source.index(";", begin) + 1
    binding = source[begin:end]
    driver = tmp_path / "spin.cpp"
    driver.write_text(
        '#include "scf/df_preparation_budget.hpp"\n'
        "namespace vibeqc::core { struct System { int electron_count=2; int multiplicity=1; }; }\n"
        "namespace vibeqc::scf {\n"
        "DfBudgetWorkload df_budget_workload(const core::System&,const core::System&,std::size_t b,unsigned h,bool f) {return {2,2,2,b,h,f};}\n"
        "std::size_t preferred_automatic_resident_df_value_peak(const DfBudgetWorkload&,std::size_t,bool u) {return u?0:4096;}\n"
        "DfResolvedBudget resolve_df_budget_for_workload(DfBudgetWorkload w,int,std::size_t) {DfResolvedBudget r{};r.total_bytes=w.preferred_value_peak_bytes;return r;}\n"
        + resolver
        + "\nstd::size_t prepare(bool unrestricted) { core::System system,auxiliary_system;int cuda_device_id=0;std::size_t output_budget_bytes=0;bool include_derivatives=true;unsigned diis_history=8;struct {DfResolvedBudget resolved_budget;} data;\n"
        + binding
        + "\nreturn data.resolved_budget.total_bytes;}\n}\n"
        "int main(){using namespace vibeqc::scf; if(prepare(false)!=4096)return 1; if(prepare(true)!=0)return 2;}\n"
    )
    executable = tmp_path / "spin"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            f"-I{root / 'src'}",
            str(driver),
            "-o",
            str(executable),
        ],
        check=True,
    )
    result = subprocess.run([str(executable)], check=False)
    assert result.returncode == 0, "UHF singlet acquired an RHF-only preferred budget"
