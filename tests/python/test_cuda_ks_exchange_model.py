"""Exercise the actual final-state model predicate, without a CUDA runtime.

Small record fixtures isolate admission from grid/provider numerical validators;
those validators and the full GPU state-export tests remain separate gates.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PREFIX = r"""
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
enum class SemilocalFamily : unsigned { Lda=0, Pbe=1, R2scan=2, B3lyp=3, Wb97mv=4 };
SemilocalFamily semilocal_family_from_code(unsigned code) {
 if(code>4)throw std::invalid_argument("family");
 return static_cast<SemilocalFamily>(code);
}
unsigned semilocal_family_domain_version(SemilocalFamily family) {
 return family==SemilocalFamily::Wb97mv?3:family==SemilocalFamily::B3lyp?2:1;
}
bool semilocal_family_has_cuda_ks(SemilocalFamily family) {
 return family==SemilocalFamily::Lda || family==SemilocalFamily::Pbe ||
        family==SemilocalFamily::R2scan;
}
namespace nlc { enum class Vv10DensityDomain { MolecularV1 }; }
namespace scf {
enum class FockBackend {Cpu,Cuda};
enum class FockSpin {Restricted,Unrestricted};
enum class FockPrecision {Float64,Float32};
enum class FockApproximation {Exact,DensityFitted};
enum class FockOperator {FullRange,LongRange};
struct Term {bool present=true; double coefficient=1,omega=0; FockOperator op=FockOperator::FullRange;
 FockApproximation approximation=FockApproximation::Exact;};
struct Spec {FockSpin spin=FockSpin::Restricted; unsigned derivative_order=0;Term coulomb,exchange;};
struct Fock {FockBackend backend=FockBackend::Cuda;FockPrecision precision=FockPrecision::Float64;
 Spec spec;double screening_tolerance=1e-12;};
// Preserve the basic finite-coefficient contract of the independent validator.
void validate_resolved_fock_build(const Fock& f) {
 if(!std::isfinite(f.spec.exchange.coefficient))throw std::invalid_argument("coefficient");
}
void require_wb97mv_composition(const Fock&,const Fock&,int) {throw std::invalid_argument("not this scope");}
}
struct Model {unsigned version=1,functional=1,scf_domain_version=1,tile_points=257,owner=1,spins=1;
 int device=0,grid=1;double semilocal_exchange_scale=1,semilocal_correlation_scale=1;
 std::optional<scf::Fock> range_correction;
 std::optional<int> nonlocal_correlation;
 nlc::Vv10DensityDomain nonlocal_density_domain=nlc::Vv10DensityDomain::MolecularV1;};
struct KsFinalStateIdentity {
 Model model;
 struct {scf::Fock model;std::vector<unsigned> occupied{1};} determinant;
};
void validate_grid_spec(int grid) {if(grid!=1)throw std::invalid_argument("grid");}
"""
DRIVER = r"""
int main(int argc,char** argv) {
 if(argc!=5)return 99;
 const std::string mode=argv[1];
 const unsigned functional=std::strtoul(argv[2],nullptr,10),spins=std::strtoul(argv[3],nullptr,10);
 const bool expected=std::atoi(argv[4])!=0;
 KsFinalStateIdentity state;
 state.model.functional=functional;state.model.spins=spins;
 state.determinant.occupied.assign(spins,1);
 auto& f=state.determinant.model;
 f.spec.spin=spins==1?scf::FockSpin::Restricted:scf::FockSpin::Unrestricted;
 f.spec.exchange.coefficient=spins==1?-0.125:-0.25;
 if(mode=="no-exchange")f.spec.exchange.present=false;
 else if(mode=="cpu" || mode=="cpu-scaled") {
   f.backend=scf::FockBackend::Cpu;state.model.device=-1;
   if(mode=="cpu-scaled")state.model.semilocal_exchange_scale=0.75;
 }
 else if(mode=="df-j")f.spec.coulomb.approximation=scf::FockApproximation::DensityFitted;
 else if(mode=="df-k")f.spec.exchange.approximation=scf::FockApproximation::DensityFitted;
 else if(mode=="range-k")f.spec.exchange.op=scf::FockOperator::LongRange;
 else if(mode=="range-decoration")state.model.range_correction=scf::Fock{};
 else if(mode=="nlc-decoration")state.model.nonlocal_correlation=1;
 else if(mode=="scale-x")state.model.semilocal_exchange_scale=0.73;
 else if(mode=="scale-c")state.model.semilocal_correlation_scale=0.5;
 else if(mode=="pbe0")state.model.semilocal_exchange_scale=0.75;
 else if(mode=="pbe0-wrong-k") {
   state.model.semilocal_exchange_scale=0.75;f.spec.exchange.coefficient=-0.2;
 }
 else if(mode=="pbe0-range-decoration") {
   state.model.semilocal_exchange_scale=0.75;state.model.range_correction=scf::Fock{};
 }
 else if(mode=="pbe0-nlc-decoration") {
   state.model.semilocal_exchange_scale=0.75;state.model.nonlocal_correlation=1;
 }
 else if(mode=="positive-k")f.spec.exchange.coefficient=0.25;
 else if(mode=="nan-k")f.spec.exchange.coefficient=std::nan("");
 else if(mode=="float32")f.precision=scf::FockPrecision::Float32;
 else if(mode=="wrong-device")state.model.device=-1;
 else if(mode=="derivative")f.spec.derivative_order=1;
 else if(mode=="bad-grid")state.model.grid=0;
 else if(mode=="bad-spins")state.determinant.occupied.clear();
 else if(mode=="wrong-domain")state.model.scf_domain_version=2;
 if(mode.rfind("rsh",0)==0) {
   scf::Fock correction=f;
   correction.spec.coulomb.present=false;
   correction.spec.exchange.op=scf::FockOperator::LongRange;
   correction.spec.exchange.omega=0.33;
   if(mode=="rsh-no-primary")f.spec.exchange.present=false;
   else if(mode=="rsh-wrong-spin")correction.spec.spin=spins==1?scf::FockSpin::Unrestricted:scf::FockSpin::Restricted;
   else if(mode=="rsh-cpu-correction")correction.backend=scf::FockBackend::Cpu;
   else if(mode=="rsh-derivative")correction.spec.derivative_order=1;
   else if(mode=="rsh-coulomb")correction.spec.coulomb.present=true;
   else if(mode=="rsh-missing-k")correction.spec.exchange.present=false;
   else if(mode=="rsh-full-range")correction.spec.exchange.op=scf::FockOperator::FullRange;
   else if(mode=="rsh-zero-omega")correction.spec.exchange.omega=0;
   else if(mode=="rsh-nan-omega")correction.spec.exchange.omega=std::nan("");
   else if(mode=="rsh-screening")correction.screening_tolerance=1e-8;
   else if(mode=="rsh-df")correction.spec.exchange.approximation=scf::FockApproximation::DensityFitted;
   else if(mode=="rsh-scaled")state.model.semilocal_exchange_scale=0.75;
   else if(mode=="rsh-nonlocal")state.model.nonlocal_correlation=1;
   state.model.range_correction=correction;
 }
 if(valid_model(state)!=expected) {std::cerr<<mode<<" functional="<<functional<<" spins="<<spins;return 1;}
}
"""


@pytest.fixture(scope="module")
def model_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a C++20 compiler")
    source = (ROOT / "src/dft/ks_final_state.cpp").read_text()
    start = source.index("bool valid_model(")
    end = source.index("bool finite_components(", start)
    directory = tmp_path_factory.mktemp("ks-exchange-model")
    unit, executable = directory / "probe.cpp", directory / "probe"
    unit.write_text(PREFIX + source[start:end] + DRIVER)
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O2",
            "-fsanitize=undefined",
            str(unit),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return executable


@pytest.mark.parametrize("spins", (1, 2))
@pytest.mark.parametrize(
    "mode,functional,accepted",
    [
        *(
            (mode, family, True)
            for mode in ("exact", "no-exchange")
            for family in (0, 1, 2)
        ),
        *(
            (mode, 1, False)
            for mode in (
                "df-j",
                "df-k",
                "range-k",
                "range-decoration",
                "nlc-decoration",
                "scale-x",
                "scale-c",
                "positive-k",
                "nan-k",
                "float32",
                "wrong-device",
                "derivative",
                "bad-grid",
                "bad-spins",
                "wrong-domain",
            )
        ),
        *(("exact", family, False) for family in (3, 4, 99)),
        ("pbe0", 1, True),
        ("pbe0-wrong-k", 1, False),
        ("pbe0-range-decoration", 1, False),
        ("pbe0-nlc-decoration", 1, False),
        ("cpu", 1, True),
        ("cpu-scaled", 1, True),
        ("cpu", 0, False),
        ("cpu", 2, False),
    ],
)
def test_final_identity_matches_bounded_primary_exchange_scope(
    model_probe: Path, mode: str, functional: int, spins: int, accepted: bool
) -> None:
    result = subprocess.run(
        [str(model_probe), mode, str(functional), str(spins), str(int(accepted))],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("spins", (1, 2))
@pytest.mark.parametrize(
    "mode,accepted",
    [
        ("rsh", True),
        ("rsh-no-primary", True),
        *(
            (mode, False)
            for mode in (
                "rsh-wrong-spin",
                "rsh-cpu-correction",
                "rsh-derivative",
                "rsh-coulomb",
                "rsh-missing-k",
                "rsh-full-range",
                "rsh-zero-omega",
                "rsh-nan-omega",
                "rsh-screening",
                "rsh-df",
                "rsh-scaled",
                "rsh-nonlocal",
            )
        ),
    ],
)
def test_final_identity_binds_complete_rsh_correction(
    model_probe: Path, mode: str, spins: int, accepted: bool
) -> None:
    result = subprocess.run(
        [str(model_probe), mode, "1", str(spins), str(int(accepted))],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr
