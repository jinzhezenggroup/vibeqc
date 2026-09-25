"""Check resident VV10 host admission before any device-side effect."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest
from test_nonlocal_host_staging_lifetime import PREFIX

ROOT = Path(__file__).resolve().parents[2]

DRIVER = r"""
}
int main(int argc,char** argv) {
 using namespace vibeqc::dft::nlc;
 if(argc!=2)return 99;
 const int test=std::atoi(argv[1]);
 Vv10Parameters parameters;
 switch(test) {
  case 0:break;
  case 1:parameters.variant=Vv10Variant::rvv10;break;
  case 2:parameters.variant=static_cast<Vv10Variant>(0);break;
  case 3:parameters.variant=static_cast<Vv10Variant>(3);break;
  default: {
   const double invalid[]={0.0,-1.0,NAN,INFINITY,-INFINITY};
   const int index=test-4;
   double* fields[]={&parameters.b,&parameters.c,&parameters.coefficient};
   *fields[index/5]=invalid[index%5];
  }
 }
 double points[3]{},weight=1.0,rho=1.0,gradient[3]{},energy=-13.0;
 double workspace[4]{};
 int error=41;
 bool rejected=false;
 try {
  enqueue_vv10_cuda_device(vv10_cuda_device_layout(1,1,false,false),parameters,
   0,&test_stream,points,&weight,&rho,gradient,workspace,sizeof(workspace),
   &energy,nullptr,nullptr,nullptr,nullptr,&error);
 } catch(const std::invalid_argument&) {rejected=true;}
 if(rejected!=(test>=2)) {std::cerr<<"incorrect parameter admission";return 1;}
 if(rejected && error!=41) {std::cerr<<"enqueued before rejection";return 2;}
 return 0;
}
"""


@pytest.fixture(scope="module")
def parameter_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    source = (ROOT / "src/dft/nonlocal_correlation/vv10_runtime_cuda.cu").read_text()
    body = source.split("Vv10CudaDeviceLayout vv10_cuda_device_layout(", 1)[1]
    body = (
        "Vv10CudaDeviceLayout vv10_cuda_device_layout("
        + body.split("void execute_vv10_cuda(", 1)[0]
    )
    body = re.sub(r"<<<.*?>>>", "", body, flags=re.DOTALL)
    folder = tmp_path_factory.mktemp("vv10-parameter-admission")
    unit, binary = folder / "probe.cpp", folder / "probe"
    unit.write_text(PREFIX + body + DRIVER)
    subprocess.run(
        [compiler, "-std=c++20", "-O0", str(unit), "-o", str(binary)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return binary


@pytest.mark.parametrize("case", range(19))
def test_resident_parameters_precede_device_effects(
    parameter_probe: Path, case: int
) -> None:
    result = subprocess.run(
        [str(parameter_probe), str(case)],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr
