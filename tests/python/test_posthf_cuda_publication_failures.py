"""Execute actual MO publication control flow with fault-injected host CUDA stubs.

These tests validate lifecycle and publication, not GPU/cuBLAS mathematics.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

_SHIM = r"""
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <vector>
using cudaStream_t = int;
constexpr int cudaMemcpyHostToDevice=1, cudaMemcpyDeviceToHost=2;
constexpr int CUBLAS_OP_N=0,CUBLAS_OP_T=1;
static std::vector<std::function<void()>> pending;
static bool late_library=false, late_input=false, fail_sync=false, wrong_device=false;
static int scans=0, drains=0, fail_output_copy=0;
static bool late_output=false;
void flush() { auto work=std::move(pending); pending.clear(); for(auto& fn:work) fn(); }
int cudaMemcpyAsync(void* d,const void* s,std::size_t n,int kind,cudaStream_t) {
  if(kind==cudaMemcpyDeviceToHost && n==sizeof(double) && fail_output_copy>0 &&
      --fail_output_copy==0)return 1;
  pending.emplace_back([=]{std::memcpy(d,s,n);}); return 0;
}
int cudaStreamSynchronize(cudaStream_t) {
  ++drains;
  if(fail_sync) {fail_sync=false;return 1;}
  flush();return 0;
}
int cudaGetLastError(){return 0;}
int cublasDgemm(int,int,int,int,int,int,const double*,const double*,int,
                const double* in,int,const double*,double* out,int) {
  pending.emplace_back([=]{*out=*in;});return 0;
}
int cublasDaxpy(int,std::size_t n,const double*,const double* in,int,double* out,int) {
  pending.emplace_back([=]{for(std::size_t i=0;i<n;++i)out[i]+=in[i];});return 0;
}
namespace vibeqc::runtime {
std::size_t size_add(std::size_t a,std::size_t b){return a+b;}
std::size_t size_mul(std::size_t a,std::size_t b){return a*b;}
}
namespace vibeqc_tensor {
struct DeviceAllocationError:std::bad_alloc {};
void error_text(char* out,std::size_t n,const char* text){if(out&&n)std::snprintf(out,n,"%s",text);}
void cuda_check(int status){if(status)throw std::runtime_error("injected CUDA failure");}
void blas_check(int status){cuda_check(status);}
struct Metrics {double input_ms{},library_ms{},kernel_ms{},output_ms{};};
struct Context {
  std::mutex mutex; cudaStream_t stream=1; int handle=1; int* error{}; Metrics metrics;
  void check_device(){if(wrong_device)throw std::runtime_error("wrong device");}
  template<class F> void section(bool,double& metric,F fn){
    fn();
    if(&metric==&metrics.input_ms && late_input){late_input=false;throw std::runtime_error("input event failed");}
    if(&metric==&metrics.output_ms && late_output){late_output=false;throw std::runtime_error("output event failed");}
    flush();
    if(&metric==&metrics.library_ms && late_library){late_library=false;throw std::runtime_error("elapsed-time failed");}
  }
};
void check_scale(double* values,std::size_t n,double,int* error,int){
  ++scans;
  for(std::size_t i=0;i<n;++i)if(!std::isfinite(values[i]))*error=1;
}
}
"""

_MAIN = r"""
int main(int argc,char**argv){
  if(argc!=2)return 99;
  int mode=std::atoi(argv[1]);
  Transform p; p.nbf=p.stage=p.output=1; p.m.fill(1);p.tile.fill(1);
  double coefficient=1,first=0,second=0,result=0;
  int invalid=0;
  p.c=&coefficient;p.first=&first;p.second=&second;p.result=&result;p.context.error=&invalid;
  std::size_t begin[4]{},count[4]{1,1,1,1}; char error[256]{};
  double value=2,download=12345;
  auto add=[&]{return posthf_cuda_add_v1(&p,&value,begin,count,error,sizeof(error));};
  auto validate=[&]{return posthf_cuda_validate_v1(&p,error,sizeof(error));};
  auto read=[&]{return posthf_cuda_download_v1(&p,&download,1,error,sizeof(error));};
  if(mode==0){
    if(add()||scans||validate()||scans!=1||read()||scans!=1||download!=2)return 10;
    if(add()||validate()||scans!=2||read()||download!=4)return 11;
  }else if(mode==1){
    value=std::numeric_limits<double>::quiet_NaN();
    if(add()||posthf_cuda_pointer_v1(&p)!=nullptr||scans!=1)return 12;
  }else if(mode==2||mode==3){
    value=mode==2?std::numeric_limits<double>::quiet_NaN():2;
    late_library=true;
    if(add()==0||validate()==0||read()==0||download!=12345)return 13;
    if(posthf_cuda_pointer_v1(&p)!=nullptr)return 14;
  }else if(mode==4){
    late_input=true;
    if(add()==0||!pending.empty()||drains==0)return 15;
    if(validate()==0)return 16;
  }else if(mode==5){
    if(add())return 17;
    fail_sync=true;
    if(validate()==0||!pending.empty()||drains<2)return 18;
    if(validate()||read()||download!=2)return 19;
  }else if(mode==6){
    if(add()||validate())return 20;
    count[0]=0;
    if(add()==0||validate()||read()||download!=2||scans!=1)return 21;
  }else if(mode==7){
    wrong_device=true;
    if(posthf_cuda_pointer_v1(&p)!=nullptr)return 22;
  }else if(mode==8){
    late_library=true;
    if(add()==0||add()==0||validate()==0)return 23;
  }else if(mode==9){
    if(add()||posthf_cuda_pointer_v1(&p)!=p.result||scans!=1)return 24;
    if(posthf_cuda_pointer_v1(&p)!=p.result||scans!=1)return 25;
    if(posthf_cuda_pointer_v1(nullptr)!=nullptr)return 26;
  }else if(mode==12){
    if(add()||validate())return 32;
    const auto before=drains;
    late_output=true;
    if(read()==0||!pending.empty()||drains!=before+1)return 33;
    download=12345;
    flush();
    if(download!=12345||read()||download!=2)return 34;
  }else if(mode==10||mode==11||mode==13||mode==14){
    BatchTransform batch;
    batch.nbf=1; batch.tile.fill(1); batch.states.resize(2); batch.context.error=&invalid;
    double raw=0,coeff[8]{1,1,1,1,1,1,1,1};
    double first[2]{},second_batch[2]{},result_batch[2]{};
    batch.raw=&raw;
    for(std::size_t request=0;request<2;++request){
      auto& state=batch.states[request];
      state.stage=state.output=1; state.coefficients=4; state.m.fill(1);
      state.c_offset={0,1,2,3}; state.c=coeff+4*request;
      state.first=&first[request];state.second=&second_batch[request];
      state.result=&result_batch[request];
    }
    double batch_value=mode==11?std::numeric_limits<double>::quiet_NaN():2;
    double out0=12345,out1=12345; double* outs[2]{&out0,&out1};
    std::size_t sizes[2]{1,1};
    if(posthf_cuda_batch_add_v1(&batch,&batch_value,begin,count,error,sizeof(error)))return 28;
    const auto before=drains;
    if(mode==13)late_output=true;
    if(mode==14)fail_output_copy=2;
    int status=posthf_cuda_batch_download_v1(&batch,outs,sizes,2,error,sizeof(error));
    if(mode==10){
      if(status||scans!=2||out0!=2||out1!=2)return 29;
    }else if(mode==11){
      if(status==0||scans!=2||out0!=12345||out1!=12345)return 30;
    }else{
      // A transport error may partially write outputs. It must not retain
      // borrowed host pointers after return, and a retry may read valid data.
      if(status==0||!pending.empty()||drains!=before+2)return 35;
      out0=out1=12345;
      flush();
      if(out0!=12345||out1!=12345)return 36;
      if(posthf_cuda_batch_download_v1(&batch,outs,sizes,2,error,sizeof(error))||
          out0!=2||out1!=2)return 37;
    }
  }else return 98;
  return pending.empty()?0:31;
}
"""


@pytest.fixture(scope="module")
def executable(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    text = (ROOT / "src/posthf/cuda_transform.cu").read_text()
    helpers = text[text.index("namespace {") : text.index('extern "C"')]
    actions = text[
        text.index("int posthf_cuda_add_v1") : text.index("int posthf_cuda_metrics_v1")
    ]
    pointer = text[
        text.index("void* posthf_cuda_pointer_v1") : text.index(
            "int posthf_cuda_versions_v1"
        )
    ]
    batch_actions = text[
        text.index("int posthf_cuda_batch_add_v1") : text.index(
            "int posthf_cuda_batch_metrics_v1"
        )
    ]
    # Only erase the CUDA launch syntax; actual validation/guard/add bodies run.
    code = re.sub(
        r"<<<.*?>>>", "", helpers + actions + pointer + batch_actions, flags=re.DOTALL
    )
    directory = tmp_path_factory.mktemp("mo-publication")
    source = directory / "publication.cpp"
    source.write_text(_SHIM + code + _MAIN)
    output = directory / "publication"
    subprocess.run(
        [compiler, "-std=c++20", "-O0", str(source), "-o", str(output)],
        check=True,
        timeout=30,
    )
    return output


@pytest.mark.parametrize("mode", range(15))
def test_actual_native_publication_boundaries(executable: Path, mode: int) -> None:
    subprocess.run([str(executable), str(mode)], check=True, timeout=10)
