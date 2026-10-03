"""Real-device Gram checks cover ragged tails and an independent scalar oracle."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest
from generativeqc_compiler.method.df_exchange_schedule import native_header

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_RESOURCE_CUDA_TEST") != "1",
    reason="requires a finite Slurm GPU allocation",
)


def test_generated_gram_ragged_shapes_and_capture(tmp_path: Path) -> None:
    """Compare every output with BLAS and selected entries with scalar FP64.

    The first small case deliberately bypasses the performance admission gate
    to exercise both tails cheaply. Partial upper triangles are never written;
    initcheck must therefore reject any accidental read from them.
    """
    assert os.environ.get("SLURM_JOB_ID")
    ccache = shutil.which("ccache")
    assert ccache, "the repository requires cached native test compilation"
    cuda = Path(os.environ.get("CUDA_PATH", "/usr/local/cuda"))
    source = tmp_path / "gram.cu"
    source.write_text(
        r"""
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
"""
        + native_header()
        + r"""
void check(cudaError_t e) {if(e!=cudaSuccess){std::fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(e));std::exit(1);}}
void blas_check(cublasStatus_t e) {if(e!=CUBLAS_STATUS_SUCCESS)std::exit(2);}
void run(int n,int k,double weight) {
 using namespace generativeqc::scf::generated;
 const auto nn=std::size_t(n)*n;
 const int splits=(k+32767)/32768,tiles=(n+31)/32;
 std::vector<double> host(std::size_t(n)*k), actual(nn), reference(nn);
 for(std::size_t i=0;i<host.size();++i)host[i]=std::sin(double(i%104729)*0.17)*0.013;
 double *u,*partial,*out,*ref;
 check(cudaMalloc(&u,host.size()*8));check(cudaMalloc(&partial,nn*splits*8));
 check(cudaMalloc(&out,nn*8));check(cudaMalloc(&ref,nn*8));
 check(cudaMemcpy(u,host.data(),host.size()*8,cudaMemcpyHostToDevice));
 cudaStream_t stream;check(cudaStreamCreate(&stream));
 cublasHandle_t h;blas_check(cublasCreate(&h));blas_check(cublasSetStream(h,stream));
 const double zero=0;
 blas_check(cublasDgemm(h,CUBLAS_OP_T,CUBLAS_OP_N,n,n,k,&weight,u,k,u,k,&zero,ref,n));
 check(cudaStreamSynchronize(stream));
 const auto launch=[&]{
  df_occupied_gram_partials<<<dim3(tiles,tiles,splits),dim3(16,16),0,stream>>>(n,k,u,partial);
  df_occupied_gram_reduce<<<(nn+255)/256,256,0,stream>>>(n,splits,weight,partial,out);
  check(cudaGetLastError());
 };
 // Capture includes both producer and reduction on the owner stream.
 cudaGraph_t graph;cudaGraphExec_t executable;
 check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));launch();
 check(cudaStreamEndCapture(stream,&graph));check(cudaGraphInstantiate(&executable,graph,0));
 for(int replay=0;replay<2;++replay) {
  if(replay) { // Captured work must use new factors, not retain old partials.
   for(auto& value:host)value=-0.71*value+0.003;
   check(cudaMemcpyAsync(u,host.data(),host.size()*8,cudaMemcpyHostToDevice,stream));
   blas_check(cublasDgemm(h,CUBLAS_OP_T,CUBLAS_OP_N,n,n,k,&weight,u,k,u,k,&zero,ref,n));
  }
  check(cudaGraphLaunch(executable,stream));check(cudaStreamSynchronize(stream));
  check(cudaMemcpy(actual.data(),out,nn*8,cudaMemcpyDeviceToHost));
  check(cudaMemcpy(reference.data(),ref,nn*8,cudaMemcpyDeviceToHost));
  double maximum=0;
  for(std::size_t i=0;i<nn;++i) {
   const auto error=std::abs(actual[i]-reference[i]);
   if(!std::isfinite(error)||error>2e-10)std::exit(3);
   if(actual[i]!=actual[(i%n)*n+i/n])std::exit(4);
   maximum=std::max(maximum,error);
  }
  for(auto ij: {std::pair{0,0},std::pair{n-1,0},std::pair{n-1,n-1},std::pair{n/2,n/3}}) {
   long double oracle=0;
   for(int l=0;l<k;++l)oracle+=static_cast<long double>(host[std::size_t(ij.first)*k+l])*host[std::size_t(ij.second)*k+l];
   if(std::abs(actual[ij.first+std::size_t(ij.second)*n]-weight*oracle)>2e-10)std::exit(5);
  }
  std::printf("n=%d k=%d weight=%g replay=%d maximum=%.17g\n",n,k,weight,replay,maximum);
 }
 check(cudaGraphExecDestroy(executable));check(cudaGraphDestroy(graph));
 blas_check(cublasDestroy(h));check(cudaStreamDestroy(stream));
 check(cudaFree(u));check(cudaFree(partial));check(cudaFree(out));check(cudaFree(ref));
}
int main(int argc,char**) {
 if(argc>1) { // Small racecheck domain retains AO and reduction tails/two slices.
  run(33,79,2);run(17,32769,1);return 0;
 }
 run(17,79,2);run(33,32769,1);run(385,65539,-0.5);run(773,131075,2);
}
"""
    )
    binary = tmp_path / "gram"
    subprocess.run(
        [
            ccache,
            str(cuda / "bin/nvcc"),
            "-O3",
            "-std=c++20",
            "-arch=sm_120",
            str(source),
            "-lcublas",
            "-o",
            str(binary),
        ],
        check=True,
    )
    # An external sanitizer may follow this child with --target-processes all.
    subprocess.run([str(binary)], check=True)
