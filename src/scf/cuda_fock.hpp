#ifndef VIBEQC_SCF_CUDA_FOCK_HPP
#define VIBEQC_SCF_CUDA_FOCK_HPP

#include "vibeqc/vibeqc.h"

#include <cstddef>
#include <string>
#include <vector>

namespace vibeqc::scf {

struct CudaFockBucketHandle;

/** Upload static Hcore/ERI data and allocate a persistent CUDA bucket. */
vibeqc_status create_cuda_fock_bucket(int device_id,
                                   std::size_t batch_size,
                                   std::size_t nbf,
                                   const std::vector<double>& hcore,
                                   const std::vector<double>& eri,
                                   CudaFockBucketHandle** handle,
                                   std::string& detail);

/** Build Fock matrices; only density and Fock cross the PCIe boundary. */
vibeqc_status execute_cuda_fock_bucket(CudaFockBucketHandle* handle,
                                    const std::vector<double>& density,
                                    std::vector<double>& fock,
                                    std::string& detail);

void destroy_cuda_fock_bucket(CudaFockBucketHandle* handle);

}  // namespace vibeqc::scf

#endif
