#ifndef QCE_SCF_CUDA_FOCK_HPP
#define QCE_SCF_CUDA_FOCK_HPP

#include "qce/qce.h"

#include <cstddef>
#include <string>
#include <vector>

namespace qce::scf {

struct CudaFockBucketHandle;

/** Upload static Hcore/ERI data and allocate a persistent CUDA bucket. */
qce_status create_cuda_fock_bucket(int device_id,
                                   std::size_t batch_size,
                                   std::size_t nbf,
                                   const std::vector<double>& hcore,
                                   const std::vector<double>& eri,
                                   CudaFockBucketHandle** handle,
                                   std::string& detail);

/** Build Fock matrices; only density and Fock cross the PCIe boundary. */
qce_status execute_cuda_fock_bucket(CudaFockBucketHandle* handle,
                                    const std::vector<double>& density,
                                    std::vector<double>& fock,
                                    std::string& detail);

void destroy_cuda_fock_bucket(CudaFockBucketHandle* handle);

}  // namespace qce::scf

#endif
