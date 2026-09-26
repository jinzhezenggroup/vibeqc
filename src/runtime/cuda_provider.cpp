#include "runtime/cuda_provider.hpp"

#ifndef VIBEQC_HAS_CUDA
#define VIBEQC_HAS_CUDA 0
#endif
#ifndef VIBEQC_CUDA_PROVIDER_CUMETAL
#define VIBEQC_CUDA_PROVIDER_CUMETAL 0
#endif

namespace vibeqc::runtime {

const CudaProviderCapabilities& active_cuda_provider() noexcept {
#if !VIBEQC_HAS_CUDA
  static constexpr auto provider = cuda_provider_capabilities(CudaProviderKind::None);
#elif VIBEQC_CUDA_PROVIDER_CUMETAL
  static constexpr auto provider = cuda_provider_capabilities(CudaProviderKind::CuMetal);
#else
  static constexpr auto provider = cuda_provider_capabilities(CudaProviderKind::Nvidia);
#endif
  return provider;
}

}  // namespace vibeqc::runtime
