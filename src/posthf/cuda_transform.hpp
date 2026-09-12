#pragma once
#include <cstddef>

#include "tensor/metrics.hpp"

// Existing CG10 private schema-1 runtime, shared by Python and native consumers.
extern "C" {
int posthf_cuda_create_v1(int, std::size_t, const std::size_t*, const std::size_t*, const double*,
                          std::size_t, void**, char*, std::size_t);
void posthf_cuda_destroy_v1(void*);
int posthf_cuda_add_v1(void*, const double*, const std::size_t*, const std::size_t*, char*,
                       std::size_t);
int posthf_cuda_download_v1(void*, double*, std::size_t, char*, std::size_t);
int posthf_cuda_metrics_v1(void*, vibeqc_tensor::Metrics*, char*, std::size_t);
void* posthf_cuda_pointer_v1(void*);
}
