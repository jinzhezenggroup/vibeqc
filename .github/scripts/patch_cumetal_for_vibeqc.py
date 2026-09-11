"""Patch the pinned CuMetal source tree for VibeQC's CUDA CI surface.

Keep these changes in the compatibility provider rather than weakening VibeQC's
production CUDA implementation.  Every patch is anchored to the pinned CuMetal
commit and fails loudly if upstream source changes make the assumption stale.
"""

from __future__ import annotations

import os
from pathlib import Path

SOURCE = Path(os.environ["CUMETAL_SOURCE"])


def replace_once(path: Path, old: str, new: str, description: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(
            f"{description}: expected one patch anchor, found {count} in {path}"
        )
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


# VibeQC's launch policy consumes standard cudaDeviceProp occupancy fields that
# are absent from CuMetal 0.5.0.  Use two of CuMetal's reserved ABI slots so the
# public structure size remains unchanged.
cuda_header = SOURCE / "runtime/api/cuda_runtime.h"
replace_once(
    cuda_header,
    "#pragma once\n",
    "#pragma once\n"
    "// Match the pinned provider's cudaRuntimeGetVersion compatibility identity.\n"
    "#ifndef CUDART_VERSION\n#define CUDART_VERSION 12000\n#endif\n",
    "CUDA runtime version macro for native toolkit provenance",
)
replace_once(
    cuda_header,
    "    int maxThreadsPerMultiProcessor; // Max threads per SM\n",
    "    int maxThreadsPerMultiProcessor; // Max threads per SM\n"
    "    int maxBlocksPerMultiProcessor;  // Max resident blocks per SM\n"
    "    int regsPerMultiprocessor;       // Register file size per SM\n",
    "cudaDeviceProp occupancy fields",
)
replace_once(
    cuda_header,
    "    int cumetalReserved[55];",
    "    int cumetalReserved[53];",
    "cudaDeviceProp reserved ABI slots",
)

# Clang checks device bodies during its host pass, when __CUDA_ARCH__ is unset.
# The pinned shim only exposes nan() in the device pass, so generated invalid-
# domain sentinels otherwise resolve to the host-only libc declaration. Keep
# the constant-NaN shim callable in both passes, including host CUDA functions.
replace_once(
    cuda_header,
    "#if defined(__CUDA_ARCH__)\n"
    "static __device__ __forceinline__ double __cumetal_nan(const char*) {\n"
    '    return __builtin_nan("");\n'
    "}\n#define nan __cumetal_nan\n#endif\n",
    "#if defined(__clang__) && defined(__CUDA__)\n"
    "static __host__ __device__ __forceinline__ double __cumetal_nan(const char*) {\n"
    '    return __builtin_nan("");\n'
    "}\n#define nan __cumetal_nan\n#endif\n",
    "CUDA nan shim during Clang host/device parsing",
)

cuda_runtime = SOURCE / "runtime/rt/cuda_runtime.cpp"
replace_once(
    cuda_runtime,
    "    prop->maxThreadsPerMultiProcessor = 2048; // Conservative estimate for M-series\n",
    "    prop->maxThreadsPerMultiProcessor = 2048; // Conservative estimate for M-series\n"
    "    prop->maxBlocksPerMultiProcessor = 32; // CUDA resident-block architectural ceiling\n"
    "    prop->regsPerMultiprocessor = 65536; // Synthetic sm_80 register budget\n",
    "cudaGetDeviceProperties occupancy values",
)

# CuMetal supports ordinary host graph replay but not CUDA's device-side
# conditional/tail graph launch used by VibeQC's qualification probe.  Make the
# device query return null and the tail stream name available.  The surrounding
# VibeQC code therefore does not issue a device graph launch and its provider
# probe records the feature as unavailable instead of claiming support.
replace_once(
    cuda_header,
    "cudaError_t cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream);\n",
    "cudaError_t cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream);\n"
    "#ifndef cudaStreamGraphTailLaunch\n"
    "#define cudaStreamGraphTailLaunch ((cudaStream_t)0x3)\n"
    "#endif\n"
    "#if defined(__CUDA_ARCH__)\n"
    "#define cudaGetCurrentGraphExec() ((cudaGraphExec_t)nullptr)\n"
    "#define cudaGraphLaunch(graphExec, stream) (cudaErrorNotSupported)\n"
    "#endif\n",
    "device-tail graph compatibility declarations",
)

# cusolverDn.h redeclares cuBLAS enums independently, so a normal translation
# unit that includes both headers fails.  Reuse cublas_v2.h's canonical types.
cublas_header = SOURCE / "runtime/api/cublas_v2.h"
replace_once(
    cublas_header,
    "typedef enum cublasFillMode_t {\n"
    "    CUBLAS_FILL_MODE_LOWER = 0,\n"
    "    CUBLAS_FILL_MODE_UPPER = 1,\n"
    "} cublasFillMode_t;",
    "typedef enum cublasFillMode_t {\n"
    "    CUBLAS_FILL_MODE_LOWER = 0,\n"
    "    CUBLAS_FILL_MODE_UPPER = 1,\n"
    "    CUBLAS_FILL_MODE_FULL = 2,\n"
    "} cublasFillMode_t;",
    "cuBLAS full fill-mode spelling",
)

cusolver_header = SOURCE / "runtime/api/cusolverDn.h"
replace_once(
    cusolver_header,
    '#include "cusolver_common.h"\n',
    '#include "cusolver_common.h"\n#include "cublas_v2.h"\n',
    "cuSOLVER canonical cuBLAS types include",
)
replace_once(
    cusolver_header,
    "typedef enum cublasFillMode_t {\n"
    "    CUBLAS_FILL_MODE_LOWER = 0,\n"
    "    CUBLAS_FILL_MODE_UPPER = 1,\n"
    "    CUBLAS_FILL_MODE_FULL = 2,\n"
    "} cublasFillMode_t;\n\n"
    "typedef enum cublasSideMode_t {\n"
    "    CUBLAS_SIDE_LEFT = 0,\n"
    "    CUBLAS_SIDE_RIGHT = 1,\n"
    "} cublasSideMode_t;\n\n",
    "",
    "duplicate cuBLAS enum declarations",
)

# CuMetal already implements double-precision symmetric eigensolve through
# Apple Accelerate.  Add the newer generic/batched cuSOLVER entry points VibeQC
# uses and implement them below as compatibility wrappers around that solver.
solver_types_and_api = r"""
typedef struct cusolverDnParams* cusolverDnParams_t;
typedef struct syevjInfo* syevjInfo_t;

cusolverStatus_t cusolverGetProperty(libraryPropertyType type, int* value);
cusolverStatus_t cusolverDnCreateParams(cusolverDnParams_t* params);
cusolverStatus_t cusolverDnDestroyParams(cusolverDnParams_t params);

cusolverStatus_t cusolverDnCreateSyevjInfo(syevjInfo_t* info);
cusolverStatus_t cusolverDnDestroySyevjInfo(syevjInfo_t info);
cusolverStatus_t cusolverDnXsyevjSetTolerance(syevjInfo_t info, double tolerance);
cusolverStatus_t cusolverDnXsyevjSetMaxSweeps(syevjInfo_t info, int max_sweeps);
cusolverStatus_t cusolverDnXsyevjSetSortEig(syevjInfo_t info, int sort_eig);

cusolverStatus_t cusolverDnDsyevjBatched_bufferSize(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo,
    int n, const double* A, int lda, const double* W, int* lwork,
    syevjInfo_t params, int batch_size);
cusolverStatus_t cusolverDnDsyevjBatched(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo,
    int n, double* A, int lda, double* W, double* work, int lwork,
    int* dev_info, syevjInfo_t params, int batch_size);

cusolverStatus_t cusolverDnXsyevd_bufferSize(
    cusolverDnHandle_t handle, cusolverDnParams_t params, cusolverEigMode_t jobz,
    cublasFillMode_t uplo, int64_t n, cudaDataType data_type_a, const void* A,
    int64_t lda, cudaDataType data_type_w, const void* W, cudaDataType compute_type,
    size_t* workspace_device_bytes, size_t* workspace_host_bytes);
cusolverStatus_t cusolverDnXsyevd(
    cusolverDnHandle_t handle, cusolverDnParams_t params, cusolverEigMode_t jobz,
    cublasFillMode_t uplo, int64_t n, cudaDataType data_type_a, void* A,
    int64_t lda, cudaDataType data_type_w, void* W, cudaDataType compute_type,
    void* workspace_device, size_t workspace_device_bytes, void* workspace_host,
    size_t workspace_host_bytes, int* dev_info);

cusolverStatus_t cusolverDnXsyevBatched_bufferSize(
    cusolverDnHandle_t handle, cusolverDnParams_t params, cusolverEigMode_t jobz,
    cublasFillMode_t uplo, int n, cudaDataType data_type_a, const void* A,
    int lda, cudaDataType data_type_w, const void* W, cudaDataType compute_type,
    size_t* workspace_device_bytes, size_t* workspace_host_bytes, int batch_size);
cusolverStatus_t cusolverDnXsyevBatched(
    cusolverDnHandle_t handle, cusolverDnParams_t params, cusolverEigMode_t jobz,
    cublasFillMode_t uplo, int n, cudaDataType data_type_a, void* A,
    int lda, cudaDataType data_type_w, void* W, cudaDataType compute_type,
    void* workspace_device, size_t workspace_device_bytes, void* workspace_host,
    size_t workspace_host_bytes, int* dev_info, int batch_size);

"""
replace_once(
    cusolver_header,
    "typedef struct cusolverDnContext* cusolverDnHandle_t;\n\n",
    "typedef struct cusolverDnContext* cusolverDnHandle_t;\n\n" + solver_types_and_api,
    "generic cuSOLVER API declarations",
)

cusolver_source = SOURCE / "runtime/rt/cusolver.cpp"
compat_impl = r"""

// VibeQC compatibility surface: newer cuSOLVER APIs mapped to CuMetal's
// existing Accelerate-backed double-precision symmetric eigensolver.
extern "C" {

struct cusolverDnParams {};
struct syevjInfo {
    double tolerance = 0.0;
    int max_sweeps = 100;
    int sort_eig = 1;
};

cusolverStatus_t cusolverGetProperty(libraryPropertyType type, int* value) {
    if (value == nullptr) return CUSOLVER_STATUS_INVALID_VALUE;
    switch (type) {
        case MAJOR_VERSION: *value = 12; break;
        case MINOR_VERSION: *value = 9; break;
        case PATCH_LEVEL: *value = 0; break;
        default: return CUSOLVER_STATUS_INVALID_VALUE;
    }
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnCreateParams(cusolverDnParams_t* params) {
    if (params == nullptr) return CUSOLVER_STATUS_INVALID_VALUE;
    *params = new (std::nothrow) cusolverDnParams();
    return *params == nullptr ? CUSOLVER_STATUS_ALLOC_FAILED : CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDestroyParams(cusolverDnParams_t params) {
    delete params;
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnCreateSyevjInfo(syevjInfo_t* info) {
    if (info == nullptr) return CUSOLVER_STATUS_INVALID_VALUE;
    *info = new (std::nothrow) syevjInfo();
    return *info == nullptr ? CUSOLVER_STATUS_ALLOC_FAILED : CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDestroySyevjInfo(syevjInfo_t info) {
    delete info;
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnXsyevjSetTolerance(syevjInfo_t info, double tolerance) {
    if (info == nullptr || !(tolerance >= 0.0)) return CUSOLVER_STATUS_INVALID_VALUE;
    info->tolerance = tolerance;
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnXsyevjSetMaxSweeps(syevjInfo_t info, int max_sweeps) {
    if (info == nullptr || max_sweeps <= 0) return CUSOLVER_STATUS_INVALID_VALUE;
    info->max_sweeps = max_sweeps;
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnXsyevjSetSortEig(syevjInfo_t info, int sort_eig) {
    if (info == nullptr || (sort_eig != 0 && sort_eig != 1))
        return CUSOLVER_STATUS_INVALID_VALUE;
    info->sort_eig = sort_eig;
    return CUSOLVER_STATUS_SUCCESS;
}

static bool vibeqc_double_solver_types(cudaDataType a, cudaDataType w,
                                       cudaDataType compute) {
    return a == CUDA_R_64F && w == CUDA_R_64F && compute == CUDA_R_64F;
}

static cusolverStatus_t vibeqc_syevd_workspace_bytes(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo,
    int n, const double* A, int lda, const double* W,
    size_t* workspace_device_bytes, size_t* workspace_host_bytes) {
    if (workspace_device_bytes == nullptr || workspace_host_bytes == nullptr)
        return CUSOLVER_STATUS_INVALID_VALUE;
    int lwork = 0;
    const cusolverStatus_t status = cusolverDnDsyevd_bufferSize(
        handle, jobz, uplo, n, A, lda, W, &lwork);
    if (status != CUSOLVER_STATUS_SUCCESS) return status;
    *workspace_device_bytes = static_cast<size_t>(lwork) * sizeof(double);
    *workspace_host_bytes = 0;
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnXsyevd_bufferSize(
    cusolverDnHandle_t handle, cusolverDnParams_t params, cusolverEigMode_t jobz,
    cublasFillMode_t uplo, int64_t n, cudaDataType data_type_a, const void* A,
    int64_t lda, cudaDataType data_type_w, const void* W, cudaDataType compute_type,
    size_t* workspace_device_bytes, size_t* workspace_host_bytes) {
    if (params == nullptr || n < 0 || lda < 0 || n > INT_MAX || lda > INT_MAX ||
        !vibeqc_double_solver_types(data_type_a, data_type_w, compute_type))
        return CUSOLVER_STATUS_INVALID_VALUE;
    return vibeqc_syevd_workspace_bytes(
        handle, jobz, uplo, static_cast<int>(n), static_cast<const double*>(A),
        static_cast<int>(lda), static_cast<const double*>(W),
        workspace_device_bytes, workspace_host_bytes);
}

cusolverStatus_t cusolverDnXsyevd(
    cusolverDnHandle_t handle, cusolverDnParams_t params, cusolverEigMode_t jobz,
    cublasFillMode_t uplo, int64_t n, cudaDataType data_type_a, void* A,
    int64_t lda, cudaDataType data_type_w, void* W, cudaDataType compute_type,
    void* workspace_device, size_t workspace_device_bytes, void* /*workspace_host*/,
    size_t /*workspace_host_bytes*/, int* dev_info) {
    if (params == nullptr || n < 0 || lda < 0 || n > INT_MAX || lda > INT_MAX ||
        !vibeqc_double_solver_types(data_type_a, data_type_w, compute_type) ||
        workspace_device_bytes % sizeof(double) != 0 ||
        workspace_device_bytes / sizeof(double) > static_cast<size_t>(INT_MAX))
        return CUSOLVER_STATUS_INVALID_VALUE;
    return cusolverDnDsyevd(
        handle, jobz, uplo, static_cast<int>(n), static_cast<double*>(A),
        static_cast<int>(lda), static_cast<double*>(W),
        static_cast<double*>(workspace_device),
        static_cast<int>(workspace_device_bytes / sizeof(double)), dev_info);
}

cusolverStatus_t cusolverDnXsyevBatched_bufferSize(
    cusolverDnHandle_t handle, cusolverDnParams_t params, cusolverEigMode_t jobz,
    cublasFillMode_t uplo, int n, cudaDataType data_type_a, const void* A,
    int lda, cudaDataType data_type_w, const void* W, cudaDataType compute_type,
    size_t* workspace_device_bytes, size_t* workspace_host_bytes, int batch_size) {
    if (params == nullptr || batch_size <= 0 ||
        !vibeqc_double_solver_types(data_type_a, data_type_w, compute_type))
        return CUSOLVER_STATUS_INVALID_VALUE;
    return vibeqc_syevd_workspace_bytes(
        handle, jobz, uplo, n, static_cast<const double*>(A), lda,
        static_cast<const double*>(W), workspace_device_bytes, workspace_host_bytes);
}

cusolverStatus_t cusolverDnXsyevBatched(
    cusolverDnHandle_t handle, cusolverDnParams_t params, cusolverEigMode_t jobz,
    cublasFillMode_t uplo, int n, cudaDataType data_type_a, void* A,
    int lda, cudaDataType data_type_w, void* W, cudaDataType compute_type,
    void* workspace_device, size_t workspace_device_bytes, void* /*workspace_host*/,
    size_t /*workspace_host_bytes*/, int* dev_info, int batch_size) {
    if (params == nullptr || batch_size <= 0 || n < 0 || lda < n ||
        !vibeqc_double_solver_types(data_type_a, data_type_w, compute_type) ||
        workspace_device_bytes % sizeof(double) != 0 ||
        workspace_device_bytes / sizeof(double) > static_cast<size_t>(INT_MAX) ||
        A == nullptr || W == nullptr || workspace_device == nullptr || dev_info == nullptr)
        return CUSOLVER_STATUS_INVALID_VALUE;
    auto* matrices = static_cast<double*>(A);
    auto* values = static_cast<double*>(W);
    const size_t matrix_stride = static_cast<size_t>(lda) * static_cast<size_t>(n);
    const int lwork = static_cast<int>(workspace_device_bytes / sizeof(double));
    for (int batch = 0; batch < batch_size; ++batch) {
        const cusolverStatus_t status = cusolverDnDsyevd(
            handle, jobz, uplo, n, matrices + static_cast<size_t>(batch) * matrix_stride,
            lda, values + static_cast<size_t>(batch) * static_cast<size_t>(n),
            static_cast<double*>(workspace_device), lwork, dev_info + batch);
        if (status != CUSOLVER_STATUS_SUCCESS) return status;
    }
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDsyevjBatched_bufferSize(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo,
    int n, const double* A, int lda, const double* W, int* lwork,
    syevjInfo_t params, int batch_size) {
    if (params == nullptr || batch_size <= 0) return CUSOLVER_STATUS_INVALID_VALUE;
    return cusolverDnDsyevd_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork);
}

cusolverStatus_t cusolverDnDsyevjBatched(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo,
    int n, double* A, int lda, double* W, double* work, int lwork,
    int* dev_info, syevjInfo_t params, int batch_size) {
    if (params == nullptr || batch_size <= 0 || n < 0 || lda < n || A == nullptr ||
        W == nullptr || work == nullptr || dev_info == nullptr)
        return CUSOLVER_STATUS_INVALID_VALUE;
    const size_t matrix_stride = static_cast<size_t>(lda) * static_cast<size_t>(n);
    for (int batch = 0; batch < batch_size; ++batch) {
        const cusolverStatus_t status = cusolverDnDsyevd(
            handle, jobz, uplo, n, A + static_cast<size_t>(batch) * matrix_stride, lda,
            W + static_cast<size_t>(batch) * static_cast<size_t>(n), work, lwork,
            dev_info + batch);
        if (status != CUSOLVER_STATUS_SUCCESS) return status;
    }
    return CUSOLVER_STATUS_SUCCESS;
}

}  // extern "C"
"""
# INT_MAX is used by the generic wrappers.
replace_once(
    cusolver_source,
    "#include <limits>\n",
    "#include <limits>\n#include <climits>\n",
    "cuSOLVER compatibility integer limits include",
)
with cusolver_source.open("a", encoding="utf-8") as handle:
    handle.write(compat_impl)
