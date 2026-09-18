#pragma once

/*
 * Minimal NVIDIA host-library ABI declarations used by VibeQC's provider-free
 * Linux wheels. This VibeQC-owned interface header declares only opaque types,
 * stable enum values, and C function signatures that VibeQC actually calls.
 *
 * Native SDK builds continue to use NVIDIA's cublas_v2.h/cusolverDn.h. Wheel
 * builds put this directory first on the include path so nvcc needs only its
 * compiler/runtime headers; Implib.so trampolines resolve the deployed CUDA-12
 * providers lazily at runtime.
 */

#include <cuda_runtime_api.h>
#include <library_types.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cublasContext* cublasHandle_t;
typedef uint32_t cublasStatus_t;
enum {
  CUBLAS_STATUS_SUCCESS = 0,
  CUBLAS_STATUS_NOT_INITIALIZED = 1,
  CUBLAS_STATUS_ALLOC_FAILED = 3,
  CUBLAS_STATUS_INVALID_VALUE = 7,
  CUBLAS_STATUS_ARCH_MISMATCH = 8,
  CUBLAS_STATUS_MAPPING_ERROR = 11,
  CUBLAS_STATUS_EXECUTION_FAILED = 13,
  CUBLAS_STATUS_INTERNAL_ERROR = 14,
  CUBLAS_STATUS_NOT_SUPPORTED = 15,
  CUBLAS_STATUS_LICENSE_ERROR = 16,
};

typedef enum { CUBLAS_OP_N = 0, CUBLAS_OP_T = 1, CUBLAS_OP_C = 2 } cublasOperation_t;
typedef enum {
  CUBLAS_FILL_MODE_LOWER = 0,
  CUBLAS_FILL_MODE_UPPER = 1,
  CUBLAS_FILL_MODE_FULL = 2,
} cublasFillMode_t;
typedef enum { CUBLAS_POINTER_MODE_HOST = 0, CUBLAS_POINTER_MODE_DEVICE = 1 } cublasPointerMode_t;
typedef enum { CUBLAS_DEFAULT_MATH = 0, CUBLAS_PEDANTIC_MATH = 2 } cublasMath_t;

cublasStatus_t cublasCreate_v2(cublasHandle_t* handle);
cublasStatus_t cublasDestroy_v2(cublasHandle_t handle);
cublasStatus_t cublasGetProperty(libraryPropertyType type, int* value);
cublasStatus_t cublasGetVersion_v2(cublasHandle_t handle, int* version);
cublasStatus_t cublasSetWorkspace_v2(cublasHandle_t handle, void* workspace,
                                     size_t workspace_size_in_bytes);
cublasStatus_t cublasSetStream_v2(cublasHandle_t handle, cudaStream_t stream);
cublasStatus_t cublasSetPointerMode_v2(cublasHandle_t handle, cublasPointerMode_t mode);
cublasStatus_t cublasSetMathMode(cublasHandle_t handle, cublasMath_t mode);
cublasStatus_t cublasDaxpy_v2(cublasHandle_t handle, int n, const double* alpha, const double* x,
                              int incx, double* y, int incy);
cublasStatus_t cublasDdot_v2(cublasHandle_t handle, int n, const double* x, int incx,
                             const double* y, int incy, double* result);
cublasStatus_t cublasDgemv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n,
                              const double* alpha, const double* a, int lda, const double* x,
                              int incx, const double* beta, double* y, int incy);
cublasStatus_t cublasDgemm_v2(cublasHandle_t handle, cublasOperation_t transa,
                              cublasOperation_t transb, int m, int n, int k, const double* alpha,
                              const double* a, int lda, const double* b, int ldb,
                              const double* beta, double* c, int ldc);
cublasStatus_t cublasDsyrk_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans,
                              int n, int k, const double* alpha, const double* a, int lda,
                              const double* beta, double* c, int ldc);
cublasStatus_t cublasDgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa,
                                         cublasOperation_t transb, int m, int n, int k,
                                         const double* alpha, const double* a, int lda,
                                         long long int stride_a, const double* b, int ldb,
                                         long long int stride_b, const double* beta, double* c,
                                         int ldc, long long int stride_c, int batch_count);
cublasStatus_t cublasDgeam(cublasHandle_t handle, cublasOperation_t transa,
                           cublasOperation_t transb, int m, int n, const double* alpha,
                           const double* a, int lda, const double* beta, const double* b, int ldb,
                           double* c, int ldc);

#define cublasCreate cublasCreate_v2
#define cublasDestroy cublasDestroy_v2
#define cublasGetVersion cublasGetVersion_v2
#define cublasSetWorkspace cublasSetWorkspace_v2
#define cublasSetStream cublasSetStream_v2
#define cublasSetPointerMode cublasSetPointerMode_v2
#define cublasDaxpy cublasDaxpy_v2
#define cublasDdot cublasDdot_v2
#define cublasDgemv cublasDgemv_v2
#define cublasDgemm cublasDgemm_v2
#define cublasDsyrk cublasDsyrk_v2

typedef struct cusolverDnContext* cusolverDnHandle_t;
typedef struct cusolverDnParams* cusolverDnParams_t;
typedef struct syevjInfo* syevjInfo_t;
typedef uint32_t cusolverStatus_t;
enum {
  CUSOLVER_STATUS_SUCCESS = 0,
  CUSOLVER_STATUS_NOT_INITIALIZED = 1,
  CUSOLVER_STATUS_ALLOC_FAILED = 2,
  CUSOLVER_STATUS_INVALID_VALUE = 3,
  CUSOLVER_STATUS_ARCH_MISMATCH = 4,
  CUSOLVER_STATUS_MAPPING_ERROR = 5,
  CUSOLVER_STATUS_EXECUTION_FAILED = 6,
  CUSOLVER_STATUS_INTERNAL_ERROR = 7,
  CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED = 8,
  CUSOLVER_STATUS_NOT_SUPPORTED = 9,
};
typedef enum { CUSOLVER_EIG_MODE_NOVECTOR = 0, CUSOLVER_EIG_MODE_VECTOR = 1 } cusolverEigMode_t;

cusolverStatus_t cusolverDnCreate(cusolverDnHandle_t* handle);
cusolverStatus_t cusolverDnDestroy(cusolverDnHandle_t handle);
cusolverStatus_t cusolverDnSetStream(cusolverDnHandle_t handle, cudaStream_t stream);
cusolverStatus_t cusolverDnCreateParams(cusolverDnParams_t* params);
cusolverStatus_t cusolverDnDestroyParams(cusolverDnParams_t params);
cusolverStatus_t cusolverDnCreateSyevjInfo(syevjInfo_t* info);
cusolverStatus_t cusolverDnDestroySyevjInfo(syevjInfo_t info);
cusolverStatus_t cusolverDnXsyevjSetTolerance(syevjInfo_t info, double tolerance);
cusolverStatus_t cusolverDnXsyevjSetMaxSweeps(syevjInfo_t info, int max_sweeps);
cusolverStatus_t cusolverDnXsyevjSetSortEig(syevjInfo_t info, int sort_eig);
cusolverStatus_t cusolverDnDsyevjBatched_bufferSize(cusolverDnHandle_t handle,
                                                    cusolverEigMode_t jobz, cublasFillMode_t uplo,
                                                    int n, const double* a, int lda,
                                                    const double* w, int* lwork, syevjInfo_t params,
                                                    int batch_size);
cusolverStatus_t cusolverDnDsyevjBatched(cusolverDnHandle_t handle, cusolverEigMode_t jobz,
                                         cublasFillMode_t uplo, int n, double* a, int lda,
                                         double* w, double* work, int lwork, int* info,
                                         syevjInfo_t params, int batch_size);
cusolverStatus_t cusolverDnXsyevd_bufferSize(cusolverDnHandle_t handle, cusolverDnParams_t params,
                                             cusolverEigMode_t jobz, cublasFillMode_t uplo,
                                             int64_t n, cudaDataType data_type_a, const void* a,
                                             int64_t lda, cudaDataType data_type_w, const void* w,
                                             cudaDataType compute_type,
                                             size_t* workspace_bytes_device,
                                             size_t* workspace_bytes_host);
cusolverStatus_t cusolverDnXsyevd(cusolverDnHandle_t handle, cusolverDnParams_t params,
                                  cusolverEigMode_t jobz, cublasFillMode_t uplo, int64_t n,
                                  cudaDataType data_type_a, void* a, int64_t lda,
                                  cudaDataType data_type_w, void* w, cudaDataType compute_type,
                                  void* workspace_device, size_t workspace_bytes_device,
                                  void* workspace_host, size_t workspace_bytes_host, int* info);
cusolverStatus_t cusolverDnXsyevBatched_bufferSize(
    cusolverDnHandle_t handle, cusolverDnParams_t params, cusolverEigMode_t jobz,
    cublasFillMode_t uplo, int64_t n, cudaDataType data_type_a, const void* a, int64_t lda,
    cudaDataType data_type_w, const void* w, cudaDataType compute_type,
    size_t* workspace_bytes_device, size_t* workspace_bytes_host, int64_t batch_size);
cusolverStatus_t cusolverDnXsyevBatched(cusolverDnHandle_t handle, cusolverDnParams_t params,
                                        cusolverEigMode_t jobz, cublasFillMode_t uplo, int64_t n,
                                        cudaDataType data_type_a, void* a, int64_t lda,
                                        cudaDataType data_type_w, void* w,
                                        cudaDataType compute_type, void* workspace_device,
                                        size_t workspace_bytes_device, void* workspace_host,
                                        size_t workspace_bytes_host, int* info, int64_t batch_size);
cusolverStatus_t cusolverGetProperty(libraryPropertyType type, int* value);

#ifdef __cplusplus
}  // extern "C"
#endif
