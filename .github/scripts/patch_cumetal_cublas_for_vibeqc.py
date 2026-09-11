"""Provide cuBLAS workspace and atomics controls used by VibeQC's CUDA backend."""

from __future__ import annotations

import os
from pathlib import Path

source = Path(os.environ["CUMETAL_SOURCE"])
header = source / "runtime/api/cublas_v2.h"
implementation = source / "runtime/rt/cublas.cpp"


def replace_once(path: Path, old: str, new: str, description: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(
            f"{description}: expected one patch anchor, found {count} in {path}"
        )
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


replace_once(
    header,
    "typedef enum cublasMath_t {\n",
    "typedef enum cublasAtomicsMode_t {\n"
    "    CUBLAS_ATOMICS_NOT_ALLOWED = 0,\n"
    "    CUBLAS_ATOMICS_ALLOWED = 1,\n"
    "} cublasAtomicsMode_t;\n\n"
    "typedef enum cublasMath_t {\n",
    "cuBLAS atomics mode enum",
)
replace_once(
    header,
    "cublasStatus_t cublasSetMathMode(cublasHandle_t handle, cublasMath_t mode);\n",
    "cublasStatus_t cublasSetMathMode(cublasHandle_t handle, cublasMath_t mode);\n"
    "cublasStatus_t cublasSetAtomicsMode(cublasHandle_t handle, cublasAtomicsMode_t mode);\n"
    "cublasStatus_t cublasGetAtomicsMode(cublasHandle_t handle, cublasAtomicsMode_t* mode);\n",
    "cuBLAS atomics mode declarations",
)
replace_once(
    header,
    "cublasStatus_t cublasGetStream(cublasHandle_t handle, cudaStream_t* stream_id);\n",
    "cublasStatus_t cublasGetStream(cublasHandle_t handle, cudaStream_t* stream_id);\n"
    "cublasStatus_t cublasSetWorkspace(cublasHandle_t handle, void* workspace, size_t workspace_size);\n",
    "cublasSetWorkspace declaration",
)

replace_once(
    implementation,
    "    cublasPointerMode_t pointer_mode = CUBLAS_POINTER_MODE_HOST;\n"
    "    std::mutex mutex;\n",
    "    cublasPointerMode_t pointer_mode = CUBLAS_POINTER_MODE_HOST;\n"
    "    cublasAtomicsMode_t atomics_mode = CUBLAS_ATOMICS_NOT_ALLOWED;\n"
    "    void* workspace = nullptr;\n"
    "    std::size_t workspace_size = 0;\n"
    "    std::mutex mutex;\n",
    "cublasContext workspace state",
)

workspace_impl = r"""

// The pinned provider's GEMM paths do not select atomics-based alternatives.
// Retain the standard handle control; ALLOWED permits but never requires such
// an implementation. VibeQC explicitly chooses NOT_ALLOWED for tensor work.
cublasStatus_t cublasSetAtomicsMode(cublasHandle_t handle, cublasAtomicsMode_t mode) {
    if (handle == nullptr) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (mode != CUBLAS_ATOMICS_NOT_ALLOWED && mode != CUBLAS_ATOMICS_ALLOWED)
        return CUBLAS_STATUS_INVALID_VALUE;
    std::lock_guard<std::mutex> lock(handle->mutex);
    handle->atomics_mode = mode;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetAtomicsMode(cublasHandle_t handle, cublasAtomicsMode_t* mode) {
    if (handle == nullptr) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (mode == nullptr) return CUBLAS_STATUS_INVALID_VALUE;
    std::lock_guard<std::mutex> lock(handle->mutex);
    *mode = handle->atomics_mode;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetWorkspace(cublasHandle_t handle, void* workspace,
                                  size_t workspace_size) {
    if (handle == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    if (workspace_size != 0) {
        if (workspace == nullptr || cumetalRuntimeIsDevicePointer(workspace) == 0) {
            return CUBLAS_STATUS_INVALID_VALUE;
        }
    }
    std::lock_guard<std::mutex> lock(handle->mutex);
    handle->workspace = workspace;
    handle->workspace_size = workspace_size;
    return CUBLAS_STATUS_SUCCESS;
}
"""
replace_once(
    implementation,
    "cublasStatus_t cublasGetStream(cublasHandle_t handle, cudaStream_t* stream_id) {\n"
    "    if (handle == nullptr || stream_id == nullptr) {\n"
    "        return CUBLAS_STATUS_NOT_INITIALIZED;\n"
    "    }\n"
    "    std::lock_guard<std::mutex> lock(handle->mutex);\n"
    "    *stream_id = handle->stream;\n"
    "    return CUBLAS_STATUS_SUCCESS;\n"
    "}\n",
    "cublasStatus_t cublasGetStream(cublasHandle_t handle, cudaStream_t* stream_id) {\n"
    "    if (handle == nullptr || stream_id == nullptr) {\n"
    "        return CUBLAS_STATUS_NOT_INITIALIZED;\n"
    "    }\n"
    "    std::lock_guard<std::mutex> lock(handle->mutex);\n"
    "    *stream_id = handle->stream;\n"
    "    return CUBLAS_STATUS_SUCCESS;\n"
    "}\n" + workspace_impl,
    "cublasSetWorkspace implementation",
)
