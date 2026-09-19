# Provider-free CUDA host-library trampolines for Python wheels.
#
# VibeQC's ordinary native SDK build still links CUDA::cudart/cuBLAS/cuSOLVER.
# Python wheels instead compile against CUDA headers and define the referenced
# host symbols through hidden Implib.so trampolines. Each trampoline lazily
# dlopens the reviewed CUDA-12 SONAME at runtime, so wheel construction never
# needs the proprietary provider shared libraries and the final ELF carries no
# CUDA provider DT_NEEDED entries.

function(vibeqc_write_cuda_symbol_file output_path)
  file(WRITE "${output_path}" "")
  foreach(symbol IN LISTS ARGN)
    file(APPEND "${output_path}" "${symbol}\n")
  endforeach()
endfunction()

function(vibeqc_attach_cuda_implib target)
  if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "VibeQC provider-free CUDA wheels currently require Linux ELF")
  endif()
  string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" processor)
  if(processor MATCHES "^(x86_64|amd64)$")
    set(implib_target x86_64)
  elseif(processor MATCHES "^(aarch64|arm64)$")
    set(implib_target aarch64)
  else()
    message(FATAL_ERROR "unsupported CUDA wheel architecture: ${CMAKE_SYSTEM_PROCESSOR}")
  endif()

  set(implib_root "${CMAKE_CURRENT_SOURCE_DIR}/cmake/3rdparty/implib")
  set(generator "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_cuda_implib.py")
  set(output_dir "${CMAKE_CURRENT_BINARY_DIR}/generated/cuda_implib")
  file(MAKE_DIRECTORY "${output_dir}")

  # Object-level names after CUDA header macro expansion. Keep these curated:
  # -z defs on the final wheel target turns any newly introduced CUDA host API
  # into a link failure instead of silently adding a provider dependency.
  set(VIBEQC_CUDART_SYMBOLS
    __cudaInitModule
    __cudaPopCallConfiguration
    __cudaPushCallConfiguration
    __cudaRegisterFatBinary
    __cudaRegisterFatBinaryEnd
    __cudaRegisterFunction
    __cudaRegisterVar
    __cudaUnregisterFatBinary
    cudaDeviceGetAttribute
    cudaDeviceGetLimit
    cudaDeviceSetLimit
    cudaDriverGetVersion
    cudaEventCreate
    cudaEventCreateWithFlags
    cudaEventDestroy
    cudaEventElapsedTime
    cudaEventRecord
    cudaEventSynchronize
    cudaFree
    cudaFreeAsync
    cudaFreeHost
    cudaFuncGetAttributes
    cudaGetDevice
    cudaGetDeviceCount
    cudaGetDeviceProperties_v2
    cudaGetErrorString
    cudaGetLastError
    cudaGraphDestroy
    cudaGraphExecDestroy
    cudaGraphInstantiate
    cudaGraphLaunch
    cudaGraphUpload
    cudaLaunchKernel
    cudaMalloc
    cudaMallocAsync
    cudaMallocHost
    cudaMemGetInfo
    cudaMemcpy
    cudaMemcpy2DAsync
    cudaMemcpyAsync
    cudaMemsetAsync
    cudaOccupancyMaxActiveBlocksPerMultiprocessor
    cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
    cudaPeekAtLastError
    cudaPointerGetAttributes
    cudaRuntimeGetVersion
    cudaSetDevice
    cudaStreamBeginCapture
    cudaStreamCreateWithFlags
    cudaStreamDestroy
    cudaStreamEndCapture
    cudaStreamIsCapturing
    cudaStreamSynchronize
  )
  set(VIBEQC_CUBLAS_SYMBOLS
    cublasCreate_v2
    cublasDaxpy_v2
    cublasDdot_v2
    cublasDestroy_v2
    cublasDgeam
    cublasDgemmStridedBatched
    cublasDgemm_v2
    cublasDgemv_v2
    cublasDsyrk_v2
    cublasGetProperty
    cublasGetVersion_v2
    cublasSetMathMode
    cublasSetPointerMode_v2
    cublasSetStream_v2
    cublasSetWorkspace_v2
    cublasSgemm_v2
    cublasSgemmStridedBatched
  )
  set(VIBEQC_CUSOLVER_SYMBOLS
    cusolverDnCreate
    cusolverDnCreateParams
    cusolverDnCreateSyevjInfo
    cusolverDnDestroy
    cusolverDnDestroyParams
    cusolverDnDestroySyevjInfo
    cusolverDnDsyevjBatched
    cusolverDnDsyevjBatched_bufferSize
    cusolverDnSetStream
    cusolverDnXsyevBatched
    cusolverDnXsyevBatched_bufferSize
    cusolverDnXsyevd
    cusolverDnXsyevd_bufferSize
    cusolverDnXsyevjSetMaxSweeps
    cusolverDnXsyevjSetSortEig
    cusolverDnXsyevjSetTolerance
    cusolverGetProperty
  )

  set(bases libcudart.so libcublas.so libcusolver.so)
  set(sonames libcudart.so.12 libcublas.so.12 libcusolver.so.11)
  set(symbol_sets VIBEQC_CUDART_SYMBOLS VIBEQC_CUBLAS_SYMBOLS VIBEQC_CUSOLVER_SYMBOLS)
  foreach(base soname symbol_set IN ZIP_LISTS bases sonames symbol_sets)
    set(symbol_file "${output_dir}/${base}.symbols")
    vibeqc_write_cuda_symbol_file("${symbol_file}" ${${symbol_set}})
    execute_process(
      COMMAND "${Python3_EXECUTABLE}" "${generator}"
              --base-name "${base}"
              --symbol-list "${symbol_file}"
              --load-name "${soname}"
              --target "${implib_target}"
              --implib-root "${implib_root}"
              --outdir "${output_dir}"
      COMMAND_ERROR_IS_FATAL ANY)
    target_sources(${target} PRIVATE
      "${output_dir}/${base}.tramp.S"
      "${output_dir}/${base}.init.c")
  endforeach()

  target_include_directories(${target} BEFORE PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/runtime/nvidia_host_api")
  target_include_directories(${target} PRIVATE ${VIBEQC_CUDA_TOOLKIT_INCLUDE_DIRS})
  target_link_libraries(${target} PRIVATE ${CMAKE_DL_LIBS})
  target_link_options(${target} PRIVATE "LINKER:-z,defs")
endfunction()
