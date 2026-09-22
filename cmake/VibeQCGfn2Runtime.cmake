include_guard(GLOBAL)

include(GNUInstallDirs)

function(vibeqc_add_gfn2_runtime target)
  set(_gfn2_root "${CMAKE_CURRENT_SOURCE_DIR}/src/xtb/gfn2_runtime")
  target_sources(${target} PRIVATE
    ${_gfn2_root}/src/cpu_dispatch/features.cpp
    ${_gfn2_root}/src/model/common/integrals_kernels_baseline.cpp
    ${_gfn2_root}/src/model/gfn2/mulliken_kernels_baseline.cpp
    ${_gfn2_root}/src/model/common/sto.cpp
    ${_gfn2_root}/src/model/common/integrals.cpp
    ${_gfn2_root}/src/model/common/scc_mixer.cpp
    ${_gfn2_root}/src/model/gfn2/eigensolver.cpp
    ${_gfn2_root}/src/model/gfn2/periodic_embedding.cpp
    ${_gfn2_root}/src/model/gfn2/es2.cpp
    ${_gfn2_root}/src/backends/common/gfn2_plan_schema.cpp
    ${_gfn2_root}/src/model/gfn2/aes2.cpp
    ${_gfn2_root}/src/model/gfn2/alpb.cpp
    ${_gfn2_root}/src/model/gfn2/basis.cpp
    ${_gfn2_root}/src/model/gfn2/coordination.cpp
    ${_gfn2_root}/src/model/gfn2/d4.cpp
    ${_gfn2_root}/src/model/gfn2/es3.cpp
    ${_gfn2_root}/src/model/gfn2/external_point_charges.cpp
    ${_gfn2_root}/src/model/gfn2/force.cpp
    ${_gfn2_root}/src/model/gfn2/h0.cpp
    ${_gfn2_root}/src/model/gfn2/lattice.cpp
    ${_gfn2_root}/src/model/gfn2/mulliken.cpp
    ${_gfn2_root}/src/model/gfn2/mulliken_kernels.cpp
    ${_gfn2_root}/src/model/gfn2/periodic_ewald.cpp
    ${_gfn2_root}/src/model/gfn2/periodic_integrals.cpp
    ${_gfn2_root}/src/model/gfn2/periodic_multipole.cpp
    ${_gfn2_root}/src/model/gfn2/periodic_topology.cpp
    ${_gfn2_root}/src/model/gfn2/repulsion.cpp
    ${_gfn2_root}/src/model/gfn2/scc_driver.cpp
    ${_gfn2_root}/src/model/gfn2/scc_mixer.cpp
    ${_gfn2_root}/src/model/gfn2/spin.cpp
    ${_gfn2_root}/src/model/gfn2/wavefunction.cpp
    ${_gfn2_root}/src/runtime/gfn2_cpu_execution.cpp)

  # Native molecular GFN2 CUDA bootstrap. The CUDA owner is pinned separately
  # from the later CPU snapshot so no GFN1 runtime is pulled into VibeQC.
  # See CUDA_SOURCE_PROVENANCE.json for the exact source cohort and adaptations.
  if(VIBEQC_ENABLE_CUDA AND NOT VIBEQC_PYTHON_WHEEL AND
     NOT VIBEQC_CUDA_PROVIDER STREQUAL "cumetal")
    set(_gfn2_cuda_sources
      ${_gfn2_root}/src/backends/cuda/cuda_runtime.cu
      ${_gfn2_root}/src/runtime/cuda_descriptor_validation.cu
      ${_gfn2_root}/src/runtime/gfn2_cuda_topology_staging.cu
      ${_gfn2_root}/src/runtime/gfn2_cuda_execution.cu
      ${_gfn2_root}/src/runtime/result_owner_cuda.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_aes2.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_classical_force.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_d4.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_density.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_electric_field.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_electronic_gradient.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_energy_force_execution.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_eigensolver.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_es2.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_es3.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_external_point_charges.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_force_composition.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_geometry.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_preprocessing.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_public_result_bridge.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_h0_force.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_hamiltonian.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_hamiltonian_force.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_inference_publication.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_integrals.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_mulliken.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_occupations.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_pairlist.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_parameters.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_plan_schema.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_periodic_embedding.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_post_scc_potential.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_repulsion.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_scc.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_scc_bridge.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_scc_classical_energy.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_scc_energy.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_scc_free_energy.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_scc_iteration.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_scc_loop.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_scc_iteration_arena.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_scc_iteration_control.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_scc_iteration_initialize.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_scc_iteration_reports.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_scc_mixer.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_scc_potential.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_scc_publication.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_scc_setup_eigensolver.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_scc_setup_inputs.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_scc_setup_topology.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_spin.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_terminal_classical_energy.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_total_energy.cu
    )
    add_library(vibeqc_gfn2_cuda STATIC ${_gfn2_cuda_sources})
    set(VIBEQC_GFN2_ELECTRONIC_CUDA_HEADER
        "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_gfn2_electronic_native.cuh")
    vibeqc_register_generated_sources(
      NAME vibeqc_gfn2_electronic_cuda_codegen
      TARGET vibeqc_gfn2_cuda
      GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_gfn2_electronic_cuda.py"
      OUTPUTS "${VIBEQC_GFN2_ELECTRONIC_CUDA_HEADER}"
      DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/method/gfn2_electronic.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/method/gfn2_electronic_contract.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/method/gfn2_electronic_runtime.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/tensor/ad_program.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/tensor/scalar_cpp.py"
      ARGS --output "${VIBEQC_GFN2_ELECTRONIC_CUDA_HEADER}"
      COMMENT "Generating compiler-owned GFN2 CUDA electronic pair science")
    # Both CPU and CUDA consume the one compiler-owned pair artifact.
    add_dependencies(vibeqc_gfn2_cuda vibeqc_gfn2_pair_cpu_codegen)
    target_include_directories(vibeqc_gfn2_cuda PRIVATE
      "${CMAKE_CURRENT_BINARY_DIR}/generated"
      ${_gfn2_root}
      ${_gfn2_root}/include
      ${_gfn2_root}/src
      ${CMAKE_CURRENT_BINARY_DIR}/generated
      ${CMAKE_CURRENT_SOURCE_DIR}/include
      ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_compile_definitions(vibeqc_gfn2_cuda PRIVATE XTBLOOM_HAS_CUDA=1)
    set_target_properties(vibeqc_gfn2_cuda PROPERTIES
      POSITION_INDEPENDENT_CODE ON
      CUDA_STANDARD 20
      CUDA_STANDARD_REQUIRED ON
      CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_RESOLVE_DEVICE_SYMBOLS ON
      JOB_POOL_COMPILE vibeqc_cuda_compile)
    set_source_files_properties(
      ${_gfn2_root}/src/backends/cuda/gfn2_pairlist.cu
      ${_gfn2_root}/src/backends/cuda/gfn2_geometry.cu
      PROPERTIES COMPILE_OPTIONS "-fmad=false")
    target_compile_definitions(${target} PRIVATE VIBEQC_HAS_GFN2_CUDA=1)

    # The CUDA archive resolves its own device symbols. Consume the complete
    # archive so CUDA registration/device-link objects cannot be discarded,
    # without propagating separable compilation to unrelated VibeQC CUDA TUs.
    target_link_libraries(${target} PRIVATE
      "$<LINK_LIBRARY:WHOLE_ARCHIVE,vibeqc_gfn2_cuda>")
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
      vibeqc_attach_cuda_driver_implib(${target})
    else()
      target_link_libraries(${target} PRIVATE CUDA::cuda_driver)
    endif()
  endif()

  target_include_directories(${target} PRIVATE
    ${_gfn2_root}
    ${_gfn2_root}/include
    ${_gfn2_root}/src)
  if(CMAKE_DL_LIBS)
    target_link_libraries(${target} PRIVATE ${CMAKE_DL_LIBS})
  endif()

  if(VIBEQC_BUNDLE_XTB_OPENBLAS)
    if(NOT VIBEQC_PYTHON_WHEEL)
      message(FATAL_ERROR "VIBEQC_BUNDLE_XTB_OPENBLAS requires VIBEQC_PYTHON_WHEEL=ON")
    endif()
    if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
      message(FATAL_ERROR "The scoped GFN2 wheel provider is currently qualified on Linux only")
    endif()
    string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _gfn2_processor)
    if(_gfn2_processor MATCHES "^(x86_64|amd64)$")
      set(_gfn2_arch x86_64)
    elseif(_gfn2_processor MATCHES "^(aarch64|arm64)$")
      set(_gfn2_arch aarch64)
    else()
      message(FATAL_ERROR "Unsupported GFN2 wheel architecture: ${CMAKE_SYSTEM_PROCESSOR}")
    endif()
    execute_process(
      COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/xtb/resolve-openblas-wheel.py"
        --manifest "${CMAKE_CURRENT_SOURCE_DIR}/tools/xtb/scipy_openblas32_manifest.json"
        --platform linux
        --architecture "${_gfn2_arch}"
      RESULT_VARIABLE _gfn2_openblas_status
      OUTPUT_VARIABLE _gfn2_openblas_json
      ERROR_VARIABLE _gfn2_openblas_error
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT _gfn2_openblas_status EQUAL 0)
      message(FATAL_ERROR
        "Failed to resolve the reviewed GFN2 OpenBLAS provider:\n${_gfn2_openblas_error}")
    endif()
    string(JSON _gfn2_openblas_library GET "${_gfn2_openblas_json}" provider_path)
    string(JSON _gfn2_openblas_prefix GET "${_gfn2_openblas_json}" expected_config_prefix)
    get_filename_component(_gfn2_openblas_dir "${_gfn2_openblas_library}" DIRECTORY)

    set_source_files_properties(
      ${_gfn2_root}/src/runtime/openblas_lp64_shim.c PROPERTIES LANGUAGE CXX)
    add_library(vibeqc_gfn2_openblas_shim SHARED
      ${_gfn2_root}/src/runtime/openblas_lp64_shim.c)
    target_link_libraries(vibeqc_gfn2_openblas_shim PRIVATE "${_gfn2_openblas_library}")
    target_link_options(vibeqc_gfn2_openblas_shim PRIVATE "LINKER:--no-as-needed")
    set_target_properties(vibeqc_gfn2_openblas_shim PROPERTIES
      OUTPUT_NAME xtbloom_openblas_lp64_shim
      CXX_VISIBILITY_PRESET hidden
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
      BUILD_RPATH "${_gfn2_openblas_dir}"
      INSTALL_RPATH "")
    install(TARGETS vibeqc_gfn2_openblas_shim
      LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
    target_compile_definitions(${target} PRIVATE
      XTBLOOM_CONFIGURED_WHEEL_OPENBLAS=1
      "XTBLOOM_CONFIGURED_WHEEL_OPENBLAS_CONFIG_PREFIX=\"${_gfn2_openblas_prefix}\"")
    add_dependencies(${target} vibeqc_gfn2_openblas_shim)
  elseif(XTBLOOM_CPU_LINALG_LIBRARY)
    target_compile_definitions(${target} PRIVATE
      "XTBLOOM_CONFIGURED_CPU_LINALG_RUNTIME=\"${XTBLOOM_CPU_LINALG_LIBRARY}\"")
  endif()
endfunction()
