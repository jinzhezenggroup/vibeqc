include_guard(GLOBAL)
include(CMakeParseArguments)

function(generativeqc_native_test target source)
  set(options NO_GENERATIVEQC NO_SRC_INCLUDE SKIP_77)
  set(multi_value_args LIBRARIES)
  cmake_parse_arguments(VNT "${options}" "" "${multi_value_args}" ${ARGN})
  add_executable(${target} ${source})
  if(NOT VNT_NO_GENERATIVEQC)
    target_link_libraries(${target} PRIVATE generativeqc ${VNT_LIBRARIES})
  elseif(VNT_LIBRARIES)
    target_link_libraries(${target} PRIVATE ${VNT_LIBRARIES})
  endif()
  if(NOT VNT_NO_SRC_INCLUDE)
    target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
  endif()
  add_test(NAME ${target} COMMAND ${target})
  if(VNT_SKIP_77)
    set_tests_properties(${target} PROPERTIES SKIP_RETURN_CODE 77)
  endif()
endfunction()

macro(generativeqc_add_native_tests)
  enable_testing()
  if(GENERATIVEQC_ENABLE_CUDA)
    generativeqc_native_test(generativeqc_hf_resource_layout_tests tests/native/test_hf_resource_layout.cpp)
    generativeqc_native_test(generativeqc_incremental_direct_jk_cuda_tests
                       tests/native/test_incremental_direct_jk_cuda.cpp
                       LIBRARIES CUDA::cudart SKIP_77)
    generativeqc_native_test(generativeqc_mean_field_setup_cuda_tests tests/native/test_mean_field_setup_cuda.cpp
                       LIBRARIES CUDA::cudart SKIP_77)
    generativeqc_native_test(generativeqc_cuda_quadrature_tests tests/native/test_cuda_quadrature.cpp
                       LIBRARIES CUDA::cudart SKIP_77)
    generativeqc_native_test(generativeqc_xc_response_cuda_tests tests/native/test_xc_response_cuda.cu
                       NO_GENERATIVEQC LIBRARIES CUDA::cudart SKIP_77)
    target_compile_definitions(generativeqc_xc_response_cuda_tests PRIVATE
                               GENERATIVEQC_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    # Run the same analytic/negative shared-policy suite through device algebra.
    generativeqc_native_test(generativeqc_df_final_validation_tests tests/native/test_final_state.cpp
                       LIBRARIES CUDA::cudart SKIP_77)
    target_compile_definitions(generativeqc_df_final_validation_tests PRIVATE GENERATIVEQC_TEST_DEVICE_FINAL_STATE=1)
    generativeqc_native_test(generativeqc_df_density_seed_tests tests/native/test_df_density_seed.cpp
                       LIBRARIES CUDA::cudart CUDA::cublas CUDA::cusolver SKIP_77)
    set_property(SOURCE "${GENERATIVEQC_GRID_SOURCE}" src/dft/cuda_ks.cpp src/dft/cuda_xc.cpp
                        src/dft/dispersion/d4_runtime_cuda.cu
                 APPEND PROPERTY COMPILE_DEFINITIONS GENERATIVEQC_TEST_HOOKS=1)
    add_executable(generativeqc_weighted_eri_probe tests/native/weighted_eri_probe.cpp)
    target_link_libraries(generativeqc_weighted_eri_probe PRIVATE generativeqc CUDA::cudart)
    target_include_directories(generativeqc_weighted_eri_probe PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    add_executable(generativeqc_df_value_probe tests/native/df_value_probe.cpp)
    target_link_libraries(generativeqc_df_value_probe PRIVATE generativeqc CUDA::cudart)
    target_include_directories(generativeqc_df_value_probe PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
  endif()

  generativeqc_native_test(generativeqc_tracked_allocator_tests tests/native/test_tracked_allocator.cpp
                     NO_GENERATIVEQC LIBRARIES Threads::Threads)
  generativeqc_native_test(generativeqc_fock_build_tests tests/native/test_fock_build.cpp)
  generativeqc_native_test(generativeqc_ecp_projector_tests tests/native/test_ecp_projector.cpp NO_GENERATIVEQC)
  generativeqc_native_test(generativeqc_ecp_capability_tests tests/native/test_ecp_capabilities.cpp)
  generativeqc_native_test(generativeqc_ecp_cpu_tests tests/native/test_ecp_cpu.cpp)
  target_sources(generativeqc_ecp_cpu_tests PRIVATE tests/native/ecp_reference.cpp)
  add_dependencies(generativeqc_ecp_projector_tests generativeqc_ecp_codegen)
  target_include_directories(generativeqc_ecp_projector_tests PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
  generativeqc_native_test(generativeqc_fock_provider_tests tests/native/test_fock_provider.cpp)
  generativeqc_native_test(generativeqc_fock_api_tests tests/native/test_fock_api.cpp NO_SRC_INCLUDE)
  generativeqc_native_test(generativeqc_native_tests tests/native/test_rhf.cpp)

  if(NOT WIN32)
    generativeqc_native_test(generativeqc_final_state_tests tests/native/test_final_state.cpp)
    generativeqc_native_test(generativeqc_ks_final_state_tests tests/native/test_ks_final_state.cpp)
    generativeqc_native_test(generativeqc_eigen_frame_tests tests/native/test_eigen_frame.cpp)
    generativeqc_native_test(generativeqc_warm_subspace_tests tests/native/test_warm_subspace.cpp)
    generativeqc_native_test(generativeqc_initial_density_tests tests/native/test_initial_density.cpp)
    generativeqc_native_test(generativeqc_preliminary_initial_guess_tests tests/native/test_preliminary_initial_guess.cpp)
    generativeqc_native_test(generativeqc_mp2_contract_tests tests/native/test_mp2_contract.cpp)
    generativeqc_native_test(generativeqc_native_gmres_tests tests/native/test_native_gmres.cpp)
    generativeqc_native_test(generativeqc_cc_lambda_preconditioner_tests tests/native/test_cc_lambda_preconditioner.cpp)
    generativeqc_native_test(generativeqc_mp2_gradient_tests tests/native/test_mp2_gradient.cpp)
    generativeqc_native_test(generativeqc_posthf_rank2_tests tests/native/test_posthf_rank2_transform.cpp)
  endif()

  if(GENERATIVEQC_ENABLE_CUDA AND NOT WIN32)
    generativeqc_native_test(generativeqc_cuda_reference_export_tests tests/native/test_cuda_reference_export.cpp
                       LIBRARIES CUDA::cudart)
    generativeqc_native_test(generativeqc_mp2_cuda_status_tests tests/native/test_mp2_cuda_status.cu
                       LIBRARIES CUDA::cudart CUDA::cublas SKIP_77)
    generativeqc_native_test(generativeqc_triples_response_cuda_tests tests/native/test_triples_response_cuda.cu
                       LIBRARIES CUDA::cudart SKIP_77)
    target_include_directories(generativeqc_triples_response_cuda_tests PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
    add_dependencies(generativeqc_triples_response_cuda_tests generativeqc_rccsd_cpu_codegen)
  endif()

  generativeqc_native_test(generativeqc_scf_proposal_tests tests/native/test_scf_proposals.cpp)
  generativeqc_native_test(generativeqc_self_consistent_tests tests/native/test_self_consistent.cpp NO_GENERATIVEQC)
  generativeqc_native_test(generativeqc_batch_tests tests/native/test_batch.cpp NO_SRC_INCLUDE)
  generativeqc_native_test(generativeqc_cpp_api_tests tests/native/test_cpp_batch.cpp NO_SRC_INCLUDE)
  generativeqc_native_test(generativeqc_cartesian_integral_tests tests/native/test_cartesian_integrals.cpp)
  generativeqc_native_test(generativeqc_density_fitting_tests tests/native/test_density_fitting.cpp)
  generativeqc_native_test(generativeqc_density_factor_tests tests/native/test_density_factor.cpp)
  generativeqc_native_test(generativeqc_scf_array_native_tests tests/native/test_scf_array_native.cpp)
  if(GENERATIVEQC_ENABLE_CUDA)
    add_executable(generativeqc_df_occupied_probe benchmarks/df_occupied_probe.cpp)
    target_link_libraries(generativeqc_df_occupied_probe PRIVATE generativeqc)
    target_include_directories(generativeqc_df_occupied_probe PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    add_executable(generativeqc_df_source_probe benchmarks/df_source_probe.cpp)
    target_link_libraries(generativeqc_df_source_probe PRIVATE generativeqc CUDA::cudart)
    target_include_directories(generativeqc_df_source_probe PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
  endif()
  generativeqc_native_test(generativeqc_cuda_eigensolver_policy_tests tests/native/test_cuda_eigensolver_policy.cpp)
  generativeqc_native_test(generativeqc_uhf_tests tests/native/test_uhf.cpp)
  generativeqc_native_test(generativeqc_spherical_tests tests/native/test_spherical.cpp)
  generativeqc_native_test(generativeqc_basis_contract_tests tests/native/test_basis_contract.cpp)
  generativeqc_native_test(generativeqc_grid_tests tests/native/test_grid.cpp)
  generativeqc_native_test(generativeqc_runtime_workspace_tests tests/native/test_runtime_workspace.cpp NO_GENERATIVEQC)
  generativeqc_native_test(generativeqc_execution_context_tests tests/native/test_execution_context.cpp)
  generativeqc_native_test(generativeqc_cpu_linalg_tests tests/native/test_cpu_linalg.cpp)
  add_executable(generativeqc_cpu_linalg_probe benchmarks/cpu_linalg_probe.cpp)
  target_link_libraries(generativeqc_cpu_linalg_probe PRIVATE generativeqc)
  target_include_directories(generativeqc_cpu_linalg_probe PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
  generativeqc_native_test(generativeqc_cosx_reference_tests tests/native/test_cosx_reference.cpp)

  # These host-side DFT support sources are compiled with identical settings by
  # several standalone native/CUDA tests. Reuse their objects instead of
  # reparsing the same translation units for every test executable.
  add_library(generativeqc_dft_grid_test_objects OBJECT
    src/dft/ao_grid.cpp
    src/dft/grid.cpp
    src/molecule/basis.cpp)
  target_include_directories(generativeqc_dft_grid_test_objects PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include" "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_BINARY_DIR}/generated")

  add_library(generativeqc_dft_xc_test_objects OBJECT
    src/scf/density_factor.cpp
    src/dft/xc.cpp)
  add_dependencies(generativeqc_dft_xc_test_objects
    generativeqc_xc_cpu_codegen generativeqc_scf_array_cpu_codegen)
  target_include_directories(generativeqc_dft_xc_test_objects PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include" "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_BINARY_DIR}/generated")

  add_executable(generativeqc_dft_tests
    tests/native/test_dft.cpp
    $<TARGET_OBJECTS:generativeqc_dft_grid_test_objects>
    $<TARGET_OBJECTS:generativeqc_dft_xc_test_objects>)
  add_dependencies(generativeqc_dft_tests generativeqc_xc_cpu_codegen generativeqc_scf_array_cpu_codegen)
  target_include_directories(generativeqc_dft_tests PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include" "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_BINARY_DIR}/generated")
  target_compile_definitions(generativeqc_dft_tests PRIVATE
    GENERATIVEQC_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
  add_test(NAME generativeqc_dft_tests COMMAND generativeqc_dft_tests)

  generativeqc_native_test(generativeqc_d3_atm_reference_tests tests/native/test_d3_atm.cpp NO_GENERATIVEQC)
  add_dependencies(generativeqc_d3_atm_reference_tests generativeqc_d3_codegen)
  target_include_directories(generativeqc_d3_atm_reference_tests PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
  generativeqc_native_test(generativeqc_d3_zero_reference_tests tests/native/test_d3_zero.cpp NO_GENERATIVEQC)
  add_dependencies(generativeqc_d3_zero_reference_tests generativeqc_d3_codegen)
  target_include_directories(generativeqc_d3_zero_reference_tests PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
  generativeqc_native_test(generativeqc_d3_ragged_tests tests/native/test_d3_ragged.cpp)
  add_dependencies(generativeqc_d3_ragged_tests generativeqc)
  target_include_directories(generativeqc_d3_ragged_tests PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
  generativeqc_native_test(generativeqc_d3_public_variant_tests tests/native/test_d3_public_variants.cpp)
  add_dependencies(generativeqc_d3_public_variant_tests generativeqc generativeqc_d3_codegen)
  target_include_directories(generativeqc_d3_public_variant_tests PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
  if(GENERATIVEQC_ENABLE_CUDA)
    generativeqc_native_test(generativeqc_d3_cooperative_cuda_tests tests/native/test_d3_cooperative_cuda.cpp)
    add_dependencies(generativeqc_d3_cooperative_cuda_tests generativeqc generativeqc_d3_codegen)
    target_include_directories(generativeqc_d3_cooperative_cuda_tests PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
    target_link_libraries(generativeqc_d3_cooperative_cuda_tests PRIVATE CUDA::cudart)
    add_executable(generativeqc_d3_cuda_endpoint_benchmark tools/benchmark_d3_cuda_endpoint.cpp)
    target_link_libraries(generativeqc_d3_cuda_endpoint_benchmark PRIVATE generativeqc CUDA::cudart)
    add_dependencies(generativeqc_d3_cuda_endpoint_benchmark generativeqc generativeqc_d3_codegen)
  endif()
  generativeqc_native_test(generativeqc_d4_production_tests tests/native/test_d4_production.cpp)
  generativeqc_native_test(generativeqc_d4_ragged_tests tests/native/test_d4_ragged.cpp)
  foreach(_generativeqc_d4_production_test IN ITEMS generativeqc_d4_production_tests generativeqc_d4_ragged_tests)
    add_dependencies(${_generativeqc_d4_production_test} generativeqc generativeqc_method_parameters_codegen
                     generativeqc_d4_derivative_codegen)
    target_include_directories(
      ${_generativeqc_d4_production_test} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
  endforeach()
  generativeqc_native_test(generativeqc_d4_reference_tests tests/native/test_d4_reference.cpp NO_GENERATIVEQC)
  generativeqc_native_test(generativeqc_d4_eeq_tests tests/native/test_d4_eeq.cpp NO_GENERATIVEQC)
  generativeqc_native_test(generativeqc_gcp_r2scan3c_tests tests/native/test_gcp_r2scan3c.cpp NO_GENERATIVEQC)
  foreach(_generativeqc_parameter_test IN ITEMS
          generativeqc_d4_reference_tests generativeqc_d4_eeq_tests generativeqc_gcp_r2scan3c_tests)
    add_dependencies(${_generativeqc_parameter_test} generativeqc_method_parameters_codegen)
    target_include_directories(
      ${_generativeqc_parameter_test} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
  endforeach()
  generativeqc_native_test(generativeqc_xc_point_tests tests/native/test_xc_point.cpp NO_GENERATIVEQC)
  target_compile_definitions(generativeqc_xc_point_tests PRIVATE
    GENERATIVEQC_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
  generativeqc_native_test(generativeqc_rks_response_tests tests/native/test_rks_response.cpp)
  target_compile_definitions(generativeqc_rks_response_tests PRIVATE
    GENERATIVEQC_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
  generativeqc_native_test(generativeqc_uks_response_tests tests/native/test_uks_response.cpp)
  target_compile_definitions(generativeqc_uks_response_tests PRIVATE
    GENERATIVEQC_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
  generativeqc_native_test(generativeqc_uks_state_tests tests/native/test_uks_state.cpp)

  if(GENERATIVEQC_ENABLE_CUDA)
    generativeqc_native_test(generativeqc_d3_atm_cuda_tests tests/native/test_d3_atm_cuda.cu
                       NO_GENERATIVEQC LIBRARIES CUDA::cudart SKIP_77)
    add_dependencies(generativeqc_d3_atm_cuda_tests generativeqc_d3_codegen)
    target_include_directories(generativeqc_d3_atm_cuda_tests PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
    set_target_properties(generativeqc_d3_atm_cuda_tests PROPERTIES CUDA_STANDARD 20)

    generativeqc_native_test(generativeqc_d3_zero_cuda_tests tests/native/test_d3_zero_cuda.cu
                       NO_GENERATIVEQC LIBRARIES CUDA::cudart SKIP_77)
    add_dependencies(generativeqc_d3_zero_cuda_tests generativeqc_d3_codegen)
    target_include_directories(generativeqc_d3_zero_cuda_tests PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
    set_target_properties(generativeqc_d3_zero_cuda_tests PROPERTIES CUDA_STANDARD 20)

    generativeqc_native_test(generativeqc_d3_cuda_replay_failure_tests
                       tests/native/test_d3_cuda_replay_failure.cu
                       NO_GENERATIVEQC LIBRARIES CUDA::cudart SKIP_77)
    target_sources(generativeqc_d3_cuda_replay_failure_tests PRIVATE src/dft/dispersion/d3_cuda.cu)
    add_dependencies(generativeqc_d3_cuda_replay_failure_tests generativeqc_d3_codegen)
    target_include_directories(generativeqc_d3_cuda_replay_failure_tests PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/include" "${CMAKE_CURRENT_BINARY_DIR}/generated")
    set_target_properties(generativeqc_d3_cuda_replay_failure_tests PROPERTIES CUDA_STANDARD 20)
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
      target_compile_definitions(generativeqc_d3_cuda_replay_failure_tests
                                 PRIVATE GENERATIVEQC_D3_REPLAY_TEST_INTERPOSE=1)
      target_link_options(generativeqc_d3_cuda_replay_failure_tests PRIVATE
        "LINKER:--wrap=cudaMemcpyAsync"
        "LINKER:--wrap=cudaMemsetAsync"
        "LINKER:--wrap=cudaGetLastError"
        "LINKER:--wrap=cudaStreamSynchronize")
    endif()

    target_link_libraries(generativeqc_d4_production_tests PRIVATE CUDA::cudart)
    target_link_libraries(generativeqc_d4_ragged_tests PRIVATE CUDA::cudart)
    add_test(NAME generativeqc_d4_production_cuda_tests COMMAND generativeqc_d4_production_tests cuda)
    add_test(NAME generativeqc_d4_ragged_cuda_tests COMMAND generativeqc_d4_ragged_tests cuda)
    set_tests_properties(generativeqc_d4_production_cuda_tests generativeqc_d4_ragged_cuda_tests
                         PROPERTIES SKIP_RETURN_CODE 77)

    generativeqc_native_test(generativeqc_d4_reference_cuda_tests tests/native/test_d4_reference_cuda.cu
                       NO_GENERATIVEQC SKIP_77)
    add_dependencies(generativeqc_d4_reference_cuda_tests generativeqc_method_parameters_codegen)
    target_include_directories(
      generativeqc_d4_reference_cuda_tests PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
    set_target_properties(generativeqc_d4_reference_cuda_tests PROPERTIES CUDA_STANDARD 20)

    generativeqc_native_test(generativeqc_d4_schedule_cuda_tests tests/native/test_d4_schedule_cuda.cu
                       LIBRARIES CUDA::cudart SKIP_77)
    add_dependencies(generativeqc_d4_schedule_cuda_tests generativeqc_method_parameters_codegen)
    target_include_directories(
      generativeqc_d4_schedule_cuda_tests PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
    set_target_properties(generativeqc_d4_schedule_cuda_tests PROPERTIES CUDA_STANDARD 20
                          BUILD_RPATH "$<TARGET_FILE_DIR:CUDA::cudart>")

    add_executable(generativeqc_d4_schedule_probe EXCLUDE_FROM_ALL benchmarks/d4_cuda_schedule_probe.cu)
    target_link_libraries(generativeqc_d4_schedule_probe PRIVATE generativeqc CUDA::cudart)
    target_include_directories(generativeqc_d4_schedule_probe PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src" "${CMAKE_CURRENT_BINARY_DIR}/generated")
    add_dependencies(generativeqc_d4_schedule_probe generativeqc_method_parameters_codegen)
    set_target_properties(generativeqc_d4_schedule_probe PROPERTIES CUDA_STANDARD 20
                          BUILD_RPATH "$<TARGET_FILE_DIR:CUDA::cudart>")

    generativeqc_native_test(generativeqc_d4_eeq_cuda_tests tests/native/test_d4_eeq_cuda.cu
                       NO_GENERATIVEQC SKIP_77)
    add_dependencies(generativeqc_d4_eeq_cuda_tests generativeqc_method_parameters_codegen)
    target_include_directories(
      generativeqc_d4_eeq_cuda_tests PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
    set_target_properties(generativeqc_d4_eeq_cuda_tests PROPERTIES CUDA_STANDARD 20)

    generativeqc_native_test(generativeqc_xc_point_cuda_tests tests/native/test_xc_point_cuda.cu
                       NO_GENERATIVEQC SKIP_77)
    target_compile_definitions(generativeqc_xc_point_cuda_tests PRIVATE
      GENERATIVEQC_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    set_target_properties(generativeqc_xc_point_cuda_tests PROPERTIES CUDA_STANDARD 20)

    add_executable(generativeqc_dft_cuda_tests tests/native/test_dft_cuda.cu
      src/dft/cuda_xc.cpp "${GENERATIVEQC_GRID_SOURCE}"
      $<TARGET_OBJECTS:generativeqc_dft_grid_test_objects>
      $<TARGET_OBJECTS:generativeqc_dft_xc_test_objects>)
    add_dependencies(generativeqc_dft_cuda_tests generativeqc_xc_cpu_codegen generativeqc_scf_array_cpu_codegen)
    target_include_directories(generativeqc_dft_cuda_tests PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/include" "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/dft" "${CMAKE_CURRENT_BINARY_DIR}/generated")
    target_link_libraries(generativeqc_dft_cuda_tests PRIVATE CUDA::cudart CUDA::cublas)
    set_target_properties(generativeqc_dft_cuda_tests PROPERTIES CUDA_STANDARD 20)
    add_test(NAME generativeqc_dft_cuda_tests COMMAND generativeqc_dft_cuda_tests)
    add_test(NAME generativeqc_dft_cuda_matrix_tests COMMAND generativeqc_dft_cuda_tests --matrix-schedule)
    set_tests_properties(generativeqc_dft_cuda_matrix_tests PROPERTIES SKIP_RETURN_CODE 77)
    set_tests_properties(generativeqc_dft_cuda_tests PROPERTIES SKIP_RETURN_CODE 77)
    generativeqc_native_test(generativeqc_ks_cuda_tests tests/native/test_ks_cuda.cpp
                       LIBRARIES CUDA::cudart SKIP_77)
    target_include_directories(generativeqc_ks_cuda_tests PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
    target_sources(generativeqc_ks_cuda_tests PRIVATE
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_split_hybrid_registry.cuh")
    add_executable(generativeqc_cosx_cuda_tests
      tests/native/test_cosx_cuda.cu
      src/dft/cuda_cosx.cu
      src/dft/cuda_cosx_derivative.cu
      "${GENERATIVEQC_ONE_ELECTRON_HEADER}"
      "${GENERATIVEQC_ONE_ELECTRON_DERIVATIVE_HEADER}"
      "${GENERATIVEQC_COSX_DERIVATIVE_CONTRACTION_HEADER}"
      "${GENERATIVEQC_ONE_ELECTRON_ST_CPU_HEADER}"
      "${GENERATIVEQC_DF_VALUE_CPU_HEADER}"
      "${GENERATIVEQC_DF_DERIVATIVE_CPU_HEADER}"
      "${GENERATIVEQC_GRID_SOURCE}"
      $<TARGET_OBJECTS:generativeqc_dft_grid_test_objects>
      src/dft/cosx_reference.cpp
      src/integrals/s_integrals.cpp
      src/integrals/generated_df_cpu.cpp
      src/integrals/ecp.cpp)
    add_dependencies(generativeqc_cosx_cuda_tests generativeqc_ecp_codegen)
    target_include_directories(generativeqc_cosx_cuda_tests PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/include" "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/dft" "${CMAKE_CURRENT_BINARY_DIR}/generated")
    target_link_libraries(generativeqc_cosx_cuda_tests PRIVATE CUDA::cudart CUDA::cublas)
    set_target_properties(generativeqc_cosx_cuda_tests PROPERTIES CUDA_STANDARD 20)
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
      target_compile_definitions(generativeqc_cosx_cuda_tests PRIVATE GENERATIVEQC_COSX_TEST_INTERPOSE=1)
      target_link_options(generativeqc_cosx_cuda_tests PRIVATE
        "LINKER:--wrap=cudaMemcpyAsync" "LINKER:--wrap=cudaStreamSynchronize"
        "LINKER:--wrap=cudaGetDevice")
    endif()
    add_test(NAME generativeqc_cosx_cuda_tests COMMAND generativeqc_cosx_cuda_tests)
    set_tests_properties(generativeqc_cosx_cuda_tests PROPERTIES SKIP_RETURN_CODE 77)
    generativeqc_native_test(generativeqc_cosx_fock_provider_tests
                       tests/native/test_cosx_fock_provider.cpp
                       LIBRARIES CUDA::cudart SKIP_77)
    generativeqc_native_test(generativeqc_cosx_scf_tests
                       tests/native/test_cosx_scf.cpp
                       LIBRARIES CUDA::cudart SKIP_77)
  endif()

  generativeqc_native_test(generativeqc_dft_api_tests tests/native/test_dft_api.cpp NO_SRC_INCLUDE)
  generativeqc_native_test(generativeqc_scf_diagnostic_tests tests/native/test_scf_diagnostic.cpp)
  generativeqc_native_test(generativeqc_dft_density_source_tests tests/native/test_dft_density_source.cpp)
  generativeqc_native_test(generativeqc_uks_tests tests/native/test_uks.cpp)
  if(NOT WIN32)
    generativeqc_native_test(generativeqc_wb97mv_scf_tests tests/native/test_wb97mv_scf.cpp)
  endif()
  generativeqc_native_test(generativeqc_mixed_precision_tests tests/native/test_mixed_precision.cpp)
  generativeqc_native_test(generativeqc_precision_policy_tests tests/native/test_precision_policy.cpp)

  if(GENERATIVEQC_ENABLE_CUDA)
    generativeqc_native_test(generativeqc_cuda_runtime_tests tests/native/test_cuda_runtime.cu
                       LIBRARIES CUDA::cudart CUDA::cublas)
    generativeqc_native_test(generativeqc_df_eigensystem_tests tests/native/test_df_eigensystem.cpp
                       LIBRARIES CUDA::cudart CUDA::cublas CUDA::cusolver)
    generativeqc_native_test(generativeqc_df_capture_recovery_tests tests/native/test_df_capture_recovery.cpp
                       LIBRARIES CUDA::cudart CUDA::cusolver)
    generativeqc_native_test(generativeqc_df_final_snapshot_tests tests/native/test_df_final_snapshot.cpp
                       LIBRARIES CUDA::cudart CUDA::cublas CUDA::cusolver)
    generativeqc_native_test(generativeqc_df_occupied_response_tests tests/native/test_df_occupied_response.cpp
                       LIBRARIES CUDA::cudart CUDA::cublas CUDA::cusolver)
    generativeqc_native_test(generativeqc_cuda_diis_tests tests/native/test_cuda_diis.cpp
                       LIBRARIES CUDA::cudart SKIP_77)
    generativeqc_native_test(generativeqc_direct_streaming_graph_cuda_tests
                       tests/native/test_direct_streaming_graph_cuda.cpp
                       LIBRARIES CUDA::cudart SKIP_77)
    generativeqc_native_test(generativeqc_cuda_force_convergence_tests tests/native/test_cuda_force_convergence.cpp
                       LIBRARIES CUDA::cudart SKIP_77)
    generativeqc_native_test(generativeqc_df_shell_pairs_tests tests/native/test_df_shell_pairs.cpp
                       LIBRARIES CUDA::cudart SKIP_77)
    generativeqc_native_test(generativeqc_cuda_fock_provider_tests tests/native/test_cuda_fock_provider.cpp
                       LIBRARIES CUDA::cudart)
    # The value suite has a separate CLI entry point; register it so CTest does
    # not silently omit screening, projections, and budget-fallback coverage.
    add_test(NAME generativeqc_cuda_fock_canonical_tests
             COMMAND generativeqc_cuda_fock_provider_tests --canonical-values-only)
    set_tests_properties(generativeqc_cuda_fock_canonical_tests PROPERTIES TIMEOUT 900)
    generativeqc_native_test(generativeqc_cuda_stream_eigensolver_tests tests/native/test_cuda_stream_eigensolver.cpp
                       LIBRARIES CUDA::cudart)
    generativeqc_native_test(generativeqc_ecp_cuda_error_tests tests/native/test_ecp_cuda_errors.cpp
                       LIBRARIES CUDA::cudart SKIP_77)
    generativeqc_native_test(generativeqc_ecp_policy_cuda_tests tests/native/test_ecp_policy_cuda.cu
                       NO_GENERATIVEQC LIBRARIES CUDA::cudart SKIP_77)
    add_dependencies(generativeqc_ecp_policy_cuda_tests generativeqc_ecp_codegen)
    target_include_directories(generativeqc_ecp_policy_cuda_tests PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
    generativeqc_native_test(generativeqc_cuda_fock_composition_tests tests/native/test_cuda_fock_composition.cpp)
  endif()

  if(GENERATIVEQC_ENABLE_CUDA AND GENERATIVEQC_ENABLE_AOT_SHELLS)
    generativeqc_native_test(generativeqc_aot_profile_tests tests/native/test_aot_profile.cpp
                       LIBRARIES CUDA::cudart)
  endif()
endmacro()
