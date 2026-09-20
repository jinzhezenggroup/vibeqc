include_guard(GLOBAL)
include(CMakeParseArguments)

function(vibeqc_native_test target source)
  set(options NO_VIBEQC NO_SRC_INCLUDE SKIP_77)
  set(multi_value_args LIBRARIES)
  cmake_parse_arguments(VNT "${options}" "" "${multi_value_args}" ${ARGN})
  add_executable(${target} ${source})
  if(NOT VNT_NO_VIBEQC)
    target_link_libraries(${target} PRIVATE vibeqc ${VNT_LIBRARIES})
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

macro(vibeqc_add_native_tests)
  enable_testing()
  if(VIBEQC_ENABLE_CUDA)
    vibeqc_native_test(vibeqc_xc_response_cuda_tests tests/native/test_xc_response_cuda.cu
                       NO_VIBEQC LIBRARIES CUDA::cudart SKIP_77)
    target_compile_definitions(vibeqc_xc_response_cuda_tests PRIVATE
                               VIBEQC_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    # Run the same analytic/negative shared-policy suite through device algebra.
    vibeqc_native_test(vibeqc_df_final_validation_tests tests/native/test_final_state.cpp
                       LIBRARIES CUDA::cudart SKIP_77)
    target_compile_definitions(vibeqc_df_final_validation_tests PRIVATE VIBEQC_TEST_DEVICE_FINAL_STATE=1)
    vibeqc_native_test(vibeqc_df_density_seed_tests tests/native/test_df_density_seed.cpp
                       LIBRARIES CUDA::cudart CUDA::cublas CUDA::cusolver SKIP_77)
    set_property(SOURCE "${VIBEQC_GRID_SOURCE}" src/dft/cuda_ks.cpp src/dft/cuda_xc.cpp
                 APPEND PROPERTY COMPILE_DEFINITIONS VIBEQC_TEST_HOOKS=1)
    add_executable(vibeqc_weighted_eri_probe tests/native/weighted_eri_probe.cpp)
    target_link_libraries(vibeqc_weighted_eri_probe PRIVATE vibeqc CUDA::cudart)
    target_include_directories(vibeqc_weighted_eri_probe PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    add_executable(vibeqc_df_value_probe tests/native/df_value_probe.cpp)
    target_link_libraries(vibeqc_df_value_probe PRIVATE vibeqc CUDA::cudart)
    target_include_directories(vibeqc_df_value_probe PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
  endif()

  vibeqc_native_test(vibeqc_tracked_allocator_tests tests/native/test_tracked_allocator.cpp
                     NO_VIBEQC LIBRARIES Threads::Threads)
  vibeqc_native_test(vibeqc_fock_build_tests tests/native/test_fock_build.cpp)
  vibeqc_native_test(vibeqc_ecp_projector_tests tests/native/test_ecp_projector.cpp NO_VIBEQC)
  vibeqc_native_test(vibeqc_ecp_capability_tests tests/native/test_ecp_capabilities.cpp)
  vibeqc_native_test(vibeqc_ecp_cpu_tests tests/native/test_ecp_cpu.cpp)
  target_sources(vibeqc_ecp_cpu_tests PRIVATE tests/native/ecp_reference.cpp)
  add_dependencies(vibeqc_ecp_projector_tests vibeqc_ecp_codegen)
  target_include_directories(vibeqc_ecp_projector_tests PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
  vibeqc_native_test(vibeqc_fock_provider_tests tests/native/test_fock_provider.cpp)
  vibeqc_native_test(vibeqc_fock_api_tests tests/native/test_fock_api.cpp NO_SRC_INCLUDE)
  vibeqc_native_test(vibeqc_native_tests tests/native/test_rhf.cpp)

  if(NOT WIN32)
    vibeqc_native_test(vibeqc_final_state_tests tests/native/test_final_state.cpp)
    vibeqc_native_test(vibeqc_ks_final_state_tests tests/native/test_ks_final_state.cpp)
    vibeqc_native_test(vibeqc_eigen_frame_tests tests/native/test_eigen_frame.cpp)
    vibeqc_native_test(vibeqc_initial_density_tests tests/native/test_initial_density.cpp)
    vibeqc_native_test(vibeqc_mp2_contract_tests tests/native/test_mp2_contract.cpp)
    vibeqc_native_test(vibeqc_native_gmres_tests tests/native/test_native_gmres.cpp)
    vibeqc_native_test(vibeqc_mp2_gradient_tests tests/native/test_mp2_gradient.cpp)
  endif()

  if(VIBEQC_ENABLE_CUDA AND NOT WIN32)
    vibeqc_native_test(vibeqc_cuda_reference_export_tests tests/native/test_cuda_reference_export.cpp
                       LIBRARIES CUDA::cudart)
    vibeqc_native_test(vibeqc_mp2_cuda_status_tests tests/native/test_mp2_cuda_status.cu
                       LIBRARIES CUDA::cudart CUDA::cublas SKIP_77)
  endif()

  vibeqc_native_test(vibeqc_scf_proposal_tests tests/native/test_scf_proposals.cpp)
  vibeqc_native_test(vibeqc_self_consistent_tests tests/native/test_self_consistent.cpp NO_VIBEQC)
  vibeqc_native_test(vibeqc_batch_tests tests/native/test_batch.cpp NO_SRC_INCLUDE)
  vibeqc_native_test(vibeqc_cpp_api_tests tests/native/test_cpp_batch.cpp NO_SRC_INCLUDE)
  vibeqc_native_test(vibeqc_cartesian_integral_tests tests/native/test_cartesian_integrals.cpp)
  vibeqc_native_test(vibeqc_density_fitting_tests tests/native/test_density_fitting.cpp)
  vibeqc_native_test(vibeqc_density_factor_tests tests/native/test_density_factor.cpp)
  vibeqc_native_test(vibeqc_scf_array_native_tests tests/native/test_scf_array_native.cpp)
  if(VIBEQC_ENABLE_CUDA)
    add_executable(vibeqc_df_occupied_probe benchmarks/df_occupied_probe.cpp)
    target_link_libraries(vibeqc_df_occupied_probe PRIVATE vibeqc)
    target_include_directories(vibeqc_df_occupied_probe PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    add_executable(vibeqc_df_source_probe benchmarks/df_source_probe.cpp)
    target_link_libraries(vibeqc_df_source_probe PRIVATE vibeqc CUDA::cudart)
    target_include_directories(vibeqc_df_source_probe PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
  endif()
  vibeqc_native_test(vibeqc_cuda_eigensolver_policy_tests tests/native/test_cuda_eigensolver_policy.cpp)
  vibeqc_native_test(vibeqc_uhf_tests tests/native/test_uhf.cpp)
  vibeqc_native_test(vibeqc_spherical_tests tests/native/test_spherical.cpp)
  vibeqc_native_test(vibeqc_basis_contract_tests tests/native/test_basis_contract.cpp)
  vibeqc_native_test(vibeqc_grid_tests tests/native/test_grid.cpp)
  vibeqc_native_test(vibeqc_runtime_workspace_tests tests/native/test_runtime_workspace.cpp NO_VIBEQC)
  vibeqc_native_test(vibeqc_cpu_linalg_tests tests/native/test_cpu_linalg.cpp)
  add_executable(vibeqc_cpu_linalg_probe benchmarks/cpu_linalg_probe.cpp)
  target_link_libraries(vibeqc_cpu_linalg_probe PRIVATE vibeqc)
  target_include_directories(vibeqc_cpu_linalg_probe PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
  vibeqc_native_test(vibeqc_cosx_reference_tests tests/native/test_cosx_reference.cpp)

  add_executable(vibeqc_dft_tests
    tests/native/test_dft.cpp src/scf/density_factor.cpp src/dft/ao_grid.cpp
    src/dft/grid.cpp src/dft/xc.cpp src/molecule/basis.cpp)
  add_dependencies(vibeqc_dft_tests vibeqc_xc_cpu_codegen)
  target_include_directories(vibeqc_dft_tests PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include" "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_BINARY_DIR}/generated")
  add_test(NAME vibeqc_dft_tests COMMAND vibeqc_dft_tests)

  vibeqc_native_test(vibeqc_d3_atm_reference_tests tests/native/test_d3_atm.cpp NO_VIBEQC)
  add_dependencies(vibeqc_d3_atm_reference_tests vibeqc_d3_codegen)
  target_include_directories(vibeqc_d3_atm_reference_tests PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
  vibeqc_native_test(vibeqc_d3_ragged_tests tests/native/test_d3_ragged.cpp)
  add_dependencies(vibeqc_d3_ragged_tests vibeqc)
  target_include_directories(vibeqc_d3_ragged_tests PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
  vibeqc_native_test(vibeqc_d4_reference_tests tests/native/test_d4_reference.cpp NO_VIBEQC)
  vibeqc_native_test(vibeqc_d4_eeq_tests tests/native/test_d4_eeq.cpp NO_VIBEQC)
  vibeqc_native_test(vibeqc_gcp_r2scan3c_tests tests/native/test_gcp_r2scan3c.cpp NO_VIBEQC)
  foreach(_vibeqc_parameter_test IN ITEMS
          vibeqc_d4_reference_tests vibeqc_d4_eeq_tests vibeqc_gcp_r2scan3c_tests)
    add_dependencies(${_vibeqc_parameter_test} vibeqc_method_parameters_codegen)
    target_include_directories(
      ${_vibeqc_parameter_test} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
  endforeach()
  vibeqc_native_test(vibeqc_xc_point_tests tests/native/test_xc_point.cpp NO_VIBEQC)
  target_compile_definitions(vibeqc_xc_point_tests PRIVATE
    VIBEQC_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
  vibeqc_native_test(vibeqc_rks_response_tests tests/native/test_rks_response.cpp)
  target_compile_definitions(vibeqc_rks_response_tests PRIVATE
    VIBEQC_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
  vibeqc_native_test(vibeqc_uks_response_tests tests/native/test_uks_response.cpp)
  target_compile_definitions(vibeqc_uks_response_tests PRIVATE
    VIBEQC_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
  vibeqc_native_test(vibeqc_uks_state_tests tests/native/test_uks_state.cpp)

  if(VIBEQC_ENABLE_CUDA)
    vibeqc_native_test(vibeqc_d3_atm_cuda_tests tests/native/test_d3_atm_cuda.cu
                       NO_VIBEQC LIBRARIES CUDA::cudart SKIP_77)
    add_dependencies(vibeqc_d3_atm_cuda_tests vibeqc_d3_codegen)
    target_include_directories(vibeqc_d3_atm_cuda_tests PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
    set_target_properties(vibeqc_d3_atm_cuda_tests PROPERTIES CUDA_STANDARD 20)

    vibeqc_native_test(vibeqc_d4_reference_cuda_tests tests/native/test_d4_reference_cuda.cu
                       NO_VIBEQC SKIP_77)
    add_dependencies(vibeqc_d4_reference_cuda_tests vibeqc_method_parameters_codegen)
    target_include_directories(
      vibeqc_d4_reference_cuda_tests PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
    set_target_properties(vibeqc_d4_reference_cuda_tests PROPERTIES CUDA_STANDARD 20)

    vibeqc_native_test(vibeqc_d4_schedule_cuda_tests tests/native/test_d4_schedule_cuda.cu
                       LIBRARIES CUDA::cudart SKIP_77)
    add_dependencies(vibeqc_d4_schedule_cuda_tests vibeqc_method_parameters_codegen)
    target_include_directories(
      vibeqc_d4_schedule_cuda_tests PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
    set_target_properties(vibeqc_d4_schedule_cuda_tests PROPERTIES CUDA_STANDARD 20
                          BUILD_RPATH "$<TARGET_FILE_DIR:CUDA::cudart>")

    add_executable(vibeqc_d4_schedule_probe EXCLUDE_FROM_ALL benchmarks/d4_cuda_schedule_probe.cu)
    target_link_libraries(vibeqc_d4_schedule_probe PRIVATE vibeqc CUDA::cudart)
    target_include_directories(vibeqc_d4_schedule_probe PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src" "${CMAKE_CURRENT_BINARY_DIR}/generated")
    add_dependencies(vibeqc_d4_schedule_probe vibeqc_method_parameters_codegen)
    set_target_properties(vibeqc_d4_schedule_probe PROPERTIES CUDA_STANDARD 20
                          BUILD_RPATH "$<TARGET_FILE_DIR:CUDA::cudart>")

    vibeqc_native_test(vibeqc_d4_eeq_cuda_tests tests/native/test_d4_eeq_cuda.cu
                       NO_VIBEQC SKIP_77)
    add_dependencies(vibeqc_d4_eeq_cuda_tests vibeqc_method_parameters_codegen)
    target_include_directories(
      vibeqc_d4_eeq_cuda_tests PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
    set_target_properties(vibeqc_d4_eeq_cuda_tests PROPERTIES CUDA_STANDARD 20)

    vibeqc_native_test(vibeqc_xc_point_cuda_tests tests/native/test_xc_point_cuda.cu
                       NO_VIBEQC SKIP_77)
    target_compile_definitions(vibeqc_xc_point_cuda_tests PRIVATE
      VIBEQC_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    set_target_properties(vibeqc_xc_point_cuda_tests PROPERTIES CUDA_STANDARD 20)

    add_executable(vibeqc_dft_cuda_tests tests/native/test_dft_cuda.cu
      src/dft/cuda_xc.cpp "${VIBEQC_GRID_SOURCE}" src/dft/ao_grid.cpp
      src/dft/grid.cpp src/dft/xc.cpp src/scf/density_factor.cpp src/molecule/basis.cpp)
    add_dependencies(vibeqc_dft_cuda_tests vibeqc_xc_cpu_codegen)
    target_include_directories(vibeqc_dft_cuda_tests PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/include" "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/dft" "${CMAKE_CURRENT_BINARY_DIR}/generated")
    target_link_libraries(vibeqc_dft_cuda_tests PRIVATE CUDA::cudart CUDA::cublas)
    set_target_properties(vibeqc_dft_cuda_tests PROPERTIES CUDA_STANDARD 20)
    add_test(NAME vibeqc_dft_cuda_tests COMMAND vibeqc_dft_cuda_tests)
    set_tests_properties(vibeqc_dft_cuda_tests PROPERTIES SKIP_RETURN_CODE 77)
    vibeqc_native_test(vibeqc_ks_cuda_tests tests/native/test_ks_cuda.cpp
                       LIBRARIES CUDA::cudart SKIP_77)
    add_executable(vibeqc_cosx_cuda_tests
      tests/native/test_cosx_cuda.cu
      src/dft/cuda_cosx.cu
      src/dft/cuda_cosx_derivative.cu
      "${VIBEQC_ONE_ELECTRON_HEADER}"
      "${VIBEQC_ONE_ELECTRON_DERIVATIVE_HEADER}"
      "${VIBEQC_ONE_ELECTRON_ST_CPU_HEADER}"
      "${VIBEQC_DF_VALUE_CPU_HEADER}"
      "${VIBEQC_DF_DERIVATIVE_CPU_HEADER}"
      "${VIBEQC_GRID_SOURCE}"
      src/dft/ao_grid.cpp
      src/dft/grid.cpp
      src/dft/cosx_reference.cpp
      src/integrals/s_integrals.cpp
      src/integrals/generated_df_cpu.cpp
      src/integrals/ecp.cpp
      src/molecule/basis.cpp)
    add_dependencies(vibeqc_cosx_cuda_tests vibeqc_ecp_codegen)
    target_include_directories(vibeqc_cosx_cuda_tests PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/include" "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/dft" "${CMAKE_CURRENT_BINARY_DIR}/generated")
    target_link_libraries(vibeqc_cosx_cuda_tests PRIVATE CUDA::cudart CUDA::cublas)
    set_target_properties(vibeqc_cosx_cuda_tests PROPERTIES CUDA_STANDARD 20)
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
      target_compile_definitions(vibeqc_cosx_cuda_tests PRIVATE VIBEQC_COSX_TEST_INTERPOSE=1)
      target_link_options(vibeqc_cosx_cuda_tests PRIVATE
        "LINKER:--wrap=cudaMemcpyAsync" "LINKER:--wrap=cudaStreamSynchronize"
        "LINKER:--wrap=cudaGetDevice")
    endif()
    add_test(NAME vibeqc_cosx_cuda_tests COMMAND vibeqc_cosx_cuda_tests)
    set_tests_properties(vibeqc_cosx_cuda_tests PROPERTIES SKIP_RETURN_CODE 77)
    vibeqc_native_test(vibeqc_cosx_fock_provider_tests
                       tests/native/test_cosx_fock_provider.cpp
                       LIBRARIES CUDA::cudart SKIP_77)
    vibeqc_native_test(vibeqc_cosx_scf_tests
                       tests/native/test_cosx_scf.cpp
                       LIBRARIES CUDA::cudart SKIP_77)
  endif()

  vibeqc_native_test(vibeqc_dft_api_tests tests/native/test_dft_api.cpp NO_SRC_INCLUDE)
  vibeqc_native_test(vibeqc_scf_diagnostic_tests tests/native/test_scf_diagnostic.cpp)
  vibeqc_native_test(vibeqc_dft_density_source_tests tests/native/test_dft_density_source.cpp)
  vibeqc_native_test(vibeqc_uks_tests tests/native/test_uks.cpp)
  vibeqc_native_test(vibeqc_mixed_precision_tests tests/native/test_mixed_precision.cpp)
  vibeqc_native_test(vibeqc_precision_policy_tests tests/native/test_precision_policy.cpp)

  if(VIBEQC_ENABLE_CUDA)
    vibeqc_native_test(vibeqc_cuda_runtime_tests tests/native/test_cuda_runtime.cu
                       LIBRARIES CUDA::cudart CUDA::cublas)
    vibeqc_native_test(vibeqc_df_eigensystem_tests tests/native/test_df_eigensystem.cpp
                       LIBRARIES CUDA::cudart CUDA::cublas CUDA::cusolver)
    vibeqc_native_test(vibeqc_df_capture_recovery_tests tests/native/test_df_capture_recovery.cpp
                       LIBRARIES CUDA::cudart CUDA::cusolver)
    vibeqc_native_test(vibeqc_df_final_snapshot_tests tests/native/test_df_final_snapshot.cpp
                       LIBRARIES CUDA::cudart CUDA::cublas CUDA::cusolver)
    vibeqc_native_test(vibeqc_df_occupied_response_tests tests/native/test_df_occupied_response.cpp
                       LIBRARIES CUDA::cudart CUDA::cublas CUDA::cusolver)
    vibeqc_native_test(vibeqc_cuda_diis_tests tests/native/test_cuda_diis.cpp
                       LIBRARIES CUDA::cudart SKIP_77)
    vibeqc_native_test(vibeqc_cuda_force_convergence_tests tests/native/test_cuda_force_convergence.cpp
                       LIBRARIES CUDA::cudart SKIP_77)
    vibeqc_native_test(vibeqc_df_shell_pairs_tests tests/native/test_df_shell_pairs.cpp
                       LIBRARIES CUDA::cudart SKIP_77)
    vibeqc_native_test(vibeqc_cuda_fock_provider_tests tests/native/test_cuda_fock_provider.cpp
                       LIBRARIES CUDA::cudart)
    vibeqc_native_test(vibeqc_ecp_cuda_error_tests tests/native/test_ecp_cuda_errors.cpp
                       LIBRARIES CUDA::cudart SKIP_77)
    vibeqc_native_test(vibeqc_ecp_policy_cuda_tests tests/native/test_ecp_policy_cuda.cu
                       NO_VIBEQC LIBRARIES CUDA::cudart SKIP_77)
    add_dependencies(vibeqc_ecp_policy_cuda_tests vibeqc_ecp_codegen)
    target_include_directories(vibeqc_ecp_policy_cuda_tests PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
    vibeqc_native_test(vibeqc_cuda_fock_composition_tests tests/native/test_cuda_fock_composition.cpp)
  endif()

  if(VIBEQC_ENABLE_CUDA AND VIBEQC_ENABLE_AOT_SHELLS)
    vibeqc_native_test(vibeqc_aot_profile_tests tests/native/test_aot_profile.cpp
                       LIBRARIES CUDA::cudart)
  endif()
endmacro()
