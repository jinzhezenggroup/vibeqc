include_guard(GLOBAL)

# Keep production source enumeration explicit and component-owned. Generated
# outputs are registered separately in VibeQCGeneratedSources.cmake.
function(vibeqc_add_api_sources target)
  target_sources(${target} PRIVATE
    src/api/c_api_batch.cpp
    src/api/c_api_calculation.cpp
    src/api/c_api_ks_snapshot.cpp
    src/api/c_api_fock.cpp
    src/api/c_api_one_electron_gradient.cpp
    src/api/c_api_projection.cpp
    src/api/c_api_df_gradient.cpp
    src/api/c_api_d3.cpp
    src/api/c_api_d4.cpp
    src/api/c_api_common.cpp
    src/api/c_api_context.cpp
    src/api/c_api_resources.cpp
    src/api/c_api_tuning.cpp
    src/api/c_api_ecp.cpp
    src/api/error.cpp
    src/methods/registry.cpp)
endfunction()

function(vibeqc_add_runtime_sources target)
  target_sources(${target} PRIVATE
    src/runtime/context.cpp
    src/runtime/cuda_provider.cpp
    src/tensor/cpu_linalg.cpp)
  if(VIBEQC_ENABLE_CUDA)
    target_sources(${target} PRIVATE
      src/runtime/cuda_runtime.cu
      src/runtime/cuda_component_trace.cpp)
  endif()
endfunction()

function(vibeqc_add_dft_sources target)
  target_sources(${target} PRIVATE
    src/dft/ao_grid.cpp
    src/dft/cosx_reference.cpp
    src/dft/bridge.cpp
    src/dft/grid.cpp
    src/dft/ks_final_state.cpp
    src/dft/rks.cpp
    src/dft/uks.cpp
    src/dft/xc.cpp
    src/dft/dispersion/d3_runtime.cpp
    src/dft/dispersion/d4_runtime.cpp
    src/methods/dft_method.cpp)
  if(VIBEQC_ENABLE_CUDA)
    target_sources(${target} PRIVATE
      src/dft/cosx_fock_provider.cpp
      src/dft/cosx_scf.cpp
      src/dft/cuda_cosx.cu
      src/dft/cuda_cosx_derivative.cu
      src/dft/cuda_xc.cpp
      src/dft/cuda_ks.cpp
      src/dft/cuda_ks_kernels.cu
      src/dft/dispersion/d3_cuda.cu
      src/dft/dispersion/d4_cuda.cu
      src/dft/dispersion/d4_runtime_cuda.cu)
  endif()
endfunction()

function(vibeqc_add_posthf_cc_sources target)
  target_sources(${target} PRIVATE
    src/cc/solver.cpp
    src/methods/mp2_method.cpp
    src/methods/rccsd_method.cpp
    src/methods/rccsdt_method.cpp
    src/posthf/bridge.cpp
    src/posthf/cuda_derivative.cpp
    src/posthf/mp2_derivative_common.cpp
    src/posthf/mp2_derivative_cpu.cpp
    src/posthf/mp2_derivative_cuda.cpp
    src/posthf/mp2_energy.cpp
    src/posthf/mp2_force.cpp
    src/posthf/mp2_gradient.cpp
    src/posthf/native_provider.cpp
    src/response/native_gmres.cpp
    src/methods/xtb_method.cpp)
  if(VIBEQC_ENABLE_CUDA)
    target_sources(${target} PRIVATE
      src/cc/cuda_solver.cu
      src/posthf/df_bridge.cu
      src/posthf/cuda_transform.cu)
  endif()
endfunction()

function(vibeqc_add_integrals_scf_sources target)
  target_sources(${target} PRIVATE
    src/integrals/s_integrals.cpp
    src/integrals/generated_df_cpu.cpp
    src/integrals/ecp.cpp
    src/methods/hf_method.cpp
    src/molecule/basis.cpp
    src/scf/density_fitting.cpp
    src/scf/density_factor.cpp
    src/scf/df_response_weights.cpp
    src/scf/fleet.cpp
    src/scf/fock_build.cpp
    src/scf/fock_provider.cpp
    src/scf/cuda_fock_provider.cpp
    src/scf/fock_prepared.cpp
    src/scf/fock_execution.cpp
    src/scf/rhf.cpp
    src/scf/solver/diis.cpp
    src/scf/solver/proposal_control.cpp
    src/scf/solver/mean_field_driver.cpp
    src/scf/solver/eigen_frame.cpp
    src/scf/solver/final_state.cpp
    src/scf/gradient/hf_gradient.cpp
    src/scf/reference/linalg.cpp
    src/scf/reference/mean_field.cpp
    src/scf/initial_guess/density.cpp
    src/tensor/symmetric_matrix_function.cpp
    src/scf/cuda/rhf_policy.cpp)

  if(VIBEQC_ENABLE_CUDA)
    target_sources(${target} PRIVATE
      src/integrals/ecp_cuda.cu
      src/scf/cuda/df_derivatives.cu
      src/scf/cuda/df_shell_derivatives.cu
      src/scf/cuda/df_gradient_bridge.cu
      src/scf/cuda/df_response_weights.cu
      src/scf/cuda/one_electron_values.cu
      src/scf/cuda/one_electron_derivatives.cu
      src/scf/cuda/one_electron_gradient_bridge.cu
      src/scf/cuda/df_coulomb.cpp
      src/scf/cuda/df_exchange.cpp
      src/scf/cuda/df_occupied_exchange.cpp
      src/scf/cuda/df_scf_factor.cpp
      src/scf/cuda/df_scf_final_state.cpp
      src/scf/cuda/df_force_response.cpp
      src/scf/cuda/df_jk.cpp
      src/scf/cuda/df_jk_kernels.cu
      src/scf/cuda/df_packed_values.cu
      src/scf/cuda/df_metric_kernels.cu
      src/scf/cuda/df_plan.cpp
      src/scf/cuda/df_plan_lifetime.cpp
      src/scf/cuda/df_plan_setup.cpp
      src/scf/cuda/df_rhf_scf.cpp
      src/scf/cuda/df_runtime.cpp
      src/scf/cuda/df_scf_kernels.cu
      src/scf/cuda/df_scf_library.cpp
      src/scf/cuda/df_scf_diis.cpp
      src/scf/cuda/df_eigensystem.cpp
      src/scf/cuda/df_final_validation.cpp
      src/scf/cuda/final_validation_kernels.cu
      src/scf/cuda/df_uhf_scf.cpp
      src/scf/cuda_eigensolver_probe.cu
      src/scf/cuda_rhf.cpp
      src/scf/cuda/rhf_bucket.cpp
      src/scf/cuda/rhf_graph.cpp
      src/scf/cuda/one_electron_reference.cu
      src/scf/cuda/one_electron_force_reference.cu
      src/scf/cuda/nuclear_kernels.cu
      src/scf/cuda/direct_pair_cache.cu
      src/scf/cuda/direct_jk.cpp
      src/scf/cuda/one_electron_view.cpp
      src/scf/cuda/one_electron_export.cpp
      src/scf/cuda/one_electron_export_batch.cpp
      src/scf/cuda/direct_tile_validation.cu
      src/scf/cuda/direct_density_bounds.cu
      src/scf/cuda/direct_tile_compaction.cu
      src/scf/cuda/direct_generated_tasks.cu
      src/scf/cuda/direct_resident_tasks.cu
      src/scf/cuda/direct_bounded_pages.cu
      src/scf/cuda/direct_bounded_tasks.cu
      src/scf/cuda/direct_queue_scan.cu
      src/scf/cuda/direct_queue_diagnostics.cu
      src/scf/cuda/resources.cpp
      src/scf/cuda/matrix_library.cpp
      src/scf/cuda/runtime_support.cpp
      src/scf/cuda/scf_state_kernels.cu
      src/scf/cuda/scf_matrix_kernels.cu
      src/scf/cuda/scf_density_kernels.cu
      src/scf/cuda/scf_diis_kernels.cu
      src/scf/cuda/scf_convergence_kernels.cu
      src/scf/cuda/basis_transform_kernels.cu
      src/scf/cuda/df_source_setup.cpp
      src/scf/cuda/df_source.cpp
      src/scf/cuda/df_source_kernels.cu
      src/scf/cuda/df_integral_export.cpp
      src/scf/cuda/df_integral_export_batch.cpp
      src/scf/cuda/arena.cpp
      src/scf/cuda/topology.cpp
      src/scf/cuda/queue_plan.cpp
      src/scf/cuda/queue_profile.cpp
      src/scf/cuda/eigensolver.cpp
      src/scf/cuda/eigensolver_kernels.cu
      src/scf/cuda_hf_entry.cpp)
  else()
    target_sources(${target} PRIVATE
      src/scf/cuda_density_fitting_stub.cpp
      src/integrals/ecp_cuda_stub.cpp
      src/scf/cuda_direct_jk_stub.cpp
      src/scf/cuda_density_fitting_integrals_stub.cpp
      src/scf/cuda_weighted_eri_stub.cpp)
  endif()
endfunction()

function(vibeqc_add_component_sources target)
  vibeqc_add_api_sources(${target})
  vibeqc_add_runtime_sources(${target})
  vibeqc_add_integrals_scf_sources(${target})
  vibeqc_add_dft_sources(${target})
  vibeqc_add_posthf_cc_sources(${target})
endfunction()
