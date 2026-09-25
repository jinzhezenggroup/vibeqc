include_guard(GLOBAL)

# Project-level generated-source declarations. The generic command mechanics
# live in VibeQCGenerated.cmake; this file owns generator inputs/outputs and the
# target(s) that consume each generated family.
macro(vibeqc_register_host_generated_sources target)
  set(VIBEQC_QUADRATURE_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_quadrature.cuh")
  vibeqc_register_generated_sources(
    NAME vibeqc_quadrature_codegen
    TARGET ${target}
    ADD_TO_TARGET
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_quadrature_cuda.py"
    OUTPUTS "${VIBEQC_QUADRATURE_HEADER}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/xc/quadrature_cuda.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/xc/grid_response_ir.py"
    ARGS --output "${VIBEQC_QUADRATURE_HEADER}"
    COMMENT "Generating bounded CUDA molecular quadrature")
  set(VIBEQC_DF_EXCHANGE_SCHEDULE_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_df_exchange_schedule.hpp")
  vibeqc_register_generated_sources(
    NAME vibeqc_df_exchange_schedule_codegen
    TARGET ${target}
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_df_exchange_schedule.py"
    OUTPUTS "${VIBEQC_DF_EXCHANGE_SCHEDULE_HEADER}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/method/df_exchange_schedule.py"
    ARGS --output "${VIBEQC_DF_EXCHANGE_SCHEDULE_HEADER}"
    COMMENT "Generating compiler-owned DF source-reuse schedule")

  # The native host policy is built even when CUDA execution is disabled.
  # Generate its CUDA-independent constants once for both build variants.
  set(VIBEQC_ONE_ELECTRON_DERIVATIVE_POLICY_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_one_electron_derivative_policy.cuh")
  vibeqc_register_generated_sources(
    NAME vibeqc_one_electron_derivative_policy_codegen
    TARGET ${target}
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_one_electron_kernels.py"
    OUTPUTS "${VIBEQC_ONE_ELECTRON_DERIVATIVE_POLICY_HEADER}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_df_kernels.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/integral/cooperative_schedule.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/integral/one_electron_derivative_policy_cuda.py"
    ARGS --derivatives --derivative-policy-output "${VIBEQC_ONE_ELECTRON_DERIVATIVE_POLICY_HEADER}")

  set(VIBEQC_METHOD_PARAMETERS_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_method_parameters.hpp")
  vibeqc_register_generated_sources(
    NAME vibeqc_method_parameters_codegen
    TARGET ${target}
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_method_parameters.py"
    OUTPUTS "${VIBEQC_METHOD_PARAMETERS_HEADER}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/method/method_parameters.json"
    ARGS
      --source "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/method/method_parameters.json"
      --cpp-output "${VIBEQC_METHOD_PARAMETERS_HEADER}"
    COMMENT "Generating audited method parameter constants")

  set(VIBEQC_D4_DERIVATIVE_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_d4_derivative.hpp")
  vibeqc_register_generated_sources(
    NAME vibeqc_d4_derivative_codegen
    TARGET ${target}
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_d4_derivative.py"
    OUTPUTS "${VIBEQC_D4_DERIVATIVE_HEADER}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/method/d4_derivative.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/common/provenance.py"
    ARGS --output "${VIBEQC_D4_DERIVATIVE_HEADER}"
    COMMENT "Generating compiler-owned D4 EEQ derivative lowering")

  set(VIBEQC_D3_DATA_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/d3_data.hpp")
  vibeqc_register_generated_sources(
    NAME vibeqc_d3_codegen
    TARGET ${target}
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/vibeqc_d3/generate_native_data.py"
    OUTPUTS "${VIBEQC_D3_DATA_HEADER}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/data/parameters/d3_production.bin"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/common/d3_data.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/manifests/xtbloom-d3.json"
    ARGS --output "${VIBEQC_D3_DATA_HEADER}"
    COMMENT "Generating pinned compact D3(BJ) tables")
  set(VIBEQC_ONE_ELECTRON_ST_CPU_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_one_electron_st_cpu.hpp")
  vibeqc_register_generated_sources(
    NAME vibeqc_one_electron_st_cpu_codegen
    TARGET ${target}
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_one_electron_kernels.py"
    OUTPUTS "${VIBEQC_ONE_ELECTRON_ST_CPU_HEADER}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_df_kernels.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/integral/one_electron_cpu.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/integral/one_electron_cuda.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/integral/one_electron_derivatives_cuda.py"
    ARGS --cpu-st-output "${VIBEQC_ONE_ELECTRON_ST_CPU_HEADER}")

  set(VIBEQC_DF_VALUE_CPU_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_df_values_cpu.hpp")
  vibeqc_register_generated_sources(
    NAME vibeqc_df_values_cpu_codegen
    TARGET ${target}
    ADD_TO_TARGET
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_df_kernels.py"
    OUTPUTS "${VIBEQC_DF_VALUE_CPU_HEADER}"
    ARGS
      --cpu
      --output "${VIBEQC_DF_VALUE_CPU_HEADER}")

  set(VIBEQC_DF_DERIVATIVE_CPU_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_df_derivatives_cpu.hpp")
  vibeqc_register_generated_sources(
    NAME vibeqc_df_derivatives_cpu_codegen
    TARGET ${target}
    ADD_TO_TARGET
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_df_kernels.py"
    OUTPUTS "${VIBEQC_DF_DERIVATIVE_CPU_HEADER}"
    ARGS
      --derivatives
      --cpu
      --output "${VIBEQC_DF_DERIVATIVE_CPU_HEADER}")

  set(VIBEQC_XC_CPU_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/xc_cpu_generated.hpp")
  vibeqc_register_generated_sources(
    NAME vibeqc_xc_cpu_codegen
    TARGET ${target}
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_xc_cpu.py"
    OUTPUTS "${VIBEQC_XC_CPU_HEADER}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/xc/semilocal_codegen.py"
    ARGS --output "${VIBEQC_XC_CPU_HEADER}")

  set(VIBEQC_SCF_ARRAY_CPU_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_scf_array_native.hpp")
  vibeqc_register_generated_sources(
    NAME vibeqc_scf_array_cpu_codegen
    TARGET ${target}
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_scf_array_native.py"
    OUTPUTS "${VIBEQC_SCF_ARRAY_CPU_HEADER}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/array_api/scf.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/tensor/scf.py"
    ARGS --output "${VIBEQC_SCF_ARRAY_CPU_HEADER}"
    COMMENT "Generating Array frontend SCF CPU tensor helpers")

  set(VIBEQC_GFN2_PAIR_CPU_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_gfn2_pair_native.hpp")
  vibeqc_register_generated_sources(
    NAME vibeqc_gfn2_pair_cpu_codegen
    TARGET ${target}
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_gfn2_pair_native.py"
    OUTPUTS "${VIBEQC_GFN2_PAIR_CPU_HEADER}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/geometry/gfn2_pair.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/tensor/ad_program.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/tensor/scalar_cpp.py"
    ARGS --output "${VIBEQC_GFN2_PAIR_CPU_HEADER}"
    COMMENT "Generating compiler-owned GFN2 CPU pair kernels")

  set(VIBEQC_GFN2_AES2_CPU_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_gfn2_aes2_native.hpp")
  vibeqc_register_generated_sources(
    NAME vibeqc_gfn2_aes2_cpu_codegen
    TARGET ${target}
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_gfn2_aes2_native.py"
    OUTPUTS "${VIBEQC_GFN2_AES2_CPU_HEADER}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/method/gfn2_aes2.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/tensor/ad_program.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/tensor/scalar_cpp.py"
    ARGS --cpu-output "${VIBEQC_GFN2_AES2_CPU_HEADER}"
    COMMENT "Generating compiler-owned GFN2 AES2 CPU kernels")

  set(VIBEQC_GFN2_ES2_NATIVE_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_gfn2_es2_native.hpp")
  vibeqc_register_generated_sources(
    NAME vibeqc_gfn2_es2_native_codegen
    TARGET ${target}
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_gfn2_es2_native.py"
    OUTPUTS "${VIBEQC_GFN2_ES2_NATIVE_HEADER}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/method/gfn2_es2_runtime.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/tensor/ad_program.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/tensor/scalar_cpp.py"
    ARGS --output "${VIBEQC_GFN2_ES2_NATIVE_HEADER}"
    COMMENT "Generating compiler-owned GFN2 ES2 scalar kernels")

  set(VIBEQC_GFN2_H0_NATIVE_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_gfn2_h0_native.hpp")
  vibeqc_register_generated_sources(
    NAME vibeqc_gfn2_h0_native_codegen
    TARGET ${target}
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_gfn2_h0_native.py"
    OUTPUTS "${VIBEQC_GFN2_H0_NATIVE_HEADER}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/method/gfn2_h0_force_runtime.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/tensor/ad_program.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/tensor/scalar_cpp.py"
    ARGS --output "${VIBEQC_GFN2_H0_NATIVE_HEADER}"
    COMMENT "Generating compiler-owned GFN2 CPU/CUDA H0 values and adjoints")
  if(TARGET vibeqc_gfn2_cuda)
    add_dependencies(vibeqc_gfn2_cuda vibeqc_gfn2_h0_native_codegen)
  endif()

  set(VIBEQC_GFN2_SPIN_NATIVE_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_gfn2_spin_native.hpp")
  vibeqc_register_generated_sources(
    NAME vibeqc_gfn2_spin_native_codegen
    TARGET ${target}
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_gfn2_spin_native.py"
    OUTPUTS "${VIBEQC_GFN2_SPIN_NATIVE_HEADER}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/method/gfn2_spin_runtime.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/tensor/ad_program.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/tensor/scalar_cpp.py"
    ARGS --output "${VIBEQC_GFN2_SPIN_NATIVE_HEADER}"
    COMMENT "Generating compiler-owned GFN2 CPU/CUDA spin energy/potential")
  if(TARGET vibeqc_gfn2_cuda)
    add_dependencies(vibeqc_gfn2_cuda vibeqc_gfn2_spin_native_codegen)
  endif()

  set(VIBEQC_GFN2_ES3_NATIVE_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_gfn2_es3_native.cuh")
  vibeqc_register_generated_sources(
    NAME vibeqc_gfn2_es3_codegen
    TARGET ${target}
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_gfn2_es3_native.py"
    OUTPUTS "${VIBEQC_GFN2_ES3_NATIVE_HEADER}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/method/gfn2_es3_runtime.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/tensor/ad_program.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/tensor/scalar_cpp.py"
    ARGS --output "${VIBEQC_GFN2_ES3_NATIVE_HEADER}"
    COMMENT "Generating compiler-owned GFN2 ES3 shell energy/potential")
  if(TARGET vibeqc_gfn2_cuda)
    add_dependencies(vibeqc_gfn2_cuda vibeqc_gfn2_es3_codegen)
    target_include_directories(
      vibeqc_gfn2_cuda PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
  endif()

  set(VIBEQC_GFN2_SCC_FREE_ENERGY_NATIVE_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_gfn2_scc_free_energy_native.hpp")
  vibeqc_register_generated_sources(
    NAME vibeqc_gfn2_scc_free_energy_codegen
    TARGET ${target}
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_gfn2_scc_free_energy_native.py"
    OUTPUTS "${VIBEQC_GFN2_SCC_FREE_ENERGY_NATIVE_HEADER}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/method/gfn2_scc_free_energy_runtime.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/tensor/scalar_cpp.py"
    ARGS --output "${VIBEQC_GFN2_SCC_FREE_ENERGY_NATIVE_HEADER}"
    COMMENT "Generating compiler-owned GFN2 SCC internal/free-energy composition")
  if(TARGET vibeqc_gfn2_cuda)
    add_dependencies(vibeqc_gfn2_cuda vibeqc_gfn2_scc_free_energy_codegen)
    target_include_directories(
      vibeqc_gfn2_cuda PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
  endif()

  set(VIBEQC_GFN2_ELECTRONIC_CPU_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_gfn2_electronic_native.hpp")
  vibeqc_register_generated_sources(
    NAME vibeqc_gfn2_electronic_cpu_codegen
    TARGET ${target}
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_gfn2_electronic_native.py"
    OUTPUTS "${VIBEQC_GFN2_ELECTRONIC_CPU_HEADER}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/method/gfn2_electronic_runtime.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/tensor/ad_program.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/tensor/scalar_cpp.py"
    ARGS --output "${VIBEQC_GFN2_ELECTRONIC_CPU_HEADER}"
    COMMENT "Generating compiler-owned GFN2 CPU electronic kernels")

  file(GLOB VIBEQC_RCCSD_GENERATOR_INPUTS CONFIGURE_DEPENDS
       "${CMAKE_CURRENT_SOURCE_DIR}/tools/vibeqc_cc/*.py")
  set(VIBEQC_GFN2_SDQ_CPU_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_gfn2_sdq_native.hpp")
  vibeqc_register_generated_sources(
    NAME vibeqc_gfn2_sdq_cpu_codegen
    TARGET ${target}
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_gfn2_sdq_native.py"
    OUTPUTS "${VIBEQC_GFN2_SDQ_CPU_HEADER}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/integral/gfn2_sdq.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/integral/gfn2_sdq_cpu.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/integral/one_electron_values.py"
    ARGS --output "${VIBEQC_GFN2_SDQ_CPU_HEADER}"
    COMMENT "Generating compiler-owned GFN2 S/D/Q CPU primitive kernels")

  set(VIBEQC_RCCSD_CPU_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_rccsd_cpu.hpp")
  vibeqc_register_generated_sources(
    NAME vibeqc_rccsd_cpu_codegen
    TARGET ${target}
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_rccsd_native.py"
    OUTPUTS "${VIBEQC_RCCSD_CPU_HEADER}"
    DEPENDS ${VIBEQC_RCCSD_GENERATOR_INPUTS}
    ARGS --cpu-header "${VIBEQC_RCCSD_CPU_HEADER}")

  set(VIBEQC_RCCSDT_CPU_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_rccsdt_cpu.hpp")
  vibeqc_register_generated_sources(
    NAME vibeqc_rccsdt_cpu_codegen
    TARGET ${target}
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_rccsdt_native.py"
    OUTPUTS "${VIBEQC_RCCSDT_CPU_HEADER}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tools/vibeqc_cc/triples.py"
    ARGS --output "${VIBEQC_RCCSDT_CPU_HEADER}")

  set(VIBEQC_DF_HF_RESPONSE_CONTRACT_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_df_hf_response_contract.hpp")
  vibeqc_register_generated_sources(
    NAME vibeqc_df_hf_response_contract_codegen
    TARGET ${target}
    ADD_TO_TARGET
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_df_hf_response.py"
    OUTPUTS "${VIBEQC_DF_HF_RESPONSE_CONTRACT_HEADER}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/method/df_hf_response_contract.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/method/df_hf_response_cuda.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/tensor/cuda_dtype.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/tensor/cuda_gemm.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/tensor/ir.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/tensor/types.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/common/layout.py"
    ARGS --contract-output "${VIBEQC_DF_HF_RESPONSE_CONTRACT_HEADER}")

  set(VIBEQC_ECP_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_ecp_ao.cuh")
  vibeqc_register_generated_sources(
    NAME vibeqc_ecp_codegen
    TARGET ${target}
    ADD_TO_TARGET
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_ecp_kernels.py"
    OUTPUTS "${VIBEQC_ECP_HEADER}"
    ARGS --output "${VIBEQC_ECP_HEADER}")
endmacro()

macro(vibeqc_register_cuda_generated_sources target)
  set(VIBEQC_MEAN_FIELD_SETUP_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_mean_field_setup.cuh")
  vibeqc_register_generated_sources(
    TARGET ${target}
    ADD_TO_TARGET
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_mean_field_setup_cuda.py"
    OUTPUTS "${VIBEQC_MEAN_FIELD_SETUP_HEADER}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_scf_array_native.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/method/mean_field_setup_cuda.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/tensor/scf.py"
    ARGS --output "${VIBEQC_MEAN_FIELD_SETUP_HEADER}"
    COMMENT "Generating shared CUDA mean-field setup projectors")

  set(VIBEQC_MATRIX_FUNCTION_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_symmetric_matrix_function.cuh")
  vibeqc_register_generated_sources(
    TARGET ${target}
    ADD_TO_TARGET
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_matrix_function_cuda.py"
    OUTPUTS "${VIBEQC_MATRIX_FUNCTION_HEADER}"
    ARGS --output "${VIBEQC_MATRIX_FUNCTION_HEADER}")

  set(VIBEQC_DF_HF_RESPONSE_CUDA_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_df_hf_response.cuh")
  vibeqc_register_generated_sources(
    TARGET ${target}
    ADD_TO_TARGET
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_df_hf_response.py"
    OUTPUTS "${VIBEQC_DF_HF_RESPONSE_CUDA_HEADER}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/method/df_hf_response_contract.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/method/df_hf_response_cuda.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/tensor/cuda_dtype.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/tensor/cuda_gemm.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/tensor/ir.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/tensor/types.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/common/layout.py"
    ARGS --cuda-output "${VIBEQC_DF_HF_RESPONSE_CUDA_HEADER}")

  set(VIBEQC_DF_GENERATED_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/df_values.cuh")
  set(VIBEQC_DF_POLICY_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_df_policy.cuh")
  set(VIBEQC_DF_VALUE_CANDIDATE_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_df_value_candidates.cuh")
  vibeqc_register_generated_sources(
    TARGET ${target}
    ADD_TO_TARGET
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_df_kernels.py"
    OUTPUTS
      "${VIBEQC_DF_GENERATED_HEADER}"
      "${VIBEQC_DF_POLICY_HEADER}"
      "${VIBEQC_DF_VALUE_CANDIDATE_HEADER}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/integral/production_df_values.json"
    ARGS
      --output "${VIBEQC_DF_GENERATED_HEADER}"
      --policy-output "${VIBEQC_DF_POLICY_HEADER}")

  # The native views and compiler both admit all 4^3 s/p/d/f classes. Keep
  # stable per-class output paths; profile edits must not reshuffle units.
  set(VIBEQC_DF_SHELL_UNIT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated/df_shells")
  set(VIBEQC_DF_SHELL_SOURCES)
  set(VIBEQC_DF_SHELL_UNIT_OUTPUTS
      "${VIBEQC_DF_SHELL_UNIT_DIRECTORY}/generated_df_shell_dispatch.hpp")
  foreach(_a RANGE 0 3)
    foreach(_b RANGE 0 3)
      foreach(_c RANGE 0 3)
        set(_class "${_a}${_b}${_c}")
        list(APPEND VIBEQC_DF_SHELL_SOURCES
             "${VIBEQC_DF_SHELL_UNIT_DIRECTORY}/df_shell_${_class}.cu")
        foreach(_header IN ITEMS polynomial math)
          list(APPEND VIBEQC_DF_SHELL_UNIT_OUTPUTS
               "${VIBEQC_DF_SHELL_UNIT_DIRECTORY}/df_shell_${_header}_${_class}.cuh")
        endforeach()
        list(APPEND VIBEQC_DF_SHELL_UNIT_OUTPUTS
             "${VIBEQC_DF_SHELL_UNIT_DIRECTORY}/df_shell_policy_${_class}.hpp")
      endforeach()
    endforeach()
  endforeach()
  target_include_directories(${target} PRIVATE "${VIBEQC_DF_SHELL_UNIT_DIRECTORY}")

  set(VIBEQC_DF_DERIVATIVE_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_df_derivatives.cuh")
  set(VIBEQC_DF_DERIVATIVE_POLICY_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_df_derivative_policy.cuh")
  set(VIBEQC_DF_DERIVATIVE_SCHEDULE_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_df_derivative_schedule.cuh")
  set(VIBEQC_DF_SHELL_DERIVATIVE_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_df_shell_derivatives.cuh")
  set(VIBEQC_DF_RYS_HEADERS
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_df_rys.cuh"
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_df_rys_policy.hpp"
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_df_production.hpp"
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_df_screening.cuh"
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_df_rys_shell.cuh")
  vibeqc_register_generated_sources(
    TARGET ${target}
    ADD_TO_TARGET
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_df_kernels.py"
    OUTPUTS
      "${VIBEQC_DF_DERIVATIVE_HEADER}"
      "${VIBEQC_DF_DERIVATIVE_POLICY_HEADER}"
      "${VIBEQC_DF_DERIVATIVE_SCHEDULE_HEADER}"
      "${VIBEQC_DF_SHELL_DERIVATIVE_HEADER}"
      ${VIBEQC_DF_RYS_HEADERS}
      ${VIBEQC_DF_SHELL_SOURCES}
      ${VIBEQC_DF_SHELL_UNIT_OUTPUTS}
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/integral/production_df_derivatives.json"
    ARGS
      --derivatives
      --output "${VIBEQC_DF_DERIVATIVE_HEADER}"
      --policy-output "${VIBEQC_DF_DERIVATIVE_POLICY_HEADER}"
      --schedule-output "${VIBEQC_DF_DERIVATIVE_SCHEDULE_HEADER}"
      --shell-output "${VIBEQC_DF_SHELL_DERIVATIVE_HEADER}"
      --shell-units-directory "${VIBEQC_DF_SHELL_UNIT_DIRECTORY}")

  set(VIBEQC_ONE_ELECTRON_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_one_electron_values.cuh")
  set(VIBEQC_ONE_ELECTRON_POLICY_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_one_electron_policy.cuh")
  vibeqc_register_generated_sources(
    TARGET ${target}
    ADD_TO_TARGET
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_one_electron_kernels.py"
    OUTPUTS
      "${VIBEQC_ONE_ELECTRON_HEADER}"
      "${VIBEQC_ONE_ELECTRON_POLICY_HEADER}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_df_kernels.py"
    ARGS
      --output "${VIBEQC_ONE_ELECTRON_HEADER}"
      --policy-output "${VIBEQC_ONE_ELECTRON_POLICY_HEADER}")

  set(VIBEQC_ONE_ELECTRON_DERIVATIVE_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_one_electron_derivatives.cuh")
  vibeqc_register_generated_sources(
    TARGET ${target}
    ADD_TO_TARGET
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_one_electron_kernels.py"
    OUTPUTS "${VIBEQC_ONE_ELECTRON_DERIVATIVE_HEADER}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_df_kernels.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/integral/cooperative_schedule.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/integral/one_electron_derivative_policy_cuda.py"
    ARGS --derivatives --output "${VIBEQC_ONE_ELECTRON_DERIVATIVE_HEADER}")

  set(VIBEQC_COSX_DERIVATIVE_CONTRACTION_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_cosx_derivative_contractions.cuh")
  vibeqc_register_generated_sources(
    NAME vibeqc_cosx_derivative_contraction_codegen
    TARGET ${target}
    ADD_TO_TARGET
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_cosx_derivative_native.py"
    OUTPUTS "${VIBEQC_COSX_DERIVATIVE_CONTRACTION_HEADER}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/method/cosx_derivative_runtime.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/tensor/scalar_cpp.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/tensor/ir.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/tensor/program.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/tensor/types.py"
    ARGS --output "${VIBEQC_COSX_DERIVATIVE_CONTRACTION_HEADER}"
    COMMENT "Generating compiler-owned COSX derivative contractions")

  set(VIBEQC_GFN2_SDQ_CUDA_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_gfn2_sdq_cuda.cuh")
  vibeqc_register_generated_sources(
    NAME vibeqc_gfn2_sdq_cuda_codegen
    TARGET ${target}
    ADD_TO_TARGET
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_gfn2_sdq_native.py"
    OUTPUTS "${VIBEQC_GFN2_SDQ_CUDA_HEADER}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/integral/gfn2_sdq.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/integral/gfn2_sdq_cpu.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/integral/one_electron_values.py"
    ARGS --cuda-output "${VIBEQC_GFN2_SDQ_CUDA_HEADER}"
    COMMENT "Generating compiler-owned GFN2 S/D/Q CUDA primitive kernels")

  set(VIBEQC_DIRECT_FOCK_ACCUMULATION_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_direct_fock_accumulation.cuh")
  vibeqc_register_generated_sources(
    TARGET ${target}
    ADD_TO_TARGET
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_shell_kernels.py"
    OUTPUTS "${VIBEQC_DIRECT_FOCK_ACCUMULATION_HEADER}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/integral/lowering/fock_accumulation.py"
    ARGS
      --direct-fock-accumulation-output "${VIBEQC_DIRECT_FOCK_ACCUMULATION_HEADER}"
    COMMENT "Generating compiler-owned Direct-Fock scatter contraction")

  set(VIBEQC_WEIGHTED_ERI_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/weighted_eri.cuh")
  vibeqc_register_generated_sources(
    TARGET ${target}
    ADD_TO_TARGET
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_weighted_eri_kernels.py"
    OUTPUTS "${VIBEQC_WEIGHTED_ERI_HEADER}"
    ARGS --output "${VIBEQC_WEIGHTED_ERI_HEADER}")

  set(VIBEQC_DIRECT_RESIDENT_PSSS_SCHEDULE_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_direct_resident_psss_schedule.cuh")
  vibeqc_register_generated_sources(
    NAME vibeqc_direct_resident_psss_schedule_codegen
    TARGET ${target}
    ADD_TO_TARGET
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_direct_resident_schedule.py"
    OUTPUTS "${VIBEQC_DIRECT_RESIDENT_PSSS_SCHEDULE_HEADER}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/integral/direct_resident_schedule.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/integral/cuda_schedule.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/integral/shell_spec.py"
    ARGS --output "${VIBEQC_DIRECT_RESIDENT_PSSS_SCHEDULE_HEADER}"
    COMMENT "Generating compiler-owned Direct-HF resident-PSSS schedule")

  set(VIBEQC_DIRECT_HIGH_ORDER_PAIR_GRADIENT_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_direct_high_order_pair_gradient.cuh")
  vibeqc_register_generated_sources(
    NAME vibeqc_direct_high_order_pair_gradient_codegen
    TARGET ${target}
    ADD_TO_TARGET
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_direct_pair_gradient.py"
    OUTPUTS "${VIBEQC_DIRECT_HIGH_ORDER_PAIR_GRADIENT_HEADER}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/integral/direct_pair_gradient_cuda.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/integral/coulomb_recurrence_cuda.py"
    ARGS --output "${VIBEQC_DIRECT_HIGH_ORDER_PAIR_GRADIENT_HEADER}"
    COMMENT "Generating compiler-owned Direct-HF high-order pair-gradient helper")

  set(VIBEQC_B3LYP_CUDA_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_b3lyp_device.cuh")
  vibeqc_register_generated_sources(
    TARGET ${target}
    ADD_TO_TARGET
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_xc_b3lyp_cuda.py"
    OUTPUTS "${VIBEQC_B3LYP_CUDA_HEADER}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/xc/semilocal_codegen.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/xc/b3lyp_production_policy.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/method/spec.py"
    ARGS --output "${VIBEQC_B3LYP_CUDA_HEADER}")

  set(VIBEQC_R2SCAN_CUDA_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_r2scan_device.cuh")
  vibeqc_register_generated_sources(
    TARGET ${target}
    ADD_TO_TARGET
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_xc_r2scan_cuda.py"
    OUTPUTS "${VIBEQC_R2SCAN_CUDA_HEADER}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/xc/semilocal_codegen.py"
    ARGS --output "${VIBEQC_R2SCAN_CUDA_HEADER}")

  set(VIBEQC_WB97MV_CUDA_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_wb97mv_device.cuh")
  vibeqc_register_generated_sources(
    TARGET ${target}
    ADD_TO_TARGET
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_xc_wb97mv_cuda.py"
    OUTPUTS "${VIBEQC_WB97MV_CUDA_HEADER}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/xc/semilocal_codegen.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/xc/wb97mv_maple.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/method/spec.py"
    ARGS --output "${VIBEQC_WB97MV_CUDA_HEADER}")

  set(VIBEQC_GRID_SOURCE
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_grid_policy.cu")
  # The r2SCAN minority-spin derivative is sensitive to contraction of 1-zeta
  # near the work-density floor. Match the compiler XC FP64 policy and the
  # independent Libxc boundary qualification; do not introduce FMA. CuMetal's
  # nvcc-compatible driver delegates to Clang and requires its native spelling.
  if(VIBEQC_CUDA_PROVIDER STREQUAL "cumetal")
    set(_vibeqc_grid_fp_contract_option -ffp-contract=off)
  else()
    set(_vibeqc_grid_fp_contract_option --fmad=false)
  endif()
  vibeqc_register_generated_sources(
    TARGET ${target}
    ADD_TO_TARGET
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_grid_kernels.py"
    OUTPUTS "${VIBEQC_GRID_SOURCE}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/src/dft/cuda_xc_kernels.cuh"
      "${VIBEQC_R2SCAN_CUDA_HEADER}"
      "${VIBEQC_WB97MV_CUDA_HEADER}"
    COMPILE_OPTIONS "${_vibeqc_grid_fp_contract_option}"
    ARGS --output "${VIBEQC_GRID_SOURCE}")

  set(VIBEQC_XC_GRADIENT_SOURCE
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_xc_gradient.cu")
  vibeqc_register_generated_sources(
    TARGET ${target}
    ADD_TO_TARGET
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_xc_gradient_cuda.py"
    OUTPUTS "${VIBEQC_XC_GRADIENT_SOURCE}"
    ARGS --output "${VIBEQC_XC_GRADIENT_SOURCE}")

  set(VIBEQC_RCCSD_CUDA_SOURCE
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_rccsd_cuda.cu")
  vibeqc_register_generated_sources(
    NAME vibeqc_rccsd_cuda_codegen
    TARGET ${target}
    ADD_TO_TARGET
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_rccsd_native.py"
    OUTPUTS "${VIBEQC_RCCSD_CUDA_SOURCE}"
    DEPENDS ${VIBEQC_RCCSD_GENERATOR_INPUTS}
    ARGS --cuda-source "${VIBEQC_RCCSD_CUDA_SOURCE}")

  set(VIBEQC_RCCSDT_CUDA_SOURCE
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_rccsdt_cuda.cu")
  vibeqc_register_generated_sources(
    NAME vibeqc_rccsdt_cuda_codegen
    TARGET ${target}
    ADD_TO_TARGET
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_rccsdt_cuda.py"
    OUTPUTS "${VIBEQC_RCCSDT_CUDA_SOURCE}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_rccsdt_cuda.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_rccsdt_native.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/vibeqc_cc/triples.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/cc/triples_cuda.hpp"
    ARGS --output "${VIBEQC_RCCSDT_CUDA_SOURCE}")

  set(VIBEQC_MP2_GENERATED_DIRECTORY
      "${CMAKE_CURRENT_BINARY_DIR}/generated/mp2")
  set(VIBEQC_MP2_GENERATED_SOURCES
      "${VIBEQC_MP2_GENERATED_DIRECTORY}/mp2_cuda_table.cu")
  set(VIBEQC_MP2_ARCHITECTURES "")
  foreach(_arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
    string(REGEX REPLACE "-.*$" "" _numeric_arch "${_arch}")
    list(APPEND VIBEQC_MP2_ARCHITECTURES "${_numeric_arch}")
  endforeach()
  list(REMOVE_DUPLICATES VIBEQC_MP2_ARCHITECTURES)
  foreach(_arch IN LISTS VIBEQC_MP2_ARCHITECTURES)
    foreach(_tile 1 2 4 8)
      list(APPEND VIBEQC_MP2_GENERATED_SOURCES
           "${VIBEQC_MP2_GENERATED_DIRECTORY}/mp2_sm${_arch}_t${_tile}_runtime.cu")
    endforeach()
  endforeach()
  file(GLOB VIBEQC_MP2_GENERATOR_INPUTS CONFIGURE_DEPENDS
       "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/tensor/*.py"
       "python/vibeqc_compiler/common/cuda_target.py"
       "${CMAKE_CURRENT_SOURCE_DIR}/tools/vibeqc_mp2/*.py"
       "${CMAKE_CURRENT_SOURCE_DIR}/tools/vibeqc_posthf/*.py")
  vibeqc_register_generated_sources(
    TARGET ${target}
    ADD_TO_TARGET
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_mp2_native.py"
    OUTPUTS ${VIBEQC_MP2_GENERATED_SOURCES}
    DEPENDS ${VIBEQC_MP2_GENERATOR_INPUTS}
    ARGS
      --cuda-dir "${VIBEQC_MP2_GENERATED_DIRECTORY}"
      --architectures "${VIBEQC_MP2_ARCHITECTURES}")
endmacro()

macro(vibeqc_add_codegen_pilot)
  if(Python3_Interpreter_FOUND)
    set(VIBEQC_CODEGEN_PILOT_DIRECTORY
        "${CMAKE_CURRENT_BINARY_DIR}/generated/shell_kernels")
    set(VIBEQC_CODEGEN_PILOT_OUTPUTS)
    foreach(axis x y z)
      set(output
          "${VIBEQC_CODEGEN_PILOT_DIRECTORY}/eri_psss_${axis}_gradient.cuh")
      vibeqc_register_generated_sources(
        GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_shell_kernels.py"
        OUTPUTS "${output}"
        ARGS --shell-class psss --axis "${axis}" --output "${output}"
        COMMENT "Generating symbolic/CSE psss ${axis}-gradient CUDA pilot")
      list(APPEND VIBEQC_CODEGEN_PILOT_OUTPUTS "${output}")
    endforeach()

    set(output
        "${VIBEQC_CODEGEN_PILOT_DIRECTORY}/eri_dppp_xy_xyz_factored_gradient.cuh")
    vibeqc_register_generated_sources(
      GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_shell_kernels.py"
      OUTPUTS "${output}"
      ARGS
        --shell-class dppp --d-component xy --p-components xyz
        --lowering factored --output "${output}"
      COMMENT "Generating factored symbolic/CSE dppp xy/xyz CUDA candidate")
    list(APPEND VIBEQC_CODEGEN_PILOT_OUTPUTS "${output}")
    add_custom_target(vibeqc_codegen_pilot DEPENDS ${VIBEQC_CODEGEN_PILOT_OUTPUTS})
  endif()
endmacro()
