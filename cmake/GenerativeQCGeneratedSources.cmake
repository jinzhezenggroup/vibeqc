include_guard(GLOBAL)

# Project-level generated-source declarations. The generic command mechanics
# live in GenerativeQCGenerated.cmake; this file owns generator inputs/outputs and the
# target(s) that consume each generated family.
macro(generativeqc_register_host_generated_sources target)
  if(GENERATIVEQC_ENABLE_STATIONARY_CPU_FORCE_AOT)
    set(GENERATIVEQC_STATIONARY_CPU_AOT_DIRECTORY
        "${CMAKE_CURRENT_BINARY_DIR}/generated/stationary_cpu_derivatives")
    set(GENERATIVEQC_STATIONARY_CPU_AOT_SOURCES)
    foreach(_generativeqc_stationary_cpu_shard RANGE 0 45)
      list(APPEND GENERATIVEQC_STATIONARY_CPU_AOT_SOURCES
           "${GENERATIVEQC_STATIONARY_CPU_AOT_DIRECTORY}/generativeqc_stationary_cpu_derivative_${_generativeqc_stationary_cpu_shard}.cpp")
    endforeach()
    generativeqc_register_generated_sources(
      NAME generativeqc_stationary_cpu_derivatives_codegen
      TARGET ${target}
      ADD_TO_TARGET
      GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_stationary_cpu_derivative_aot.py"
      OUTPUTS ${GENERATIVEQC_STATIONARY_CPU_AOT_SOURCES}
      DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/integral/expr.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/integral/first_derivative_native.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/integral/first_derivative_schedule.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/integral/one_electron_derivatives.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/integral/range_separation.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/integral/scalar_c.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/integral/shell_spec.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/integral/weighted_eri.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/integral/weighted_eri_cuda.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/integral/weighted_eri_native.py"
      ARGS --output-directory "${GENERATIVEQC_STATIONARY_CPU_AOT_DIRECTORY}"
      COMPILE_OPTIONS "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-ffp-contract=off>"
      COMMENT "Generating packaged stationary CPU s/p/d derivative inventory")
  endif()

  if(GENERATIVEQC_ENABLE_STATIONARY_CPU_FORCE_AOT)
    set(GENERATIVEQC_RANGE_DERIVATIVE_CPU_AOT_DIRECTORY
        "${CMAKE_CURRENT_BINARY_DIR}/generated/range_derivative_cpu")
    set(GENERATIVEQC_RANGE_DERIVATIVE_CPU_AOT_SOURCES)
    foreach(_generativeqc_range_family IN ITEMS sr lr)
      foreach(_a RANGE 0 1)
        foreach(_b RANGE 0 1)
          foreach(_c RANGE 0 1)
            foreach(_d RANGE 0 1)
              set(_generativeqc_range_shell "${_a}${_b}${_c}${_d}")
              list(APPEND GENERATIVEQC_RANGE_DERIVATIVE_CPU_AOT_SOURCES
                   "${GENERATIVEQC_RANGE_DERIVATIVE_CPU_AOT_DIRECTORY}/generativeqc_derivative_range_${_generativeqc_range_family}_${_generativeqc_range_shell}_0.cpp")
              if(_generativeqc_range_shell STREQUAL "1111")
                list(APPEND GENERATIVEQC_RANGE_DERIVATIVE_CPU_AOT_SOURCES
                     "${GENERATIVEQC_RANGE_DERIVATIVE_CPU_AOT_DIRECTORY}/generativeqc_derivative_range_${_generativeqc_range_family}_1111_1.cpp")
              endif()
            endforeach()
          endforeach()
        endforeach()
      endforeach()
    endforeach()
    generativeqc_register_generated_sources(
      NAME generativeqc_range_derivative_cpu_aot_codegen
      TARGET ${target}
      ADD_TO_TARGET
      GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_derivative_range_aot.py"
      OUTPUTS ${GENERATIVEQC_RANGE_DERIVATIVE_CPU_AOT_SOURCES}
      DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/manifests/derivative_aot_radials.json"
        "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/integral/rsh_cpu_aot.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/integral/derivative_aot_registry.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/integral/weighted_eri.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/integral/weighted_eri_native.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/integral/weighted_eri_cuda.py"
      ARGS
        --output-directory "${GENERATIVEQC_RANGE_DERIVATIVE_CPU_AOT_DIRECTORY}"
        --radial-manifest "${CMAKE_CURRENT_SOURCE_DIR}/manifests/derivative_aot_radials.json"
      COMPILE_OPTIONS "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-ffp-contract=off>"
      COMMENT "Generating manifest-owned CPU SR/LR derivative programs")
  endif()

  set(GENERATIVEQC_QUADRATURE_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_quadrature.cuh")
  generativeqc_register_generated_sources(
    NAME generativeqc_quadrature_codegen
    TARGET ${target}
    ADD_TO_TARGET
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_quadrature_cuda.py"
    OUTPUTS "${GENERATIVEQC_QUADRATURE_HEADER}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/xc/quadrature_cuda.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/xc/grid_response_ir.py"
    ARGS --output "${GENERATIVEQC_QUADRATURE_HEADER}"
    COMMENT "Generating bounded CUDA molecular quadrature")
  set(GENERATIVEQC_DF_EXCHANGE_SCHEDULE_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_df_exchange_schedule.hpp")
  generativeqc_register_generated_sources(
    NAME generativeqc_df_exchange_schedule_codegen
    TARGET ${target}
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_df_exchange_schedule.py"
    OUTPUTS "${GENERATIVEQC_DF_EXCHANGE_SCHEDULE_HEADER}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/method/df_exchange_schedule.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/method/df_occupied_gram_cuda.py"
    ARGS --output "${GENERATIVEQC_DF_EXCHANGE_SCHEDULE_HEADER}"
    COMMENT "Generating compiler-owned DF source-reuse schedule")

  # The native host policy is built even when CUDA execution is disabled.
  # Generate its CUDA-independent constants once for both build variants.
  set(GENERATIVEQC_ONE_ELECTRON_DERIVATIVE_POLICY_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_one_electron_derivative_policy.cuh")
  generativeqc_register_generated_sources(
    NAME generativeqc_one_electron_derivative_policy_codegen
    TARGET ${target}
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_one_electron_kernels.py"
    OUTPUTS "${GENERATIVEQC_ONE_ELECTRON_DERIVATIVE_POLICY_HEADER}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_df_kernels.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/integral/cooperative_schedule.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/integral/one_electron_derivative_policy_cuda.py"
    ARGS --derivatives --derivative-policy-output "${GENERATIVEQC_ONE_ELECTRON_DERIVATIVE_POLICY_HEADER}")

  set(GENERATIVEQC_METHOD_PARAMETERS_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_method_parameters.hpp")
  generativeqc_register_generated_sources(
    NAME generativeqc_method_parameters_codegen
    TARGET ${target}
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_method_parameters.py"
    OUTPUTS "${GENERATIVEQC_METHOD_PARAMETERS_HEADER}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/method/method_parameters.json"
    ARGS
      --source "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/method/method_parameters.json"
      --cpp-output "${GENERATIVEQC_METHOD_PARAMETERS_HEADER}"
    COMMENT "Generating audited method parameter constants")

  set(GENERATIVEQC_D4_DERIVATIVE_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_d4_derivative.hpp")
  generativeqc_register_generated_sources(
    NAME generativeqc_d4_derivative_codegen
    TARGET ${target}
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_d4_derivative.py"
    OUTPUTS "${GENERATIVEQC_D4_DERIVATIVE_HEADER}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/method/d4_derivative.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/common/provenance.py"
    ARGS --output "${GENERATIVEQC_D4_DERIVATIVE_HEADER}"
    COMMENT "Generating compiler-owned D4 EEQ derivative lowering")

  set(GENERATIVEQC_D3_DATA_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/d3_data.hpp")
  generativeqc_register_generated_sources(
    NAME generativeqc_d3_codegen
    TARGET ${target}
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generativeqc_d3/generate_native_data.py"
    OUTPUTS "${GENERATIVEQC_D3_DATA_HEADER}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/data/parameters/d3_production.bin"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/common/d3_data.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/manifests/xtbloom-d3.json"
    ARGS --output "${GENERATIVEQC_D3_DATA_HEADER}"
    COMMENT "Generating pinned compact D3(BJ) tables")
  set(GENERATIVEQC_ONE_ELECTRON_ST_CPU_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_one_electron_st_cpu.hpp")
  generativeqc_register_generated_sources(
    NAME generativeqc_one_electron_st_cpu_codegen
    TARGET ${target}
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_one_electron_kernels.py"
    OUTPUTS "${GENERATIVEQC_ONE_ELECTRON_ST_CPU_HEADER}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_df_kernels.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/integral/one_electron_cpu.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/integral/one_electron_cuda.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/integral/one_electron_derivatives_cuda.py"
    ARGS --cpu-st-output "${GENERATIVEQC_ONE_ELECTRON_ST_CPU_HEADER}")

  set(GENERATIVEQC_ERI_CPU_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_eri_cpu.hpp")
  generativeqc_register_generated_sources(
    NAME generativeqc_eri_cpu_codegen
    TARGET ${target}
    ADD_TO_TARGET
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_eri_cpu.py"
    OUTPUTS "${GENERATIVEQC_ERI_CPU_HEADER}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/integral/eri_cpu.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/integral/shell_class.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/integral/scalar_c.py"
    ARGS --output "${GENERATIVEQC_ERI_CPU_HEADER}"
    COMMENT "Generating shared-DAG CPU s/p/d ERI values")

  set(GENERATIVEQC_DF_VALUE_CPU_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_df_values_cpu.hpp")
  generativeqc_register_generated_sources(
    NAME generativeqc_df_values_cpu_codegen
    TARGET ${target}
    ADD_TO_TARGET
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_df_kernels.py"
    OUTPUTS "${GENERATIVEQC_DF_VALUE_CPU_HEADER}"
    ARGS
      --cpu
      --output "${GENERATIVEQC_DF_VALUE_CPU_HEADER}")

  set(GENERATIVEQC_DF_DERIVATIVE_CPU_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_df_derivatives_cpu.hpp")
  generativeqc_register_generated_sources(
    NAME generativeqc_df_derivatives_cpu_codegen
    TARGET ${target}
    ADD_TO_TARGET
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_df_kernels.py"
    OUTPUTS "${GENERATIVEQC_DF_DERIVATIVE_CPU_HEADER}"
    ARGS
      --derivatives
      --cpu
      --output "${GENERATIVEQC_DF_DERIVATIVE_CPU_HEADER}")

  set(GENERATIVEQC_XC_CPU_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/xc_cpu_generated.hpp")
  generativeqc_register_generated_sources(
    NAME generativeqc_xc_cpu_codegen
    TARGET ${target}
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_xc_cpu.py"
    OUTPUTS "${GENERATIVEQC_XC_CPU_HEADER}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/xc/semilocal_codegen.py"
    ARGS --output "${GENERATIVEQC_XC_CPU_HEADER}")

  set(GENERATIVEQC_LIBXC_SEMILOCAL_CPU_DIRECTORY
      "${CMAKE_CURRENT_BINARY_DIR}/generated/libxc_semilocal_cpu")
  set(GENERATIVEQC_LIBXC_SEMILOCAL_CPU_SOURCES
      "${GENERATIVEQC_LIBXC_SEMILOCAL_CPU_DIRECTORY}/generated_libxc_semilocal_registry.hpp"
      "${GENERATIVEQC_LIBXC_SEMILOCAL_CPU_DIRECTORY}/generated_libxc_semilocal_registry.cpp")
  foreach(_generativeqc_libxc_semilocal_shard RANGE 0 7)
    list(APPEND GENERATIVEQC_LIBXC_SEMILOCAL_CPU_SOURCES
         "${GENERATIVEQC_LIBXC_SEMILOCAL_CPU_DIRECTORY}/generated_libxc_semilocal_${_generativeqc_libxc_semilocal_shard}.cpp")
  endforeach()
  generativeqc_register_generated_sources(
    NAME generativeqc_libxc_semilocal_cpu_codegen
    TARGET ${target}
    ADD_TO_TARGET
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_libxc_semilocal_cpu_registry.py"
    OUTPUTS ${GENERATIVEQC_LIBXC_SEMILOCAL_CPU_SOURCES}
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_libxc_semilocal_cpu_registry.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/integral/expr.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/integral/scalar_c.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/xc/automatic_semilocal.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/xc/bulk_runtime.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/xc/libxc_blacklist.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/xc/libxc_bulk.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/xc/libxc_bulk_capabilities.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/xc/libxc_bulk_catalog.json"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/xc/libxc_maple.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/xc/spec.py"
    ARGS --output-directory "${GENERATIVEQC_LIBXC_SEMILOCAL_CPU_DIRECTORY}"
    COMPILE_OPTIONS "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-ffp-contract=off>"
    COMMENT "Generating automatic Libxc CPU semilocal registry")

  set(GENERATIVEQC_SCF_ARRAY_CPU_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_scf_array_native.hpp")
  generativeqc_register_generated_sources(
    NAME generativeqc_scf_array_cpu_codegen
    TARGET ${target}
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_scf_array_native.py"
    OUTPUTS "${GENERATIVEQC_SCF_ARRAY_CPU_HEADER}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/array_api/scf.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/tensor/scf.py"
    ARGS --output "${GENERATIVEQC_SCF_ARRAY_CPU_HEADER}"
    COMMENT "Generating Array frontend SCF CPU tensor helpers")

  if(GENERATIVEQC_ENABLE_CUDA)
    set(GENERATIVEQC_SCF_DENSITY_CUDA_HEADER
        "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_scf_density_cuda.cuh")
    generativeqc_register_generated_sources(
      NAME generativeqc_scf_density_cuda_codegen
      TARGET ${target}
      GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_scf_density_cuda.py"
      OUTPUTS "${GENERATIVEQC_SCF_DENSITY_CUDA_HEADER}"
      DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/tensor/scf.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/tensor/scf_cuda.py"
      ARGS --output "${GENERATIVEQC_SCF_DENSITY_CUDA_HEADER}"
      COMMENT "Generating compiler-owned CUDA SCF density kernel")
  endif()

  set(GENERATIVEQC_NONLOCAL_PAIR_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_nonlocal_pair_native.hpp")
  generativeqc_register_generated_sources(
    NAME generativeqc_nonlocal_pair_codegen
    TARGET ${target}
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_nonlocal_pair_native.py"
    OUTPUTS "${GENERATIVEQC_NONLOCAL_PAIR_HEADER}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/method/nonlocal_pair.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/common/nonlocal_correlation.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/tensor/scalar_cpp.py"
    ARGS --output "${GENERATIVEQC_NONLOCAL_PAIR_HEADER}"
    COMMENT "Generating shared CPU/CUDA nonlocal correlation pair algebra")

  set(GENERATIVEQC_GFN2_PAIR_CPU_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_gfn2_pair_native.hpp")
  generativeqc_register_generated_sources(
    NAME generativeqc_gfn2_pair_cpu_codegen
    TARGET ${target}
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_gfn2_pair_native.py"
    OUTPUTS "${GENERATIVEQC_GFN2_PAIR_CPU_HEADER}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/geometry/gfn2_pair.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/tensor/ad_program.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/tensor/scalar_cpp.py"
    ARGS --output "${GENERATIVEQC_GFN2_PAIR_CPU_HEADER}"
    COMMENT "Generating compiler-owned GFN2 CPU pair kernels")

  set(GENERATIVEQC_GFN2_AES2_CPU_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_gfn2_aes2_native.hpp")
  generativeqc_register_generated_sources(
    NAME generativeqc_gfn2_aes2_cpu_codegen
    TARGET ${target}
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_gfn2_aes2_native.py"
    OUTPUTS "${GENERATIVEQC_GFN2_AES2_CPU_HEADER}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/method/gfn2_aes2.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/method/gfn2_aes2_schedule.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/tensor/ad_program.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/tensor/scalar_cpp.py"
    ARGS --cpu-output "${GENERATIVEQC_GFN2_AES2_CPU_HEADER}"
    COMMENT "Generating compiler-owned GFN2 AES2 CPU kernels")

  set(GENERATIVEQC_GFN2_ES2_NATIVE_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_gfn2_es2_native.hpp")
  generativeqc_register_generated_sources(
    NAME generativeqc_gfn2_es2_native_codegen
    TARGET ${target}
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_gfn2_es2_native.py"
    OUTPUTS "${GENERATIVEQC_GFN2_ES2_NATIVE_HEADER}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/method/gfn2_es2_runtime.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/tensor/ad_program.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/tensor/scalar_cpp.py"
    ARGS --output "${GENERATIVEQC_GFN2_ES2_NATIVE_HEADER}"
    COMMENT "Generating compiler-owned GFN2 ES2 scalar kernels")

  set(GENERATIVEQC_GFN2_EXTERNAL_POINT_CHARGE_FORCE_NATIVE_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_gfn2_external_point_charge_force.hpp")
  generativeqc_register_generated_sources(
    NAME generativeqc_gfn2_external_point_charge_force_native_codegen
    TARGET ${target}
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_gfn2_external_point_charge_force.py"
    OUTPUTS "${GENERATIVEQC_GFN2_EXTERNAL_POINT_CHARGE_FORCE_NATIVE_HEADER}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/method/gfn2_external_point_charge_force.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/tensor/ad_program.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/tensor/scalar_cpp.py"
    ARGS --output "${GENERATIVEQC_GFN2_EXTERNAL_POINT_CHARGE_FORCE_NATIVE_HEADER}"
    COMMENT "Generating compiler-owned GFN2 external point-charge force response")

  set(GENERATIVEQC_GFN2_H0_NATIVE_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_gfn2_h0_native.hpp")
  generativeqc_register_generated_sources(
    NAME generativeqc_gfn2_h0_native_codegen
    TARGET ${target}
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_gfn2_h0_native.py"
    OUTPUTS "${GENERATIVEQC_GFN2_H0_NATIVE_HEADER}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/method/gfn2_h0_force_runtime.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/method/gfn2_h0_force_schedule.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/common/schedule.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/tensor/ad_program.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/tensor/scalar_cpp.py"
    ARGS --output "${GENERATIVEQC_GFN2_H0_NATIVE_HEADER}"
    COMMENT "Generating compiler-owned GFN2 CPU/CUDA H0 values and adjoints")
  if(TARGET generativeqc_gfn2_cuda)
    add_dependencies(generativeqc_gfn2_cuda generativeqc_gfn2_h0_native_codegen)
  endif()

  set(GENERATIVEQC_GFN2_SPIN_NATIVE_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_gfn2_spin_native.hpp")
  generativeqc_register_generated_sources(
    NAME generativeqc_gfn2_spin_native_codegen
    TARGET ${target}
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_gfn2_spin_native.py"
    OUTPUTS "${GENERATIVEQC_GFN2_SPIN_NATIVE_HEADER}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/method/gfn2_spin_runtime.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/tensor/ad_program.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/tensor/scalar_cpp.py"
    ARGS --output "${GENERATIVEQC_GFN2_SPIN_NATIVE_HEADER}"
    COMMENT "Generating compiler-owned GFN2 CPU/CUDA spin energy/potential")
  if(TARGET generativeqc_gfn2_cuda)
    add_dependencies(generativeqc_gfn2_cuda generativeqc_gfn2_spin_native_codegen)
  endif()

  set(GENERATIVEQC_GFN2_ES3_NATIVE_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_gfn2_es3_native.cuh")
  generativeqc_register_generated_sources(
    NAME generativeqc_gfn2_es3_codegen
    TARGET ${target}
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_gfn2_es3_native.py"
    OUTPUTS "${GENERATIVEQC_GFN2_ES3_NATIVE_HEADER}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/method/gfn2_es3_runtime.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/tensor/ad_program.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/tensor/scalar_cpp.py"
    ARGS --output "${GENERATIVEQC_GFN2_ES3_NATIVE_HEADER}"
    COMMENT "Generating compiler-owned GFN2 ES3 shell energy/potential")
  if(TARGET generativeqc_gfn2_cuda)
    add_dependencies(generativeqc_gfn2_cuda generativeqc_gfn2_es3_codegen)
    target_include_directories(
      generativeqc_gfn2_cuda PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
  endif()

  set(GENERATIVEQC_GFN2_SCC_FREE_ENERGY_NATIVE_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_gfn2_scc_free_energy_native.hpp")
  generativeqc_register_generated_sources(
    NAME generativeqc_gfn2_scc_free_energy_codegen
    TARGET ${target}
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_gfn2_scc_free_energy_native.py"
    OUTPUTS "${GENERATIVEQC_GFN2_SCC_FREE_ENERGY_NATIVE_HEADER}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/method/gfn2_scc_free_energy_runtime.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/tensor/scalar_cpp.py"
    ARGS --output "${GENERATIVEQC_GFN2_SCC_FREE_ENERGY_NATIVE_HEADER}"
    COMMENT "Generating compiler-owned GFN2 SCC internal/free-energy composition")
  if(TARGET generativeqc_gfn2_cuda)
    add_dependencies(generativeqc_gfn2_cuda generativeqc_gfn2_scc_free_energy_codegen)
    target_include_directories(
      generativeqc_gfn2_cuda PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
  endif()

  set(GENERATIVEQC_GFN2_ELECTRONIC_CPU_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_gfn2_electronic_native.hpp")
  generativeqc_register_generated_sources(
    NAME generativeqc_gfn2_electronic_cpu_codegen
    TARGET ${target}
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_gfn2_electronic_native.py"
    OUTPUTS "${GENERATIVEQC_GFN2_ELECTRONIC_CPU_HEADER}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/method/gfn2_electronic_runtime.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/tensor/ad_program.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/tensor/scalar_cpp.py"
    ARGS --output "${GENERATIVEQC_GFN2_ELECTRONIC_CPU_HEADER}"
    COMMENT "Generating compiler-owned GFN2 CPU electronic kernels")

  file(GLOB GENERATIVEQC_RCCSD_GENERATOR_INPUTS CONFIGURE_DEPENDS
       "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/cc/*.py"
       "${CMAKE_CURRENT_SOURCE_DIR}/tools/generativeqc_cc/*.py")
  set(GENERATIVEQC_GFN2_SDQ_CPU_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_gfn2_sdq_native.hpp")
  generativeqc_register_generated_sources(
    NAME generativeqc_gfn2_sdq_cpu_codegen
    TARGET ${target}
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_gfn2_sdq_native.py"
    OUTPUTS "${GENERATIVEQC_GFN2_SDQ_CPU_HEADER}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/integral/gfn2_sdq.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/integral/gfn2_sdq_cpu.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/integral/one_electron_values.py"
    ARGS --output "${GENERATIVEQC_GFN2_SDQ_CPU_HEADER}"
    COMMENT "Generating compiler-owned GFN2 S/D/Q CPU primitive kernels")

  set(GENERATIVEQC_RCCSD_CPU_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_rccsd_cpu.hpp")
  generativeqc_register_generated_sources(
    NAME generativeqc_rccsd_cpu_codegen
    TARGET ${target}
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_rccsd_native.py"
    OUTPUTS "${GENERATIVEQC_RCCSD_CPU_HEADER}"
    DEPENDS ${GENERATIVEQC_RCCSD_GENERATOR_INPUTS}
    ARGS --cpu-header "${GENERATIVEQC_RCCSD_CPU_HEADER}")

  set(GENERATIVEQC_RCCSDT_CPU_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_rccsdt_cpu.hpp")
  generativeqc_register_generated_sources(
    NAME generativeqc_rccsdt_cpu_codegen
    TARGET ${target}
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_rccsdt_native.py"
    OUTPUTS "${GENERATIVEQC_RCCSDT_CPU_HEADER}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tools/generativeqc_cc/triples.py"
    ARGS --output "${GENERATIVEQC_RCCSDT_CPU_HEADER}")

  set(GENERATIVEQC_DF_HF_RESPONSE_CONTRACT_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_df_hf_response_contract.hpp")
  generativeqc_register_generated_sources(
    NAME generativeqc_df_hf_response_contract_codegen
    TARGET ${target}
    ADD_TO_TARGET
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_df_hf_response.py"
    OUTPUTS "${GENERATIVEQC_DF_HF_RESPONSE_CONTRACT_HEADER}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/method/df_hf_response_contract.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/method/df_hf_response_cuda.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/method/df_occupied_response_cuda.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/tensor/cuda_dtype.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/tensor/cuda_gemm.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/tensor/ir.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/tensor/types.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/common/layout.py"
    ARGS --contract-output "${GENERATIVEQC_DF_HF_RESPONSE_CONTRACT_HEADER}")

  set(GENERATIVEQC_ECP_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_ecp_ao.cuh")
  generativeqc_register_generated_sources(
    NAME generativeqc_ecp_codegen
    TARGET ${target}
    ADD_TO_TARGET
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_ecp_kernels.py"
    OUTPUTS "${GENERATIVEQC_ECP_HEADER}"
    ARGS --output "${GENERATIVEQC_ECP_HEADER}")
endmacro()

macro(generativeqc_register_cuda_generated_sources target)
  set(GENERATIVEQC_MEAN_FIELD_SETUP_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_mean_field_setup.cuh")
  generativeqc_register_generated_sources(
    TARGET ${target}
    ADD_TO_TARGET
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_mean_field_setup_cuda.py"
    OUTPUTS "${GENERATIVEQC_MEAN_FIELD_SETUP_HEADER}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_scf_array_native.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/method/mean_field_setup_cuda.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/tensor/scf.py"
    ARGS --output "${GENERATIVEQC_MEAN_FIELD_SETUP_HEADER}"
    COMMENT "Generating shared CUDA mean-field setup projectors")

  set(GENERATIVEQC_MATRIX_FUNCTION_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_symmetric_matrix_function.cuh")
  generativeqc_register_generated_sources(
    TARGET ${target}
    ADD_TO_TARGET
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_matrix_function_cuda.py"
    OUTPUTS "${GENERATIVEQC_MATRIX_FUNCTION_HEADER}"
    ARGS --output "${GENERATIVEQC_MATRIX_FUNCTION_HEADER}")

  set(GENERATIVEQC_DF_HF_RESPONSE_CUDA_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_df_hf_response.cuh")
  generativeqc_register_generated_sources(
    TARGET ${target}
    ADD_TO_TARGET
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_df_hf_response.py"
    OUTPUTS "${GENERATIVEQC_DF_HF_RESPONSE_CUDA_HEADER}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/method/df_hf_response_contract.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/method/df_hf_response_cuda.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/method/df_occupied_response_cuda.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/tensor/cuda_dtype.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/tensor/cuda_gemm.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/tensor/ir.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/tensor/types.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/common/layout.py"
    ARGS --cuda-output "${GENERATIVEQC_DF_HF_RESPONSE_CUDA_HEADER}")

  set(GENERATIVEQC_DF_GENERATED_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/df_values.cuh")
  set(GENERATIVEQC_DF_POLICY_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_df_policy.cuh")
  set(GENERATIVEQC_DF_VALUE_CANDIDATE_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_df_value_candidates.cuh")
  generativeqc_register_generated_sources(
    TARGET ${target}
    ADD_TO_TARGET
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_df_kernels.py"
    OUTPUTS
      "${GENERATIVEQC_DF_GENERATED_HEADER}"
      "${GENERATIVEQC_DF_POLICY_HEADER}"
      "${GENERATIVEQC_DF_VALUE_CANDIDATE_HEADER}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/integral/production_df_values.json"
    ARGS
      --output "${GENERATIVEQC_DF_GENERATED_HEADER}"
      --policy-output "${GENERATIVEQC_DF_POLICY_HEADER}")

  # The native views and compiler both admit all 4^3 s/p/d/f classes. Keep
  # stable per-class output paths; profile edits must not reshuffle units.
  set(GENERATIVEQC_DF_SHELL_UNIT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated/df_shells")
  set(GENERATIVEQC_DF_SHELL_SOURCES)
  set(GENERATIVEQC_DF_SHELL_UNIT_OUTPUTS
      "${GENERATIVEQC_DF_SHELL_UNIT_DIRECTORY}/generated_df_shell_dispatch.hpp")
  foreach(_a RANGE 0 3)
    foreach(_b RANGE 0 3)
      foreach(_c RANGE 0 3)
        set(_class "${_a}${_b}${_c}")
        list(APPEND GENERATIVEQC_DF_SHELL_SOURCES
             "${GENERATIVEQC_DF_SHELL_UNIT_DIRECTORY}/df_shell_${_class}.cu")
        foreach(_header IN ITEMS polynomial math)
          list(APPEND GENERATIVEQC_DF_SHELL_UNIT_OUTPUTS
               "${GENERATIVEQC_DF_SHELL_UNIT_DIRECTORY}/df_shell_${_header}_${_class}.cuh")
        endforeach()
        list(APPEND GENERATIVEQC_DF_SHELL_UNIT_OUTPUTS
             "${GENERATIVEQC_DF_SHELL_UNIT_DIRECTORY}/df_shell_policy_${_class}.hpp")
      endforeach()
    endforeach()
  endforeach()
  target_include_directories(${target} PRIVATE "${GENERATIVEQC_DF_SHELL_UNIT_DIRECTORY}")

  set(GENERATIVEQC_DF_DERIVATIVE_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_df_derivatives.cuh")
  set(GENERATIVEQC_DF_DERIVATIVE_POLICY_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_df_derivative_policy.cuh")
  set(GENERATIVEQC_DF_DERIVATIVE_SCHEDULE_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_df_derivative_schedule.cuh")
  set(GENERATIVEQC_DF_SHELL_DERIVATIVE_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_df_shell_derivatives.cuh")
  set(GENERATIVEQC_DF_RYS_HEADERS
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_df_rys.cuh"
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_df_rys_policy.hpp"
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_df_production.hpp"
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_df_screening.cuh"
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_df_pair_screening.cuh"
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_df_rys_shell.cuh")
  generativeqc_register_generated_sources(
    TARGET ${target}
    ADD_TO_TARGET
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_df_kernels.py"
    OUTPUTS
      "${GENERATIVEQC_DF_DERIVATIVE_HEADER}"
      "${GENERATIVEQC_DF_DERIVATIVE_POLICY_HEADER}"
      "${GENERATIVEQC_DF_DERIVATIVE_SCHEDULE_HEADER}"
      "${GENERATIVEQC_DF_SHELL_DERIVATIVE_HEADER}"
      ${GENERATIVEQC_DF_RYS_HEADERS}
      ${GENERATIVEQC_DF_SHELL_SOURCES}
      ${GENERATIVEQC_DF_SHELL_UNIT_OUTPUTS}
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/integral/production_df_derivatives.json"
    ARGS
      --derivatives
      --output "${GENERATIVEQC_DF_DERIVATIVE_HEADER}"
      --policy-output "${GENERATIVEQC_DF_DERIVATIVE_POLICY_HEADER}"
      --schedule-output "${GENERATIVEQC_DF_DERIVATIVE_SCHEDULE_HEADER}"
      --shell-output "${GENERATIVEQC_DF_SHELL_DERIVATIVE_HEADER}"
      --shell-units-directory "${GENERATIVEQC_DF_SHELL_UNIT_DIRECTORY}")

  set(GENERATIVEQC_ONE_ELECTRON_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_one_electron_values.cuh")
  set(GENERATIVEQC_ONE_ELECTRON_POLICY_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_one_electron_policy.cuh")
  generativeqc_register_generated_sources(
    TARGET ${target}
    ADD_TO_TARGET
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_one_electron_kernels.py"
    OUTPUTS
      "${GENERATIVEQC_ONE_ELECTRON_HEADER}"
      "${GENERATIVEQC_ONE_ELECTRON_POLICY_HEADER}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_df_kernels.py"
    ARGS
      --output "${GENERATIVEQC_ONE_ELECTRON_HEADER}"
      --policy-output "${GENERATIVEQC_ONE_ELECTRON_POLICY_HEADER}")

  set(GENERATIVEQC_ONE_ELECTRON_DERIVATIVE_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_one_electron_derivatives.cuh")
  generativeqc_register_generated_sources(
    TARGET ${target}
    ADD_TO_TARGET
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_one_electron_kernels.py"
    OUTPUTS "${GENERATIVEQC_ONE_ELECTRON_DERIVATIVE_HEADER}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_df_kernels.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/integral/cooperative_schedule.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/integral/one_electron_derivative_policy_cuda.py"
    ARGS --derivatives --output "${GENERATIVEQC_ONE_ELECTRON_DERIVATIVE_HEADER}")

  set(GENERATIVEQC_COSX_DERIVATIVE_CONTRACTION_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_cosx_derivative_contractions.cuh")
  generativeqc_register_generated_sources(
    NAME generativeqc_cosx_derivative_contraction_codegen
    TARGET ${target}
    ADD_TO_TARGET
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_cosx_derivative_native.py"
    OUTPUTS "${GENERATIVEQC_COSX_DERIVATIVE_CONTRACTION_HEADER}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/method/cosx_derivative_runtime.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/tensor/scalar_cpp.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/tensor/ir.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/tensor/program.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/tensor/types.py"
    ARGS --output "${GENERATIVEQC_COSX_DERIVATIVE_CONTRACTION_HEADER}"
    COMMENT "Generating compiler-owned COSX derivative contractions")

  set(GENERATIVEQC_GFN2_SDQ_CUDA_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_gfn2_sdq_cuda.cuh")
  generativeqc_register_generated_sources(
    NAME generativeqc_gfn2_sdq_cuda_codegen
    TARGET ${target}
    ADD_TO_TARGET
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_gfn2_sdq_native.py"
    OUTPUTS "${GENERATIVEQC_GFN2_SDQ_CUDA_HEADER}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/integral/gfn2_force_schedule.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/integral/gfn2_sdq.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/integral/gfn2_sdq_cpu.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/integral/one_electron_values.py"
    ARGS --cuda-output "${GENERATIVEQC_GFN2_SDQ_CUDA_HEADER}"
    COMMENT "Generating compiler-owned GFN2 S/D/Q CUDA primitive kernels")

  set(GENERATIVEQC_DIRECT_FOCK_ACCUMULATION_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_direct_fock_accumulation.cuh")
  generativeqc_register_generated_sources(
    TARGET ${target}
    ADD_TO_TARGET
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_shell_kernels.py"
    OUTPUTS "${GENERATIVEQC_DIRECT_FOCK_ACCUMULATION_HEADER}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/integral/lowering/fock_accumulation.py"
    ARGS
      --direct-fock-accumulation-output "${GENERATIVEQC_DIRECT_FOCK_ACCUMULATION_HEADER}"
    COMMENT "Generating compiler-owned Direct-Fock scatter contraction")

  set(GENERATIVEQC_WEIGHTED_ERI_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/weighted_eri.cuh")
  generativeqc_register_generated_sources(
    TARGET ${target}
    ADD_TO_TARGET
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_weighted_eri_kernels.py"
    OUTPUTS "${GENERATIVEQC_WEIGHTED_ERI_HEADER}"
    ARGS --output "${GENERATIVEQC_WEIGHTED_ERI_HEADER}")

  set(GENERATIVEQC_DIRECT_RESIDENT_PSSS_SCHEDULE_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_direct_resident_psss_schedule.cuh")
  generativeqc_register_generated_sources(
    NAME generativeqc_direct_resident_psss_schedule_codegen
    TARGET ${target}
    ADD_TO_TARGET
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_direct_resident_schedule.py"
    OUTPUTS "${GENERATIVEQC_DIRECT_RESIDENT_PSSS_SCHEDULE_HEADER}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/integral/direct_resident_schedule.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/integral/cuda_schedule.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/integral/shell_spec.py"
    ARGS --output "${GENERATIVEQC_DIRECT_RESIDENT_PSSS_SCHEDULE_HEADER}"
    COMMENT "Generating compiler-owned Direct-HF resident-PSSS schedule")

  set(GENERATIVEQC_DIRECT_HIGH_ORDER_PAIR_GRADIENT_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_direct_high_order_pair_gradient.cuh")
  generativeqc_register_generated_sources(
    NAME generativeqc_direct_high_order_pair_gradient_codegen
    TARGET ${target}
    ADD_TO_TARGET
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_direct_pair_gradient.py"
    OUTPUTS "${GENERATIVEQC_DIRECT_HIGH_ORDER_PAIR_GRADIENT_HEADER}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/integral/direct_pair_gradient_cuda.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/integral/coulomb_recurrence_cuda.py"
    ARGS --output "${GENERATIVEQC_DIRECT_HIGH_ORDER_PAIR_GRADIENT_HEADER}"
    COMMENT "Generating compiler-owned Direct-HF high-order pair-gradient helper")

  set(GENERATIVEQC_DIRECT_SOURCE_CONTRACTION_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_direct_source_contraction.cuh")
  generativeqc_register_generated_sources(
    NAME generativeqc_direct_source_contraction_codegen
    TARGET ${target}
    ADD_TO_TARGET
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_direct_source_contraction.py"
    OUTPUTS "${GENERATIVEQC_DIRECT_SOURCE_CONTRACTION_HEADER}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/integral/direct_source_contraction_cuda.py"
    ARGS --output "${GENERATIVEQC_DIRECT_SOURCE_CONTRACTION_HEADER}"
    COMMENT "Generating compiler-owned Direct-HF source-contraction helper")

  set(GENERATIVEQC_DERIVATIVE_SHELL_AOT_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_derivative_cuda_shell_aot.cuh")
  generativeqc_register_generated_sources(
    NAME generativeqc_derivative_shell_aot_codegen
    TARGET ${target}
    ADD_TO_TARGET
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_derivative_cuda_shell_aot.py"
    OUTPUTS "${GENERATIVEQC_DERIVATIVE_SHELL_AOT_HEADER}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_derivative_cuda_shell_aot.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/manifests/derivative_aot_radials.json"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/integral/derivative_aot_registry.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/integral/derivative_cuda_shell_aot.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/integral/production_cost.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/integral/production_profile.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/integral/production_selection.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/integral/production_shell_classes.json"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/integral/shell_spec.py"
    ARGS
      --output "${GENERATIVEQC_DERIVATIVE_SHELL_AOT_HEADER}"
      --radial-manifest "${CMAKE_CURRENT_SOURCE_DIR}/manifests/derivative_aot_radials.json"
      --production-manifest "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/integral/production_shell_classes.json"
      --target-architecture "sm_120"
    COMMENT "Generating method-neutral CUDA exact-shell derivative AOT package registry")

  set(GENERATIVEQC_DIRECT_RECURRENCE_HEADERS
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_direct_eri_order2.cuh"
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_direct_eri_order3.cuh"
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_direct_eri_order4.cuh"
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_direct_shell_class.cuh")
  generativeqc_register_generated_sources(
    NAME generativeqc_direct_recurrence_codegen
    TARGET ${target}
    ADD_TO_TARGET
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_direct_recurrence.py"
    OUTPUTS ${GENERATIVEQC_DIRECT_RECURRENCE_HEADERS}
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/integral/direct_recurrence_cuda.py"
    ARGS --output-directory "${CMAKE_CURRENT_BINARY_DIR}/generated"
    COMMENT "Generating compiler-owned Direct-HF primitive recurrence support")

  set(GENERATIVEQC_DIRECT_PAIR_SUPPORT_HEADERS
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_direct_pair_order2.cuh"
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_direct_pair_order3.cuh"
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_direct_shell_pair_hermite.cuh")
  generativeqc_register_generated_sources(
    NAME generativeqc_direct_pair_support_codegen
    TARGET ${target}
    ADD_TO_TARGET
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_direct_pair_support.py"
    OUTPUTS ${GENERATIVEQC_DIRECT_PAIR_SUPPORT_HEADERS}
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/integral/direct_pair_support_cuda.py"
    ARGS --output-directory "${CMAKE_CURRENT_BINARY_DIR}/generated"
    COMMENT "Generating compiler-owned Direct-HF pair/Hermite support")

  set(GENERATIVEQC_DIRECT_ORDER2_SHELL_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_direct_order2_shell.cuh")
  generativeqc_register_generated_sources(
    NAME generativeqc_direct_order2_shell_codegen
    TARGET ${target}
    ADD_TO_TARGET
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_direct_order2_shell.py"
    OUTPUTS "${GENERATIVEQC_DIRECT_ORDER2_SHELL_HEADER}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/integral/direct_order2_shell_cuda.py"
    ARGS --output "${GENERATIVEQC_DIRECT_ORDER2_SHELL_HEADER}"
    COMMENT "Generating compiler-owned Direct-HF order-two shell contraction")

  set(GENERATIVEQC_DIRECT_CARTESIAN_CONTRACTION_HEADERS
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_direct_cartesian.cuh"
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_direct_contraction.cuh"
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_direct_eri_materialization.cuh")
  generativeqc_register_generated_sources(
    NAME generativeqc_direct_cartesian_contraction_codegen
    TARGET ${target}
    ADD_TO_TARGET
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_direct_cartesian_contraction.py"
    OUTPUTS ${GENERATIVEQC_DIRECT_CARTESIAN_CONTRACTION_HEADERS}
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/integral/direct_cartesian_contraction_cuda.py"
    ARGS --output-directory "${CMAKE_CURRENT_BINARY_DIR}/generated"
    COMMENT "Generating compiler-owned Direct-HF Cartesian/contraction support")

  set(GENERATIVEQC_DIRECT_PAIR_CACHE_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_direct_pair_cache.cuh")
  generativeqc_register_generated_sources(
    NAME generativeqc_direct_pair_cache_codegen
    TARGET ${target}
    ADD_TO_TARGET
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_direct_pair_cache.py"
    OUTPUTS "${GENERATIVEQC_DIRECT_PAIR_CACHE_HEADER}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/integral/direct_pair_cache_cuda.py"
    ARGS --output "${GENERATIVEQC_DIRECT_PAIR_CACHE_HEADER}"
    COMMENT "Generating compiler-owned Direct-HF primitive-pair cache geometry")

  set(GENERATIVEQC_B3LYP_CUDA_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_b3lyp_device.cuh")
  generativeqc_register_generated_sources(
    TARGET ${target}
    ADD_TO_TARGET
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_xc_b3lyp_cuda.py"
    OUTPUTS "${GENERATIVEQC_B3LYP_CUDA_HEADER}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/xc/semilocal_codegen.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/xc/b3lyp_production_policy.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/method/spec.py"
    ARGS --output "${GENERATIVEQC_B3LYP_CUDA_HEADER}")

  set(GENERATIVEQC_R2SCAN_CUDA_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_r2scan_device.cuh")
  generativeqc_register_generated_sources(
    TARGET ${target}
    ADD_TO_TARGET
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_xc_r2scan_cuda.py"
    OUTPUTS "${GENERATIVEQC_R2SCAN_CUDA_HEADER}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/xc/semilocal_codegen.py"
    ARGS --output "${GENERATIVEQC_R2SCAN_CUDA_HEADER}")

  set(GENERATIVEQC_WB97MV_CUDA_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_wb97mv_device.cuh")
  generativeqc_register_generated_sources(
    TARGET ${target}
    ADD_TO_TARGET
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_xc_wb97mv_cuda.py"
    OUTPUTS "${GENERATIVEQC_WB97MV_CUDA_HEADER}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/xc/semilocal_codegen.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/xc/wb97mv_maple.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/method/spec.py"
    ARGS --output "${GENERATIVEQC_WB97MV_CUDA_HEADER}")

  set(GENERATIVEQC_SPLIT_HYBRID_CUDA_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_split_hybrid_registry.cuh")
  generativeqc_register_generated_sources(
    TARGET ${target}
    ADD_TO_TARGET
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_xc_split_hybrid_registry.py"
    OUTPUTS "${GENERATIVEQC_SPLIT_HYBRID_CUDA_HEADER}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/manifests/cuda_split_hybrids.json"
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_xc_split_hybrid_cuda.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/libxc_split_hybrid.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/libxc_bulk_metadata.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/libxc_method_metadata.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/xc/libxc_bulk.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/xc/libxc_bulk_catalog.json"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/xc/libxc_maple.py"
    ARGS --output "${GENERATIVEQC_SPLIT_HYBRID_CUDA_HEADER}")

  set(GENERATIVEQC_GRID_SOURCE
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_grid_policy.cu")
  # The r2SCAN minority-spin derivative is sensitive to contraction of 1-zeta
  # near the work-density floor. Match the compiler XC FP64 policy and the
  # independent Libxc boundary qualification; do not introduce FMA. CuMetal's
  # nvcc-compatible driver delegates to Clang and requires its native spelling.
  if(GENERATIVEQC_CUDA_PROVIDER STREQUAL "cumetal")
    set(_generativeqc_grid_fp_contract_option -ffp-contract=off)
  else()
    set(_generativeqc_grid_fp_contract_option --fmad=false)
  endif()
  generativeqc_register_generated_sources(
    TARGET ${target}
    ADD_TO_TARGET
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_grid_kernels.py"
    OUTPUTS "${GENERATIVEQC_GRID_SOURCE}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/src/dft/cuda_xc_kernels.cuh"
      "${GENERATIVEQC_B3LYP_CUDA_HEADER}"
      "${GENERATIVEQC_R2SCAN_CUDA_HEADER}"
      "${GENERATIVEQC_WB97MV_CUDA_HEADER}"
      "${GENERATIVEQC_SPLIT_HYBRID_CUDA_HEADER}"
    COMPILE_OPTIONS "${_generativeqc_grid_fp_contract_option}"
    ARGS --output "${GENERATIVEQC_GRID_SOURCE}")

  set(GENERATIVEQC_XC_GRADIENT_SOURCE
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_xc_gradient.cu")
  generativeqc_register_generated_sources(
    TARGET ${target}
    ADD_TO_TARGET
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_xc_gradient_cuda.py"
    OUTPUTS "${GENERATIVEQC_XC_GRADIENT_SOURCE}"
    ARGS --output "${GENERATIVEQC_XC_GRADIENT_SOURCE}")

  set(GENERATIVEQC_RCCSD_CUDA_SOURCE
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_rccsd_cuda.cu")
  generativeqc_register_generated_sources(
    NAME generativeqc_rccsd_cuda_codegen
    TARGET ${target}
    ADD_TO_TARGET
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_rccsd_native.py"
    OUTPUTS "${GENERATIVEQC_RCCSD_CUDA_SOURCE}"
    DEPENDS ${GENERATIVEQC_RCCSD_GENERATOR_INPUTS}
    ARGS --cuda-source "${GENERATIVEQC_RCCSD_CUDA_SOURCE}")

  set(GENERATIVEQC_RCCSDT_CUDA_SOURCE
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_rccsdt_cuda.cu")
  generativeqc_register_generated_sources(
    NAME generativeqc_rccsdt_cuda_codegen
    TARGET ${target}
    ADD_TO_TARGET
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_rccsdt_cuda.py"
    OUTPUTS "${GENERATIVEQC_RCCSDT_CUDA_SOURCE}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_rccsdt_cuda.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_rccsdt_native.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/generativeqc_cc/triples.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/cc/triples_cuda.hpp"
    ARGS --output "${GENERATIVEQC_RCCSDT_CUDA_SOURCE}")

  set(GENERATIVEQC_MP2_GENERATED_DIRECTORY
      "${CMAKE_CURRENT_BINARY_DIR}/generated/mp2")
  set(GENERATIVEQC_MP2_GENERATED_SOURCES
      "${GENERATIVEQC_MP2_GENERATED_DIRECTORY}/mp2_cuda_table.cu"
      "${GENERATIVEQC_MP2_GENERATED_DIRECTORY}/mp2_pair_energy.cuh")
  set(GENERATIVEQC_MP2_ARCHITECTURES "")
  foreach(_arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
    string(REGEX REPLACE "-.*$" "" _numeric_arch "${_arch}")
    list(APPEND GENERATIVEQC_MP2_ARCHITECTURES "${_numeric_arch}")
  endforeach()
  list(REMOVE_DUPLICATES GENERATIVEQC_MP2_ARCHITECTURES)
  foreach(_arch IN LISTS GENERATIVEQC_MP2_ARCHITECTURES)
    foreach(_tile 1 2 4 8)
      list(APPEND GENERATIVEQC_MP2_GENERATED_SOURCES
           "${GENERATIVEQC_MP2_GENERATED_DIRECTORY}/mp2_sm${_arch}_t${_tile}_runtime.cu")
    endforeach()
  endforeach()
  file(GLOB GENERATIVEQC_MP2_GENERATOR_INPUTS CONFIGURE_DEPENDS
       "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/tensor/*.py"
       "python/generativeqc_compiler/common/cuda_target.py"
       "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/common/source_reuse.py"
       "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/method/mp2_schedule.py"
       "${CMAKE_CURRENT_SOURCE_DIR}/python/generativeqc_compiler/mp2/*.py"
       "${CMAKE_CURRENT_SOURCE_DIR}/tools/generativeqc_mp2/*.py"
       "${CMAKE_CURRENT_SOURCE_DIR}/tools/generativeqc_posthf/*.py")
  generativeqc_register_generated_sources(
    TARGET ${target}
    ADD_TO_TARGET
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_mp2_native.py"
    OUTPUTS ${GENERATIVEQC_MP2_GENERATED_SOURCES}
    DEPENDS ${GENERATIVEQC_MP2_GENERATOR_INPUTS}
    ARGS
      --cuda-dir "${GENERATIVEQC_MP2_GENERATED_DIRECTORY}"
      --architectures "${GENERATIVEQC_MP2_ARCHITECTURES}")
endmacro()

macro(generativeqc_add_codegen_pilot)
  if(Python3_Interpreter_FOUND)
    set(GENERATIVEQC_CODEGEN_PILOT_DIRECTORY
        "${CMAKE_CURRENT_BINARY_DIR}/generated/shell_kernels")
    set(GENERATIVEQC_CODEGEN_PILOT_OUTPUTS)
    foreach(axis x y z)
      set(output
          "${GENERATIVEQC_CODEGEN_PILOT_DIRECTORY}/eri_psss_${axis}_gradient.cuh")
      generativeqc_register_generated_sources(
        GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_shell_kernels.py"
        OUTPUTS "${output}"
        ARGS --shell-class psss --axis "${axis}" --output "${output}"
        COMMENT "Generating symbolic/CSE psss ${axis}-gradient CUDA pilot")
      list(APPEND GENERATIVEQC_CODEGEN_PILOT_OUTPUTS "${output}")
    endforeach()

    set(output
        "${GENERATIVEQC_CODEGEN_PILOT_DIRECTORY}/eri_dppp_xy_xyz_factored_gradient.cuh")
    generativeqc_register_generated_sources(
      GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_shell_kernels.py"
      OUTPUTS "${output}"
      ARGS
        --shell-class dppp --d-component xy --p-components xyz
        --lowering factored --output "${output}"
      COMMENT "Generating factored symbolic/CSE dppp xy/xyz CUDA candidate")
    list(APPEND GENERATIVEQC_CODEGEN_PILOT_OUTPUTS "${output}")
    add_custom_target(generativeqc_codegen_pilot DEPENDS ${GENERATIVEQC_CODEGEN_PILOT_OUTPUTS})
  endif()
endmacro()
