include_guard(GLOBAL)

# Project-level generated-source declarations. The generic command mechanics
# live in VibeQCGenerated.cmake; this file owns generator inputs/outputs and the
# target(s) that consume each generated family.
macro(vibeqc_register_host_generated_sources target)
  set(VIBEQC_XC_CPU_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/xc_cpu_generated.hpp")
  vibeqc_register_generated_sources(
    NAME vibeqc_xc_cpu_codegen
    TARGET ${target}
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_xc_cpu.py"
    OUTPUTS "${VIBEQC_XC_CPU_HEADER}"
    DEPENDS ${VIBEQC_SCIENTIFIC_COMPILER_INPUTS}
    ARGS --output "${VIBEQC_XC_CPU_HEADER}")

  set(VIBEQC_ECP_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_ecp_ao.cuh")
  vibeqc_register_generated_sources(
    NAME vibeqc_ecp_codegen
    TARGET ${target}
    ADD_TO_TARGET
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_ecp_kernels.py"
    OUTPUTS "${VIBEQC_ECP_HEADER}"
    DEPENDS ${VIBEQC_SCIENTIFIC_COMPILER_INPUTS}
    ARGS --output "${VIBEQC_ECP_HEADER}")
endmacro()

macro(vibeqc_register_cuda_generated_sources target)
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
      ${VIBEQC_SCIENTIFIC_COMPILER_INPUTS}
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
      ${VIBEQC_SCIENTIFIC_COMPILER_INPUTS}
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
      ${VIBEQC_SCIENTIFIC_COMPILER_INPUTS}
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
      ${VIBEQC_SCIENTIFIC_COMPILER_INPUTS}
    ARGS --derivatives --output "${VIBEQC_ONE_ELECTRON_DERIVATIVE_HEADER}")

  set(VIBEQC_WEIGHTED_ERI_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/generated/weighted_eri.cuh")
  vibeqc_register_generated_sources(
    TARGET ${target}
    ADD_TO_TARGET
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_weighted_eri_kernels.py"
    OUTPUTS "${VIBEQC_WEIGHTED_ERI_HEADER}"
    DEPENDS ${VIBEQC_SCIENTIFIC_COMPILER_INPUTS}
    ARGS --output "${VIBEQC_WEIGHTED_ERI_HEADER}")

  set(VIBEQC_GRID_SOURCE
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_grid_policy.cu")
  vibeqc_register_generated_sources(
    TARGET ${target}
    ADD_TO_TARGET
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_grid_kernels.py"
    OUTPUTS "${VIBEQC_GRID_SOURCE}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/src/dft/cuda_xc_kernels.cuh"
      ${VIBEQC_SCIENTIFIC_COMPILER_INPUTS}
    ARGS --output "${VIBEQC_GRID_SOURCE}")

  set(VIBEQC_XC_GRADIENT_SOURCE
      "${CMAKE_CURRENT_BINARY_DIR}/generated/generated_xc_gradient.cu")
  vibeqc_register_generated_sources(
    TARGET ${target}
    ADD_TO_TARGET
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_xc_gradient_cuda.py"
    OUTPUTS "${VIBEQC_XC_GRADIENT_SOURCE}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/src/dft/grid_response_adjoint.hpp"
      ${VIBEQC_SCIENTIFIC_COMPILER_INPUTS}
    ARGS --output "${VIBEQC_XC_GRADIENT_SOURCE}")

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
       "${CMAKE_CURRENT_SOURCE_DIR}/tools/vibeqc_tensor/*.py"
       "${CMAKE_CURRENT_SOURCE_DIR}/tools/vibeqc_mp2/*.py"
       "${CMAKE_CURRENT_SOURCE_DIR}/tools/vibeqc_posthf/*.py")
  vibeqc_register_generated_sources(
    TARGET ${target}
    ADD_TO_TARGET
    GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_mp2_native.py"
    OUTPUTS ${VIBEQC_MP2_GENERATED_SOURCES}
    DEPENDS ${VIBEQC_MP2_GENERATOR_INPUTS} ${VIBEQC_SCIENTIFIC_COMPILER_INPUTS}
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
        DEPENDS ${VIBEQC_SCIENTIFIC_COMPILER_INPUTS}
        ARGS --shell-class psss --axis "${axis}" --output "${output}"
        COMMENT "Generating symbolic/CSE psss ${axis}-gradient CUDA pilot")
      list(APPEND VIBEQC_CODEGEN_PILOT_OUTPUTS "${output}")
    endforeach()

    set(output
        "${VIBEQC_CODEGEN_PILOT_DIRECTORY}/eri_dppp_xy_xyz_factored_gradient.cuh")
    vibeqc_register_generated_sources(
      GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_shell_kernels.py"
      OUTPUTS "${output}"
      DEPENDS ${VIBEQC_SCIENTIFIC_COMPILER_INPUTS}
      ARGS
        --shell-class dppp --d-component xy --p-components xyz
        --lowering factored --output "${output}"
      COMMENT "Generating factored symbolic/CSE dppp xy/xyz CUDA candidate")
    list(APPEND VIBEQC_CODEGEN_PILOT_OUTPUTS "${output}")
    add_custom_target(vibeqc_codegen_pilot DEPENDS ${VIBEQC_CODEGEN_PILOT_OUTPUTS})
  endif()
endmacro()
