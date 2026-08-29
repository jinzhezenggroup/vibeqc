cmake_minimum_required(VERSION 3.24)

project(VibeQC VERSION 0.1.0 LANGUAGES CXX)

find_package(Python3 COMPONENTS Interpreter QUIET)

option(VIBEQC_ENABLE_CUDA "Build the CUDA runtime backend" ON)
option(VIBEQC_ENABLE_AOT_SHELLS
       "Build generated shell kernels in addition to the generic CUDA path" ON)
option(VIBEQC_BUILD_TESTS "Build native tests" ON)
set(VIBEQC_COMPILER_CACHE "auto" CACHE STRING
    "Compiler cache launcher: auto, off, or an executable name/path")
set(VIBEQC_CUDA_ARCHITECTURES "" CACHE STRING
    "CUDA architectures shipped by VibeQC; defaults to CMAKE_CUDA_ARCHITECTURES")
set(VIBEQC_CUDA_SPLIT_COMPILE_THREADS "1" CACHE STRING
    "NVCC split-compile threads for cuda_rhf.cu; 1 preserves production code generation")
set(VIBEQC_AOT_PROFILE "auto" CACHE STRING
    "AOT profile for every target: auto, portable, sm_XX, or a named profile")
set(VIBEQC_AOT_PROFILES "" CACHE STRING
    "Optional explicit list of AOT profiles, independent of architecture order")
set(VIBEQC_AOT_SHARDS "8" CACHE STRING
    "Number of independently compiled CUDA shards per AOT architecture")

# Repeated CUDA experiments frequently return to byte-identical source after an
# A/B test or branch switch.  Reuse those expensive objects automatically when
# a supported cache is installed, while preserving explicit user launchers.
unset(_vibeqc_compiler_cache)
if(VIBEQC_COMPILER_CACHE STREQUAL "auto")
  find_program(_vibeqc_compiler_cache NAMES sccache ccache NO_CACHE)
elseif(NOT VIBEQC_COMPILER_CACHE STREQUAL "off"
       AND NOT VIBEQC_COMPILER_CACHE STREQUAL "none"
       AND NOT VIBEQC_COMPILER_CACHE STREQUAL "")
  if(IS_ABSOLUTE "${VIBEQC_COMPILER_CACHE}")
    if(NOT EXISTS "${VIBEQC_COMPILER_CACHE}")
      message(FATAL_ERROR
              "VIBEQC_COMPILER_CACHE does not exist: ${VIBEQC_COMPILER_CACHE}")
    endif()
    set(_vibeqc_compiler_cache "${VIBEQC_COMPILER_CACHE}")
  else()
    find_program(_vibeqc_compiler_cache
                 NAMES "${VIBEQC_COMPILER_CACHE}" NO_CACHE)
    if(NOT _vibeqc_compiler_cache)
      message(FATAL_ERROR
              "Compiler cache executable not found: ${VIBEQC_COMPILER_CACHE}")
    endif()
  endif()
endif()
if(_vibeqc_compiler_cache)
  if(NOT CMAKE_CXX_COMPILER_LAUNCHER)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${_vibeqc_compiler_cache}")
  endif()
  if(NOT CMAKE_CUDA_COMPILER_LAUNCHER)
    set(CMAKE_CUDA_COMPILER_LAUNCHER "${_vibeqc_compiler_cache}")
  endif()
  message(STATUS "VibeQC compiler cache: ${_vibeqc_compiler_cache}")
endif()

if(NOT VIBEQC_CUDA_SPLIT_COMPILE_THREADS MATCHES "^[0-9]+$")
  message(FATAL_ERROR
          "VIBEQC_CUDA_SPLIT_COMPILE_THREADS must be a non-negative integer")
endif()
if(NOT VIBEQC_AOT_SHARDS MATCHES "^[1-9][0-9]*$")
  message(FATAL_ERROR "VIBEQC_AOT_SHARDS must be a positive integer")
endif()
math(EXPR VIBEQC_AOT_LAST_SHARD "${VIBEQC_AOT_SHARDS} - 1")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(VIBEQC_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    if(VIBEQC_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES ${VIBEQC_CUDA_ARCHITECTURES})
    endif()
    if(NOT CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES 120)
    endif()
    # XsyevBatched is the capture-safe production eigensolver for realistic
    # AO buckets and is available in the supported CUDA 12.9 toolkit.
    find_package(CUDAToolkit 12.9 REQUIRED)
  else()
    message(WARNING "CUDA compiler not found; building the CPU reference backend only")
    set(VIBEQC_ENABLE_CUDA OFF)
  endif()
endif()

add_library(vibeqc SHARED
  src/api/c_api_batch.cpp
  src/api/c_api_calculation.cpp
  src/api/c_api_common.cpp
  src/api/c_api_context.cpp
  src/api/error.cpp
  src/integrals/s_integrals.cpp
  src/methods/hf_method.cpp
  src/methods/registry.cpp
  src/molecule/basis.cpp
  src/runtime/context.cpp
  src/scf/density_fitting.cpp
  src/scf/fleet.cpp
  src/scf/rhf.cpp
)

target_include_directories(vibeqc
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_definitions(vibeqc PRIVATE VIBEQC_BUILDING_LIBRARY=1)
target_compile_options(vibeqc PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>)

if(VIBEQC_ENABLE_CUDA)
  target_sources(vibeqc PRIVATE
    src/runtime/cuda_runtime.cu
    src/scf/cuda_density_fitting.cu
    src/scf/cuda_eigensolver_probe.cu
    src/scf/cuda_rhf.cu
  )
  if(NOT VIBEQC_CUDA_SPLIT_COMPILE_THREADS STREQUAL "1")
    # Split compilation is useful for syntax/resource experiments on the
    # monolithic fallback TU, but NVCC may make different optimization choices.
    # Keep it opt-in so production benchmark binaries remain comparable.
    set_property(SOURCE src/scf/cuda_rhf.cu APPEND PROPERTY COMPILE_OPTIONS
                 "--split-compile=${VIBEQC_CUDA_SPLIT_COMPILE_THREADS}")
  endif()
  if(VIBEQC_ENABLE_AOT_SHELLS)
    if(NOT Python3_Interpreter_FOUND)
      find_package(Python3 COMPONENTS Interpreter REQUIRED)
    endif()
    set(VIBEQC_AOT_SHELL_MANIFEST
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/vibeqc_codegen/production_shell_classes.json"
        CACHE FILEPATH "Accepted generated shell classes for production CUDA")
    set(VIBEQC_AOT_GENERATED_DIRECTORY
        "${CMAKE_CURRENT_BINARY_DIR}/generated/production_shell_kernels")

    # Normalize and sort the compile targets so generated behavior and source
    # assignment do not depend on CMAKE_CUDA_ARCHITECTURES list order.
    set(VIBEQC_AOT_TARGET_ARCHITECTURES)
    foreach(architecture IN LISTS CMAKE_CUDA_ARCHITECTURES)
      string(REGEX REPLACE "-(real|virtual)$" "" architecture "${architecture}")
      if(NOT architecture MATCHES "^[0-9]+$")
        message(FATAL_ERROR
                "VibeQC AOT requires numeric CUDA architectures, got ${architecture}")
      endif()
      list(APPEND VIBEQC_AOT_TARGET_ARCHITECTURES "${architecture}")
    endforeach()
    list(REMOVE_DUPLICATES VIBEQC_AOT_TARGET_ARCHITECTURES)
    list(SORT VIBEQC_AOT_TARGET_ARCHITECTURES COMPARE NATURAL)

    set(VIBEQC_AOT_GENERATOR_ARGUMENTS)
    set(VIBEQC_AOT_GENERATED_PROFILE_SOURCES)
    foreach(architecture IN LISTS VIBEQC_AOT_TARGET_ARCHITECTURES)
      set(profile_architecture "sm_${architecture}")
      list(APPEND VIBEQC_AOT_GENERATOR_ARGUMENTS
           --target-architecture "${profile_architecture}")
      foreach(shard RANGE 0 ${VIBEQC_AOT_LAST_SHARD})
        list(APPEND VIBEQC_AOT_GENERATED_PROFILE_SOURCES
             "${VIBEQC_AOT_GENERATED_DIRECTORY}/${profile_architecture}/vibeqc_generated_shell_sm${architecture}_shard_${shard}.cu")
      endforeach()
    endforeach()
    if(NOT VIBEQC_AOT_PROFILE STREQUAL "auto")
      list(APPEND VIBEQC_AOT_GENERATOR_ARGUMENTS
           --profile "${VIBEQC_AOT_PROFILE}")
    endif()
    foreach(profile IN LISTS VIBEQC_AOT_PROFILES)
      if(profile MATCHES "^sm_([0-9]+)$")
        set(profile_architecture "${CMAKE_MATCH_1}")
        if(NOT profile_architecture IN_LIST VIBEQC_AOT_TARGET_ARCHITECTURES)
          message(FATAL_ERROR
                  "AOT profile ${profile} has no matching CUDA compile target")
        endif()
        list(APPEND VIBEQC_AOT_GENERATOR_ARGUMENTS
             --profile-map "${profile}=${profile}")
      elseif(NOT profile STREQUAL "")
        list(LENGTH VIBEQC_AOT_TARGET_ARCHITECTURES target_count)
        if(NOT target_count EQUAL 1)
          message(FATAL_ERROR
                  "Named/portable VIBEQC_AOT_PROFILES require one CUDA target; use VIBEQC_AOT_PROFILE to apply one profile to all targets")
        endif()
        list(GET VIBEQC_AOT_TARGET_ARCHITECTURES 0 profile_architecture)
        list(APPEND VIBEQC_AOT_GENERATOR_ARGUMENTS
             --profile-map "sm_${profile_architecture}=${profile}")
      endif()
    endforeach()

    set(VIBEQC_AOT_GENERATED_REGISTRY_SOURCE
        "${VIBEQC_AOT_GENERATED_DIRECTORY}/vibeqc_generated_shell_registry.cu")
    set(VIBEQC_AOT_GENERATED_HEADER
        "${VIBEQC_AOT_GENERATED_DIRECTORY}/vibeqc_generated_shell_registry.hpp")
    add_custom_command(
      OUTPUT ${VIBEQC_AOT_GENERATED_PROFILE_SOURCES}
             "${VIBEQC_AOT_GENERATED_REGISTRY_SOURCE}"
             "${VIBEQC_AOT_GENERATED_HEADER}"
      COMMAND "${Python3_EXECUTABLE}"
              "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_shell_kernels.py"
              --production-manifest "${VIBEQC_AOT_SHELL_MANIFEST}"
              --output-directory "${VIBEQC_AOT_GENERATED_DIRECTORY}"
              --shards "${VIBEQC_AOT_SHARDS}"
              ${VIBEQC_AOT_GENERATOR_ARGUMENTS}
      DEPENDS
        "${VIBEQC_AOT_SHELL_MANIFEST}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_shell_kernels.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/vibeqc_codegen/__init__.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/vibeqc_codegen/backend.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/vibeqc_codegen/cuda.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/vibeqc_codegen/cuda_adapter.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/vibeqc_codegen/cuda_emitter.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/vibeqc_codegen/cuda_schedule.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/vibeqc_codegen/cuda_target.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/vibeqc_codegen/dppp_specialization.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/vibeqc_codegen/dppp_dispatch.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/vibeqc_codegen/expr.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/vibeqc_codegen/fused_schedule.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/vibeqc_codegen/ir.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/vibeqc_codegen/production.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/vibeqc_codegen/rys.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/vibeqc_codegen/rys2_data.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/vibeqc_codegen/rys3_data.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/vibeqc_codegen/rys4_data.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/vibeqc_codegen/shell_class.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/vibeqc_codegen/shell_spec.py"
      COMMENT "Generating target-specific fused shell-class CUDA bundles"
      VERBATIM)
    set_source_files_properties(
        ${VIBEQC_AOT_GENERATED_PROFILE_SOURCES}
        "${VIBEQC_AOT_GENERATED_REGISTRY_SOURCE}"
        PROPERTIES GENERATED TRUE)

    # Each object target compiles only the schedule selected for that SM. The
    # handwritten CUDA sources and host registry still use the full fatbin set.
    foreach(architecture IN LISTS VIBEQC_AOT_TARGET_ARCHITECTURES)
      set(profile_architecture "sm_${architecture}")
      set(profile_sources)
      foreach(shard RANGE 0 ${VIBEQC_AOT_LAST_SHARD})
        list(APPEND profile_sources
             "${VIBEQC_AOT_GENERATED_DIRECTORY}/${profile_architecture}/vibeqc_generated_shell_sm${architecture}_shard_${shard}.cu")
      endforeach()
      add_library(vibeqc_aot_${profile_architecture} OBJECT ${profile_sources})
      target_include_directories(vibeqc_aot_${profile_architecture} PRIVATE
          "${CMAKE_CURRENT_SOURCE_DIR}/src"
          "${VIBEQC_AOT_GENERATED_DIRECTORY}")
      target_compile_definitions(vibeqc_aot_${profile_architecture}
                                 PRIVATE VIBEQC_HAS_CUDA=1)
      set_target_properties(vibeqc_aot_${profile_architecture} PROPERTIES
          CUDA_ARCHITECTURES "${architecture}"
          CUDA_STANDARD 20
          CUDA_STANDARD_REQUIRED ON
          POSITION_INDEPENDENT_CODE ON)
      target_sources(vibeqc PRIVATE
          $<TARGET_OBJECTS:vibeqc_aot_${profile_architecture}>)
    endforeach()
    target_sources(vibeqc PRIVATE "${VIBEQC_AOT_GENERATED_REGISTRY_SOURCE}")
    target_include_directories(vibeqc PRIVATE
        "${VIBEQC_AOT_GENERATED_DIRECTORY}")
  else()
    target_sources(vibeqc PRIVATE src/scf/aot_shell_registry_stub.cpp)
  endif()
  target_link_libraries(vibeqc PRIVATE CUDA::cudart CUDA::cublas CUDA::cusolver)
  target_compile_definitions(vibeqc PUBLIC VIBEQC_HAS_CUDA=1)
  set_target_properties(vibeqc PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
else()
  target_sources(vibeqc PRIVATE src/scf/cuda_density_fitting_stub.cpp)
  target_compile_definitions(vibeqc PUBLIC VIBEQC_HAS_CUDA=0)
endif()

set_target_properties(vibeqc PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION 0
  OUTPUT_NAME vibeqc
)

if(VIBEQC_BUILD_TESTS)
  enable_testing()
  add_executable(vibeqc_native_tests tests/native/test_rhf.cpp)
  target_link_libraries(vibeqc_native_tests PRIVATE vibeqc)
  target_include_directories(vibeqc_native_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  add_test(NAME vibeqc_native_tests COMMAND vibeqc_native_tests)

  add_executable(vibeqc_batch_tests tests/native/test_batch.cpp)
  target_link_libraries(vibeqc_batch_tests PRIVATE vibeqc)
  add_test(NAME vibeqc_batch_tests COMMAND vibeqc_batch_tests)

  add_executable(vibeqc_cpp_api_tests tests/native/test_cpp_batch.cpp)
  target_link_libraries(vibeqc_cpp_api_tests PRIVATE vibeqc)
  add_test(NAME vibeqc_cpp_api_tests COMMAND vibeqc_cpp_api_tests)

  add_executable(vibeqc_cartesian_integral_tests
    tests/native/test_cartesian_integrals.cpp)
  target_link_libraries(vibeqc_cartesian_integral_tests PRIVATE vibeqc)
  target_include_directories(vibeqc_cartesian_integral_tests
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  add_test(NAME vibeqc_cartesian_integral_tests
    COMMAND vibeqc_cartesian_integral_tests)

  add_executable(vibeqc_density_fitting_tests
    tests/native/test_density_fitting.cpp)
  target_link_libraries(vibeqc_density_fitting_tests PRIVATE vibeqc)
  target_include_directories(vibeqc_density_fitting_tests
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  add_test(NAME vibeqc_density_fitting_tests
    COMMAND vibeqc_density_fitting_tests)

  add_executable(vibeqc_cuda_eigensolver_policy_tests
    tests/native/test_cuda_eigensolver_policy.cpp)
  target_link_libraries(vibeqc_cuda_eigensolver_policy_tests PRIVATE vibeqc)
  target_include_directories(vibeqc_cuda_eigensolver_policy_tests
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  add_test(NAME vibeqc_cuda_eigensolver_policy_tests
    COMMAND vibeqc_cuda_eigensolver_policy_tests)

  add_executable(vibeqc_uhf_tests tests/native/test_uhf.cpp)
  target_link_libraries(vibeqc_uhf_tests PRIVATE vibeqc)
  target_include_directories(vibeqc_uhf_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  add_test(NAME vibeqc_uhf_tests COMMAND vibeqc_uhf_tests)

  add_executable(vibeqc_spherical_tests tests/native/test_spherical.cpp)
  target_link_libraries(vibeqc_spherical_tests PRIVATE vibeqc)
  target_include_directories(vibeqc_spherical_tests
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  add_test(NAME vibeqc_spherical_tests COMMAND vibeqc_spherical_tests)

  if(VIBEQC_ENABLE_CUDA)
    add_executable(vibeqc_aot_profile_tests tests/native/test_aot_profile.cpp)
    target_link_libraries(vibeqc_aot_profile_tests PRIVATE vibeqc CUDA::cudart)
    target_include_directories(vibeqc_aot_profile_tests
      PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    add_test(NAME vibeqc_aot_profile_tests COMMAND vibeqc_aot_profile_tests)
  endif()
endif()

if(Python3_Interpreter_FOUND)
  set(VIBEQC_CODEGEN_PILOT_DIRECTORY
      "${CMAKE_CURRENT_BINARY_DIR}/generated/shell_kernels")
  set(VIBEQC_CODEGEN_PILOT_OUTPUTS)
  foreach(axis x y z)
    set(output
        "${VIBEQC_CODEGEN_PILOT_DIRECTORY}/eri_psss_${axis}_gradient.cuh")
    add_custom_command(
      OUTPUT "${output}"
      COMMAND "${Python3_EXECUTABLE}"
              "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_shell_kernels.py"
              --shell-class psss --axis "${axis}" --output "${output}"
      DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_shell_kernels.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/vibeqc_codegen/cuda.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/vibeqc_codegen/expr.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/vibeqc_codegen/shell_class.py"
      COMMENT "Generating symbolic/CSE psss ${axis}-gradient CUDA pilot"
      VERBATIM)
    list(APPEND VIBEQC_CODEGEN_PILOT_OUTPUTS "${output}")
  endforeach()
  set(output
      "${VIBEQC_CODEGEN_PILOT_DIRECTORY}/eri_dppp_xy_xyz_factored_gradient.cuh")
  add_custom_command(
    OUTPUT "${output}"
    COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_shell_kernels.py"
            --shell-class dppp --d-component xy --p-components xyz
            --lowering factored --output "${output}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_shell_kernels.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/vibeqc_codegen/cuda.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/vibeqc_codegen/expr.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/vibeqc_codegen/shell_class.py"
    COMMENT "Generating factored symbolic/CSE dppp xy/xyz CUDA candidate"
    VERBATIM)
  list(APPEND VIBEQC_CODEGEN_PILOT_OUTPUTS "${output}")
  add_custom_target(vibeqc_codegen_pilot DEPENDS ${VIBEQC_CODEGEN_PILOT_OUTPUTS})
endif()

include(GNUInstallDirs)
install(TARGETS vibeqc
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
