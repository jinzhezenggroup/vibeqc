cmake_minimum_required(VERSION 3.24)

project(VibeQC VERSION 0.1.0 LANGUAGES CXX)

find_package(Python3 COMPONENTS Interpreter QUIET)

option(VIBEQC_ENABLE_CUDA "Build the CUDA runtime backend" ON)
option(VIBEQC_BUILD_TESTS "Build native tests" ON)

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
  src/api/c_api.cpp
  src/integrals/s_integrals.cpp
  src/molecule/basis.cpp
  src/runtime/context.cpp
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
    src/scf/cuda_rhf.cu
  )
  if(NOT Python3_Interpreter_FOUND)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
  endif()
  set(VIBEQC_AOT_SHELL_MANIFEST
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/vibeqc_codegen/production_shell_classes.json"
      CACHE FILEPATH "Accepted generated shell classes for production CUDA")
  set(VIBEQC_AOT_GENERATED_DIRECTORY
      "${CMAKE_CURRENT_BINARY_DIR}/generated/production_shell_kernels")
  list(GET CMAKE_CUDA_ARCHITECTURES 0 VIBEQC_AOT_PRIMARY_CUDA_ARCHITECTURE)
  set(VIBEQC_AOT_CODEGEN_ARCHITECTURE
      "sm_${VIBEQC_AOT_PRIMARY_CUDA_ARCHITECTURE}"
      CACHE STRING "CUDA architecture profile used by shell code generation")
  set(VIBEQC_AOT_GENERATED_SOURCES
      "${VIBEQC_AOT_GENERATED_DIRECTORY}/vibeqc_generated_shell_registry.cu")
  foreach(shard RANGE 0 3)
    list(APPEND VIBEQC_AOT_GENERATED_SOURCES
         "${VIBEQC_AOT_GENERATED_DIRECTORY}/vibeqc_generated_shell_shard_${shard}.cu")
  endforeach()
  set(VIBEQC_AOT_GENERATED_HEADER
      "${VIBEQC_AOT_GENERATED_DIRECTORY}/vibeqc_generated_shell_registry.hpp")
  add_custom_command(
    OUTPUT ${VIBEQC_AOT_GENERATED_SOURCES} "${VIBEQC_AOT_GENERATED_HEADER}"
    COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_shell_kernels.py"
            --production-manifest "${VIBEQC_AOT_SHELL_MANIFEST}"
            --output-directory "${VIBEQC_AOT_GENERATED_DIRECTORY}"
            --architecture "${VIBEQC_AOT_CODEGEN_ARCHITECTURE}"
            --shards 4
    DEPENDS
      "${VIBEQC_AOT_SHELL_MANIFEST}"
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_shell_kernels.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/vibeqc_codegen/__init__.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/vibeqc_codegen/cuda.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/vibeqc_codegen/dppp_dispatch.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/vibeqc_codegen/expr.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/vibeqc_codegen/fused_schedule.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/vibeqc_codegen/ir.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/vibeqc_codegen/low_order_force.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/vibeqc_codegen/production.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/vibeqc_codegen/shell_class.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/vibeqc_codegen/shell_spec.py"
    COMMENT "Generating accepted fused shell-class CUDA shards"
    VERBATIM)
  set_source_files_properties(${VIBEQC_AOT_GENERATED_SOURCES}
                              PROPERTIES GENERATED TRUE)
  target_sources(vibeqc PRIVATE ${VIBEQC_AOT_GENERATED_SOURCES})
  target_include_directories(vibeqc PRIVATE "${VIBEQC_AOT_GENERATED_DIRECTORY}")
  target_link_libraries(vibeqc PRIVATE CUDA::cudart CUDA::cublas CUDA::cusolver)
  target_compile_definitions(vibeqc PUBLIC VIBEQC_HAS_CUDA=1)
  set_target_properties(vibeqc PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
else()
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

  add_executable(vibeqc_uhf_tests tests/native/test_uhf.cpp)
  target_link_libraries(vibeqc_uhf_tests PRIVATE vibeqc)
  target_include_directories(vibeqc_uhf_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  add_test(NAME vibeqc_uhf_tests COMMAND vibeqc_uhf_tests)

  add_executable(vibeqc_spherical_tests tests/native/test_spherical.cpp)
  target_link_libraries(vibeqc_spherical_tests PRIVATE vibeqc)
  target_include_directories(vibeqc_spherical_tests
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  add_test(NAME vibeqc_spherical_tests COMMAND vibeqc_spherical_tests)
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
