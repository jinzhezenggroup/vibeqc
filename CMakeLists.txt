cmake_minimum_required(VERSION 3.24)

project(qce VERSION 0.1.0 LANGUAGES CXX)

find_package(Python3 COMPONENTS Interpreter QUIET)

option(QCE_ENABLE_CUDA "Build the CUDA runtime backend" ON)
option(QCE_BUILD_TESTS "Build native tests" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(QCE_ENABLE_CUDA)
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
    set(QCE_ENABLE_CUDA OFF)
  endif()
endif()

add_library(qce SHARED
  src/api/c_api.cpp
  src/integrals/s_integrals.cpp
  src/molecule/basis.cpp
  src/runtime/context.cpp
  src/scf/fleet.cpp
  src/scf/rhf.cpp
)

target_include_directories(qce
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_definitions(qce PRIVATE QCE_BUILDING_LIBRARY=1)
target_compile_options(qce PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>)

if(QCE_ENABLE_CUDA)
  target_sources(qce PRIVATE
    src/runtime/cuda_runtime.cu
    src/scf/cuda_rhf.cu
  )
  if(NOT Python3_Interpreter_FOUND)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
  endif()
  set(QCE_AOT_SHELL_MANIFEST
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/qce_codegen/production_shell_classes.json"
      CACHE FILEPATH "Accepted generated shell classes for production CUDA")
  set(QCE_AOT_GENERATED_DIRECTORY
      "${CMAKE_CURRENT_BINARY_DIR}/generated/production_shell_kernels")
  set(QCE_AOT_GENERATED_SOURCES
      "${QCE_AOT_GENERATED_DIRECTORY}/qce_generated_shell_registry.cu")
  foreach(shard RANGE 0 3)
    list(APPEND QCE_AOT_GENERATED_SOURCES
         "${QCE_AOT_GENERATED_DIRECTORY}/qce_generated_shell_shard_${shard}.cu")
  endforeach()
  set(QCE_AOT_GENERATED_HEADER
      "${QCE_AOT_GENERATED_DIRECTORY}/qce_generated_shell_registry.hpp")
  add_custom_command(
    OUTPUT ${QCE_AOT_GENERATED_SOURCES} "${QCE_AOT_GENERATED_HEADER}"
    COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_shell_kernels.py"
            --production-manifest "${QCE_AOT_SHELL_MANIFEST}"
            --output-directory "${QCE_AOT_GENERATED_DIRECTORY}"
            --shards 4
    DEPENDS
      "${QCE_AOT_SHELL_MANIFEST}"
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_shell_kernels.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/qce_codegen/__init__.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/qce_codegen/dppp_dispatch.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/qce_codegen/fused_schedule.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/qce_codegen/low_order_force.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/qce_codegen/production.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/qce_codegen/shell_spec.py"
    COMMENT "Generating accepted fused shell-class CUDA shards"
    VERBATIM)
  set_source_files_properties(${QCE_AOT_GENERATED_SOURCES}
                              PROPERTIES GENERATED TRUE)
  target_sources(qce PRIVATE ${QCE_AOT_GENERATED_SOURCES})
  target_include_directories(qce PRIVATE "${QCE_AOT_GENERATED_DIRECTORY}")
  target_link_libraries(qce PRIVATE CUDA::cudart CUDA::cublas CUDA::cusolver)
  target_compile_definitions(qce PUBLIC QCE_HAS_CUDA=1)
  set_target_properties(qce PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
else()
  target_compile_definitions(qce PUBLIC QCE_HAS_CUDA=0)
endif()

set_target_properties(qce PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION 0
  OUTPUT_NAME qce
)

if(QCE_BUILD_TESTS)
  enable_testing()
  add_executable(qce_native_tests tests/native/test_rhf.cpp)
  target_link_libraries(qce_native_tests PRIVATE qce)
  target_include_directories(qce_native_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  add_test(NAME qce_native_tests COMMAND qce_native_tests)

  add_executable(qce_batch_tests tests/native/test_batch.cpp)
  target_link_libraries(qce_batch_tests PRIVATE qce)
  add_test(NAME qce_batch_tests COMMAND qce_batch_tests)

  add_executable(qce_cpp_api_tests tests/native/test_cpp_batch.cpp)
  target_link_libraries(qce_cpp_api_tests PRIVATE qce)
  add_test(NAME qce_cpp_api_tests COMMAND qce_cpp_api_tests)

  add_executable(qce_cartesian_integral_tests
    tests/native/test_cartesian_integrals.cpp)
  target_link_libraries(qce_cartesian_integral_tests PRIVATE qce)
  target_include_directories(qce_cartesian_integral_tests
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  add_test(NAME qce_cartesian_integral_tests
    COMMAND qce_cartesian_integral_tests)

  add_executable(qce_uhf_tests tests/native/test_uhf.cpp)
  target_link_libraries(qce_uhf_tests PRIVATE qce)
  target_include_directories(qce_uhf_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  add_test(NAME qce_uhf_tests COMMAND qce_uhf_tests)

  add_executable(qce_spherical_tests tests/native/test_spherical.cpp)
  target_link_libraries(qce_spherical_tests PRIVATE qce)
  target_include_directories(qce_spherical_tests
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  add_test(NAME qce_spherical_tests COMMAND qce_spherical_tests)
endif()

if(Python3_Interpreter_FOUND)
  set(QCE_CODEGEN_PILOT_DIRECTORY
      "${CMAKE_CURRENT_BINARY_DIR}/generated/shell_kernels")
  set(QCE_CODEGEN_PILOT_OUTPUTS)
  foreach(axis x y z)
    set(output
        "${QCE_CODEGEN_PILOT_DIRECTORY}/eri_psss_${axis}_gradient.cuh")
    add_custom_command(
      OUTPUT "${output}"
      COMMAND "${Python3_EXECUTABLE}"
              "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_shell_kernels.py"
              --shell-class psss --axis "${axis}" --output "${output}"
      DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_shell_kernels.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/qce_codegen/cuda.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/qce_codegen/expr.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/qce_codegen/shell_class.py"
      COMMENT "Generating symbolic/CSE psss ${axis}-gradient CUDA pilot"
      VERBATIM)
    list(APPEND QCE_CODEGEN_PILOT_OUTPUTS "${output}")
  endforeach()
  set(output
      "${QCE_CODEGEN_PILOT_DIRECTORY}/eri_dppp_xy_xyz_factored_gradient.cuh")
  add_custom_command(
    OUTPUT "${output}"
    COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_shell_kernels.py"
            --shell-class dppp --d-component xy --p-components xyz
            --lowering factored --output "${output}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_shell_kernels.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/qce_codegen/cuda.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/qce_codegen/expr.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/qce_codegen/shell_class.py"
    COMMENT "Generating factored symbolic/CSE dppp xy/xyz CUDA candidate"
    VERBATIM)
  list(APPEND QCE_CODEGEN_PILOT_OUTPUTS "${output}")
  add_custom_target(qce_codegen_pilot DEPENDS ${QCE_CODEGEN_PILOT_OUTPUTS})
endif()

include(GNUInstallDirs)
install(TARGETS qce
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
