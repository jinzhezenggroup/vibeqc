cmake_minimum_required(VERSION 3.24)

project(VibeQC VERSION 0.1.0 LANGUAGES CXX)

find_package(Python3 COMPONENTS Interpreter QUIET)
find_package(Threads REQUIRED)

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/VibeQCGenerated.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/VibeQCGeneratedSources.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/VibeQCSources.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/VibeQCCuda.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/VibeQCCudaImplib.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/VibeQCCpuLinalg.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/VibeQCGfn2Runtime.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/VibeQCTests.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/VibeQCSourceIdentity.cmake")

option(VIBEQC_ENABLE_CUDA "Build the CUDA runtime backend" ON)
option(VIBEQC_ENABLE_AOT_SHELLS
       "Build generated shell kernels in addition to the generic CUDA path" ON)
option(VIBEQC_BUILD_TESTS "Build native tests" ON)
option(VIBEQC_PYTHON_WHEEL
       "Install one unversioned shared library for a Python wheel" OFF)
option(VIBEQC_BUNDLE_XTB_OPENBLAS
       "Bundle the reviewed private LP64 OpenBLAS provider for GFN2 wheels"
       ${VIBEQC_PYTHON_WHEEL})
option(VIBEQC_CUDA_SEPARABLE_COMPILATION
       "Enable CUDA relocatable device code and the device-link stage" OFF)
option(VIBEQC_CUDA_DIRECT_DEVICE_LINK
       "Share native direct numerical code through a bounded NVIDIA device link" ON)
option(VIBEQC_CUDA_FAST_COMPILE
       "Use compiler-oriented CUDA options (never for release measurements)" OFF)
set(VIBEQC_COMPILER_CACHE "auto" CACHE STRING
    "Compiler cache launcher: auto, off, or an executable name/path")
set(VIBEQC_CUDA_ARCHITECTURES "" CACHE STRING
    "CUDA architectures shipped by VibeQC; defaults to CMAKE_CUDA_ARCHITECTURES")
set(VIBEQC_CUDA_COMPILE_ARCHITECTURES "" CACHE STRING
    "NVCC architecture values, retaining 120-real/120-virtual intent")
set(VIBEQC_CUDA_COMPILE_JOBS "2" CACHE STRING
    "Maximum concurrent CUDA compiler jobs in the Ninja job pool")
set(VIBEQC_CUDA_SPLIT_COMPILE_THREADS "1" CACHE STRING
    "NVCC split-compile threads for native direct kernels; 1 preserves production code generation")
set(VIBEQC_CUDA_PROVIDER "nvidia" CACHE STRING
    "CUDA execution provider identity: nvidia or cumetal")
set_property(CACHE VIBEQC_CUDA_PROVIDER PROPERTY STRINGS nvidia cumetal)
set(VIBEQC_AOT_PROFILE "auto" CACHE STRING
    "AOT profile for every target: auto, portable, sm_XX, or a named profile")
set(VIBEQC_AOT_PROFILES "" CACHE STRING
    "Optional explicit list of AOT profiles, independent of architecture order")
set(VIBEQC_AOT_SHARDS "8" CACHE STRING
    "Number of independently compiled CUDA shards per AOT architecture")
set(VIBEQC_AOT_UNIT_MODE "stable-shards" CACHE STRING
    "Generated source layout: stable-shards (release) or class (development)")
set_property(CACHE VIBEQC_AOT_UNIT_MODE PROPERTY STRINGS stable-shards class)

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
  get_filename_component(_vibeqc_compiler_cache_name
                         "${_vibeqc_compiler_cache}" NAME)
  set(_vibeqc_compiler_cache_launcher "${_vibeqc_compiler_cache}")
  if(_vibeqc_compiler_cache_name STREQUAL "ccache")
    # Generated CUDA/AOT sources live below the build tree. Without a stable
    # base directory, their absolute paths make byte-identical objects miss
    # when the same source is rebuilt in another worktree or build directory.
    set(_vibeqc_compiler_cache_launcher
        "${CMAKE_COMMAND};-E;env;CCACHE_BASEDIR=${CMAKE_BINARY_DIR};${_vibeqc_compiler_cache}")
  endif()
  if(NOT CMAKE_CXX_COMPILER_LAUNCHER)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${_vibeqc_compiler_cache_launcher}")
  endif()
  if(NOT CMAKE_CUDA_COMPILER_LAUNCHER)
    set(CMAKE_CUDA_COMPILER_LAUNCHER "${_vibeqc_compiler_cache_launcher}")
  endif()
  message(STATUS "VibeQC compiler cache: ${_vibeqc_compiler_cache}")
endif()

if(NOT Python3_Interpreter_FOUND)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
endif()

if(NOT VIBEQC_CUDA_SPLIT_COMPILE_THREADS MATCHES "^[0-9]+$")
  message(FATAL_ERROR
          "VIBEQC_CUDA_SPLIT_COMPILE_THREADS must be a non-negative integer")
endif()
if(NOT VIBEQC_AOT_SHARDS MATCHES "^[1-9][0-9]*$")
  message(FATAL_ERROR "VIBEQC_AOT_SHARDS must be a positive integer")
endif()
if(NOT VIBEQC_CUDA_COMPILE_JOBS MATCHES "^[1-9][0-9]*$")
  message(FATAL_ERROR "VIBEQC_CUDA_COMPILE_JOBS must be a positive integer")
endif()
if(NOT VIBEQC_CUDA_PROVIDER STREQUAL "nvidia"
   AND NOT VIBEQC_CUDA_PROVIDER STREQUAL "cumetal")
  message(FATAL_ERROR "VIBEQC_CUDA_PROVIDER must be nvidia or cumetal")
endif()
if(NOT VIBEQC_AOT_UNIT_MODE STREQUAL "stable-shards"
   AND NOT VIBEQC_AOT_UNIT_MODE STREQUAL "class")
  message(FATAL_ERROR "VIBEQC_AOT_UNIT_MODE must be stable-shards or class")
endif()
math(EXPR VIBEQC_AOT_LAST_SHARD "${VIBEQC_AOT_SHARDS} - 1")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(VIBEQC_ENABLE_CUDA)
  # Resolve only caller-supplied target sources before CUDA language enablement.
  # CMake may otherwise initialize CMAKE_CUDA_ARCHITECTURES from NVCC's legacy
  # compiler default, which must not become an implicit VibeQC target.
  set(_vibeqc_cuda_requested_architectures "")
  if(VIBEQC_CUDA_COMPILE_ARCHITECTURES)
    set(_vibeqc_cuda_requested_architectures
        ${VIBEQC_CUDA_COMPILE_ARCHITECTURES})
  elseif(VIBEQC_CUDA_ARCHITECTURES)
    set(_vibeqc_cuda_requested_architectures ${VIBEQC_CUDA_ARCHITECTURES})
  elseif(DEFINED CMAKE_CUDA_ARCHITECTURES AND CMAKE_CUDA_ARCHITECTURES)
    set(_vibeqc_cuda_requested_architectures ${CMAKE_CUDA_ARCHITECTURES})
  elseif(DEFINED ENV{CUDAARCHS} AND NOT "$ENV{CUDAARCHS}" STREQUAL "")
    set(_vibeqc_cuda_requested_architectures "$ENV{CUDAARCHS}")
  endif()

  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    if(NOT _vibeqc_cuda_requested_architectures)
      message(FATAL_ERROR
        "CUDA target architecture is required; set VIBEQC_CUDA_ARCHITECTURES, "
        "CMAKE_CUDA_ARCHITECTURES, VIBEQC_CUDA_COMPILE_ARCHITECTURES, or CUDAARCHS")
    endif()
    set(CMAKE_CUDA_ARCHITECTURES ${_vibeqc_cuda_requested_architectures})
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    # Preserve NVCC's real/virtual suffixes for compilation. A separate
    # numeric profile identity is derived below for manifest and cache paths.
    set(_vibeqc_cuda_compile_architectures ${CMAKE_CUDA_ARCHITECTURES})
    set_property(GLOBAL PROPERTY JOB_POOLS
                 vibeqc_cuda_compile=${VIBEQC_CUDA_COMPILE_JOBS})
    # XsyevBatched is the capture-safe production eigensolver for realistic
    # AO buckets and is available in the supported CUDA 12.9 toolkit. Native
    # SDK builds link the toolkit providers normally. Python wheels instead
    # need only compiler-owned headers: hidden Implib.so trampolines resolve
    # the provider SONAMEs lazily at runtime.
    if(VIBEQC_PYTHON_WHEEL)
      if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.9)
        message(FATAL_ERROR "VibeQC CUDA wheels require CUDA 12.9 or newer")
      endif()
      set(CMAKE_CUDA_RUNTIME_LIBRARY None)
      enable_language(C ASM)
      set(VIBEQC_CUDA_TOOLKIT_INCLUDE_DIRS ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
      foreach(_vibeqc_cuda_header IN ITEMS cuda_runtime_api.h library_types.h)
        set(_vibeqc_cuda_header_found OFF)
        foreach(_vibeqc_cuda_include IN LISTS VIBEQC_CUDA_TOOLKIT_INCLUDE_DIRS)
          if(EXISTS "${_vibeqc_cuda_include}/${_vibeqc_cuda_header}")
            set(_vibeqc_cuda_header_found ON)
            break()
          endif()
        endforeach()
        if(NOT _vibeqc_cuda_header_found)
          message(FATAL_ERROR "CUDA compiler include set lacks ${_vibeqc_cuda_header}")
        endif()
      endforeach()
    else()
      if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT VIBEQC_CUDA_PROVIDER STREQUAL "cumetal")
        # Native GFN2 resolves its driver metadata calls lazily as well. Enable
        # import-source languages here, not inside a function-local scope.
        enable_language(C ASM)
      endif()
      find_package(CUDAToolkit 12.9 REQUIRED)
      set(VIBEQC_CUDA_TOOLKIT_INCLUDE_DIRS ${CUDAToolkit_INCLUDE_DIRS})
    endif()
  else()
    message(WARNING "CUDA compiler not found; building the CPU reference backend only")
    set(VIBEQC_ENABLE_CUDA OFF)
  endif()
endif()

# Scoped VibeQC-owned GFN2 CPU runtime is registered below.
add_library(vibeqc SHARED)
target_link_libraries(vibeqc PRIVATE Threads::Threads)
vibeqc_configure_cpu_linalg(vibeqc)
vibeqc_add_gfn2_runtime(vibeqc)
vibeqc_add_component_sources(vibeqc)
vibeqc_register_host_generated_sources(vibeqc)

# A local tuned library may replace this build only when its scientific source
# and generator are identical. Paths and Git metadata are deliberately excluded
# so the same installed build can be reused across homogeneous cluster nodes.
vibeqc_collect_source_identity_inputs(_vibeqc_identity_inputs)
set(_vibeqc_identity_text "")
foreach(_vibeqc_identity_path IN LISTS _vibeqc_identity_inputs)
  file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/${_vibeqc_identity_path}" _vibeqc_file_hash)
  string(APPEND _vibeqc_identity_text "${_vibeqc_identity_path}:${_vibeqc_file_hash}\n")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
               "${CMAKE_CURRENT_SOURCE_DIR}/${_vibeqc_identity_path}")
endforeach()
string(SHA256 VIBEQC_SOURCE_IDENTITY "${_vibeqc_identity_text}")
set(VIBEQC_TUNING_RELEASE_BUILD OFF)
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  set(VIBEQC_TUNING_RELEASE_BUILD ON)
endif()
configure_file(src/api/build_identity.hpp.in generated/build_identity.hpp @ONLY)
target_include_directories(vibeqc PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")

target_include_directories(vibeqc
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_definitions(vibeqc PRIVATE VIBEQC_BUILDING_LIBRARY=1)
if(VIBEQC_CUDA_PROVIDER STREQUAL "cumetal")
  target_compile_definitions(vibeqc PRIVATE VIBEQC_CUDA_PROVIDER_CUMETAL=1)
else()
  target_compile_definitions(vibeqc PRIVATE VIBEQC_CUDA_PROVIDER_CUMETAL=0)
endif()
target_compile_options(vibeqc PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>)

if(VIBEQC_ENABLE_CUDA)
  # A single real architecture may use that architecture's measured host-side
  # candidate schedule. Fat binaries and virtual/PTX-only targets keep host
  # scheduling portable: their active runtime device is not a compile-time
  # property of shared host code and may differ from the PTX target.
  list(LENGTH _vibeqc_cuda_compile_architectures _vibeqc_cuda_architecture_count)
  if(_vibeqc_cuda_architecture_count EQUAL 1)
    list(GET _vibeqc_cuda_compile_architectures 0 _vibeqc_cuda_profile_architecture)
    if(NOT _vibeqc_cuda_profile_architecture MATCHES "-virtual$")
      string(REGEX REPLACE "-real$" "" _vibeqc_cuda_profile_architecture
             "${_vibeqc_cuda_profile_architecture}")
      target_compile_definitions(
        vibeqc PRIVATE VIBEQC_CUDA_PROFILE_ARCHITECTURE=${_vibeqc_cuda_profile_architecture})
    endif()
  endif()
  vibeqc_register_cuda_generated_sources(vibeqc)
  vibeqc_configure_cuda_backend(vibeqc)
else()
  target_compile_definitions(vibeqc PUBLIC VIBEQC_HAS_CUDA=0)
endif()

set_target_properties(vibeqc PROPERTIES OUTPUT_NAME vibeqc)
if(NOT VIBEQC_PYTHON_WHEEL)
  # Native SDK installs retain the normal SONAME chain. Wheel archives often
  # dereference those symlinks and would otherwise store the large CUDA shared
  # object multiple times, so Python wheels install one unversioned library.
  set_target_properties(vibeqc PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 0
  )
endif()

if(VIBEQC_BUILD_TESTS)
  vibeqc_add_native_tests()
endif()

vibeqc_add_codegen_pilot()

include(GNUInstallDirs)
install(TARGETS vibeqc
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
