include_guard(GLOBAL)

macro(vibeqc_configure_cuda_backend target)
  foreach(_arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
    if(_arch STREQUAL "native" OR _arch STREQUAL "all" OR _arch STREQUAL "all-major")
      message(FATAL_ERROR
        "VIBEQC MP2 AOT generation requires numeric CMAKE_CUDA_ARCHITECTURES; "
        "special value '${_arch}' is unsupported. Configure with numeric architectures, "
        "for example 75;90.")
    endif()
  endforeach()

  target_include_directories(${target} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/dft"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/tensor")

  # These consumers share substantial retained numerical functions. Resolve
  # only this archive's device symbols so NVCC can coalesce those definitions
  # without coupling generated kernels or unrelated SCF/DF owners to its link.
  set(VIBEQC_DIRECT_NATIVE_SOURCES
    src/scf/cuda/direct_jk_kernels.cu
    src/scf/cuda/weighted_eri_kernels.cu
    src/scf/cuda/direct_cached_tensor_kernels.cu
    src/scf/cuda/direct_schwarz_kernels.cu
    src/scf/cuda/direct_packed_fock_kernels.cu
    src/scf/cuda/direct_angular_fock.cu
    src/scf/cuda/direct_reference_force.cu
    src/scf/cuda/direct_bounded_dddd.cu
    src/scf/cuda/direct_bounded_exact_force.cu
    src/scf/cuda/direct_bounded_fallback.cu
  )
  # The resident angular-force kernels have launch-bound register ceilings.
  # NVCC 12.9 needs whole-program compilation to propagate those ceilings into
  # retained callees, including for PTX JIT. Keep this owner outside device
  # linking even when the caller enables whole-library separable compilation.
  add_library(vibeqc_direct_angular_force OBJECT
    src/scf/cuda/direct_angular_force.cu "${VIBEQC_WEIGHTED_ERI_HEADER}")
  target_include_directories(vibeqc_direct_angular_force PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_BINARY_DIR}/generated")
  target_compile_definitions(vibeqc_direct_angular_force PRIVATE
    VIBEQC_BUILDING_LIBRARY=1 VIBEQC_HAS_CUDA=1)
  if(NOT VIBEQC_PYTHON_WHEEL)
    target_link_libraries(vibeqc_direct_angular_force PRIVATE CUDA::cudart)
  endif()
  set_target_properties(vibeqc_direct_angular_force PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES "${_vibeqc_cuda_compile_architectures}"
    JOB_POOL_COMPILE vibeqc_cuda_compile)
  if(VIBEQC_CUDA_FAST_COMPILE)
    target_compile_options(vibeqc_direct_angular_force PRIVATE --Ofast-compile=max)
  endif()
  target_sources(${target} PRIVATE $<TARGET_OBJECTS:vibeqc_direct_angular_force>)
  # The CuMetal toolkit shim on Apple hosts does not provide NVIDIA's device
  # linker. Keep independently compiled launch owners for that backend and for
  # other CUDA compilers; callers can also select this path explicitly.
  if(VIBEQC_CUDA_DIRECT_DEVICE_LINK AND
     CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA" AND NOT APPLE)
    add_library(vibeqc_direct_native STATIC ${VIBEQC_DIRECT_NATIVE_SOURCES}
                "${VIBEQC_WEIGHTED_ERI_HEADER}")
    target_include_directories(vibeqc_direct_native PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/include"
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${CMAKE_CURRENT_BINARY_DIR}/generated")
    target_compile_definitions(vibeqc_direct_native PRIVATE
      VIBEQC_BUILDING_LIBRARY=1 VIBEQC_HAS_CUDA=1)
    if(NOT VIBEQC_PYTHON_WHEEL)
      target_link_libraries(vibeqc_direct_native PRIVATE CUDA::cudart)
    endif()
    set_target_properties(vibeqc_direct_native PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_RESOLVE_DEVICE_SYMBOLS ON
      CUDA_ARCHITECTURES "${_vibeqc_cuda_compile_architectures}"
      JOB_POOL_COMPILE vibeqc_cuda_compile)
    if(VIBEQC_CUDA_FAST_COMPILE)
      target_compile_options(vibeqc_direct_native PRIVATE --Ofast-compile=max)
    endif()
    target_link_libraries(${target} PRIVATE vibeqc_direct_native)
  else()
    target_sources(${target} PRIVATE ${VIBEQC_DIRECT_NATIVE_SOURCES})
  endif()
  set_property(TARGET ${target} PROPERTY JOB_POOL_COMPILE vibeqc_cuda_compile)
  if(VIBEQC_CUDA_FAST_COMPILE)
    # Explicitly opt-in: this mode is for iteration speed and must not be used
    # for release performance/resource measurements.
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--Ofast-compile=max>)
    # NVCC 12.9 fast mode over-allocates shared storage for the derivative
    # entrypoints sharing contract_shell_task (even sss exceeds 48 KiB).
    # Compile the independent class units with the normal optimizer so scratch
    # remains within the launch limit. Release builds already use this mode.
    set_property(SOURCE ${VIBEQC_DF_SHELL_SOURCES}
                 APPEND PROPERTY COMPILE_OPTIONS "--Ofast-compile=0")
  endif()
  if(NOT VIBEQC_CUDA_SPLIT_COMPILE_THREADS STREQUAL "1")
    # Split compilation is useful for syntax/resource experiments on native
    # direct kernels, but NVCC may make different optimization choices.
    # Keep it opt-in so production benchmark binaries remain comparable.
    set_property(SOURCE ${VIBEQC_DIRECT_NATIVE_SOURCES} src/scf/cuda/direct_angular_force.cu
                 APPEND PROPERTY COMPILE_OPTIONS
                 "--split-compile=${VIBEQC_CUDA_SPLIT_COMPILE_THREADS}")
  endif()
  if(VIBEQC_ENABLE_AOT_SHELLS)
    if(NOT Python3_Interpreter_FOUND)
      find_package(Python3 COMPONENTS Interpreter REQUIRED)
    endif()
    set(VIBEQC_AOT_SHELL_MANIFEST
        "${CMAKE_CURRENT_SOURCE_DIR}/python/vibeqc_compiler/integral/production_shell_classes.json"
        CACHE FILEPATH "Accepted generated shell classes for production CUDA")
    set(VIBEQC_AOT_GENERATED_DIRECTORY
        "${CMAKE_CURRENT_BINARY_DIR}/generated/production_shell_kernels")

    # Class-mode development builds need a stable output list at configure
    # time.  Read names from the manifest itself rather than relying on its
    # ordering; profile resolution still decides which units contain code.
    set(VIBEQC_AOT_CLASS_NAMES)
    file(READ "${VIBEQC_AOT_SHELL_MANIFEST}" _vibeqc_manifest_text)
    string(REGEX MATCHALL
           "\"shell_class\"[ \t\r\n]*:[ \t\r\n]*\"[a-z]+\""
           _vibeqc_manifest_class_matches "${_vibeqc_manifest_text}")
    foreach(_vibeqc_class_match IN LISTS _vibeqc_manifest_class_matches)
      string(REGEX REPLACE
             ".*\"shell_class\"[ \t\r\n]*:[ \t\r\n]*\"([a-z]+)\".*"
             "\\1" _vibeqc_class_name "${_vibeqc_class_match}")
      list(APPEND VIBEQC_AOT_CLASS_NAMES "${_vibeqc_class_name}")
    endforeach()
    list(REMOVE_DUPLICATES VIBEQC_AOT_CLASS_NAMES)
    list(SORT VIBEQC_AOT_CLASS_NAMES)

    # Normalize and sort the compile targets so generated behavior and source
    # assignment do not depend on CMAKE_CUDA_ARCHITECTURES list order.
    set(VIBEQC_AOT_TARGET_ARCHITECTURES)
    foreach(compile_architecture IN LISTS _vibeqc_cuda_compile_architectures)
      string(REGEX REPLACE "-(real|virtual)$" "" architecture
             "${compile_architecture}")
      if(NOT architecture MATCHES "^[0-9]+$")
        message(FATAL_ERROR
                "VibeQC AOT requires numeric CUDA architectures, got ${architecture}")
      endif()
      list(APPEND VIBEQC_AOT_TARGET_ARCHITECTURES "${architecture}")
      set(_vibeqc_aot_compile_architecture_${architecture}
          "${compile_architecture}")
    endforeach()
    list(REMOVE_DUPLICATES VIBEQC_AOT_TARGET_ARCHITECTURES)
    list(SORT VIBEQC_AOT_TARGET_ARCHITECTURES COMPARE NATURAL)

    set(VIBEQC_AOT_GENERATOR_ARGUMENTS)
    set(_vibeqc_aot_unit_arguments
        --unit-mode "${VIBEQC_AOT_UNIT_MODE}")
    if(VIBEQC_AOT_UNIT_MODE STREQUAL "class")
      list(APPEND _vibeqc_aot_unit_arguments --all-class-units)
    endif()
    set(VIBEQC_AOT_GENERATED_PROFILE_SOURCES)
    foreach(architecture IN LISTS VIBEQC_AOT_TARGET_ARCHITECTURES)
      set(profile_architecture "sm_${architecture}")
      list(APPEND VIBEQC_AOT_GENERATOR_ARGUMENTS
           --target-architecture "${profile_architecture}")
      if(VIBEQC_AOT_UNIT_MODE STREQUAL "class")
        foreach(shell_class IN LISTS VIBEQC_AOT_CLASS_NAMES)
          list(APPEND VIBEQC_AOT_GENERATED_PROFILE_SOURCES
               "${VIBEQC_AOT_GENERATED_DIRECTORY}/${profile_architecture}/vibeqc_generated_shell_sm${architecture}_${shell_class}.cu")
        endforeach()
      else()
        foreach(shard RANGE 0 ${VIBEQC_AOT_LAST_SHARD})
          list(APPEND VIBEQC_AOT_GENERATED_PROFILE_SOURCES
               "${VIBEQC_AOT_GENERATED_DIRECTORY}/${profile_architecture}/vibeqc_generated_shell_sm${architecture}_shard_${shard}.cu")
        endforeach()
      endif()
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
    vibeqc_register_generated_sources(
      GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_shell_kernels.py"
      OUTPUTS
        ${VIBEQC_AOT_GENERATED_PROFILE_SOURCES}
        "${VIBEQC_AOT_GENERATED_REGISTRY_SOURCE}"
        "${VIBEQC_AOT_GENERATED_HEADER}"
      DEPENDS
        "${VIBEQC_AOT_SHELL_MANIFEST}"
      ARGS
        --production-manifest "${VIBEQC_AOT_SHELL_MANIFEST}"
        --output-directory "${VIBEQC_AOT_GENERATED_DIRECTORY}"
        --shards "${VIBEQC_AOT_SHARDS}"
        ${_vibeqc_aot_unit_arguments}
        ${VIBEQC_AOT_GENERATOR_ARGUMENTS}
      COMMENT "Generating target-specific fused shell-class CUDA bundles")

    # Each object target compiles only the schedule selected for that SM. The
    # handwritten CUDA sources and host registry still use the full fatbin set.
    foreach(architecture IN LISTS VIBEQC_AOT_TARGET_ARCHITECTURES)
      set(profile_architecture "sm_${architecture}")
      set(profile_sources)
      if(VIBEQC_AOT_UNIT_MODE STREQUAL "class")
        foreach(shell_class IN LISTS VIBEQC_AOT_CLASS_NAMES)
          list(APPEND profile_sources
               "${VIBEQC_AOT_GENERATED_DIRECTORY}/${profile_architecture}/vibeqc_generated_shell_sm${architecture}_${shell_class}.cu")
        endforeach()
      else()
        foreach(shard RANGE 0 ${VIBEQC_AOT_LAST_SHARD})
          list(APPEND profile_sources
               "${VIBEQC_AOT_GENERATED_DIRECTORY}/${profile_architecture}/vibeqc_generated_shell_sm${architecture}_shard_${shard}.cu")
        endforeach()
      endif()
      if(VIBEQC_AOT_UNIT_MODE STREQUAL "class")
        # Give development builds a directly addressable object target per
        # shell class.  A dddd edit can therefore be built with
        # ``--target vibeqc_aot_sm_120_dddd`` without compiling neighboring
        # generated classes.
        foreach(shell_class IN LISTS VIBEQC_AOT_CLASS_NAMES)
          set(class_source
              "${VIBEQC_AOT_GENERATED_DIRECTORY}/${profile_architecture}/vibeqc_generated_shell_sm${architecture}_${shell_class}.cu")
          set(class_target "vibeqc_aot_${profile_architecture}_${shell_class}")
          add_library(${class_target} OBJECT "${class_source}")
          target_include_directories(${class_target} PRIVATE
              "${CMAKE_CURRENT_SOURCE_DIR}/src"
              "${VIBEQC_AOT_GENERATED_DIRECTORY}")
          target_compile_definitions(${class_target} PRIVATE VIBEQC_HAS_CUDA=1)
          set_target_properties(${class_target} PROPERTIES
              CUDA_ARCHITECTURES
              "${_vibeqc_aot_compile_architecture_${architecture}}"
              CUDA_STANDARD 20
              CUDA_STANDARD_REQUIRED ON
              POSITION_INDEPENDENT_CODE ON
              JOB_POOL_COMPILE vibeqc_cuda_compile)
          if(VIBEQC_CUDA_FAST_COMPILE)
            target_compile_options(${class_target} PRIVATE
              $<$<COMPILE_LANGUAGE:CUDA>:--Ofast-compile=max>)
          endif()
          target_sources(${target} PRIVATE $<TARGET_OBJECTS:${class_target}>)
        endforeach()
      else()
        add_library(vibeqc_aot_${profile_architecture} OBJECT ${profile_sources})
        target_include_directories(vibeqc_aot_${profile_architecture} PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/src"
            "${VIBEQC_AOT_GENERATED_DIRECTORY}")
        target_compile_definitions(vibeqc_aot_${profile_architecture}
                                   PRIVATE VIBEQC_HAS_CUDA=1)
        set_target_properties(vibeqc_aot_${profile_architecture} PROPERTIES
            CUDA_ARCHITECTURES
            "${_vibeqc_aot_compile_architecture_${architecture}}"
            CUDA_STANDARD 20
            CUDA_STANDARD_REQUIRED ON
            POSITION_INDEPENDENT_CODE ON
            JOB_POOL_COMPILE vibeqc_cuda_compile)
        if(VIBEQC_CUDA_FAST_COMPILE)
          target_compile_options(vibeqc_aot_${profile_architecture} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--Ofast-compile=max>)
        endif()
        target_sources(${target} PRIVATE
            $<TARGET_OBJECTS:vibeqc_aot_${profile_architecture}>)
      endif()
    endforeach()
    target_sources(${target} PRIVATE "${VIBEQC_AOT_GENERATED_REGISTRY_SOURCE}")
    target_include_directories(${target} PRIVATE
        "${VIBEQC_AOT_GENERATED_DIRECTORY}")
  else()
    target_sources(${target} PRIVATE src/scf/aot_shell_registry_stub.cpp)
  endif()
  if(VIBEQC_PYTHON_WHEEL)
    vibeqc_attach_cuda_implib(${target})
  else()
    target_link_libraries(${target} PRIVATE CUDA::cudart CUDA::cublas CUDA::cusolver)
  endif()
  target_compile_definitions(${target} PUBLIC VIBEQC_HAS_CUDA=1)
  set_target_properties(${target} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ${VIBEQC_CUDA_SEPARABLE_COMPILATION})
endmacro()
