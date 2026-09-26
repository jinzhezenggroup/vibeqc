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
    src/scf/cuda/direct_angular_force.cu
    "${VIBEQC_WEIGHTED_ERI_HEADER}"
    "${VIBEQC_DIRECT_RESIDENT_PSSS_SCHEDULE_HEADER}"
    "${VIBEQC_DIRECT_HIGH_ORDER_PAIR_GRADIENT_HEADER}"
    "${VIBEQC_DIRECT_FOCK_ACCUMULATION_HEADER}")
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
                "${VIBEQC_WEIGHTED_ERI_HEADER}"
                "${VIBEQC_DIRECT_HIGH_ORDER_PAIR_GRADIENT_HEADER}"
    "${VIBEQC_DIRECT_FOCK_ACCUMULATION_HEADER}")
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
              JOB_POOL_COMPILE ${_vibeqc_aot_compile_pool})
          if(VIBEQC_CUDA_FAST_COMPILE)
            target_compile_options(${class_target} PRIVATE
              $<$<COMPILE_LANGUAGE:CUDA>:--Ofast-compile=max>)
          endif()
          if(NOT VIBEQC_AOT_SPLIT_COMPILE_THREADS STREQUAL "1")
            target_compile_options(${class_target} PRIVATE
              $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CUDA_COMPILER_ID:NVIDIA>>:--split-compile=${VIBEQC_AOT_SPLIT_COMPILE_THREADS}>)
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
            JOB_POOL_COMPILE ${_vibeqc_aot_compile_pool})
        if(VIBEQC_CUDA_FAST_COMPILE)
          target_compile_options(vibeqc_aot_${profile_architecture} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--Ofast-compile=max>)
        endif()
        if(NOT VIBEQC_AOT_SPLIT_COMPILE_THREADS STREQUAL "1")
          target_compile_options(vibeqc_aot_${profile_architecture} PRIVATE
            $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CUDA_COMPILER_ID:NVIDIA>>:--split-compile=${VIBEQC_AOT_SPLIT_COMPILE_THREADS}>)
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
  if(VIBEQC_ENABLE_STATIONARY_FORCE_AOT)
    # Preserve hashed source/asset identity separately from generator imports.
    set(_vibeqc_stationary_contract_assets
      "src/dft/stationary_gradient_cuda.cuh"
      "src/dft/grid_task_view.cuh"
      "src/dft/xc_point.hpp"
      "src/integrals/eri_geometry.hpp"
      "src/integrals/range_moments.hpp"
      "src/tensor/cuda_runtime.cuh"
      "src/runtime/bounded_workspace.hpp"
      "src/runtime/cuda_resources.cuh"
      "src/runtime/resource_cuda.cuh"
      "src/runtime/resource_ledger.hpp"
      "src/tensor/cuda_error.hpp"
      "src/tensor/metrics.hpp"
      "src/runtime/allocation_measurement.hpp"
    )
    set(_vibeqc_stationary_contract_inputs)
    foreach(_input IN LISTS _vibeqc_identity_inputs)
      if(_input MATCHES "^python/vibeqc_compiler/(common|integral|xc|dft)/.*\\.(py|json)$" OR
         _input STREQUAL "python/vibeqc_compiler/__init__.py" OR
         _input IN_LIST _vibeqc_stationary_contract_assets)
        list(APPEND _vibeqc_stationary_contract_inputs "${CMAKE_CURRENT_SOURCE_DIR}/${_input}")
      endif()
    endforeach()
    set(VIBEQC_STATIONARY_AOT_DIRECTORY
        "${CMAKE_CURRENT_BINARY_DIR}/generated/stationary_force")
    set(_vibeqc_stationary_architectures)
    foreach(_vibeqc_stationary_arch IN LISTS _vibeqc_cuda_compile_architectures)
      string(REGEX REPLACE "-(real|virtual)$" "" _vibeqc_stationary_arch
             "${_vibeqc_stationary_arch}")
      list(APPEND _vibeqc_stationary_architectures
           "sm_${_vibeqc_stationary_arch}")
    endforeach()
    list(REMOVE_DUPLICATES _vibeqc_stationary_architectures)
    set(_vibeqc_stationary_architecture_args)
    foreach(_vibeqc_stationary_arch IN LISTS _vibeqc_stationary_architectures)
      list(APPEND _vibeqc_stationary_architecture_args
           --architecture "${_vibeqc_stationary_arch}")
    endforeach()
    set(_vibeqc_stationary_compile_architecture_args)
    foreach(_vibeqc_stationary_arch IN LISTS _vibeqc_cuda_compile_architectures)
      list(APPEND _vibeqc_stationary_compile_architecture_args
           --compile-architecture "${_vibeqc_stationary_arch}")
    endforeach()
    # Component-expanded s/p/d derivatives are shared compiler output: generate
    # the bounded primitive inventory once, then device-link it into each
    # method/spin wrapper. The 23 x 16 layout is part of the versioned v3
    # artifact contract and is checked again by the Python manifest writer.
    set(_vibeqc_stationary_spd_primitive_sources)
    set(_vibeqc_stationary_spd_primitive_args)
    set(_vibeqc_stationary_spd_codegen_targets)
    foreach(_vibeqc_stationary_shard RANGE 0 22)
      set(_vibeqc_stationary_shard_source
          "${VIBEQC_STATIONARY_AOT_DIRECTORY}/vibeqc_stationary_spd_primitive_${_vibeqc_stationary_shard}.cu")
      list(APPEND _vibeqc_stationary_spd_primitive_sources
           "${_vibeqc_stationary_shard_source}")
      list(APPEND _vibeqc_stationary_spd_primitive_args
           --primitive-source "${_vibeqc_stationary_shard_source}")
      set(_vibeqc_stationary_shard_codegen
          "vibeqc_stationary_spd_primitive_${_vibeqc_stationary_shard}_codegen")
      list(APPEND _vibeqc_stationary_spd_codegen_targets
           "${_vibeqc_stationary_shard_codegen}")
      vibeqc_register_generated_sources(
        NAME "${_vibeqc_stationary_shard_codegen}"
        GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_stationary_force_aot.py"
        OUTPUTS "${_vibeqc_stationary_shard_source}"
        DEPENDS
          "${CMAKE_CURRENT_SOURCE_DIR}/src/dft/stationary_gradient_cuda.cuh"
        ARGS
          --output "${_vibeqc_stationary_shard_source}"
          --component-domain spd
          --shard-index "${_vibeqc_stationary_shard}"
        COMMENT
          "Generating stationary CUDA s/p/d primitive shard ${_vibeqc_stationary_shard}")
    endforeach()
    add_library(vibeqc_stationary_spd_primitives OBJECT
                ${_vibeqc_stationary_spd_primitive_sources})
    add_dependencies(vibeqc_stationary_spd_primitives
                     ${_vibeqc_stationary_spd_codegen_targets})
    target_include_directories(vibeqc_stationary_spd_primitives PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_compile_definitions(vibeqc_stationary_spd_primitives PRIVATE
        VIBEQC_HAS_CUDA=1)
    target_compile_options(vibeqc_stationary_spd_primitives PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    set_target_properties(vibeqc_stationary_spd_primitives PROPERTIES
        CUDA_ARCHITECTURES "${_vibeqc_cuda_compile_architectures}"
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
        JOB_POOL_COMPILE vibeqc_cuda_compile)

    # Generated source weights are plan-bound after #665/#689. Keep RKS and
    # UKS artifacts distinct so one-spin D/W lowering cannot serve two-spin work.
    set(_vibeqc_stationary_functionals 0 0 1 1 2 2)
    set(_vibeqc_stationary_spins
        unpolarized polarized unpolarized polarized unpolarized polarized)
    set(_vibeqc_stationary_names
        lda_rks lda_uks pbe_rks pbe_uks r2scan_rks r2scan_uks)
    foreach(_vibeqc_stationary_functional _vibeqc_stationary_spin
            _vibeqc_stationary_name IN ZIP_LISTS
            _vibeqc_stationary_functionals _vibeqc_stationary_spins
            _vibeqc_stationary_names)
      set(_vibeqc_stationary_source
          "${VIBEQC_STATIONARY_AOT_DIRECTORY}/vibeqc_stationary_${_vibeqc_stationary_name}.cu")
      set(_vibeqc_stationary_manifest
          "${CMAKE_CURRENT_BINARY_DIR}/vibeqc_stationary_${_vibeqc_stationary_name}.json")
      vibeqc_register_generated_sources(
        NAME "vibeqc_stationary_${_vibeqc_stationary_name}_codegen"
        GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_stationary_force_aot.py"
        OUTPUTS "${_vibeqc_stationary_source}"
        DEPENDS
          "${CMAKE_CURRENT_SOURCE_DIR}/src/dft/stationary_gradient_cuda.cuh"
        ARGS
          --output "${_vibeqc_stationary_source}"
          --functional "${_vibeqc_stationary_functional}"
          --spin "${_vibeqc_stationary_spin}"
          --iterations 3
        COMMENT
          "Generating ${_vibeqc_stationary_name} stationary CUDA AOT source")
      set(_vibeqc_stationary_target
          "vibeqc_stationary_${_vibeqc_stationary_name}")
      add_library(${_vibeqc_stationary_target} SHARED
                  "${_vibeqc_stationary_source}")
      add_dependencies(${_vibeqc_stationary_target}
                       "vibeqc_stationary_${_vibeqc_stationary_name}_codegen")
      target_include_directories(${_vibeqc_stationary_target} PRIVATE
          "${CMAKE_CURRENT_SOURCE_DIR}/src")
      target_compile_definitions(${_vibeqc_stationary_target} PRIVATE
          VIBEQC_HAS_CUDA=1)
      target_compile_options(${_vibeqc_stationary_target} PRIVATE
          $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>
          $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
      if(NOT VIBEQC_AOT_SPLIT_COMPILE_THREADS STREQUAL "1")
        target_compile_options(${_vibeqc_stationary_target} PRIVATE
          $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CUDA_COMPILER_ID:NVIDIA>>:--split-compile=${VIBEQC_AOT_SPLIT_COMPILE_THREADS}>)
      endif()
      set_target_properties(${_vibeqc_stationary_target} PROPERTIES
          CUDA_ARCHITECTURES "${_vibeqc_cuda_compile_architectures}"
          CUDA_STANDARD 20
          CUDA_STANDARD_REQUIRED ON
          POSITION_INDEPENDENT_CODE ON
          JOB_POOL_COMPILE ${_vibeqc_aot_compile_pool}
          OUTPUT_NAME "vibeqc_stationary_${_vibeqc_stationary_name}")
      if(VIBEQC_PYTHON_WHEEL)
        vibeqc_attach_cuda_implib(${_vibeqc_stationary_target})
      else()
        target_link_libraries(${_vibeqc_stationary_target} PRIVATE
                              CUDA::cudart CUDA::cublas)
      endif()
      vibeqc_register_generated_sources(
        OUTPUTS "${_vibeqc_stationary_manifest}"
        GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/write_stationary_aot_manifest.py"
        ARGS
                --library "$<TARGET_FILE:${_vibeqc_stationary_target}>"
                --source "${_vibeqc_stationary_source}"
                --output "${_vibeqc_stationary_manifest}"
                --functional "${_vibeqc_stationary_functional}"
                --spin "${_vibeqc_stationary_spin}"
                --iterations 3
                ${_vibeqc_stationary_architecture_args}
                ${_vibeqc_stationary_compile_architecture_args}
        DEPENDS
          ${_vibeqc_stationary_target}
          "${_vibeqc_stationary_source}"
          ${_vibeqc_stationary_contract_inputs}
        COMMENT
          "Recording ${_vibeqc_stationary_name} stationary CUDA AOT identity")
      add_custom_target(
        "${_vibeqc_stationary_target}_manifest" ALL
        DEPENDS "${_vibeqc_stationary_manifest}")
      install(TARGETS ${_vibeqc_stationary_target}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
      install(FILES "${_vibeqc_stationary_manifest}"
        DESTINATION ${CMAKE_INSTALL_LIBDIR})
      set(_vibeqc_stationary_spd_source
          "${VIBEQC_STATIONARY_AOT_DIRECTORY}/vibeqc_stationary_${_vibeqc_stationary_name}_spd.cu")
      set(_vibeqc_stationary_spd_manifest
          "${CMAKE_CURRENT_BINARY_DIR}/vibeqc_stationary_${_vibeqc_stationary_name}_spd.json")
      set(_vibeqc_stationary_spd_codegen
          "vibeqc_stationary_${_vibeqc_stationary_name}_spd_codegen")
      vibeqc_register_generated_sources(
        NAME "${_vibeqc_stationary_spd_codegen}"
        GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_stationary_force_aot.py"
        OUTPUTS "${_vibeqc_stationary_spd_source}"
        DEPENDS
          "${CMAKE_CURRENT_SOURCE_DIR}/src/dft/stationary_gradient_cuda.cuh"
        ARGS
          --output "${_vibeqc_stationary_spd_source}"
          --functional "${_vibeqc_stationary_functional}"
          --spin "${_vibeqc_stationary_spin}"
          --iterations 3
          --component-domain spd
        COMMENT
          "Generating ${_vibeqc_stationary_name} stationary CUDA s/p/d wrapper")
      set(_vibeqc_stationary_spd_target
          "vibeqc_stationary_${_vibeqc_stationary_name}_spd")
      add_library(${_vibeqc_stationary_spd_target} SHARED
                  "${_vibeqc_stationary_spd_source}"
                  $<TARGET_OBJECTS:vibeqc_stationary_spd_primitives>)
      add_dependencies(${_vibeqc_stationary_spd_target}
                       "${_vibeqc_stationary_spd_codegen}"
                       vibeqc_stationary_spd_primitives)
      target_include_directories(${_vibeqc_stationary_spd_target} PRIVATE
          "${CMAKE_CURRENT_SOURCE_DIR}/src")
      target_compile_definitions(${_vibeqc_stationary_spd_target} PRIVATE
          VIBEQC_HAS_CUDA=1)
      target_compile_options(${_vibeqc_stationary_spd_target} PRIVATE
          $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>
          $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
      set_target_properties(${_vibeqc_stationary_spd_target} PROPERTIES
          CUDA_ARCHITECTURES "${_vibeqc_cuda_compile_architectures}"
          CUDA_STANDARD 20
          CUDA_STANDARD_REQUIRED ON
          CUDA_SEPARABLE_COMPILATION ON
          CUDA_RESOLVE_DEVICE_SYMBOLS ON
          POSITION_INDEPENDENT_CODE ON
          JOB_POOL_COMPILE vibeqc_cuda_compile
          OUTPUT_NAME "vibeqc_stationary_${_vibeqc_stationary_name}_spd")
      if(VIBEQC_PYTHON_WHEEL)
        vibeqc_attach_cuda_implib(${_vibeqc_stationary_spd_target})
      else()
        target_link_libraries(${_vibeqc_stationary_spd_target} PRIVATE
                              CUDA::cudart CUDA::cublas)
      endif()
      vibeqc_register_generated_sources(
        OUTPUTS "${_vibeqc_stationary_spd_manifest}"
        GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/tools/write_stationary_aot_manifest.py"
        ARGS
                --library "$<TARGET_FILE:${_vibeqc_stationary_spd_target}>"
                --source "${_vibeqc_stationary_spd_source}"
                --output "${_vibeqc_stationary_spd_manifest}"
                --functional "${_vibeqc_stationary_functional}"
                --spin "${_vibeqc_stationary_spin}"
                --iterations 3
                --component-domain spd
                ${_vibeqc_stationary_spd_primitive_args}
                ${_vibeqc_stationary_architecture_args}
                ${_vibeqc_stationary_compile_architecture_args}
        DEPENDS
          ${_vibeqc_stationary_spd_target}
          "${_vibeqc_stationary_spd_source}"
          ${_vibeqc_stationary_spd_primitive_sources}
          ${_vibeqc_stationary_contract_inputs}
        COMMENT
          "Recording ${_vibeqc_stationary_name} stationary CUDA s/p/d AOT identity")
      add_custom_target(
        "${_vibeqc_stationary_spd_target}_manifest" ALL
        DEPENDS "${_vibeqc_stationary_spd_manifest}")
      install(TARGETS ${_vibeqc_stationary_spd_target}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
      install(FILES "${_vibeqc_stationary_spd_manifest}"
        DESTINATION ${CMAKE_INSTALL_LIBDIR})
    endforeach()
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
