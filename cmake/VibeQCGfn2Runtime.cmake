include_guard(GLOBAL)

include(GNUInstallDirs)

function(vibeqc_add_gfn2_runtime target)
  set(_gfn2_root "${CMAKE_CURRENT_SOURCE_DIR}/src/xtb/gfn2_runtime")
  target_sources(${target} PRIVATE
    ${_gfn2_root}/src/cpu_dispatch/features.cpp
    ${_gfn2_root}/src/model/common/integrals_kernels_baseline.cpp
    ${_gfn2_root}/src/model/gfn2/mulliken_kernels_baseline.cpp
    ${_gfn2_root}/src/model/common/sto.cpp
    ${_gfn2_root}/src/model/common/integrals.cpp
    ${_gfn2_root}/src/model/common/scc_mixer.cpp
    ${_gfn2_root}/src/model/gfn2/eigensolver.cpp
    ${_gfn2_root}/src/model/gfn2/periodic_embedding.cpp
    ${_gfn2_root}/src/model/gfn2/es2.cpp
    ${_gfn2_root}/src/backends/common/gfn2_plan_schema.cpp
    ${_gfn2_root}/src/model/gfn2/aes2.cpp
    ${_gfn2_root}/src/model/gfn2/alpb.cpp
    ${_gfn2_root}/src/model/gfn2/basis.cpp
    ${_gfn2_root}/src/model/gfn2/coordination.cpp
    ${_gfn2_root}/src/model/gfn2/d4.cpp
    ${_gfn2_root}/src/model/gfn2/es3.cpp
    ${_gfn2_root}/src/model/gfn2/external_point_charges.cpp
    ${_gfn2_root}/src/model/gfn2/force.cpp
    ${_gfn2_root}/src/model/gfn2/h0.cpp
    ${_gfn2_root}/src/model/gfn2/lattice.cpp
    ${_gfn2_root}/src/model/gfn2/mulliken.cpp
    ${_gfn2_root}/src/model/gfn2/mulliken_kernels.cpp
    ${_gfn2_root}/src/model/gfn2/periodic_ewald.cpp
    ${_gfn2_root}/src/model/gfn2/periodic_integrals.cpp
    ${_gfn2_root}/src/model/gfn2/periodic_multipole.cpp
    ${_gfn2_root}/src/model/gfn2/periodic_topology.cpp
    ${_gfn2_root}/src/model/gfn2/repulsion.cpp
    ${_gfn2_root}/src/model/gfn2/scc_driver.cpp
    ${_gfn2_root}/src/model/gfn2/scc_mixer.cpp
    ${_gfn2_root}/src/model/gfn2/spin.cpp
    ${_gfn2_root}/src/model/gfn2/wavefunction.cpp
    ${_gfn2_root}/src/runtime/gfn2_cpu_execution.cpp)
  target_include_directories(${target} PRIVATE
    ${_gfn2_root}
    ${_gfn2_root}/include
    ${_gfn2_root}/src)
  if(CMAKE_DL_LIBS)
    target_link_libraries(${target} PRIVATE ${CMAKE_DL_LIBS})
  endif()

  if(VIBEQC_BUNDLE_XTB_OPENBLAS)
    if(NOT VIBEQC_PYTHON_WHEEL)
      message(FATAL_ERROR "VIBEQC_BUNDLE_XTB_OPENBLAS requires VIBEQC_PYTHON_WHEEL=ON")
    endif()
    if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
      message(FATAL_ERROR "The scoped GFN2 wheel provider is currently qualified on Linux only")
    endif()
    string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _gfn2_processor)
    if(_gfn2_processor MATCHES "^(x86_64|amd64)$")
      set(_gfn2_arch x86_64)
    elseif(_gfn2_processor MATCHES "^(aarch64|arm64)$")
      set(_gfn2_arch aarch64)
    else()
      message(FATAL_ERROR "Unsupported GFN2 wheel architecture: ${CMAKE_SYSTEM_PROCESSOR}")
    endif()
    execute_process(
      COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/xtb/resolve-openblas-wheel.py"
        --manifest "${CMAKE_CURRENT_SOURCE_DIR}/tools/xtb/scipy_openblas32_manifest.json"
        --platform linux
        --architecture "${_gfn2_arch}"
      RESULT_VARIABLE _gfn2_openblas_status
      OUTPUT_VARIABLE _gfn2_openblas_json
      ERROR_VARIABLE _gfn2_openblas_error
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT _gfn2_openblas_status EQUAL 0)
      message(FATAL_ERROR
        "Failed to resolve the reviewed GFN2 OpenBLAS provider:\n${_gfn2_openblas_error}")
    endif()
    string(JSON _gfn2_openblas_library GET "${_gfn2_openblas_json}" provider_path)
    string(JSON _gfn2_openblas_prefix GET "${_gfn2_openblas_json}" expected_config_prefix)
    get_filename_component(_gfn2_openblas_dir "${_gfn2_openblas_library}" DIRECTORY)

    set_source_files_properties(
      ${_gfn2_root}/src/runtime/openblas_lp64_shim.c PROPERTIES LANGUAGE CXX)
    add_library(vibeqc_gfn2_openblas_shim SHARED
      ${_gfn2_root}/src/runtime/openblas_lp64_shim.c)
    target_link_libraries(vibeqc_gfn2_openblas_shim PRIVATE "${_gfn2_openblas_library}")
    target_link_options(vibeqc_gfn2_openblas_shim PRIVATE "LINKER:--no-as-needed")
    set_target_properties(vibeqc_gfn2_openblas_shim PROPERTIES
      OUTPUT_NAME xtbloom_openblas_lp64_shim
      CXX_VISIBILITY_PRESET hidden
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
      BUILD_RPATH "${_gfn2_openblas_dir}"
      INSTALL_RPATH "")
    install(TARGETS vibeqc_gfn2_openblas_shim
      LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
    target_compile_definitions(${target} PRIVATE
      XTBLOOM_CONFIGURED_WHEEL_OPENBLAS=1
      "XTBLOOM_CONFIGURED_WHEEL_OPENBLAS_CONFIG_PREFIX=\"${_gfn2_openblas_prefix}\"")
    add_dependencies(${target} vibeqc_gfn2_openblas_shim)
  elseif(XTBLOOM_CPU_LINALG_LIBRARY)
    target_compile_definitions(${target} PRIVATE
      "XTBLOOM_CONFIGURED_CPU_LINALG_RUNTIME=\"${XTBLOOM_CPU_LINALG_LIBRARY}\"")
  endif()
endfunction()
