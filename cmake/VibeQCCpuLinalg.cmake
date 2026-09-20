include_guard(GLOBAL)

set(VIBEQC_CPU_LINALG_PROVIDER "auto" CACHE STRING
    "CPU dense-linear-algebra provider: auto, scalar, or openblas")
set_property(CACHE VIBEQC_CPU_LINALG_PROVIDER PROPERTY STRINGS auto scalar openblas)

function(vibeqc_configure_cpu_linalg target)
  if(NOT VIBEQC_CPU_LINALG_PROVIDER STREQUAL "auto"
     AND NOT VIBEQC_CPU_LINALG_PROVIDER STREQUAL "scalar"
     AND NOT VIBEQC_CPU_LINALG_PROVIDER STREQUAL "openblas")
    message(FATAL_ERROR
            "VIBEQC_CPU_LINALG_PROVIDER must be auto, scalar, or openblas")
  endif()

  set(_provider_libraries "")
  set(_scipy_prefix 0)
  set(_include_dirs "")
  if(NOT VIBEQC_CPU_LINALG_PROVIDER STREQUAL "scalar")
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
      pkg_check_modules(VIBEQC_OPENBLAS QUIET IMPORTED_TARGET openblas)
      if(TARGET PkgConfig::VIBEQC_OPENBLAS)
        set(_provider_libraries PkgConfig::VIBEQC_OPENBLAS)
        set(_include_dirs ${VIBEQC_OPENBLAS_INCLUDE_DIRS})
      else()
        pkg_check_modules(VIBEQC_SCIPY_OPENBLAS QUIET IMPORTED_TARGET scipy-openblas)
        if(TARGET PkgConfig::VIBEQC_SCIPY_OPENBLAS)
          set(_provider_libraries PkgConfig::VIBEQC_SCIPY_OPENBLAS)
          set(_include_dirs ${VIBEQC_SCIPY_OPENBLAS_INCLUDE_DIRS})
          set(_scipy_prefix 1)
        endif()
      endif()
    endif()

    # OpenBLAS also ships a CMake package config. This path matters on minimal
    # build hosts without pkg-config and for SciPy's redistributable OpenBLAS.
    if(NOT _provider_libraries)
      find_package(OpenBLAS CONFIG QUIET)
      if(OpenBLAS_FOUND AND OpenBLAS_LIBRARIES)
        set(_provider_libraries ${OpenBLAS_LIBRARIES})
        set(_include_dirs ${OpenBLAS_INCLUDE_DIRS})
        foreach(_library IN LISTS OpenBLAS_LIBRARIES)
          if(_library MATCHES "scipy_openblas")
            set(_scipy_prefix 1)
          endif()
        endforeach()
      endif()
    endif()
  endif()

  if(_provider_libraries)
    include(CheckCXXSourceCompiles)
    set(CMAKE_REQUIRED_INCLUDES ${_include_dirs})
    set(CMAKE_REQUIRED_LIBRARIES ${_provider_libraries})
    if(_scipy_prefix)
      set(_thread_probe
          "#include <cblas.h>\nint main(){return scipy_openblas_set_num_threads_local(1);}")
      set(_global_thread_probe
          "#include <cblas.h>\nint main(){int n=scipy_openblas_get_num_threads();scipy_openblas_set_num_threads(n);return 0;}")
      set(_lapack_probe
          "#include <lapacke.h>\nint main(){double a[1]={1},w[1];int x=scipy_LAPACKE_dpotrf(LAPACK_ROW_MAJOR,'L',1,a,1);return x+scipy_LAPACKE_dsyevd(LAPACK_ROW_MAJOR,'V','L',1,a,1,w);}")
    else()
      set(_thread_probe
          "#include <cblas.h>\nint main(){return openblas_set_num_threads_local(1);}")
      set(_global_thread_probe
          "#include <cblas.h>\nint main(){int n=openblas_get_num_threads();openblas_set_num_threads(n);return 0;}")
      set(_lapack_probe
          "#include <lapacke.h>\nint main(){double a[1]={1},w[1];int x=LAPACKE_dpotrf(LAPACK_ROW_MAJOR,'L',1,a,1);return x+LAPACKE_dsyevd(LAPACK_ROW_MAJOR,'V','L',1,a,1,w);}")
    endif()
    # Capability results depend on the selected provider, not merely the build
    # directory. Re-probe if callers switch OpenBLAS implementations in place.
    unset(VIBEQC_OPENBLAS_HAS_LOCAL_THREADS CACHE)
    unset(VIBEQC_OPENBLAS_HAS_GLOBAL_THREADS CACHE)
    unset(VIBEQC_OPENBLAS_HAS_LAPACKE CACHE)
    check_cxx_source_compiles("${_thread_probe}" VIBEQC_OPENBLAS_HAS_LOCAL_THREADS)
    check_cxx_source_compiles("${_global_thread_probe}" VIBEQC_OPENBLAS_HAS_GLOBAL_THREADS)
    check_cxx_source_compiles("${_lapack_probe}" VIBEQC_OPENBLAS_HAS_LAPACKE)
    unset(CMAKE_REQUIRED_INCLUDES)
    unset(CMAKE_REQUIRED_LIBRARIES)

    target_link_libraries(${target} PRIVATE ${_provider_libraries})
    target_include_directories(${target} PRIVATE ${_include_dirs})
    target_compile_definitions(${target} PRIVATE
      VIBEQC_HAS_OPENBLAS=1
      VIBEQC_OPENBLAS_SCIPY_PREFIX=${_scipy_prefix}
      VIBEQC_OPENBLAS_HAS_LOCAL_THREADS=$<BOOL:${VIBEQC_OPENBLAS_HAS_LOCAL_THREADS}>
      VIBEQC_OPENBLAS_HAS_GLOBAL_THREADS=$<BOOL:${VIBEQC_OPENBLAS_HAS_GLOBAL_THREADS}>
      VIBEQC_OPENBLAS_HAS_LAPACKE=$<BOOL:${VIBEQC_OPENBLAS_HAS_LAPACKE}>)
    message(STATUS
      "VibeQC CPU linear algebra: OpenBLAS (local threads=${VIBEQC_OPENBLAS_HAS_LOCAL_THREADS}, global threads=${VIBEQC_OPENBLAS_HAS_GLOBAL_THREADS}, LAPACKE=${VIBEQC_OPENBLAS_HAS_LAPACKE})")
  else()
    if(VIBEQC_CPU_LINALG_PROVIDER STREQUAL "openblas")
      message(FATAL_ERROR
              "VIBEQC_CPU_LINALG_PROVIDER=openblas requested but no OpenBLAS package metadata was found")
    endif()
    target_compile_definitions(${target} PRIVATE
      VIBEQC_HAS_OPENBLAS=0
      VIBEQC_OPENBLAS_SCIPY_PREFIX=0
      VIBEQC_OPENBLAS_HAS_LOCAL_THREADS=0
      VIBEQC_OPENBLAS_HAS_GLOBAL_THREADS=0
      VIBEQC_OPENBLAS_HAS_LAPACKE=0)
    message(STATUS "VibeQC CPU linear algebra: scalar fallback")
  endif()
endfunction()
