include_guard(GLOBAL)
include(CMakeParseArguments)

function(vibeqc_register_generated_sources)
  set(options ADD_TO_TARGET)
  set(one_value_args NAME TARGET GENERATOR LANGUAGE COMMENT)
  set(multi_value_args OUTPUTS BYPRODUCTS DEPENDS ARGS COMPILE_OPTIONS)
  # Preserve quoted semicolon-containing arguments, such as the MP2 architecture
  # list, as single command arguments when forwarding them to the generator.
  cmake_parse_arguments(PARSE_ARGV 0 VGS "${options}" "${one_value_args}" "${multi_value_args}")
  if(NOT VGS_GENERATOR OR NOT VGS_OUTPUTS)
    message(FATAL_ERROR "generated sources require GENERATOR and OUTPUTS")
  endif()

  get_filename_component(
    _vibeqc_codegen_runner
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../tools/run_codegen.py"
    ABSOLUTE)
  list(GET VGS_OUTPUTS 0 _vibeqc_primary_output)
  set(_vibeqc_depfile "${_vibeqc_primary_output}.d")
  set(_vibeqc_depfile_targets)
  foreach(_vibeqc_output IN LISTS VGS_OUTPUTS)
    list(APPEND _vibeqc_depfile_targets --target "${_vibeqc_output}")
  endforeach()

  # CMake 3.27+ can tell Ninja that the explicit DEPENDS/DEPFILE edges fully
  # describe generated-source prerequisites. This avoids inheriting transitive
  # target dependencies as implicit custom-command dependencies while keeping
  # the repository's CMake 3.24 minimum supported.
  set(_vibeqc_explicit_dependency_boundary)
  if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.27")
    set(_vibeqc_explicit_dependency_boundary DEPENDS_EXPLICIT_ONLY)
  endif()

  add_custom_command(
    OUTPUT ${VGS_OUTPUTS}
    BYPRODUCTS ${VGS_BYPRODUCTS}
    COMMAND "${Python3_EXECUTABLE}" "${_vibeqc_codegen_runner}"
            --depfile "${_vibeqc_depfile}"
            ${_vibeqc_depfile_targets}
            --source-root "${PROJECT_SOURCE_DIR}"
            "${VGS_GENERATOR}" ${VGS_ARGS}
    DEPENDS "${VGS_GENERATOR}" "${_vibeqc_codegen_runner}" ${VGS_DEPENDS}
    DEPFILE "${_vibeqc_depfile}"
    ${_vibeqc_explicit_dependency_boundary}
    COMMENT "${VGS_COMMENT}"
    VERBATIM)
  set_source_files_properties(${VGS_OUTPUTS} ${VGS_BYPRODUCTS} PROPERTIES GENERATED TRUE)

  if(VGS_ADD_TO_TARGET)
    if(NOT VGS_TARGET)
      message(FATAL_ERROR "ADD_TO_TARGET requires TARGET")
    endif()
    target_sources(${VGS_TARGET} PRIVATE ${VGS_OUTPUTS})
  endif()
  if(VGS_NAME)
    add_custom_target(${VGS_NAME} DEPENDS ${VGS_OUTPUTS})
    if(VGS_TARGET)
      add_dependencies(${VGS_TARGET} ${VGS_NAME})
    endif()
  endif()
  if(VGS_LANGUAGE)
    set_source_files_properties(${VGS_OUTPUTS} PROPERTIES LANGUAGE "${VGS_LANGUAGE}")
  endif()
  if(VGS_COMPILE_OPTIONS)
    set_source_files_properties(${VGS_OUTPUTS} PROPERTIES COMPILE_OPTIONS "${VGS_COMPILE_OPTIONS}")
  endif()
endfunction()
