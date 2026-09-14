include_guard(GLOBAL)
include(CMakeParseArguments)

function(vibeqc_register_generated_sources)
  set(options ADD_TO_TARGET)
  set(one_value_args NAME TARGET GENERATOR LANGUAGE)
  set(multi_value_args OUTPUTS BYPRODUCTS DEPENDS ARGS COMPILE_OPTIONS)
  cmake_parse_arguments(VGS "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})
  if(NOT VGS_GENERATOR OR NOT VGS_OUTPUTS)
    message(FATAL_ERROR "generated sources require GENERATOR and OUTPUTS")
  endif()

  add_custom_command(
    OUTPUT ${VGS_OUTPUTS}
    BYPRODUCTS ${VGS_BYPRODUCTS}
    COMMAND "${Python3_EXECUTABLE}" "${VGS_GENERATOR}" ${VGS_ARGS}
    DEPENDS "${VGS_GENERATOR}" ${VGS_DEPENDS}
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
