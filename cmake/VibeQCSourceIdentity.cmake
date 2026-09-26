include_guard(GLOBAL)

# Expand the source-identity inventory from one repository-owned manifest.
# CONFIGURE_DEPENDS is intentionally limited to inventory membership: adding or
# removing a covered file regenerates the build graph, while ordinary content
# edits are handled by the build-time identity generator.
function(vibeqc_collect_source_identity_inputs output_variable)
  set(_manifest_relative "cmake/VibeQCSourceIdentity.json")
  set(_manifest_path "${CMAKE_CURRENT_SOURCE_DIR}/${_manifest_relative}")
  if(NOT EXISTS "${_manifest_path}")
    message(FATAL_ERROR "VibeQC source identity manifest is missing: ${_manifest_path}")
  endif()

  # The manifest defines the configure-time membership graph itself. Re-run
  # CMake when that graph changes, while ordinary member content remains a
  # build-time dependency handled by generate_build_identity.py.
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_manifest_path}")

  file(READ "${_manifest_path}" _manifest_json)
  string(JSON _schema GET "${_manifest_json}" schema_version)
  if(NOT _schema STREQUAL "1")
    message(FATAL_ERROR "Unsupported VibeQC source identity manifest schema: ${_schema}")
  endif()

  set(_inputs "${_manifest_relative}")

  string(JSON _group_count LENGTH "${_manifest_json}" recursive_groups)
  if(_group_count GREATER 0)
    math(EXPR _group_last "${_group_count} - 1")
    foreach(_group_index RANGE 0 ${_group_last})
      string(JSON _root GET "${_manifest_json}" recursive_groups ${_group_index} root)
      if(IS_ABSOLUTE "${_root}" OR _root MATCHES "(^|/)\\.\\.(/|$)")
        message(FATAL_ERROR "Unsafe source identity root: ${_root}")
      endif()
      if(NOT IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${_root}")
        message(FATAL_ERROR "Source identity root is not a directory: ${_root}")
      endif()

      string(JSON _pattern_count LENGTH "${_manifest_json}" recursive_groups
             ${_group_index} patterns)
      if(_pattern_count LESS 1)
        message(FATAL_ERROR "Source identity root has no patterns: ${_root}")
      endif()
      math(EXPR _pattern_last "${_pattern_count} - 1")
      foreach(_pattern_index RANGE 0 ${_pattern_last})
        string(JSON _pattern GET "${_manifest_json}" recursive_groups
               ${_group_index} patterns ${_pattern_index})
        string(JSON _pattern_type TYPE "${_manifest_json}" recursive_groups
               ${_group_index} patterns ${_pattern_index})
        if(NOT _pattern_type STREQUAL "STRING" OR _pattern STREQUAL "" OR
           IS_ABSOLUTE "${_pattern}" OR _pattern MATCHES "(^|/)\\.\\.(/|$)")
          message(FATAL_ERROR "Unsafe source identity pattern: ${_pattern}")
        endif()
        file(GLOB_RECURSE _matches CONFIGURE_DEPENDS
             RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
             "${CMAKE_CURRENT_SOURCE_DIR}/${_root}/${_pattern}")
        list(APPEND _inputs ${_matches})
      endforeach()
    endforeach()
  endif()

  string(JSON _file_count LENGTH "${_manifest_json}" files)
  if(_file_count GREATER 0)
    math(EXPR _file_last "${_file_count} - 1")
    foreach(_file_index RANGE 0 ${_file_last})
      string(JSON _relative GET "${_manifest_json}" files ${_file_index})
      if(IS_ABSOLUTE "${_relative}" OR _relative MATCHES "(^|/)\\.\\.(/|$)")
        message(FATAL_ERROR "Unsafe source identity file: ${_relative}")
      endif()
      if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_relative}" OR
         IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${_relative}")
        message(FATAL_ERROR "Source identity file is missing: ${_relative}")
      endif()
      list(APPEND _inputs "${_relative}")
    endforeach()
  endif()

  list(REMOVE_DUPLICATES _inputs)
  list(SORT _inputs)
  set(${output_variable} "${_inputs}" PARENT_SCOPE)
endfunction()
