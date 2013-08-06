# File: UseDoxygen.cmake
# CMake commands to actually use doxygen
# Version: 1.0
# Author: Alexander Fabisch <afabisch@googlemail.com>
#
#
#
# The following macros are defined:
#
# add_documentation(target <configuration file> [ADD_TO_ALL])
#     Adds targets that generate the documentation.
# install_documentation(source_dir target_dir)
#     Installs the documentation, e. g. copies the html documentation to
#     /usr/share/docs/target_dir

macro(add_documentation target configuration)
  if(NOT DOXYGEN_FOUND)
    message(FATAL_ERROR "Doxygen not found")
  endif()

  set(DOXYGEN_INPUT ${configuration})
  set(COMMAND_NAME "${target}_doxygen")
  set(TARGET_NAME "${target}")
  set(TARGET_NAME_FORCED "${target}_forced")

  add_custom_command(
    OUTPUT ${COMMAND_NAME}
    COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_INPUT} > /dev/null
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDS ${DOXYGEN_INPUT}
  )
  add_custom_target(${TARGET_NAME} DEPENDS ${COMMAND_NAME})
  foreach(opt ${ARGN})
    if(opt STREQUAL "ADD_TO_ALL")
      add_custom_target(${TARGET_NAME}_all ALL DEPENDS ${COMMAND_NAME})
    endif()
  endforeach()
endmacro()

macro(install_documentation source target)
  if(UNIX)
    install(DIRECTORY ${source} DESTINATION "share/doc/${target}")
  else()
    message(WARNING "I do not know where to install the documentation!")
  endif()
endmacro()
