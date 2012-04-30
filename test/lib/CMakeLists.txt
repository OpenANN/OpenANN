if(NOT (EXISTS "CPP-Test")) # TODO this check does not work
  if(UNIX)
    exec_program("sh" ${CMAKE_CURRENT_SOURCE_DIR} ARGS "setup.sh" "CPP-Test" RETURN_VALUE result)
    if(result GREATER 0)
      message(FATAL_ERROR "I could not install CPP-Test.")
    endif()
  else()
    message(FATAL_ERROR "I do not know how to install CPP-Test.")
  endif()
endif()

add_subdirectory(CPP-Test)
