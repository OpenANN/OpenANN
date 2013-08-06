if(NOT (IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/CMA-ESpp/"))
  if(UNIX)
    exec_program("sh" ${CMAKE_CURRENT_SOURCE_DIR} ARGS "setup.sh" "CMA-ESpp" "ALGLIB" RETURN_VALUE result)
    if(result EQUAL 1)
      message(FATAL_ERROR "I could not install CMA-ESpp.")
    elseif(result EQUAL 2)
      message(FATAL_ERROR "I could not install ALGLIB.")
    endif()
  else()
    message(FATAL_ERROR "I do not know how to install CMA-ESpp and ALGLIB.")
  endif()
endif()

set(ALGLIB_DEFINITIONS "-DAE_HAVE_STDINT")
if(CMAKE_COMPILER_IS_GNUCXX AND NOT CYGWIN_FOUND)
  set(ALGLIB_DEFINITIONS "${ALGLIB_DEFINITIONS} -Wno-warnings")
endif()
if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "x86_64")
  set(ALGLIB_DEFINITIONS "${ALGLIB_DEFINITIONS} -DAE_CPU=1")
elseif(${CMAKE_SYSTEM_PROCESSOR} MATCHES "sparc")
  set(ALGLIB_DEFINITIONS "${ALGLIB_DEFINITIONS} -DAE_CPU=2")
endif()
set(alglib_src
  ALGLIB/optimization.cpp
  ALGLIB/alglibinternal.cpp
  ALGLIB/linalg.cpp
  ALGLIB/alglibmisc.cpp
  ALGLIB/ap.cpp)
add_library(alglib SHARED ${alglib_src})
set(OPENANN_LINK_LIBS "${OPENANN_LINK_LIBS} -lalglib")
set_target_properties(alglib PROPERTIES COMPILE_FLAGS
  "${OPENANN_OPTIMIZATION_FLAGS} ${ALGLIB_DEFINITIONS}")
install(TARGETS alglib DESTINATION lib)
install(DIRECTORY ALGLIB/ DESTINATION include/alglib FILES_MATCHING PATTERN "*.h")
