include(CheckCXXCompilerFlag)

# A macro that determines the current time
# source: http://www.cmake.org/pipermail/cmake/2004-September/005526.html
macro(current_time result)
  set(NEED_FLAG)
  if(WIN32)
    if(NOT CYGWIN)
      set(NEED_FLAG "/T")
    endif(NOT CYGWIN)
  endif(WIN32)
  exec_program(date ARGS ${NEED_FLAG} OUTPUT_VARIABLE ${result})
endmacro()

macro(compiler_add_flag flag)
  set(OPENANN_OPTIMIZATION_FLAGS "${OPENANN_OPTIMIZATION_FLAGS} ${flag}")
endmacro()

macro(compiler_check_add_flag flag)
  check_cxx_compiler_flag(${flag} COMPILER_SUPPORT${flag})
  if(COMPILER_SUPPORT${flag})
    compiler_add_flag(${flag})
  endif()
endmacro()