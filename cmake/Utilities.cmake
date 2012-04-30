# A macro that determines the current time
# source: http://www.cmake.org/pipermail/cmake/2004-September/005526.html
macro(CURRENT_TIME result)
  set(NEED_FLAG)
  if(WIN32)
    if(NOT CYGWIN)
      set(NEED_FLAG "/T")
    endif(NOT CYGWIN)
  endif(WIN32)
  exec_program(date ARGS ${NEED_FLAG} OUTPUT_VARIABLE ${result})
endmacro()