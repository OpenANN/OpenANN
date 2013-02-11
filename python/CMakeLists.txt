find_package(PythonInterp)

if(PYTHONINTERP_FOUND)
  file(COPY test.py DESTINATION .)
  configure_file(setup.py.in setup.py)
  configure_file(openann.pyx.in openann.pyx)

  add_custom_command(
    OUTPUT python_extension_command
    COMMAND ${CMAKE_COMMAND} -E echo "Building Python extension..."
    COMMAND ${PYTHON_EXECUTABLE} setup.py build_ext -i
    COMMAND ${CMAKE_COMMAND} -E echo "Done."
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  add_custom_target(python_extension ALL DEPENDS python_extension_command)
else()
  message(STATUS "Could not build Python extension. You must install Python, NumPy and Cython)")
endif()