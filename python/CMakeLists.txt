find_package(PythonInterp)

if(PYTHONINTERP_FOUND)
  set(PYTHONEX_LINK_LIBRARIES "\"openann\"")
  if(CUDA_FOUND)
    set(PYTHONEX_LINK_LIBRARIES "${PYTHONEX_LINK_LIBRARIES}, \"openann_cuda\"")
  else()
  endif()
  if(USE_GPL_LICENSE)
    set(PYTHONEX_LINK_LIBRARIES "${PYTHONEX_LINK_LIBRARIES}, \"alglib\"")
  endif()

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

  add_custom_command(
    OUTPUT python_extension_install_command
    COMMAND ${CMAKE_COMMAND} -E echo "Installing Python extension..."
    COMMAND ${PYTHON_EXECUTABLE} setup.py install
    COMMAND ${CMAKE_COMMAND} -E echo "Done."
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  add_custom_target(python_extension_install DEPENDS python_extension_install_command)
  install(CODE "execute_process(COMMAND \"${CMAKE_COMMAND}\" --build . --target python_extension_install)")
else()
  message(STATUS "Could not build Python extension. You must install Python, NumPy and Cython)")
endif()