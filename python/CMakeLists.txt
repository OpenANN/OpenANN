find_package(PythonInterp)

if(PYTHONINTERP_FOUND)
  set(PYTHONEX_LINK_LIBRARIES "\"openann\", \"alglib\"")
  if(CUDA_FOUND)
    set(PYTHONEX_LINK_LIBRARIES "${PYTHONEX_LINK_LIBRARIES}, \"openann_cuda\"")
  endif()

  file(GLOB CYTHON_SOURCES "*.p[xy][dix]")
  configure_file(build_info.py.in ${PROJECT_SOURCE_DIR}/python/build_info.py)

  add_custom_command(
    OUTPUT python_extension_command
    COMMAND ${CMAKE_COMMAND} -E echo "Building Python extension..."
    COMMAND ${PYTHON_EXECUTABLE} setup.py build_ext -i
    COMMAND ${CMAKE_COMMAND} -E echo "Done."
    DEPENDS ${CYTHON_SOURCES}
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/python
  )
  # HACK This target will only be build to test the python bindings!
  add_custom_target(python_extension DEPENDS openann alglib python_extension_command)

  add_custom_command(
    OUTPUT python_extension_install_command
    COMMAND ${CMAKE_COMMAND} -E echo "Installing Python extension..."
    COMMAND ${PYTHON_EXECUTABLE} setup.py install --prefix=${CMAKE_INSTALL_PREFIX}
    COMMAND ${CMAKE_COMMAND} -E echo "Done."
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/python
  )
  add_custom_target(python_extension_install DEPENDS openann alglib python_extension_install_command)
  install(CODE "execute_process(COMMAND \"${CMAKE_COMMAND}\" --build . --target python_extension_install)")
else()
  message(STATUS "Could not build Python extension. You must install Python, NumPy and Cython)")
endif()
