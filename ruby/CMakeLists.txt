find_package(Ruby)

if(RUBY_FOUND)
  message(STATUS "Building ruby extension")

  configure_file(extconf.rb.in extconf.rb)

  file(GLOB ruby_extension_src "*.cpp")
  file(GLOB ruby_extension_h "*.h")

  file(COPY ${ruby_extension_src} DESTINATION .)
  file(COPY ${ruby_extension_h} DESTINATION .)

  add_custom_command(
    OUTPUT ruby_extension_command
    COMMAND ${CMAKE_COMMAND} -E echo "Building Ruby extension..."
    COMMAND ${RUBY_EXECUTABLE} extconf.rb
    COMMAND make
    COMMAND ${CMAKE_COMMAND} -E echo "Done."
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  add_custom_target(ruby_extension ALL DEPENDS ruby_extension_command)
endif()
