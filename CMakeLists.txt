cmake_minimum_required(VERSION 2.8)

project(OpenANN)

find_package(Doxygen)
include(cmake/UseDoxygen.cmake)

set(OPENANN_VERSION_NUMBER "1.0.0")
set(OPENANN_URL "https://github.com/AlexanderFabisch/OpenANN")
set(OPENANN_BRIEF_DESCRIPTION "An open source library for artificial neural networks.")
set(OPENANN_TIME) # TODO write macro that determines the current time
set(OPENANN_COMPILER_FLAGS) # TODO determine optimization flags, warning flags etc.

include_directories(OpenANN)

add_subdirectory(src)

install(DIRECTORY OpenANN DESTINATION include)

option(BUILD_DOCUMENTATION "Build OpenANN documentation using doxygen" ON)
if(DOXYGEN_FOUND AND BUILD_DOCUMENTATION)
  if(DOXYGEN_DOT_FOUND)
    set(OPENANN_USE_DOT "YES")
  else()
    set(OPENANN_USE_DOT "NO")
  endif()
  configure_file(doxygen.cfg.in ${PROJECT_SOURCE_DIR}/doxygen.cfg)
  add_documentation(openann_doc doxygen.cfg)
  install_documentation(${PROJECT_BINARY_DIR}/html openann)
endif()
