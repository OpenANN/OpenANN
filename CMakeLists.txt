cmake_minimum_required(VERSION 2.8)

project(OpenANN)

find_package(Doxygen)
find_package(OpenMP)
include(cmake/UseDoxygen.cmake)
include(cmake/Utilities.cmake)

set(OPENANN_VERSION_NUMBER "1.0.0")
set(OPENANN_URL "https://github.com/AlexanderFabisch/OpenANN")
set(OPENANN_BRIEF_DESCRIPTION "An open source library for artificial neural networks.")
current_time(OPENANN_TIME) # cmake execution time
set(OPENANN_CUDA_FLAGS) # TODO CUDA optimization flags
set(OPENANN_OPTIMIZATION_FLAGS) # TODO optimization flags
set(OPENANN_COMPILER_FLAGS) # TODO determine optimization flags, warning flags etc.
if(CMAKE_COMPILER_IS_GNUCXX)
  set(COMPILER_WARNING_FLAGS "-Wall -Wextra -pedantic -Wno-long-long")
endif()
set(OPENANN_LINK_LIBS "-lopenann") # TODO libraries to link for external executables
if(OPENMP_FOUND AND CMAKE_COMPILER_IS_GNUCXX)
  set(OPENANN_LINK_LIBS "${OPENANN_LINK_LIBS} -lgomp")
endif()

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

if(OPENMP_FOUND AND (LINUX_FOUND OR FREEBSD_FOUND))
  set(OPENANN_LINK_LIBS "-lgomp")
endif()
configure_file(openann.pc.in openann.pc @ONLY)

string(REPLACE ":" ";" pkg_target_search "$ENV{PKG_CONFIG_LIBDIR}")
find_path(pkg_target_dir "pkgconfig" ${pkg_target_search} "${CMAKE_INSTALL_PREFIX}/share" ${pkg_target_search})
if(NOT pkg_target_dir)
  set(pkg_target_dir "${CMAKE_INSTALL_PREFIX}/share")
endif()
install(FILES "${PROJECT_BINARY_DIR}/openann.pc" DESTINATION "${pkg_target_dir}/pkgconfig")