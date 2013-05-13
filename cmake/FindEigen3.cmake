find_path(EIGEN3_INCLUDE_DIRS Eigen/Dense
  ${CMAKE_INSTALL_PREFIX}/include/eigen3
  /usr/include/eigen3
  /opt/local/include/eigen3
  DOC "Eigen 3 include directory")

set(EIGEN3_FOUND ${EIGEN3_INCLUDE_DIRS} CACHE BOOL "" FORCE)
