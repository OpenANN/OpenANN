find_path(EIGEN3_INCLUDE_DIRS Eigen/Dense /usr/local/include/eigen3)

message(STATUS ${EIGEN3_INCLUDE_DIRS})

if(EIGEN_INCLUDE_DIRS EQUAL "")
  set(EIGEN3_FOUND FALSE)
else()
  set(EIGEN3_FOUND TRUE)
endif()
