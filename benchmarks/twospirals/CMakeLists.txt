project(TwoSpiralsBenchmark)

include_directories("${CMAKE_SOURCE_DIR}/examples/twospirals")
add_definitions("${OPENANN_COMPILER_FLAGS}")
add_executable(TwoSpiralsBenchmark twospirals.cpp)
target_link_libraries(TwoSpiralsBenchmark openann)
