project(TwoSpiralsBenchmark)

add_definitions("${OPENANN_COMPILER_FLAGS}")
add_executable(TwoSpiralsBenchmark twospirals.cpp)
target_link_libraries(TwoSpiralsBenchmark openann)
