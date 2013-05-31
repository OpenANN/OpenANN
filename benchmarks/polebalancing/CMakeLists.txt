project(PoleBalancingBenchmark)

file(COPY benchmark.py DESTINATION .)
set(src_files
  ${CMAKE_SOURCE_DIR}/examples/polebalancing/SinglePoleBalancing.cpp
  ${CMAKE_SOURCE_DIR}/examples/polebalancing/DoublePoleBalancing.cpp
  ${CMAKE_SOURCE_DIR}/examples/polebalancing/NeuroEvolutionAgent.cpp
)
include_directories("${CMAKE_SOURCE_DIR}/examples/polebalancing")
add_definitions("${OPENANN_COMPILER_FLAGS}")
add_executable(PoleBalancingBenchmark polebalancing.cpp ${src_files})
target_link_libraries(PoleBalancingBenchmark openann)
