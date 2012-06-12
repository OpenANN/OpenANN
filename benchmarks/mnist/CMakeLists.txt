project(MNISTBenchmark)

add_definitions("${OPENANN_COMPILER_FLAGS}")
add_executable(MNIST mnist.cpp)
target_link_libraries(MNIST openann)
