project(ConvolutionBenchmark)

add_definitions("${OPENANN_COMPILER_FLAGS}")
add_executable(Convolution convolution.cpp)
target_link_libraries(Convolution openann)
