project(CIFAR10Benchmark)

add_definitions("${OPENANN_COMPILER_FLAGS}")
add_executable(CIFAR10 cifar10.cpp)
target_link_libraries(CIFAR10 openann)
