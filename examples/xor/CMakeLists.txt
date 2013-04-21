project(XOR)

add_definitions("${OPENANN_COMPILER_FLAGS}")
add_executable(XOR xor.cpp)
target_link_libraries(XOR openann)
