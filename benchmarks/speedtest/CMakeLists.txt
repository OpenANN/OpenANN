project(Speedtest)

add_definitions("${OPENANN_COMPILER_FLAGS}")
add_executable(Speedtest speedtest.cpp)
target_link_libraries(Speedtest openann)
