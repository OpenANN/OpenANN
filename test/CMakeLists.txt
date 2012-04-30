add_subdirectory(lib)

include_directories(lib/CPP-Test)
file(GLOB test_src "*.cpp")
add_executable(TestSuite ${test_src})
target_link_libraries(TestSuite cpptest)
