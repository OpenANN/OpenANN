project(OpenANNLibrary)

configure_file(OpenANN.cpp.in ${PROJECT_SOURCE_DIR}/OpenANN.cpp)
file(GLOB_RECURSE openann_src "*.cpp")
add_library(openann SHARED ${openann_src})
install(TARGETS openann DESTINATION lib)
