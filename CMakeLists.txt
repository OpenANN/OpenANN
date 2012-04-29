cmake_minimum_required(VERSION 2.8)

project(OpenANN)

set(OPENANN_VERSION_NUMBER "1.0.0")
set(OPENANN_URL "https://github.com/AlexanderFabisch/OpenANN")
set(OPENANN_BRIEF_DESCRIPTION "An open source library for artificial neural networks.")
set(OPENANN_TIME) # TODO write macro that determines the current time
set(OPENANN_COMPILER_FLAGS) # TODO determine optimization flags, warning flags etc.

include_directories(OpenANN)

add_subdirectory(src)
