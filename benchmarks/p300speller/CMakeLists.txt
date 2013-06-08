project(P300SpellerBenchmark)

file(COPY benchmark.py DESTINATION .)
add_definitions("${OPENANN_COMPILER_FLAGS}")
add_executable(P300Speller p300.cpp BCIDataSet.cpp)
target_link_libraries(P300Speller openann)
