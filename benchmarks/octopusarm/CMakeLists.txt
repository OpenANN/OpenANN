project(OctopusArmBenchmark)

file(COPY benchmark.py DESTINATION .)
file(COPY settings.xml DESTINATION .)
file(COPY run DESTINATION .)
file(COPY evaluate DESTINATION .)
add_definitions("${OPENANN_COMPILER_FLAGS}")
add_executable(OctopusArmBenchmark agent.cpp tcpClient.cpp protocol.cpp)
target_link_libraries(OctopusArmBenchmark openann)
