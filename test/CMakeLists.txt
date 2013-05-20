add_subdirectory(lib)

find_package(Qt4)

include_directories(lib/CPP-Test)
add_definitions(${OPENANN_COMPILER_FLAGS})
file(GLOB test_src "*.cpp")

if(QT_FOUND)
  set(QT_USE_QTMAIN TRUE)
  include(${QT_USE_FILE})
  add_definitions("${QT_DEFINITIONS} -DUSE_QT")
  add_executable(TestSuite ${test_src})
  target_link_libraries(TestSuite cpptest openann ${QT_LIBRARIES})
else()
  add_executable(TestSuite ${test_src})
  target_link_libraries(TestSuite cpptest openann)
endif()
