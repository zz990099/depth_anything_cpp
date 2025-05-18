cmake_minimum_required(VERSION 3.8)
project(depth_anything_cpp)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(easy_deploy_tool)
add_subdirectory(stereo)

if (BUILD_TESTING)
  add_subdirectory(simple_tests)
endif()
