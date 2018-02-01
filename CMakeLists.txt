cmake_minimum_required(VERSION 3.8)
project(feature_tracking)

set(CMAKE_CXX_STANDARD 11)
set(EXECUTABLE_OUTPUT_PATH bin)

find_package(OpenCV REQUIRED)
add_subdirectory(src)
