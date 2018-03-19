# features_lib_lib library
include_directories(${CMAKE_SOURCE_DIR}/include)
set(features_lib_srcs
  ${CMAKE_SOURCE_DIR}/include/features/orb_features.h orb_features.cpp
)

set(libs_list ${OpenCV_LIBRARIES})
add_library(features_lib ${features_lib_srcs})
target_link_libraries(features_lib ${libs_list})

add_subdirectory(apps)
