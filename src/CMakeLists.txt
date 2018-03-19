# feature_tracking_lib library
include_directories(${CMAKE_SOURCE_DIR}/include)
set(feature_tracking_srcs
  ${CMAKE_SOURCE_DIR}/include/feature_tracking/orb_tracker.h orb_tracker.cpp
)

set(libs_list ${OpenCV_LIBRARIES})
add_library(feature_tracking_lib ${feature_tracking_srcs})
target_link_libraries(feature_tracking_lib ${libs_list})

add_subdirectory(apps)
