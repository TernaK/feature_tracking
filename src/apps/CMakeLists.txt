add_executable(tracking tracking.cpp)
target_link_libraries(tracking features_lib ${libs_list})

add_executable(plane plane.cpp)
target_link_libraries(plane features_lib ${libs_list})
