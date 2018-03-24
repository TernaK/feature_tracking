add_executable(tracking tracking.cpp)
target_link_libraries(tracking features_lib ${libs_list})

add_executable(stereo stereo.cpp)
target_link_libraries(stereo features_lib ${libs_list})

add_executable(capture_chessboard capture_chessboard.cpp)
target_link_libraries(capture_chessboard features_lib ${libs_list})

add_executable(stereo_calibrate stereo_calibrate.cpp)
target_link_libraries(stereo_calibrate features_lib ${libs_list})
