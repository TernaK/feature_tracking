# include_directories(${OpenCV_INCLUDE_})
add_executable(orb_tracking orb_tracking.cpp)
target_link_libraries(orb_tracking ${OpenCV_LIBS})
