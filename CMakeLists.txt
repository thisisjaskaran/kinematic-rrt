cmake_minimum_required( VERSION 3.0 )

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

SET(OpenCV_DIR /usr/local/lib/cmake/opencv4)

project(rrt_star)
find_package( OpenCV REQUIRED )
add_executable( rrt_star rrt_star.cpp )
target_link_libraries(rrt_star ${OpenCV_LIBS})