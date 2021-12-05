cmake_minimum_required( VERSION 3.0 )

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

SET(OpenCV_DIR /usr/local/lib/cmake/opencv4)

project(rrt_star)

find_package( OpenCV REQUIRED )
find_package (Eigen3 3.4 REQUIRED NO_MODULE)

add_executable( rrt_star rrt_star_3.cpp )

target_link_libraries(rrt_star ${OpenCV_LIBS})
target_link_libraries (rrt_star Eigen3::Eigen)