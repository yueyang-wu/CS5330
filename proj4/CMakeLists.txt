cmake_minimum_required(VERSION 3.21)

# enable C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# set project name
project(proj4)

# config OpenCV
set(OpenCV_DIR "/usr/local/Cellar/opencv/4.5.5/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)

# set include directory (e.g. include header files)
set(PROJ4_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# set output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# include header files
include_directories(${PROJ4_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})

add_executable(arSystem arSystem.cpp processors.cpp)
add_executable(harrisCorner harrisCorner.cpp processors.cpp)

# link dynamic libraries to this executable
target_link_libraries(arSystem ${OpenCV_LIBS})
target_link_libraries(harrisCorner ${OpenCV_LIBS})
