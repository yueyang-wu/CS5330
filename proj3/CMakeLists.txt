cmake_minimum_required(VERSION 3.21)

# enable C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# set project name
project(proj3)

# config OpenCV
set(OpenCV_DIR "/usr/local/Cellar/opencv/4.5.4_3/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)

# set include directory (e.g. include header files)
set(PROJ3_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# set output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# include header files
include_directories(${PROJ3_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})

add_executable(objectRecognizer objectRecognizer.cpp processors.cpp csv_util.cpp)

# link dynamic libraries to this executable
target_link_libraries(objectRecognizer ${OpenCV_LIBS})