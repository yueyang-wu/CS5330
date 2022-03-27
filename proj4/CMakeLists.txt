cmake_minimum_required(VERSION 3.21)

# enable C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# set project name
project(proj4)

# config OpenCV
set(OpenCV_DIR "/usr/local/Cellar/opencv/4.5.5/lib/cmake/opencv4")
#set(OpenCV_DIR "/Users/yueyangwu/Desktop/CS5330/opencv_build")
find_package(OpenCV REQUIRED)

# config Aruco
#set(ARUCO_DIR     "/PATH/TO/aruco-1.2.5")
#set(ARUCO_INCLUDE "${ARUCO_DIR}/include")
#set(ARUCO_SRC     "${ARUCO_DIR}/src")
#set(ARUCO_LIB      aruco-1.2.5)

# config OpenGL
find_package(OpenGL REQUIRED)

# config glew
find_path(GLEW_INCLUDE_DIR GL/glew.h)
find_library(GLEW_LIBRARIES NAMES GLEW glew32 glew glew32s PATH_SUFFIXES lib64)
message("${GLEW_INCLUDE_DIR}")
message("${GLEW_LIBRARIES}")

# config glfw
#set(GLFW_DIR "/usr/local/Cellar/glfw/3.3.6/lib/cmake/glfw3/")
#find_package(glfw3 REQUIRED)
find_path(GLFW_INCLUDE_DIR GLFW/glfw3.h)
FIND_LIBRARY(GLFW_LIBRARIES NAMES glfw3 glfw)
message("${GLFW_INCLUDE_DIR}")
message("${GLFW_LIBRARIES}")

# set include directory (e.g. include header files)
set(PROJ4_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# set output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# include header files
include_directories(${PROJ4_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS} ${OPENGL_INCLUDE_DIRS} ${GLEW_INCLUDE_DIR} ${GLFW_INCLUDE_DIR})

add_executable(arSystem arSystem.cpp processors.cpp)
add_executable(harrisCorner harrisCorner.cpp processors.cpp)
add_executable(opengl_example opengl_example.cpp)
add_executable(aruco aruco.cpp)

# link dynamic libraries to this executable
target_link_libraries(arSystem ${OpenCV_LIBS})
target_link_libraries(harrisCorner ${OpenCV_LIBS})
target_link_libraries(opengl_example ${OpenCV_LIBS} ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${GLFW_LIBRARIES})
target_link_libraries(aruco ${OpenCV_LIBS})
