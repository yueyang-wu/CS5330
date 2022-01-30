project(imgDisplay)

# include opencv header files
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(imgDisplay imgDisplay.cpp)

# link dynamic libraries to this executable
target_link_libraries(imgDisplay ${OpenCV_LIBS})