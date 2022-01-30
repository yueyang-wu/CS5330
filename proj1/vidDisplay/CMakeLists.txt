project(vidDisplay)

# include header files
include_directories(${PROJ1_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})

add_executable(vidDisplay vidDisplay.cpp filters.cpp)

# link dynamic libraries to this executable
target_link_libraries(vidDisplay ${OpenCV_LIBS})