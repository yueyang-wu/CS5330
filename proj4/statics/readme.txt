// CS5330 Project 4
// Yueyang Wu

- Links/URLs to any videos you created and want to submit as part of your report.
    Link for a demo video is included in the report

- The URL for your wiki report for this project.
   https://wiki.khoury.northeastern.edu/display/~yueyangwu/CS5330_Project4

- What operating system and IDE you used to run and compile your code.
    MacOS 12.1 (Monterey)
    CLion, CMake, and terminal

- Instructions for running your executables.
    - There are two executables: arSystem and harrisCorner
    - arSystem is able to detect chessboard corners and ArUco targets.
      It allows the user to select calibration pictures, and then calibrate the camera.
      It can put AR objects on the two targets at the same time, and it can also overlay a picture on the ArUco targets
        - takes one input, the file path to an image, the image is used to overlay the ArUco target
        - type 'p' to display/remove the image from ArUco targets
        - for the two target, the system calibrate the camera separately
        - type 's' to select calibration images for chessboard
        - type 'h' to select calibration images for ArUco target
        - type 'c' to calibrate camera for chessboard (at least 5 calibration image required)
        - type 'x' to calibrate camera for chessboard (at least 5 calibration image required)
        - once the camera is calibrated, it will put a virtual object on the target
        - type 'q' to quit the system
    - harrisCorners is able to detect corners and highlight corners
        - takes no input
        - type 'q' to quit the system

- Instructions for testing any extensions you completed.
    No extra steps are needed for testing the extensions. Details are in the instructions for running my executables.

- Whether received accommodation for the project or you are using any time travel days and how many.
    Yes, 3 time travel days.