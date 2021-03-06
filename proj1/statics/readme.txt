// CS5330 Project 1
// Yueyang Wu

- Links/URLs to any videos you created and want to submit as part of your report.
    Demo of a saved video sequence with effects: https://drive.google.com/file/d/1SBnRlHvLVfztqg9cmsJL5NudX_1_epCI/view?usp=sharing

- The URL for your wiki report for this project.
    https://wiki.khoury.northeastern.edu/display/~yueyangwu/CS5330_Project1

- What operating system and IDE you used to run and compile your code.
    MacOS 12.1 (Monterey)
    CLion, CMake, and terminal

- Instructions for running your executables.
    - imgDisplay
        - go to the directory of the executable
        - type "./imgDisplay {path of the image you want to display}" in the terminal
        - the program will display the image
    - vidDisplay
        - go to the directory of the executable
        - type "./vidDisplay" in the terminal
        - the program will access the camera on your device and display a live video
        - press keys to see different filters and effects

- Instructions for testing any extensions you completed.
    All of my extensions are in vidDisplay.cpp.

    - switch between the original video and effected video
        - press space key to display the original version

    - bilateral blur filter(for task 10)
        - press 'f' to apply the filter to the video

    - adjust brightness(for task 10)
        - press '1' to increase the brightness
        - press '2' to decrease the brightness
        - press '3' to return to the actual brightness

    - add meme to image/video saved(for extension)
        - when you press 's' to save an image, the program will ask you to write a meme for the image
        - write the meme in terminal and press the return key to save an image with a meme
        - if you don't want to write a meme, just press the return key

    - save a video sequence with effects(for extension)
        - press 'v' key to start to save a video sequence
        - press 'v' key again to stop saving the video sequence
        - the program will also ask for a meme before starting to save the video sequence

    - the image and video will be saved in the same directory of the executables