// CS5330 Project 3
// Yueyang Wu

- Links/URLs to any videos you created and want to submit as part of your report.
    Link for a demo video is included in the report

- The URL for your wiki report for this project.
   https://wiki.khoury.northeastern.edu/display/~yueyangwu/CS5330_Project3

- What operating system and IDE you used to run and compile your code.
    MacOS 12.1 (Monterey)
    CLion, CMake, and terminal

- Instructions for running your executables.
    - The executable takes two input. The first is the path to the csv file store the class name and feature vector for each known object.
    The second is the classifier type ('n' for the nearest neighbor, 'k' for KNN).
        - example: './objectRecognizer objectDB.csv n'
    - Enter training mode: type 't'. The system will switch back to inference mode after labeling all the objects in the video

- Instructions for testing any extensions you completed.
    No extra steps are needed for testing the extensions. Details are in the report.

- Whether received accommodation for the project or you are using any time travel days and how many.
    Yes, 2 days.

- Code for the classname (used in training mode)
    Since this system is designed to recognize a specific set of objects(the 14 objects listed in my report), to make the training mode
    easier to operate, it will ask the user to type a code(one letter) on the video window instead of typing the full name in the terminal.
    Here is the code to classname map:
    {'p', "pen"}, {'a', "alligator"}, {'h', "hammer"}, {'g', "glasses"},
    {'r', "round"}, {'c', "cat"}, {'b', "bone"}, {'k', "key"},
    {'m', "mouse"}, {'x', "binder clip"},
    {'w', "watch"}, {'s', "credit card"}, {'t', "spanner"} , {'y', "pliers"}