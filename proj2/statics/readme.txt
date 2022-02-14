// CS5330 Project 2
// Yueyang Wu

- Links/URLs to any videos you created and want to submit as part of your report.
    N/A

- The URL for your wiki report for this project.
    https://wiki.khoury.northeastern.edu/display/~yueyangwu/CS5330_Project2

- What operating system and IDE you used to run and compile your code.
    MacOS 12.1 (Monterey)
    CLion, CMake, and terminal

- Instructions for running your executables.
    - There are two executables. 'featureExtractor' is used to calculate the feature vectors
    of each image in the database and write the info to a csv file. 'imageRetriever' is used to
    calculate the feature vector for the target image and then get the top N matches.
    - 'featureExtractor'
        - Takes three input. The first is the path to a directory of images. The second is the feature set.
          The third is a path to the csv file where the feature vector for each image is written.
        - example: './featureExtractor ../../olympus mgtc multigaborcolor.csv'
    - 'imageRetriever'
        - Takes five input. The first is the path to the target image. The second is the feature set.
          The third is a path to the csv file where the feature vector for each image is written.
          The forth is distance metrics. The fifth is the desired number of matches N.
        - example: './imageRetriever ../../olympus/pic.0535.jpg mgtc multigaborcolor.csv sd 5'

- Instructions for testing any extensions you completed.
    I implemented the Gabor filter, and it can be tested in my executables using the corresponding code.

- Whether you are using any time travel days and how many.
    No.

- Code for features and distance metrics (used to run the executables)
    - features
        - b: baseline (task 1)
        - c: color histogram (task 2)
        - mc: multiple histograms of color (task 3)
        - t: Sobel texture
        - tc: Sobel texture and color (task 4)
        - midc: color on middle part of the image
        - midt: Sobel texture on middle part of the image
        - midtc: Sobel color and texture on middle part of the image (task 5)
        - gt: Gabor texture (extension)
        - gtc: Gabor texture and color (extension)
        - mgtc: multiple histograms of Gabor texture and color (extension)
        - midgtc: Gabor texture and color on middle part of the image (extension)
    - distance metric
        - sd: sum of square difference
        - hi: histogram intersection