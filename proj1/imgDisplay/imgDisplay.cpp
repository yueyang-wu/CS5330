//
// Created by Yueyang Wu on 1/24/22.
//
#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main(int argc, char *argv[]) {
    // read an image
    Mat image;
    image = imread(argv[1], 1);

    // validate image data
    if (!image.data) {
        cout << "No image data\n";
        return -1;
    }

    // display image
    namedWindow("Project 1 - Task 1", WINDOW_AUTOSIZE);
    imshow("Project 1 -Task 1", image);

    // check for keypress.
    while (true) {
        char key = waitKey(10);
        // if user types 'q', quit
        if (key == 'q') {
            break;
        }
    }

    // destroy all the windows created
    destroyAllWindows();

    return 0;
}

