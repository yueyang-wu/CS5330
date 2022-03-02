#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "processors.h"

using namespace cv;
using namespace std;

int main() {
    // read an image
    string fp = "/Users/yueyangwu/Desktop/test.png";
    Mat image = imread(fp, 1);
    namedWindow("Original Image", WINDOW_AUTOSIZE);
    imshow("Original Image", image);

    if (!image.data) {
        cout << "No image data\n";
        return -1;
    }

    // threshold
    namedWindow("Threshold", WINDOW_AUTOSIZE);
    Mat thresholdImage = threshold(image);
    imshow("Threshold", thresholdImage);

    // clean up the image
    Mat cleanupImage = cleanup(thresholdImage);
    namedWindow("Clean Up", WINDOW_AUTOSIZE);
    imshow("Clean Up", cleanupImage);

    // get the region
    Mat stats, centroids;
    Mat regionImage = getRegions(cleanupImage, stats, centroids);
    namedWindow("Region", WINDOW_AUTOSIZE);
    imshow("Region", regionImage);


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