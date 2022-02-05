//
// Created by Yueyang Wu on 2/4/22.
//

#include "features.h"
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

/*
 * Given an image.
 * Use the 9x9 square in the middle of the image as a feature vector.
 * Return the feature vector.
 */
Mat baseline(Mat &image) {
    Mat featureVector;
    int x = image.cols / 2 - 4, y = image.rows / 2 - 4;
    featureVector = image(Rect(x, y, 9, 9)).clone(); // ????

    featureVector = featureVector.reshape(1, featureVector.total() * featureVector.channels());

    return featureVector;
}