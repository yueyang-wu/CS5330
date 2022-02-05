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
vector<float> baseline(Mat &image) {
    Mat middle9X9;
    vector<float> featureVector;
    int x = image.cols / 2 - 4, y = image.rows / 2 - 4;
    middle9X9 = image(Rect(x, y, 9, 9)).clone();

    Mat flat = middle9X9.reshape(1, middle9X9.total() * middle9X9.channels());
    flat.convertTo(flat, CV_32F);
    featureVector = middle9X9.isContinuous() ? flat : flat.clone();

    return featureVector;
}