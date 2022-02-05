//
// Created by Yueyang Wu on 2/4/22.
//

#ifndef PROJ2_FEATURES_H
#define PROJ2_FEATURES_H

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

/*
 * Given an image.
 * Use the 9x9 square in the middle of the image as a feature vector.
 * Return the feature vector.
 */
vector<float> baseline(Mat &image);

#endif //PROJ2_FEATURES_H
