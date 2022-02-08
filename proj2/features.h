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

/*
 * Given an image.
 * Use the whole image RGB histogram with 8 bins for each of RGB as the feature vector.
 * Return the feature vector.
 */
vector<float> histogram(Mat &image);

/*
 * Given an image.
 * Split it into 2 x 2 grids
 * Calculate the histogram for each part, using RGB histogram with 8 bins for each of RGB
 * return the result as a single 1D vector
 */
vector<float> multiHistogram(Mat &image);

/*
 * Convert a Mat to a 1D vector
 */
vector<float> matToVector(Mat &m);

#endif //PROJ2_FEATURES_H
