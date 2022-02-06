//
// Created by Yueyang Wu on 2/4/22.
//

#ifndef PROJ2_DISTANCEMETRICS_H
#define PROJ2_DISTANCEMETRICS_H

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

/*
 * Compute the sum of square difference of two images.
 */
float sumOfSquareDifference(vector<float> &target, vector<float> &image);


#endif //PROJ2_DISTANCEMETRICS_H
