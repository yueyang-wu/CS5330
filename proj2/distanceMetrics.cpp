//
// Created by Yueyang Wu on 2/4/22.
//

#include "distanceMetrics.h"
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

/*
 * Compute the sum of square difference of the two feature vectors.
 */
float sumOfSquareDifference(vector<float> &target, vector<float> &image) {
    float sum = 0;
    for (int i = 0; i < target.size(); i ++) {
        sum += (target[i] - image[i]) * (target[i] - image[i]);
    }
    return sum;
}

/*
 * Compute the histogram intersection of the two feature vectors.
 */
float histogramIntersection(vector<float> &target, vector<float> &image) {
    float intersection = 0;
    for (int i = 0; i < target.size(); i++) {
        intersection += (min(target[i], image[i]));
    }
    return intersection;
}