//
// Created by Yueyang Wu on 2/4/22.
//

#include <opencv2/opencv.hpp>
#include "distanceMetrics.h"

using namespace cv;
using namespace std;

/*
 * Compute the sum of square difference of the two feature vectors.
 */
float sumOfSquareDifference(vector<float> &target, vector<float> &image) {
    CV_Assert(target.size() == image.size()); // two features should have the same size
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
    CV_Assert(target.size() == image.size()); // two features should have the same size
    float intersection = 0;
    for (int i = 0; i < target.size(); i++) {
        intersection += (min(target[i], image[i]));
    }
    return intersection;
}