//
// Created by Yueyang Wu on 2/4/22.
//

#include "distanceMetrics.h"
#include <opencv2/opencv.hpp>

using namespace cv;

/*
 * Compute the sum of square difference of two images.
 */
int sumOfSquareDifference(Mat &target, Mat &image) {
    int sum = 0;
    for (int i = 0; i < target.cols; i ++) {
        sum += (target.at<uchar>(0, i) - image.at<uchar>(0,  i)) * (target.at<uchar>(0, i) - image.at<uchar>(0,  i));
    }
    return sum;
}