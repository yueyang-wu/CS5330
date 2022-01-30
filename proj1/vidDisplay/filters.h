//
// Created by Yueyang Wu on 1/28/22.
//

#ifndef PROJ1_FILTERS_H
#define PROJ1_FILTERS_H

#include <opencv2/opencv.hpp>

using namespace cv;

int greyscale(const Mat &src, Mat &dst);
int blur5x5(const Mat &src, Mat &dst);
int sobelX3x3(const Mat &src, Mat &dst);
int sobelY3x3(const Mat &src, Mat &dst);
int magnitude(const Mat &sx, const Mat &sy, Mat &dst);
int blurQuantize(const Mat &src, Mat &dst, int levels);
int cartoon(const Mat &src, Mat &dst, int levels, int magThreshold);
int filter1xN(const Mat &src, Mat &dst, Mat &filter, int normalizer);
int filterNx1(const Mat &src, Mat &dst, Mat &filter, int normalizer);

#endif //PROJ1_FILTERS_H
