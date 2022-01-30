//
// Created by Yueyang Wu on 1/28/22.
//

#include "filters.h"
#include <opencv2/opencv.hpp>
//#include <math.h>

using namespace cv;
using namespace std;


int greyscale(const Mat &src, Mat &dst) {
    dst.create(src.size(), CV_8UC1);
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            dst.at<uchar>(i, j) = (src.at<Vec3b>(i, j)[0] + src.at<Vec3b>(i, j)[1] + src.at<Vec3b>(i, j)[2]) / 3;
        }
    }
    return 0;
}

int blur5x5(const Mat &src, Mat &dst) {
    int filterArray[] = {1, 2, 4, 2, 1};
    Mat filter(1, 5, CV_8UC1, filterArray);

    Mat src32FC3;
    src.convertTo(src32FC3, CV_32FC3);
    Mat temp(src.size(), CV_32FC3);
    Mat dst32FC3(src.size(), CV_32FC3);

    // apply [1 2 4 2 1] horizontally
    filter1xN(src32FC3, temp, filter, 10);
    // apply [1 2 4 2 1] vertically
    filterNx1(temp, dst32FC3, filter, 10);

    dst32FC3.convertTo(dst, CV_8UC3);

    return 0;
}

int sobelX3x3(const Mat &src, Mat &dst) {
    int horizontalFilterArray[] = {-1, 0, 1};
    int verticalFilterArray[] = {1, 2, 1};
    Mat horizontalFilter(1, 3, CV_16SC1, horizontalFilterArray);
    Mat verticalFilter(1, 3, CV_8UC1, verticalFilterArray);

    Mat src32FC3;
    src.convertTo(src32FC3, CV_32FC3);
    Mat temp(src.size(), CV_32FC3);
    Mat dst32FC3(src.size(), CV_32FC3);

    // apply horizontal filter
    filter1xN(src32FC3, temp, horizontalFilter, 1);
    // apply vertical filter
    filterNx1(temp, dst32FC3, verticalFilter, 4);

    dst32FC3.convertTo(dst, CV_16SC3);

    return 0;
}

int sobelY3x3(const Mat &src, Mat &dst) {
    int horizontalFilterArray[] = {1, 2, 1};
    int verticalFilterArray[] = {1, 0, -1};
    Mat horizontalFilter(1, 3, CV_16SC1, horizontalFilterArray);
    Mat verticalFilter(1, 3, CV_8UC1, verticalFilterArray);

    Mat src32FC3;
    src.convertTo(src32FC3, CV_32FC3);
    Mat temp(src.size(), CV_32FC3);
    Mat dst32FC3(src.size(), CV_32FC3);

    // apply horizontal filter
    filter1xN(src32FC3, temp, horizontalFilter, 4);
    // apply vertical filter
    filterNx1(temp, dst32FC3, verticalFilter, 1);

    dst32FC3.convertTo(dst, CV_16SC3);
    return 0;
}

int magnitude(const Mat &sx, const Mat &sy, Mat &dst) {
    Mat sx32FC3, sy32FC3;
    sx.convertTo(sx32FC3, CV_32FC3);
    sy.convertTo(sy32FC3, CV_32FC3);

    Mat dst32FC3;
    sqrt(sx32FC3.mul(sx32FC3) + sy32FC3.mul(sy32FC3), dst32FC3);
    dst32FC3.convertTo(dst, CV_16SC3);
    return 0;
}

int blurQuantize(const Mat &src, Mat &dst, int levels) {
    // blur the image
    blur5x5(src, dst);

    int b = 255 / levels;
    for (int i = 0; i < dst.rows; i++) {
        for (int j = 0; j < dst.cols; j++) {
            for (int k = 0; k <= 2; k++) {
                dst.at<Vec3b>(i, j)[k] = dst.at<Vec3b>(i, j)[k] / b * b;
            }
        }
    }

    return 0;
}

int cartoon(const Mat &src, Mat &dst, int levels, int magThreshold) {
    Mat sx, sy;
    sobelX3x3(src, sx);
    sobelY3x3(src, sy);
    Mat mag;
    magnitude(sx, sy, mag);

    blurQuantize(src, dst, levels);

    for (int i = 0; i < mag.rows; i++) {
        for (int j = 0; j < mag.cols; j++) {
            if (mag.at<Vec3s>(i, j)[0] > magThreshold || mag.at<Vec3s>(i, j)[1] > magThreshold || mag.at<Vec3s>(i, j)[2] > magThreshold) {
                dst.at<Vec3b>(i, j)[0] = 0;
                dst.at<Vec3b>(i, j)[1] = 0;
                dst.at<Vec3b>(i, j)[2] = 0;
            }
        }
    }
    return 0;
}

int filter1xN(const Mat &src, Mat &dst, Mat &filter, int normalizer) {
    CV_Assert(filter.channels() == 1);
    CV_Assert(filter.rows == 1); // assume filter is 1xN array
    CV_Assert(dst.size() == src.size());
    CV_Assert(src.type() == CV_32FC3 && dst.type() == CV_32FC3);
    int radius = filter.cols / 2;
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            int blue = 0, green = 0, red = 0;
            for (int k = -radius; k <= radius; k++) {
                if (j + k < 0 || j + k >= src.cols) {
                    blue += filter.at<int>(radius + k) * src.at<Vec3f>(i, j - k)[0];
                    green += filter.at<int>(radius + k) * src.at<Vec3f>(i, j - k)[1];
                    red += filter.at<int>(radius + k) * src.at<Vec3f>(i, j - k)[2];
                } else {
                    blue += filter.at<int>(radius + k) * src.at<Vec3f>(i, j + k)[0];
                    green += filter.at<int>(radius + k) * src.at<Vec3f>(i, j + k)[1];
                    red += filter.at<int>(radius + k) * src.at<Vec3f>(i, j + k)[2];
                }
            }
            dst.at<Vec3f>(i, j)[0] = blue / normalizer;
            dst.at<Vec3f>(i, j)[1] = green / normalizer;
            dst.at<Vec3f>(i, j)[2] = red / normalizer;
        }
    }
    return 0;
}

int filterNx1(const Mat &src, Mat &dst, Mat &filter, int normalizer) {
    CV_Assert(filter.channels() == 1);
    CV_Assert(filter.rows == 1); // assume filter is 1xN array
    CV_Assert(dst.size() == src.size());
    CV_Assert(src.type() == CV_32FC3 && dst.type() == CV_32FC3);
    int radius = filter.cols / 2;
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            int blue = 0, green = 0, red = 0;
            for (int k = -radius; k <= radius; k++) {
                if (i + k < 0 || i + k > src.rows) {
                    blue += filter.at<int>(radius + k) * src.at<Vec3f>(i - k, j)[0];
                    green += filter.at<int>(radius + k) * src.at<Vec3f>(i - k, j)[1];
                    red += filter.at<int>(radius + k) * src.at<Vec3f>(i - k, j)[2];
                } else {
                    blue += filter.at<int>(radius + k) * src.at<Vec3f>(i + k, j)[0];
                    green += filter.at<int>(radius + k) * src.at<Vec3f>(i + k, j)[1];
                    red += filter.at<int>(radius + k) * src.at<Vec3f>(i + k, j)[2];
                }
            }
            dst.at<Vec3f>(i, j)[0] = blue / normalizer;
            dst.at<Vec3f>(i, j)[1] = green / normalizer;
            dst.at<Vec3f>(i, j)[2] = red / normalizer;
        }
    }
    return 0;
}