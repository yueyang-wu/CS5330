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
    int x = image.cols / 2 - 4, y = image.rows / 2 - 4;
    middle9X9 = image(Rect(x, y, 9, 9)).clone();

    // convert the 9 x 9 mat to a 1D vector
    return matToVector(middle9X9);
}

/*
 * Given an image.
 * Use the whole image RGB histogram with 8 bins for each of RGB as the feature vector.
 * Return the feature vector.
 */
vector<float> histogram(Mat &image) {
    int range = 256 / 8; // calculate the range in each bin

    // initialize a 3D mat
    int histSize[] = {8, 8, 8};
    Mat feature = Mat::zeros(3, histSize, CV_32S);

    // loop the image and build a 3D histogram
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            int b = image.at<Vec3b>(i, j)[0] / range;
            int g = image.at<Vec3b>(i, j)[1] / range;
            int r = image.at<Vec3b>(i, j)[2] / range;
            feature.at<int>(b, g, r)++;
        }
    }

    // convert the 3D histogram into a 1D vector
    return matToVector(feature);
}

/*
 * Given an image.
 * Split it into 2 x 2 grids
 * Calculate the histogram for each part, using RGB histogram with 8 bins for each of RGB
 * concatenate the result of each part into a singel 1D vector and return the vector
 */
vector<float> multiHistogram(Mat &image) {
    vector<float> feature;
    int x = image.cols / 2, y = image.rows / 2;
    int topX[] = {0, x};
    int topY[] = {0, y};
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            Mat m = image(Rect(topX[i], topY[j], x, y)).clone(); // get ROI
            vector<float> v = histogram(m); // calculate feature vector
            feature.insert(feature.end(), v.begin(), v.end()); // concatenate
        }
    }
    return feature;
}

/*
 * Convert a Mat to a 1D vector<float>
 */
vector<float> matToVector(Mat &m) {
    Mat flat = m.reshape(1, m.total() * m.channels());
    flat.convertTo(flat, CV_32F);
    return m.isContinuous() ? flat : flat.clone();
}


// codes from project1
int sobelX3x3(const Mat &src, Mat &dst) {
    // build the filters
    int horizontalFilterArray[] = {-1, 0, 1};
    int verticalFilterArray[] = {1, 2, 1};
    Mat horizontalFilter(1, 3, CV_16SC1, horizontalFilterArray);
    Mat verticalFilter(1, 3, CV_8UC1, verticalFilterArray);

    // both src and dst of filter1xN and filterNx1 should be type CV_32FC3
    Mat src32FC3;
    src.convertTo(src32FC3, CV_32FC3);
    Mat temp(src.size(), CV_32FC3);
    Mat dst32FC3(src.size(), CV_32FC3);

    // apply horizontal filter
    filter1xN(src32FC3, temp, horizontalFilter, 1);
    // apply vertical filter
    filterNx1(temp, dst32FC3, verticalFilter, 4);

    // convert the dst type to CV_16SC3 otherwise there be problem with the pointer
    dst32FC3.convertTo(dst, CV_16SC3);

    return 0;
}

int sobelY3x3(const Mat &src, Mat &dst) {
    // build the filters
    int horizontalFilterArray[] = {1, 2, 1};
    int verticalFilterArray[] = {1, 0, -1};
    Mat horizontalFilter(1, 3, CV_16SC1, horizontalFilterArray);
    Mat verticalFilter(1, 3, CV_8UC1, verticalFilterArray);

    // both src and dst of filter1xN and filterNx1 should be type CV_32FC3
    Mat src32FC3;
    src.convertTo(src32FC3, CV_32FC3);
    Mat temp(src.size(), CV_32FC3);
    Mat dst32FC3(src.size(), CV_32FC3);

    // apply horizontal filter
    filter1xN(src32FC3, temp, horizontalFilter, 4);
    // apply vertical filter
    filterNx1(temp, dst32FC3, verticalFilter, 1);

    // convert the dst type to CV_16SC3 otherwise there be problem with the pointer
    dst32FC3.convertTo(dst, CV_16SC3);

    return 0;
}

int magnitude(const Mat &sx, const Mat &sy, Mat &dst) {
    // to use cv::sqrt, the input and output type need to be CV_32FC3
    Mat sx32FC3, sy32FC3, dst32FC3;
    sx.convertTo(sx32FC3, CV_32FC3);
    sy.convertTo(sy32FC3, CV_32FC3);

    sqrt(sx32FC3.mul(sx32FC3) + sy32FC3.mul(sy32FC3), dst32FC3);

    // convert the dst type back to CV_16SC3
    dst32FC3.convertTo(dst, CV_16SC3);

    return 0;
}

/*
 * filter1xN and filterNx1 are two helper functions
 * Since the filters will contain both positive and negative integers and
 * there will be different types of src and dst from the functions using these helper functions,
 * the two helper functions will only input/output CV_32FC3 type, which is a more general type
 * the functions using these helpers should convert the src and dst types according to the actual needs
 */
int filter1xN(const Mat &src, Mat &dst, Mat &filter, int normalizer) {
    // validate input
    CV_Assert(filter.channels() == 1);
    CV_Assert(filter.rows == 1); // assume filter is 1d array
    CV_Assert(dst.size() == src.size());
    CV_Assert(src.type() == CV_32FC3 && dst.type() == CV_32FC3);

    // apply the 1xN filter and normalize the result
    // treat pixels off the edge as having asymmetric reflection over that edge
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
    // validate input
    CV_Assert(filter.channels() == 1);
    CV_Assert(filter.rows == 1); // assume filter is 1d array
    CV_Assert(dst.size() == src.size());
    CV_Assert(src.type() == CV_32FC3 && dst.type() == CV_32FC3);

    // apply the Nx1 filter and normalize the result
    // treat pixels off the edge as having asymmetric reflection over that edge
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