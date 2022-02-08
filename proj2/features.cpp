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