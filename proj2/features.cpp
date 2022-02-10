//
// Created by Yueyang Wu on 2/4/22.
//

#include "features.h"
#include <opencv2/opencv.hpp>
#include <math.h>

#define PI 3.14

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

    // initialize a 3D histogram
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
    // L2 normalize the histogram
//    normalize(feature, feature);

    // convert the 3D histogram into a 1D vector
    return matToVector(feature);
}

/*
 * Given an image.
 * Split it into 2 x 2 grids
 * Calculate the histogram for each part, using RGB histogram with 8 bins for each of RGB
 * concatenate the result of each part into a single 1D vector and return the vector
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
 * Given an image.
 * Convert it to grayscale and compute a 2D histogram of gradient magnitude and orientation
 * Using 8 bins for each dimension
 * the max value for magnitude is sqrt(2) * max(sx, sy) which is approximately 1.4 * 255 = 400
 * the max value for orientation is 2PI
 */
vector<float> texture(Mat &image) {
    // convert image to grayscale
    Mat grayscale;
    cvtColor(image, grayscale, COLOR_BGR2GRAY);

    // calculate gradient magnitude on grayscale
    Mat imageMagnitude = magnitude(grayscale);

    // calculate gradient orientation on grayscale
    Mat imageOrientation = orientation(grayscale);

    // initialize a 2D histogram
    int histSize[] = {8, 8};
    Mat feature = Mat::zeros(2, histSize, CV_32S);

    // calculate the range in each bin
    float rangeMagnitude = 400 / 8.0;
    float rangeOrientation = 2 * PI / 8.0;

//    cout << "orientation size: " << imageOrientation.size() << endl;
//    cout << "magnitude size: " << imageMagnitude.size() << endl;

    // loop the magnitude and orientation and build the 2D histogram
    for (int i = 0; i < imageMagnitude.rows; i++) {
        for (int j = 0; j < imageMagnitude.cols; j++) {
            int m = imageMagnitude.at<float>(i, j) / rangeMagnitude;
            int o = (imageOrientation.at<float>(i, j) + PI) / rangeOrientation;
//            cout << "i: " << i << "j: " << j << endl;
//            cout << "m: " << m << "o: " << o << endl;
//            cout << "imageOrientation.at<float>(i, j): " << imageOrientation.at<float>(i, j) << endl;
//            cout << "PI: " << PI << endl;
//            cout << "rangeOrientation: " << rangeOrientation << endl;
//            cout << "imageMagnitude.at<float>(i, j): " << imageMagnitude.at<float>(i, j) << endl;
//            cout << "rangeMagnitude: " << rangeMagnitude << endl;
            feature.at<int>(m, o)++;
        }
    }

//    cout << "after loop" << endl;

    // L2 normalize the histogram
//    normalize(feature, feature);

    // convert the 2D histogram into a 1D vector
    return matToVector(feature);
}

/*
 * Given an image.
 * Calculate a histogram of gradient orientation and magnitude and another histogram of BGR color
 * concatenate the result of each part into a single 1D vector and return the vector
 */
vector<float> textureAndColor(Mat &image) {
    vector<float> feature = texture(image);
    vector<float> color = histogram(image);
    feature.insert(feature.end(), color.begin(), color.end());
    return feature;
}

/*
 * Take a single-channel image
 * Compute sobelX, treat pixels off the edge as having asymmetric reflection over that edge
 * horizontal filter [-1, 0, 1], vertical filter [1, 2, 1]
 */
Mat sobelX(Mat &image) {
//    Mat dst(image.size(), CV_32F);
//    Mat temp(image.size(), CV_32F);
    Mat dst = Mat::zeros(image.size(), CV_32F);
    Mat temp = Mat::zeros(image.size(), CV_32F);
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            if (j > 0 && j < image.cols - 1) {
                temp.at<float>(i, j) = -image.at<uchar>(i, j - 1) + image.at<uchar>(i, j + 1);
            }
        }
    }
    for (int i = 0; i < temp.rows; i++) {
        for (int j = 0; j < temp.cols; j++) {
            if (i == 0) {
                dst.at<float>(i, j) = (temp.at<float>(i + 1, j) + 2 * temp.at<float>(i, j) + temp.at<float>(i + 1, j)) / 4;
            } else if (i == temp.rows - 1) {
                dst.at<float>(i, j) = (temp.at<float>(i - 1, j) + 2 * temp.at<float>(i, j) + temp.at<float>(i - 1, j)) / 4;
            } else {
                dst.at<float>(i, j) = (temp.at<float>(i - 1, j) + 2 * temp.at<float>(i, j) + temp.at<float>(i + 1, j)) / 4;
            }
        }
    }
    return dst;
}

/*
 * Take a single-channel image
 * Compute sobelY, treat pixels off the edge as having asymmetric reflection over that edge
 * horizontal [1, 2, 1], vertical [-1, 0, 1]
 */
Mat sobelY(Mat &image) {
//    Mat dst(image.size(), CV_32F);
//    Mat temp(image.size(), CV_32F);
    Mat dst = Mat::zeros(image.size(), CV_32F);
    Mat temp = Mat::zeros(image.size(), CV_32F);

    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            if (j == 0) {
                temp.at<float>(i, j) = (image.at<uchar>(i, j + 1) + 2 * image.at<uchar>(i, j) + image.at<uchar>(i, j + 1)) / 4;
            } else if (j == image.cols - 1) {
                temp.at<float>(i, j) = (image.at<uchar>(i, j - 1) + 2 * image.at<uchar>(i, j) + image.at<uchar>(i, j - 1)) / 4;
            } else {
                temp.at<float>(i, j) = (image.at<uchar>(i, j - 1) + 2 * image.at<uchar>(i, j) + image.at<uchar>(i, j + 1)) / 4;
            }
        }
    }
    for (int i = 0; i < temp.rows; i++) {
        for (int j = 0; j < temp.cols; j++) {
            if (i > 0 && i < temp.rows - 1) {
                dst.at<float>(i, j) = -temp.at<float>(i - 1, j) + temp.at<float>(i + 1, j);
            }
//
//            if (isnan(dst.at<float>(i, j))) {
//                cout << "i: " << i << ", j: " << j << endl;
//                cout << "dst: " << dst.at<float>(i, j) << endl;
//            }
        }
    }
    return dst;
}

/*
 * Take a single-channel image,
 * calculate the gradient magnitude of it.
 */
Mat magnitude(Mat &image) {
    // calculate sobelX and sobelY
    Mat sx = sobelX(image);
    Mat sy = sobelY(image);

    // calculate gradient magnitude
    Mat dst;
    sqrt(sx.mul(sx) + sy.mul(sy), dst);

    return dst;
}

/*
 * Take a single-channel image,
 * calculate the gradient orientation of it.
 */
Mat orientation(Mat &image) {
    // calculate sobelX and sobelY
    Mat sx = sobelX(image);
    Mat sy = sobelY(image);

    Mat dst(image.size(), CV_32F);
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            dst.at<float>(i, j) = atan2(sy.at<float>(i, j), sx.at<float>(i, j));
//            if (isnan(dst.at<float>(i, j))) {
//                cout << "i: " << i << ", j: " << j << endl;
//                cout << "sy: " << sy.at<float>(i, j) << endl;
//                cout << "sx: " << sx.at<float>(i, j) << endl;
//                cout << "dst: " << dst.at<float>(i, j) << endl;
//            }
        }
    }

    return dst;
}

/*
 * Convert a Mat to a 1D vector<float>
 */
vector<float> matToVector(Mat &m) {
    Mat flat = m.reshape(1, m.total() * m.channels());
    flat.convertTo(flat, CV_32F);
    return m.isContinuous() ? flat : flat.clone();
}