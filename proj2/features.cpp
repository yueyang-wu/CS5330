//
// Created by Yueyang Wu on 2/4/22.
//

#include <math.h>
#include <opencv2/opencv.hpp>
#include "features.h"

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
    Mat feature = Mat::zeros(3, histSize, CV_32F);

    // loop the image and build a 3D histogram
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            int b = image.at<Vec3b>(i, j)[0] / range;
            int g = image.at<Vec3b>(i, j)[1] / range;
            int r = image.at<Vec3b>(i, j)[2] / range;
            feature.at<float>(b, g, r)++;
        }
    }
    // L2 normalize the histogram
    normalize(feature, feature, 1, 0, NORM_L2, -1, Mat());

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
 * the max value for magnitude is sqrt(2) * max(sx, sy), which is approximately 1.4 * 255 = 400
 * the max value for orientation is -PI to PI
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
    Mat feature = Mat::zeros(2, histSize, CV_32F);

    // calculate the range in each bin
    float rangeMagnitude = 400 / 8.0;
    float rangeOrientation = 2 * CV_PI / 8.0;

    // loop the magnitude and orientation and build the 2D histogram
    for (int i = 0; i < imageMagnitude.rows; i++) {
        for (int j = 0; j < imageMagnitude.cols; j++) {
            int m = imageMagnitude.at<float>(i, j) / rangeMagnitude;
            int o = (imageOrientation.at<float>(i, j) + CV_PI) / rangeOrientation;
            feature.at<float>(m, o)++;
        }
    }

    // L2 normalize the histogram
    normalize(feature, feature, 1, 0, NORM_L2, -1, Mat());

    // convert the 2D histogram into a 1D vector
    return matToVector(feature);
}

/*
 * Given an image.
 * Calculate a 2D histogram of gradient orientation and magnitude and another 3D histogram of BGR color
 * concatenate the result of each part into a single 1D vector and return the vector
 */
vector<float> textureAndColor(Mat &image) {
    vector<float> feature = texture(image);
    vector<float> color = histogram(image);
    feature.insert(feature.end(), color.begin(), color.end());
    return feature;
}

/*
 * Given an image, convert it to grayscale.
 * Apply 48 gabor filters on it (5 scales and 16 orientations)
 * For each result, calculate the mean and standard deviation of it
 * Concatenate the result into a 1D vector and return the vector
 */
vector<float> gaborTexture(Mat &image) {
    vector<float> feature;

    // convert image to grayscale
    Mat grayscale;
    cvtColor(image, grayscale, COLOR_BGR2GRAY);

    // get gabor kernels and apply to the grayscale image
    float sigmaValue[] = {1.0, 2.0, 4.0};
    for (auto s : sigmaValue) {
        for (int k = 0; k < 16; k++) {
            float t = k * CV_PI / 8;
            Mat gaborKernel = getGaborKernel( Size(31,31), s, t, 10.0, 0.5, 0, CV_32F );
            Mat filteredImage;
            vector<float> hist(9, 0);
            filter2D(grayscale, filteredImage, CV_32F, gaborKernel);

            // calculate the mean and standard deviation of each filtered image
            Scalar mean, stddev;
            meanStdDev(filteredImage, mean, stddev);
            feature.push_back(mean[0]);
            feature.push_back(stddev[0]);
        }
    }

    // L2 normalize the feature vector
    normalize(feature, feature, 1, 0, NORM_L2, -1, Mat());

    return feature;
}

/*
 * Given an image.
 * Calculate a feature vector using gabor filters using `gaborTexture()`
 * Calculate another feature vector using color information using `histogram()`
 * Concatenate the two features and return the vector
 */
vector<float> gaborTextureAndColor(Mat &image) {
    vector<float> feature = gaborTexture(image);
    vector<float> color = histogram(image);
    feature.insert(feature.end(), color.begin(), color.end());
    return feature;

}

/*
 * Given an image
 * Split it into 2 x 2 grids
 * Calculate a feature vector for each part using `gaborTextureAndColor()`
 * Concatenate the result into a 1D vector and return it
 */
vector<float> multiGaborTextureAndColor(Mat &image) {
    vector<float> feature;
    int x = image.cols / 2, y = image.rows / 2;
    int topX[] = {0, x};
    int topY[] = {0, y};
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            Mat m = image(Rect(topX[i], topY[j], x, y)).clone(); // get ROI
            vector<float> v = gaborTextureAndColor(m); // calculate feature vector
            feature.insert(feature.end(), v.begin(), v.end()); // concatenate
        }
    }
    return feature;
}

/*
 * Take a single-channel image
 * Compute sobelX, treat pixels off the edge as having asymmetric reflection over that edge
 * horizontal filter [-1, 0, 1], vertical filter [1, 2, 1]
 */
Mat sobelX(Mat &image) {
    Mat dst = Mat::zeros(image.size(), CV_32F);
    Mat temp = Mat::zeros(image.size(), CV_32F);

    // apply horizontal filter
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            if (j > 0 && j < image.cols - 1) {
                temp.at<float>(i, j) = -image.at<uchar>(i, j - 1) + image.at<uchar>(i, j + 1);
            }
        }
    }
    // apply vertical filter
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
    Mat dst = Mat::zeros(image.size(), CV_32F);
    Mat temp = Mat::zeros(image.size(), CV_32F);

    // apply horizontal filter
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
    // apply vertical filter
    for (int i = 0; i < temp.rows; i++) {
        for (int j = 0; j < temp.cols; j++) {
            if (i > 0 && i < temp.rows - 1) {
                dst.at<float>(i, j) = -temp.at<float>(i - 1, j) + temp.at<float>(i + 1, j);
            }
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

    // calculate orientation
    Mat dst(image.size(), CV_32F);
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            dst.at<float>(i, j) = atan2(sy.at<float>(i, j), sx.at<float>(i, j));
        }
    }

    return dst;
}

/*
 * Given an image.
 * Treat it as a 3 * 3 grid, and take the middle part
 */
Mat getMiddle(Mat &image) {
    int x = image.cols / 3, y = image.rows / 3;
    Mat middle = image(Rect(x, y, x, y)).clone();
    return middle;
}

/*
 * Convert a Mat to a 1D vector<float>
 */
vector<float> matToVector(Mat &m) {
    Mat flat = m.reshape(1, m.total() * m.channels());
    flat.convertTo(flat, CV_32F);
    return m.isContinuous() ? flat : flat.clone();
}