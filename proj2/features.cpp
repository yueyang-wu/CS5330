//
// Created by Yueyang Wu on 2/4/22.
//

#include "features.h"
#include <opencv2/opencv.hpp>
#include <math.h>

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
//    Mat normalizedFeature(feature, CV_32F);
//    normalize(feature, normalizedFeature);

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
    Mat feature = Mat::zeros(2, histSize, CV_32F);

    // calculate the range in each bin
    float rangeMagnitude = 400 / 8.0;
    float rangeOrientation = 2 * CV_PI / 8.0;

//    cout << "orientation size: " << imageOrientation.size() << endl;
//    cout << "magnitude size: " << imageMagnitude.size() << endl;

    // loop the magnitude and orientation and build the 2D histogram
    for (int i = 0; i < imageMagnitude.rows; i++) {
        for (int j = 0; j < imageMagnitude.cols; j++) {
            int m = imageMagnitude.at<float>(i, j) / rangeMagnitude;
            int o = (imageOrientation.at<float>(i, j) + CV_PI) / rangeOrientation;
//            cout << "i: " << i << "j: " << j << endl;
//            cout << "m: " << m << "o: " << o << endl;
//            cout << "imageOrientation.at<float>(i, j): " << imageOrientation.at<float>(i, j) << endl;
//            cout << "PI: " << PI << endl;
//            cout << "rangeOrientation: " << rangeOrientation << endl;
//            cout << "imageMagnitude.at<float>(i, j): " << imageMagnitude.at<float>(i, j) << endl;
//            cout << "rangeMagnitude: " << rangeMagnitude << endl;
            feature.at<float>(m, o)++;
        }
    }

//    cout << "after loop" << endl;

    // L2 normalize the histogram
    normalize(feature, feature, 1, 0, NORM_L2, -1, Mat());

    // convert the 2D histogram into a 1D vector
    return matToVector(feature);
}

Mat getMiddle(Mat &image) {
    int x = image.cols / 3, y = image.rows / 3;
    Mat middle = image(Rect(x, y, x, y)).clone();
    return middle;
}

vector<float> middleTextureAndColor(Mat &image) {
    Mat middle = getMiddle(image);
    return textureAndColor(middle);
}

/*
 * Given an image.
 * Calculate a histogram of gradient orientation and magnitude and another histogram of BGR color
 * concatenate the result of each part into a single 1D vector and return the vector
 */
vector<float> textureAndColor(Mat &image) {
    vector<float> feature = texture(image);
    vector<float> color = histogram(image);
    for (int i = 0; i < color.size(); i++) {
        color[i] /= 2;
    }
    feature.insert(feature.end(), color.begin(), color.end());
    return feature;
}

vector<float> custom(Mat &image) {
    vector<float> feature;
    Mat hsv, hsvThreshold;
    cvtColor(image, hsv, COLOR_BGR2HSV);

    namedWindow("image", WINDOW_AUTOSIZE);
    imshow("image", image);

    inRange(hsv, Scalar(20, 100, 100), Scalar(30, 255,  255), hsvThreshold);

    namedWindow("hsvThreshold", WINDOW_AUTOSIZE);
    imshow("hsvThreshold", hsvThreshold);
    waitKey(0);
    destroyAllWindows();
    return feature;
}

vector<float> gaborTexture(Mat &image) {
    vector<float> feature;
    Mat grayscale;
    cvtColor(image, grayscale, COLOR_BGR2GRAY);

    float sigmaValue[] = {1.0, 2.0, 4.0};
//    float thetaValue[] = {-3 * CV_PI / 4, -CV_PI / 2, -CV_PI / 4, 0, CV_PI / 4, CV_PI / 2, 3 * CV_PI / 4, CV_PI};
    for (auto s : sigmaValue) {
        for (int k = 0; k < 16; k++) {
            float t = k * CV_PI / 8;
//            cout << s << ", " << t << endl;
            Mat gaborKernel = getGaborKernel( Size(31,31), s, t, 10.0, 0.5, 0, CV_32F );
            Mat filteredImage;
            vector<float> hist(9, 0);
            filter2D(grayscale, filteredImage, CV_32F, gaborKernel);

            Scalar mean, stddev;
            meanStdDev(filteredImage, mean, stddev);
            feature.push_back(mean[0]);
            feature.push_back(stddev[0]);

            // apply non-linear transformation tanh(a * value), the result will be -1 to 1
            // Reference: https://www.ee.columbia.edu/~sfchang/course/dip-S06/handout/jain-texture.pdf
//            for (int i = 0; i < filteredImage.rows; i++) {
//                for (int j = 0; j < filteredImage.cols; j++) {
//                    filteredImage.at<float>(i, j) = tanh(0.01 * filteredImage.at<float>(i, j));
////                    cout << "tanh: " << filteredImage.at<float>(i, j) << endl;
//                    int pos = (filteredImage.at<float>(i, j) + 1) / 0.25; // range -1 to 1, 9 bins
////                    cout << pos << endl;
//                    hist[pos]++;
//                }
//            }
//            cout << "hist" << endl;
//            for (int i = 0; i < hist.size(); i++) {
//                cout << hist[i] << ",";
//            }
//            cout << endl;
//            feature.insert(feature.end(), hist.begin(), hist.end()); // concatenate
        }
    }
    // L2 normalize the vector
    normalize(feature, feature, 1, 0, NORM_L2, -1, Mat());

    return feature;
}

vector<float> gaborTextureAndColor(Mat &image) {
    vector<float> feature = gaborTexture(image);
    vector<float> color = histogram(image);
//    for (int i = 0; i < color.size(); i++) {
//        color[i] /= 2;
//    }
    feature.insert(feature.end(), color.begin(), color.end());
    return feature;

}

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