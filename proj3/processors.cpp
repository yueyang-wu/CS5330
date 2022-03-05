#include <stdlib.h>
#include <map>
#include <float.h>
#include <math.h>
#include <opencv2/opencv.hpp>
#include "processors.h"

using namespace cv;
using namespace std;

Mat threshold(Mat &image) {
    int THRESHOLD = 130;
    Mat temp, processedImage, grayscale;
    cvtColor(image, temp, COLOR_BGR2HSV);
    processedImage = Mat(image.size(), CV_8UC1);

    cvtColor(image, grayscale, COLOR_BGR2GRAY);
    for (int i = 0; i < grayscale.rows; i++) {
        for (int j = 0; j < grayscale.cols; j++) {
            if (grayscale.at<uchar>(i, j) <= THRESHOLD) {
                processedImage.at<uchar>(i, j) = 255;
            } else {
                processedImage.at<uchar>(i, j) = 0;
            }
        }
    }
    return processedImage;
}

Mat cleanup(Mat &image) {
    Mat processedImage;
    const Mat kernel = getStructuringElement(MORPH_CROSS, Size(25, 25));
    morphologyEx(image, processedImage, MORPH_CLOSE, kernel);
    return processedImage;
}

Mat getRegions(Mat &image, Mat &labeledRegions, Mat &stats, Mat &centroids, vector<int> &topNLabels) {
    Mat processedImage;
    int nLabels = connectedComponentsWithStats(image, labeledRegions, stats, centroids);

    // save all region areas into a vector and sort the area descending
    Mat areas = Mat::zeros(1, nLabels - 1, CV_32S);
    Mat sortedIdx;
    for (int i = 1; i < nLabels; i++) {
        int area = stats.at<int>(i, CC_STAT_AREA);
        areas.at<int>(i - 1) = area;
    }
    if (areas.cols > 0) {
        sortIdx(areas, sortedIdx, SORT_EVERY_ROW + SORT_DESCENDING);
    }

    vector<Vec3b> colors(nLabels, Vec3b(0, 0, 0)); // label to color mapping

    int N = 3; // only take the largest 3 non-background regions
    N = (N < sortedIdx.cols) ? N : sortedIdx.cols;
    int THRESHOLD = 5000; // any region area less than 5,000 will be ignored
    for (int i = 0; i < N; i++) {
        int label = sortedIdx.at<int>(i) + 1;
        if (stats.at<int>(label, CC_STAT_AREA) > THRESHOLD) {
            colors[label] = Vec3b(rand() % 256, rand() % 256, rand() % 256);
            topNLabels.push_back(label);
        }
    }

    processedImage = Mat::zeros(labeledRegions.size(), CV_8UC3);
    for(int i = 0; i < processedImage.rows; i++) {
        for (int j = 0; j < processedImage.cols; j++) {
            int label = labeledRegions.at<int>(i, j);
            processedImage.at<Vec3b>(i, j) = colors[label];
        }
    }
    return processedImage;
}

void calcHuMoments(Mat &region, vector<double> &huMoments) {
    Moments mo = moments(region, true);
    double hu[7];
    HuMoments(mo, hu);

    // covert array to vector
    for (double d : hu) {
        huMoments.push_back(d);
    }
    return;
}

double euclideanDistance(vector<double> features1, vector<double> features2) {
    return norm(features1, features2, NORM_L2) / (norm(features1, NORM_L2) * norm(features2, NORM_L2));
}

/*
 * used normalized euclidean distance as distance metric
 */
string classifier(vector<vector<double>> featureVectors, vector<string> classNames, vector<double> currentFeature) {
    double distance = DBL_MAX;
    string className = " ";
    for (int i = 0; i < featureVectors.size(); i++) {
        vector<double> dbFeature = featureVectors[i];
        string dbClassName = classNames[i];
        double curDistance = euclideanDistance(dbFeature, currentFeature);
        if (curDistance < distance) {
            className = dbClassName;
            distance = curDistance;
        }
    }
    return className;
}

string getClassName(char c) {
    std::map<char, string> myMap {
            {'p', "pen"}, {'a', "alligator"}, {'h', "hammer"}
    };
    return myMap[c];
}

RotatedRect getBoundingBox(Mat &region, Mat &centroids, int label) {
    Moments m = moments(region, true);
    double centroidX = centroids.at<double>(label, 0);
    double centroidY = centroids.at<double>(label, 1);
    cout << "x: " << centroidX << endl;
    cout << "y: " << centroidY << endl;
    double alpha = 1.0 / 2.0 * atan2(2 * m.mu11, m.mu20 - m.mu02);
    cout << "a: " << alpha << endl;

    int maxX = INT_MIN, minX = INT_MAX, maxY = INT_MIN, minY = INT_MAX;
    for (int i = 0; i < region.rows; i++) {
        for (int j = 0; j < region.cols; j++) {
            if (region.at<uchar>(i, j) == 255) {
                int projectedX = (i - centroidX) * cos(alpha) + (j - centroidY) * sin(alpha);
                int projectedY = -(i - centroidX) * sin(alpha) + (j - centroidY) * cos(alpha);
                maxX = max(maxX, projectedX);
                minX = min(minX, projectedX);
                maxY = max(maxY, projectedY);
                minY = min(minY, projectedY);
            }
        }
    }
    int lengthX = maxX - minX;
    int lengthY = maxY - minY;

    Point centroid = Point(centroidX, centroidY);
    Size size = Size(lengthX, lengthY);

    return RotatedRect(centroid, size, alpha * 180.0 / CV_PI);
}

void drawLine(Mat &image, double x, double y, double alpha, Scalar color) {
    double length = 100.0;
    double edge1 = length * sin(alpha);
    double edge2 = sqrt(length * length - edge1 * edge1);
    double xPrime = x + edge2, yPrime = y + edge1;

    arrowedLine(image, Point(x, y), Point(xPrime, yPrime), color, 3);
}

void drawBoundingBox(Mat &image, RotatedRect boundingBox, Scalar color) {
    Point2f rect_points[4];
    boundingBox.points(rect_points);
    for (int i = 0; i < 4; i++) {
        line(image, rect_points[i], rect_points[(i + 1) % 4], color);
    }
}

