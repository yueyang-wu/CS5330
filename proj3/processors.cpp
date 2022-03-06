#include <stdlib.h>
#include <map>
#include <float.h>
#include <math.h>
#include <opencv2/opencv.hpp>
#include "processors.h"
#include "csv_util.h"

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

void calcHuMoments(Moments mo, vector<double> &huMoments) {
    double hu[7];
    HuMoments(mo, hu);

    // covert array to vector
    for (double d : hu) {
        huMoments.push_back(d);
    }
    return;
}

//double euclideanDistance(vector<double> features1, vector<double> features2) {
//    return norm(features1, features2, NORM_L2) / (norm(features1, NORM_L2) * norm(features2, NORM_L2));
//}

double euclideanDistance(vector<double> features1, vector<double> features2) {
    double sum1 = 0, sum2 = 0, sumDifference;
    for (int i = 0; i < features1.size(); i++) {
        sumDifference += (features1[i] - features2[i]) * (features1[i] - features2[i]);
        sum1 += features1[i] * features1[i];
        sum2 += features2[i] * features2[i];
    }
    return sqrt(sumDifference) / (sqrt(sum1) + sqrt(sum2));
}

/*
 * find the nearest neighbor
 * use normalized euclidean distance as distance metric
 */
string classifier(vector<vector<double>> featureVectors, vector<string> classNames, vector<double> currentFeature) {
    double THRESHOLD = 0.15;
    double distance = DBL_MAX;
    string className = " ";
    for (int i = 0; i < featureVectors.size(); i++) {
        vector<double> dbFeature = featureVectors[i];
        string dbClassName = classNames[i];
        double curDistance = euclideanDistance(dbFeature, currentFeature);
        if (curDistance < distance && curDistance < THRESHOLD) {
            className = dbClassName;
            distance = curDistance;
        }
    }
    return className;
}

/*
 * find KNN
 * use normalized euclidean distance as distance metric
 */
string classifierKNN(vector<vector<double>> featureVectors, vector<string> classNames, vector<double> currentFeature, int K) {
    double THRESHOLD = 0.15;
    // compute the distances of current feature vector with all the feature vectors in DB
    vector<double> distances;
    for (int i = 0; i < featureVectors.size(); i++) {
        vector<double> dbFeature = featureVectors[i];
        double distance = euclideanDistance(dbFeature, currentFeature);
        if (distance < THRESHOLD) {
            distances.push_back(distance);
        }
    }

    string className = " ";
    if (distances.size() > 0) {
        // sort the distances in ascending order
        vector<int> sortedIdx;
        sortIdx(distances, sortedIdx, SORT_EVERY_ROW + SORT_ASCENDING);

        // get the first K class name, and count the number of each name
        vector<string> firstKNames;
        int s = sortedIdx.size();
        map<string, int> nameCount;
        int range = min(s, K); // if less than K classnames, get all of them
        for (int i = 0; i < range; i++) {
            string name = classNames[sortedIdx[i]];
            if (nameCount.find(name) != nameCount.end()) {
                nameCount[name]++;
            } else {
                nameCount[name] = 1;
            }
        }

        int count = 0;
        for (map<string ,int>::iterator it = nameCount.begin(); it != nameCount.end(); it++) {
            if (it->second > count) {
                className = it->first;
                count = it->second;
            }
        }
    }
    return className;
}

string getClassName(char c) {
    std::map<char, string> myMap {
            {'p', "pen"}, {'a', "alligator"}, {'h', "hammer"}, {'g', "glasses"},
            {'r', "round"}, {'c', "cat"}, {'b', "bone"}, {'k', "key"},
            {'m', "mouse"}, {'x', "binder clip"},
            {'w', "watch"}, {'s', "credit card"}, {'t', "spanner"} , {'y', "pliers"}
    };
    return myMap[c];
}

RotatedRect getBoundingBox(Mat &region, double x, double y, double alpha) {
    int maxX = INT_MIN, minX = INT_MAX, maxY = INT_MIN, minY = INT_MAX;
    for (int i = 0; i < region.rows; i++) {
        for (int j = 0; j < region.cols; j++) {
            if (region.at<uchar>(i, j) == 255) {
                int projectedX = (i - x) * cos(alpha) + (j - y) * sin(alpha);
                int projectedY = -(i - x) * sin(alpha) + (j - y) * cos(alpha);
                maxX = max(maxX, projectedX);
                minX = min(minX, projectedX);
                maxY = max(maxY, projectedY);
                minY = min(minY, projectedY);
            }
        }
    }
    int lengthX = maxX - minX;
    int lengthY = maxY - minY;

    Point centroid = Point(x, y);
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
        line(image, rect_points[i], rect_points[(i + 1) % 4], color, 3);
    }
}

