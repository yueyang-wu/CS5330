#ifndef PROJ3_PROCESSORS_H
#define PROJ3_PROCESSORS_H

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

Mat threshold(Mat &image);

Mat cleanup(Mat &image);

Mat getRegions(Mat &image, Mat &labeledRegions, Mat &stats, Mat &centroids, vector<int> &topNLabels);

void calcHuMoments(Mat &region, vector<double> &huMoments);

double euclideanDistance(vector<double> features1, vector<double> features2);

string classifier(vector<vector<double>> featureVectors, vector<string> classNames, vector<double> currentFeature);
string classifierKNN(vector<vector<double>> featureVectors, vector<string> classNames, vector<double> currentFeature, int K);

string getClassName(char c);

RotatedRect getBoundingBox(Mat &region, double x, double y, double alpha);

void drawLine(Mat &image, double x, double y, double alpha, Scalar color);

void drawBoundingBox(Mat &image, RotatedRect boundingBox, Scalar color);
#endif //PROJ3_PROCESSORS_H
