#ifndef PROJ3_PROCESSORS_H
#define PROJ3_PROCESSORS_H

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

Mat threshold(Mat &image);

Mat cleanup(Mat &image);

//Mat getRegions(Mat &image, Mat stats, Mat centroids);
Mat getRegions(Mat &image, Mat &labeledRegions, Mat &stats, Mat &centroids, vector<int> &topNLabels);

void calcHuMoments(Mat &labeledRegions, vector<int> topNLabels, map<int, double*> &huMomentsMap);

#endif //PROJ3_PROCESSORS_H
