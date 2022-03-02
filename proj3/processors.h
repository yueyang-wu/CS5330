#ifndef PROJ3_PROCESSORS_H
#define PROJ3_PROCESSORS_H

#include <opencv2/opencv.hpp>

using namespace cv;

Mat threshold(Mat &image);

Mat cleanup(Mat &image);

Mat getRegions(Mat &image, Mat stats, Mat centroids);

#endif //PROJ3_PROCESSORS_H
