#ifndef PROJ4_PROCESSORS_H
#define PROJ4_PROCESSORS_H

using namespace std;
using namespace cv;

bool extractCorners(Mat &frame, Size patternSize, vector<Point2f> &corners);

#endif //PROJ4_PROCESSORS_H
