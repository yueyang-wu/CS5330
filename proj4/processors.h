#ifndef PROJ4_PROCESSORS_H
#define PROJ4_PROCESSORS_H

using namespace std;
using namespace cv;

bool extractCorners(Mat &frame, Size patternSize, vector<Point2f> &corners);

vector<Vec3f> constructObjectPoints();

void drawObjects(Mat &frame, vector<Point2f> p);

#endif //PROJ4_PROCESSORS_H
