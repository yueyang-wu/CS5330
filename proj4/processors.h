#ifndef PROJ4_PROCESSORS_H
#define PROJ4_PROCESSORS_H

using namespace std;
using namespace cv;

bool extractChessboardCorners(Mat &frame, Size patternSize, vector<Point2f> &corners);

bool extractArucoCorners(Mat &frame, vector<Point2f> &corners);

void arucoOutsidePoints(Mat &frame, vector<Point2f> &outsidePoints);

vector<Vec3f> constructWorldCoordinates(Size patternSize);

void overlayPicture(Mat &frame, Mat &displayedFrame, Mat &image);

void printMatrix(Mat &m);

vector<Vec3f> constructObjectPoints();

void drawObjects(Mat &frame, vector<Point2f> p);

void projectOutsideCorners(Mat &frame, vector<Vec3f> points, Mat rvec, Mat tvec, Mat cameraMatrix, Mat distCoeffs);

void projectVirtualObject(Mat &frame, Mat rvec, Mat tvec, Mat cameraMatrix, Mat distCoeffs);

#endif //PROJ4_PROCESSORS_H
