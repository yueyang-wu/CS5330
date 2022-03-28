#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include "processors.h"

using namespace std;
using namespace cv;

vector<Vec3f> constructWorldCoordinates(Size patternSize) {
    vector<Vec3f> points;
    for (int i = 0; i < patternSize.height; i++) {
        for (int j = 0; j < patternSize.width; j++) {
            Vec3f coordinates = Vec3f(j, -i, 0);
            points.push_back(coordinates);
        }
    }
    return points;
}

bool extractChessboardCorners(Mat &frame, Size patternSize, vector<Point2f> &corners) {
    bool foundCorners = findChessboardCorners(frame, patternSize, corners);
    if (foundCorners) {
        Mat grayscale;
        cvtColor(frame, grayscale, COLOR_BGR2GRAY); // the input image for cornerSubPix must be single-channel
        Size subPixWinSize(10, 10);
        TermCriteria termCrit(TermCriteria::COUNT|TermCriteria::EPS, 5, 0.03);
        cornerSubPix(grayscale, corners, subPixWinSize, Size(-1, -1), termCrit);
    }
    return foundCorners;
}

// only use the top left points of each target
bool extractArucoCorners(Mat &frame, vector<Point2f> &corners) {
    corners.resize(35, Point2f(0, 0));
    vector<int> markerIds;
    vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;
    Ptr<cv::aruco::DetectorParameters> parameters = aruco::DetectorParameters::create();
    Ptr<cv::aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(aruco::DICT_6X6_250);
    aruco::detectMarkers(frame, dictionary, markerCorners, markerIds, parameters, rejectedCandidates);

    for (int i = 0; i < markerIds.size(); i++) {
        int idx = markerIds[i];
        corners[idx] = markerCorners[i][0];
    }

    return markerCorners.size() == 35; // successfully extract Aruco corners
}

/*
 * Matrix type double
 */
void printMatrix(Mat &m) {
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) {
            cout << m.at<double>(i, j) << ", ";
        }
        cout << "\n";
    }
}

vector<Vec3f> constructObjectPoints() {
    vector<Vec3f> objectPoints;
    objectPoints.push_back(Vec3f(1, -1, 1));
    objectPoints.push_back(Vec3f(1, -4, 1));
    objectPoints.push_back(Vec3f(4, -1, 1));
    objectPoints.push_back(Vec3f(4, -4, 1));
    objectPoints.push_back(Vec3f(2, -2, 3));
    objectPoints.push_back(Vec3f(2, -3, 3));
    objectPoints.push_back(Vec3f(3, -2, 3));
    objectPoints.push_back(Vec3f(3, -3, 3));
    return objectPoints;
}

void drawObjects(Mat &frame, vector<Point2f> p) {
    line(frame, p[0], p[1], Scalar(147, 20, 255), 2);
    line(frame, p[0], p[2], Scalar(147, 20, 255), 2);
    line(frame, p[1], p[3], Scalar(147, 20, 255), 2);
    line(frame, p[2], p[3], Scalar(147, 20, 255), 2);
    line(frame, p[4], p[6], Scalar(147, 20, 255), 2);
    line(frame, p[4], p[5], Scalar(147, 20, 255), 2);
    line(frame, p[5], p[7], Scalar(147, 20, 255), 2);
    line(frame, p[6], p[7], Scalar(147, 20, 255), 2);
    line(frame, p[0], p[4], Scalar(147, 20, 255), 2);
    line(frame, p[1], p[5], Scalar(147, 20, 255), 2);
    line(frame, p[2], p[6], Scalar(147, 20, 255), 2);
    line(frame, p[3], p[7], Scalar(147, 20, 255), 2);
}

void projectOutsideCorners(Mat &frame, vector<Vec3f> points, Mat rvec, Mat tvec, Mat cameraMatrix, Mat distCoeffs) {
    vector<Point2f> imagePoints;
    projectPoints(points, rvec, tvec, cameraMatrix, distCoeffs, imagePoints);
    int index[] = {0, 8, 45, 53};
    for (int i : index) {
        circle(frame, imagePoints[i], 5, Scalar(147, 20, 255), 4);
    }
}

void projectVirtualObject(Mat &frame, Mat rvec, Mat tvec, Mat cameraMatrix, Mat distCoeffs) {
    vector<Vec3f> objectPoints = constructObjectPoints();
    vector<Point2f> projectedPoints;
    projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs, projectedPoints);
    for (int i = 0; i < projectedPoints.size(); i++) {
        circle(frame, projectedPoints[i], 1, Scalar(147, 20, 255), 4);
    }
    drawObjects(frame, projectedPoints);
}