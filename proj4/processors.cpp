#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "processors.h"

using namespace std;
using namespace cv;

void extractCorners(Mat &frame, Size patternSize, vector<Point2f> &corners) {
    bool foundCorners = findChessboardCorners(frame, patternSize, corners);
    cout << "number of corners: " << corners.size() << endl;
    if (foundCorners) {
        Mat grayscale;
        cvtColor(frame, grayscale, COLOR_BGR2GRAY); // the input image for cornerSubPix must be single-channel
        Size subPixWinSize(10, 10);
        TermCriteria termCrit(TermCriteria::COUNT|TermCriteria::EPS, 5, 0.03);
        cornerSubPix(grayscale, corners, subPixWinSize, Size(-1, -1), termCrit);
        cout << "coordinates of the first corner: (" << corners[0].x << ", " << corners[0].y << ")" << endl;
        drawChessboardCorners(frame, patternSize, corners, foundCorners);
    }
}