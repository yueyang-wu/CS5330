#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "processors.h"

using namespace std;
using namespace cv;

int main() {
    Size patternSize(9, 6); // the size of the chessboard, height is 6, width is 9
    Mat cameraMatrix;
    vector<Point2f> corners; // the image points found by findChessboardCorners(in calibration mode)
    vector<Vec3f> points; // the 3D world points constructed
    vector<vector<Point2f> > cornerList;
    vector<vector<Vec3f> > pointList;
    int CALIBRATION_FRAME_NEEDED = 5; // the minimum number of calibration frames needed
    Mat distCoeffs; // output arrays for calibrateCamera()
    vector<Mat> R, T; // output arrays for calibrateCamera()
    bool augmentedReality = false; // whether the system should start to put virtual objects

    // open the video device
    VideoCapture *capdev;
    capdev = new VideoCapture(0);
    if (!capdev->isOpened()) {
        cout << "Unable to open video device\n";
        return -1;
    }

    // identify window
    namedWindow("Video", 1);

    // create a vector of points that specifies the 3D position of th corners in world coordinates
    for (int i = 0; i < patternSize.height; i++) {
        for (int j = 0; j < patternSize.width; j++) {
            Vec3f coordinates = Vec3f(j, -i, 0);
            points.push_back(coordinates);
        }
    }

    // initialize a 3 x 3 camera matrix
//    double data[3][3] = {{1, 0, double(frame.cols / 2)}, {0, 1, double(frame.rows / 2)}, {0, 0, 1}};
//    Mat cameraMatrix = Mat(3, 3, CV_64FC1, data);

    Mat frame;

    while (true) {
        *capdev >> frame; // get a new frame from the camera, treat as a stream
        if (frame.empty()) {
            cout << "frame is empty\n";
            break;
        }

        // resize the frame to 1/2 of the original size
        resize(frame, frame, Size(), 0.5, 0.5);

        Mat frame2 = frame.clone();

        char key = waitKey(10); // see if there is a waiting keystroke for the video

        bool foundCorners = extractCorners(frame, patternSize, corners);
        if (foundCorners) {
            drawChessboardCorners(frame2, patternSize, corners, foundCorners);
        }

        if (key == 's') { // select calibration images
            if (foundCorners) {

                cout << "frame size: " << frame.size() << endl;

                cout << "corners: " << endl;
                cout << corners[0] << ", " << corners[1] << ", " << corners[2] << ", " << corners[3] << ", " << corners[4] << ", " << corners[5] << corners[6] << ", " << corners[7] << ", " << corners[8] << corners[9] << endl;

                cout << "select calibration image" << endl;
                // add the vector of corners found by findChessCorners() into a cornerList
                cornerList.push_back(corners);
                // add the vector of points into a pointList
                pointList.push_back(points);
            } else {
                cout << "No corners found" << endl;
            }
        } else if (key == 'c') { // calibrate the camera
            if (pointList.size() < CALIBRATION_FRAME_NEEDED) { // not enough calibration frames
                cout << "Not enough calibration frames. 5 or more needed." << endl;
            } else {
                cout << "calibrate camera" << endl;

                // calibrate the camera
                double error = calibrateCamera(pointList, cornerList, Size(frame.rows, frame.cols), cameraMatrix, distCoeffs, R, T);

                // print out the intrinsic parameters and the final re-projection error
                cout << "Camera Matrix: " << endl;
                for (int i = 0; i < cameraMatrix.rows; i++) {
                    for (int j = 0; j < cameraMatrix.cols; j++) {
                        cout << cameraMatrix.at<double>(i, j) << ", ";
                    }
                    cout << "\n";
                }
                cout << "Distortion Coefficients: " << endl;
                for (int i = 0; i < distCoeffs.rows; i++) {
                    for (int j = 0; j < distCoeffs.cols; j++) {
                        cout << distCoeffs.at<double>(i, j) << ", ";
                    }
                }
                cout << "\n";
                cout << "re-projection error: " << error << endl;
            }
        } else if (key == 't') { // start the AR test
            if (!augmentedReality && distCoeffs.rows != 0) {
                cout << "AR started" << endl;
                augmentedReality = true;
            } else if (!augmentedReality && distCoeffs.rows == 0) {
                cout << "camera not calibrated" << endl;
            } else { // switch back to calibration mode
                cout << "AR ended" << endl;
                augmentedReality = false;
            }
        }

        if (augmentedReality) { // in the AR mode, should put virtual object
            // convert pointList and cornerList to vector<>

            // extractCorners of current frame
            vector<Point2f> currCorners; // the image points found by findChessboardCorners
            bool foundCurrCorners = extractCorners(frame, patternSize, currCorners);
            if (foundCurrCorners) {
                drawChessboardCorners(frame2, patternSize, currCorners, foundCorners);
            }

//            vector<Point3f> convertedPointList;
//            for (auto && v : pointList) {
//                convertedPointList.insert(convertedPointList.end(), v.begin(), v.end());
//            }
//            vector<Point2f> convertedCornerList;
//            for (auto && v : cornerList) {
//                convertedCornerList.insert(convertedCornerList.end(), v.begin(), v.end());
//            }

            cout << "currCorners" << currCorners.size() << endl;
//            for (int i = 0; i < currCorners.size(); i++) {
//                cout << currCorners[i] << ", ";
//            }
//            cout << "\n";
            if (foundCurrCorners) {
                Mat rvec, tvec; // output arrays for solvePnP()
                bool status = solvePnP(points, currCorners, cameraMatrix, distCoeffs, rvec, tvec);

                if (status) {
                    // print the rotation and translation data
                    cout << "Rotation Data: " << endl;
                    for (int i = 0; i < rvec.rows; i++) {
                        for (int j = 0; j < rvec.cols; j++) {
                            cout << rvec.at<double>(i, j) << ", ";
                        }
                    }
                    cout << "\n";
                    cout << "Translation Data: " << endl;
                    for (int i = 0; i < tvec.rows; i++) {
                        for (int j = 0; j < tvec.cols; j++) {
                            cout << tvec.at<double>(i, j) << ", ";
                        }
                    }
                    cout << "\n";

                    // project outside corners
                    vector<Point2f> imagePoints;
                    projectPoints(points, rvec, tvec, cameraMatrix, distCoeffs, imagePoints);
                    for (int i = 0; i < imagePoints.size(); i++) {
                        circle(frame2, imagePoints[i], 5, Scalar(255, 0, 0), 4);
                    }
                }
            }
        }

        imshow("Video", frame2);
        if (key == 'q') { // press 'q' to quit the system
            break;
        }
    }

    destroyAllWindows();

    return 0;
}