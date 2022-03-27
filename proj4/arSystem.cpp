#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "processors.h"

using namespace std;
using namespace cv;

int main() {
    Size chessboardPatternSize(9, 6); // the size of the chessboard, height is 6, width is 9
    Size arucoPatternSize(5, 7); // the size of the aruco target, height is 7, width is 5
    Mat chessboardCameraMatrix, arucoCameraMatrix;
    vector<Vec3f> chessBoardPoints; // the 3D world points constructed for the chessboard target
    vector<Vec3f> arucoPoints; // the 3D world points constructed for the aruco target
    vector<vector<Point2f> > chessboardCornerList;
    vector<vector<Vec3f> > chessboardPointList;
    vector<vector<Point2f> > arucoCornerList;
    vector<vector<Vec3f> > arucoPointList;
    int CALIBRATION_FRAME_NEEDED = 5; // the minimum number of calibration frames needed
    Mat chessboardDistCoeffs, arucoDistCoeffs; // output arrays for calibrateCamera()
    vector<Mat> chessboardR, chessboardT, arucoR, arucoT; // output arrays for calibrateCamera()

    // open the video device
    VideoCapture *capdev;
    capdev = new VideoCapture(0);
    if (!capdev->isOpened()) {
        cout << "Unable to open video device\n";
        return -1;
    }

    // identify window
    namedWindow("Video", 1);

    // create a vector of points that specifies the 3D position of the corners in world coordinates
    chessBoardPoints = constructWorldCoordinates(chessboardPatternSize);
    arucoPoints = constructWorldCoordinates(arucoPatternSize);

    Mat frame; // the original frame

    while (true) {
        *capdev >> frame; // get a new frame from the camera, treat as a stream
        if (frame.empty()) {
            cout << "frame is empty\n";
            break;
        }

        // resize the frame to 1/2 of the original size
        resize(frame, frame, Size(), 0.5, 0.5);

        Mat displayedFrame = frame.clone(); // the frame displayed

        char key = waitKey(10); // see if there is a waiting keystroke for the video

        vector<Point2f> chessboardCorners; // the image points found by extractChessboardCorners()
//        bool foundChessboardCorners = false;
        bool foundChessboardCorners = extractChessboardCorners(frame, chessboardPatternSize, chessboardCorners);
        if (foundChessboardCorners) {
            drawChessboardCorners(displayedFrame, chessboardPatternSize, chessboardCorners, foundChessboardCorners);
        }

        vector<Point2f> arucoCorners; // the image points found by extractarucoCorners()
        bool foundArucoCorners = extractArucoCorners(frame, arucoCorners);
        if (foundArucoCorners) {
            for (int i = 0; i < arucoCorners.size(); i++) {
                circle(displayedFrame, arucoCorners[i], 1, Scalar(147, 200, 255), 4);
            }
        }

        if (key == 's') { // select calibration images for chessboard
            if (foundChessboardCorners) {
                cout << "select chessboard calibration image" << endl;
                // add the vector of corners found by findChessCorners() into a cornerList
                chessboardCornerList.push_back(chessboardCorners);
                // add the vector of points into a pointList
                chessboardPointList.push_back(chessBoardPoints);
            } else {
                cout << "No chessboard corners found" << endl;
            }
        } else if (key == 'h') { // select calibration images for aruco target
            if (foundArucoCorners) {
                cout << "select aruco calibration image" << endl;
                // add the vector of corners found by extractarucoCorners() into a cornerList
                arucoCornerList.push_back(arucoCorners);
                // add the vector of points into a pointList
                arucoPointList.push_back(arucoPoints);
            } else {
                cout << "No aruco corners found" << endl;
            }
        } else if (key == 'c') { // calibrate the camera for chessboard
            if (chessboardPointList.size() < CALIBRATION_FRAME_NEEDED) { // not enough calibration frames
                cout << "Not enough calibration frames. 5 or more needed." << endl;
            } else {
                cout << "calibrate camera" << endl;

                // calibrate the camera
                double chessboardError = calibrateCamera(chessboardPointList, chessboardCornerList, Size(frame.rows, frame.cols), chessboardCameraMatrix, chessboardDistCoeffs, chessboardR, chessboardT);

                // print out the intrinsic parameters and the final re-projection error
                cout << "Chessboard Camera Matrix: " << endl;
                for (int i = 0; i < chessboardCameraMatrix.rows; i++) {
                    for (int j = 0; j < chessboardCameraMatrix.cols; j++) {
                        cout << chessboardCameraMatrix.at<double>(i, j) << ", ";
                    }
                    cout << "\n";
                }
                cout << "Chessboard Distortion Coefficients: " << endl;
                for (int i = 0; i < chessboardDistCoeffs.rows; i++) {
                    for (int j = 0; j < chessboardDistCoeffs.cols; j++) {
                        cout << chessboardDistCoeffs.at<double>(i, j) << ", ";
                    }
                }
                cout << "\n";
                cout << "Chessboard Re-projection Error: " << chessboardError << endl;
            }
        } else if (key == 'x') { // calibrate the camera for aruco target
            if (arucoPointList.size() < CALIBRATION_FRAME_NEEDED) { // not enough calibration frames
                cout << "Not enough calibration frames. 5 or more needed." << endl;
            } else {
                cout << "calibrate camera" << endl;

                // calibrate the camera
                double arucoError = calibrateCamera(arucoPointList, arucoCornerList, Size(frame.rows, frame.cols), arucoCameraMatrix, arucoDistCoeffs, arucoR, arucoT);

                // print out the intrinsic parameters and the final re-projection error
                cout << "Aruco Camera Matrix: " << endl;
                for (int i = 0; i < arucoCameraMatrix.rows; i++) {
                    for (int j = 0; j < arucoCameraMatrix.cols; j++) {
                        cout << arucoCameraMatrix.at<double>(i, j) << ", ";
                    }
                    cout << "\n";
                }
                cout << "Aruco Distortion Coefficients: " << endl;
                for (int i = 0; i < arucoDistCoeffs.rows; i++) {
                    for (int j = 0; j < arucoDistCoeffs.cols; j++) {
                        cout << arucoDistCoeffs.at<double>(i, j) << ", ";
                    }
                }
                cout << "\n";
                cout << "Aruco Re-projection Error: " << arucoError << endl;
            }
        }

        if (chessboardDistCoeffs.rows != 0) {
            // extractChessboardCorners of current frame
            vector<Point2f> currCorners; // the image points found by findChessboardCorners
            bool foundCurrCorners = extractChessboardCorners(frame, chessboardPatternSize, currCorners);

            if (foundCurrCorners) {
                Mat rvec, tvec; // output arrays for solvePnP()
                bool status = solvePnP(chessBoardPoints, currCorners, chessboardCameraMatrix, chessboardDistCoeffs, rvec, tvec);

                if (status) { // solvePnP() succeed
                    // print the rotation and translation data
//                    cout << "Rotation Data: " << endl;
//                    for (int i = 0; i < rvec.rows; i++) {
//                        for (int j = 0; j < rvec.cols; j++) {
//                            cout << rvec.at<double>(i, j) << ", ";
//                        }
//                    }
//                    cout << "\n";
//                    cout << "Translation Data: " << endl;
//                    for (int i = 0; i < tvec.rows; i++) {
//                        for (int j = 0; j < tvec.cols; j++) {
//                            cout << tvec.at<double>(i, j) << ", ";
//                        }
//                    }
//                    cout << "\n";

                    // project outside corners
                    vector<Point2f> imagePoints;
                    projectPoints(chessBoardPoints, rvec, tvec, chessboardCameraMatrix, chessboardDistCoeffs, imagePoints);
                    int index[] = {0, 8, 45, 53};
                    for (int i : index) {
                        circle(displayedFrame, imagePoints[i], 5, Scalar(147, 20, 255), 4);
                    }

                    // project a virtual object
                    vector<Vec3f> objectPoints = constructObjectPoints();
                    vector<Point2f> projectedPoints;
                    projectPoints(objectPoints, rvec, tvec, chessboardCameraMatrix, chessboardDistCoeffs, projectedPoints);
                    for (int i = 0; i < projectedPoints.size(); i++) {
                        circle(displayedFrame, projectedPoints[i], 1, Scalar(147, 20, 255), 4);
                    }
                    drawObjects(displayedFrame, projectedPoints);
                }
            }
        }

        if (arucoDistCoeffs.rows != 0) {
            // extractArucoCorners of current frame
            vector<Point2f> currCorners; // the image points found by extractArucoCorners
            bool foundCurrCorners = extractArucoCorners(frame, currCorners);

            if (foundCurrCorners) {
                Mat rvec, tvec; // output arrays for solvePnP()
                bool status = solvePnP(arucoPoints, currCorners, arucoCameraMatrix, arucoDistCoeffs, rvec, tvec);

                if (status) { // solvePnP() succeed
                    // print the rotation and translation data
//                    cout << "Rotation Data: " << endl;
//                    for (int i = 0; i < rvec.rows; i++) {
//                        for (int j = 0; j < rvec.cols; j++) {
//                            cout << rvec.at<double>(i, j) << ", ";
//                        }
//                    }
//                    cout << "\n";
//                    cout << "Translation Data: " << endl;
//                    for (int i = 0; i < tvec.rows; i++) {
//                        for (int j = 0; j < tvec.cols; j++) {
//                            cout << tvec.at<double>(i, j) << ", ";
//                        }
//                    }
//                    cout << "\n";

                    // project a virtual object
                    vector<Vec3f> objectPoints = constructObjectPoints();
                    vector<Point2f> projectedPoints;
                    projectPoints(objectPoints, rvec, tvec, arucoCameraMatrix, arucoDistCoeffs, projectedPoints);
                    for (int i = 0; i < projectedPoints.size(); i++) {
                        circle(displayedFrame, projectedPoints[i], 1, Scalar(147, 20, 255), 4);
                    }
                    drawObjects(displayedFrame, projectedPoints);
                }
            }
        }

        imshow("Video", displayedFrame);

        if (key == 'q') { // press 'q' to quit the system
            break;
        }
    }

    destroyAllWindows();

    return 0;
}