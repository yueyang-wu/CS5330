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

        // extract chessboard corners
        vector<Point2f> chessboardCorners; // the image points found by extractChessboardCorners()
        bool foundChessboardCorners = extractChessboardCorners(frame, chessboardPatternSize, chessboardCorners);
        if (foundChessboardCorners) { // display the chessboard corners
            drawChessboardCorners(displayedFrame, chessboardPatternSize, chessboardCorners, foundChessboardCorners);
        }

        // extract ArUco corners
        vector<Point2f> arucoCorners; // the image points found by extractArucoCorners()
        bool foundArucoCorners = extractArucoCorners(frame, arucoCorners);
        if (foundArucoCorners) { // display the top left corner of each target
            for (int i = 0; i < arucoCorners.size(); i++) {
                circle(displayedFrame, arucoCorners[i], 1, Scalar(147, 200, 255), 4);
            }
        }

        if (key == 's') { // select calibration images for chessboard
            if (foundChessboardCorners) {
                cout << "select chessboard calibration image" << endl;
                // add the vector of corners found by findChessCorners() into a cornerList
                chessboardCornerList.push_back(chessboardCorners);
                // add the vector of real-world points into a pointList
                chessboardPointList.push_back(chessBoardPoints);
            } else {
                cout << "No chessboard corners found" << endl;
            }
        } else if (key == 'h') { // select calibration images for aruco target
            if (foundArucoCorners) {
                cout << "select aruco calibration image" << endl;
                // add the vector of corners found by extractarucoCorners() into a cornerList
                arucoCornerList.push_back(arucoCorners);
                // add the vector of real-world points into a pointList
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
                printMatrix(chessboardCameraMatrix);
                cout << "Chessboard Distortion Coefficients: " << endl;
                printMatrix(chessboardDistCoeffs);
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
                printMatrix(arucoCameraMatrix);
                cout << "Aruco Distortion Coefficients: " << endl;
                printMatrix(arucoDistCoeffs);
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
                    // project outside corners
                    projectOutsideCorners(displayedFrame, chessBoardPoints, rvec, tvec, chessboardCameraMatrix, chessboardDistCoeffs);

                    // project a virtual object
                    projectVirtualObject(displayedFrame, rvec, tvec, chessboardCameraMatrix, chessboardDistCoeffs);
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
                    // project a virtual object
                    projectVirtualObject(displayedFrame, rvec, tvec, arucoCameraMatrix, arucoDistCoeffs);
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