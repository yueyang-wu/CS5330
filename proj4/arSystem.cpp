#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "processors.h"

using namespace std;
using namespace cv;

int main() {
    Size patternSize(9, 6); // the size of the chessboard
    vector<Point2f> corners; // the image points found by findChessboardCorners
    vector<vector<Point2f> > cornerList;
    vector<vector<Vec3f> > pointList;
    int CALIBRATION_FRAME_NEEDED = 5; // the minimum number of calibration frames needed

    // open the video device
    VideoCapture *capdev;
    capdev = new VideoCapture(0);
    if (!capdev->isOpened()) {
        cout << "Unable to open video device\n";
        return -1;
    }

    // identify window
    namedWindow("Video", 1);

    Mat frame;

    while (true) {
        *capdev >> frame; // get a new frame from the camera, treat as a stream
        if (frame.empty()) {
            cout << "frame is empty\n";
            break;
        }

        // resize the frame to 1/2 of the original size
        resize(frame, frame, Size(), 0.5, 0.5);

        char key = waitKey(10); // see if there is a waiting keystroke for the video

        extractCorners(frame, patternSize, corners);

        if (key == 's') { // select calibration images
            cout << "select calibration image" << endl;
            // add the vector of corners found by findChessCorners() into a cornerList
            cornerList.push_back(corners);
            // create a vector of points that specifies the 3D position of th corners in world coordinates
            vector<Vec3f> points; // the 3D world points constructed
            for (int i = 0; i < patternSize.height; i++) {
                for (int j = 0; j < patternSize.width; j++) {
                    Vec3f coordinates = Vec3f(j, -i, 0);
                    points.push_back(coordinates);
                }
            }
            // add the vector of points into a pointList
            pointList.push_back(points);
        } else if (key == 'c') { // calibrate the camera
            if (pointList.size() < CALIBRATION_FRAME_NEEDED) { // not enough calibration frames
                cout << "Not enough calibration frames. 5 or more needed." << endl;
            } else {
                cout << "calibrate camera" << endl;
                // initialize a 3 x 3 camera matrix
                double data[3][3] = {{1, 0, double(frame.cols / 2)}, {0, 1, double(frame.rows / 2)}, {0, 0, 1}};
                Mat cameraMatrix = Mat(3, 3, CV_64FC1, data);

                // calibrate the camera
                Mat distCoeffs, rvecs, tvecs;
                double error = calibrateCamera(pointList, cornerList, Size(frame.rows, frame.cols), cameraMatrix, distCoeffs, rvecs, tvecs);

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
        } else if (key == 'w') { // write the intrinsic parameters to a file

        }

        imshow("Video", frame);
        if (key == 'q') { // press 'q' to quit the system
            break;
        }
    }

    destroyAllWindows();

    return 0;
}