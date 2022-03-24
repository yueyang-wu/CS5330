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
//            for (int i = 0; i < points.size(); i++) {
//                cout << i << ": " << points[i][0] << ", " << points[i][1] << ", " << points[i][2] << endl;
//            }
            // add the vector of points into a pointList
            pointList.push_back(points);
        } else if (key == 'c') {
            if (pointList.size() < CALIBRATION_FRAME_NEEDED) { // not enough calibration frames
                cout << "Not enough calibration frames. 5 or more needed." << endl;
            } else {
                // initialize a 3 x 3 camera matrix
                double data[3][3] = {{1, 0, double(frame.cols / 2)}, {0, 1, double(frame.rows / 2)}, {0, 0, 1}};
                Mat cameraMatrix = Mat(3, 3, CV_64FC1, data);

                // calibrate the camera
//                Mat
//                double error = calibrateCamera()
            }

        }

        imshow("Video", frame);
        if (key == 'q') { // press 'q' to quit the system
            break;
        }
    }

    destroyAllWindows();

    return 0;
}