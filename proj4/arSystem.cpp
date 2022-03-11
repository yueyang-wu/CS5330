#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "processors.h"

using namespace std;
using namespace cv;

int main() {
    Size patternSize(9, 6); // the size of the chessboard
    vector<Point2f> corners; // the image points found by findChessboardCorners
    bool patternWasFound; // whether a pattern was found by extractCorners()
    vector<vector<Point2f> > cornerList;
    vector<vector<Vec3f> > pointList;

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

        char key = waitKey(10); // see if there is a waiting keystroke for the video

        extractCorners(frame, patternSize, corners, patternWasFound);

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
            if (points.size() == corners.size()) {
                cout << "there: yes" << endl;
            }

//            for (int i = 0; i < points.size(); i++) {
//                cout << i << ": " << points[i][0] << ", " << points[i][1] << ", " << points[i][2] << endl;
//            }
            // add the vector of points into a pointList
            pointList.push_back(points);
            
            if (pointList.size() == cornerList.size()) {
                cout << "here: yes" << endl;
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