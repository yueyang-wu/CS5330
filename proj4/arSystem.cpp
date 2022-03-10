#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main() {
    Size patternSize(9, 6); // the size of the chessboard
    vector<Point2f> corners; // the image points found by findChessboardCorners
    vector<Vec3f> points; // the 3D world points constructed
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

        char key = waitKey(10);

        bool foundCorners = findChessboardCorners(frame, patternSize, corners);
        cout << "number of corners: " << corners.size() << endl;
        if (foundCorners) {
            cout << "coordinates of the first corner: (" << corners[0].x << ", " <<  corners[0].y << ")" << endl;
            for (int i = 0; i < corners.size(); i++) {
                circle(frame, corners[i], 3, Scalar(0, 0, 255), 3);
            }
        }

        imshow("Video", frame);
        if (key == 'q') {
            break;
        }
    }

    destroyAllWindows();

    return 0;
}