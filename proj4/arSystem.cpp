#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main() {
    Size patternSize(9, 6); // the size of the chessboard
    vector<Point2f> corners; // the image points found by findChessboardCorners
    Size subPixWinSize(10, 10);
    TermCriteria termCrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.03);
    bool patternWasFound;
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
            Mat grayscale;
            cvtColor(frame, grayscale, COLOR_BGR2GRAY); // the input image for cornerSubPix must be single-channel
            cornerSubPix(grayscale, corners, subPixWinSize, Size(-1, -1), termCrit);
            cout << "coordinates of the first corner: (" << corners[0].x << ", " << corners[0].y << ")" << endl;
            drawChessboardCorners(frame, patternSize, corners, patternWasFound);
        }

        imshow("Video", frame);
        if (key == 'q') { // press 'q' to quit the system
            break;
        }
    }

    destroyAllWindows();

    return 0;
}