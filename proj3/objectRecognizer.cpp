#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "processors.h"

using namespace cv;
using namespace std;

int main() {
    VideoCapture *capdev;

    // open the video device
    capdev = new VideoCapture(0);
    if (!capdev->isOpened()) {
        cout << "Unable to open video device\n";
        return -1;
    }

    // get some properties of the image
//    Size refS((int) capdev->get(cv::CAP_PROP_FRAME_WIDTH),
//              (int) capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
//    cout << "Expected size: " << refS.width << " " << refS.height << "\n";

    // identify two windows
    namedWindow("Original Video", 1);
    namedWindow("Processed Video", 1);
//    namedWindow("Test Video", 1);
    Mat frame, processedFrame;

    while (true) {
        *capdev >> frame; // get a new frame from the camera, treat as a stream
        if (frame.empty()) {
            cout << "frame is empty\n";
            break;
        }
        imshow("Original Video", frame);
        processedFrame = threshold(frame);
        imshow("Processed Video", processedFrame);

        // see if there is a waiting keystroke
        char key = waitKey(10);
        // if user types 'q', quit.
        if (key == 'q') {
            break;
        }
    }

    return 0;
}