#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "processors.h"

using namespace std;
using namespace cv;

int main() {
    int blockSize = 2;
    int kSize = 3;
    double k = 0.04;
    double thresh = 100;

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

        char key = waitKey(10);

        // run Harris corner detector
        Mat grayscale;
        cvtColor(frame, grayscale, COLOR_BGR2GRAY);
        Mat dst = Mat::zeros(grayscale.size(), CV_32FC1);
        cornerHarris(grayscale, dst, blockSize, kSize, k);

        Mat dst_norm, dst_norm_scaled;
        normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
        convertScaleAbs( dst_norm, dst_norm_scaled );
        for (int i = 0; i < dst_norm.rows ; i++) {
            for(int j = 0; j < dst_norm.cols; j++) {
                if ((int)dst_norm.at<float>(i,j) > thresh) {
                    circle(frame, Point(j,i), 5, Scalar(0), 2, 8, 0);
                }
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