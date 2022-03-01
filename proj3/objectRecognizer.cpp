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
    Mat frame, thresholdFrame, processedFrame;

    while (true) {
        *capdev >> frame; // get a new frame from the camera, treat as a stream
        if (frame.empty()) {
            cout << "frame is empty\n";
            break;
        }
        imshow("Original Video", frame);

        // threshold the image
        thresholdFrame = threshold(frame);
        // clean up the image
        const Mat kernel = getStructuringElement(MORPH_CROSS, Size(25, 25));
        morphologyEx(thresholdFrame, thresholdFrame, MORPH_CLOSE, kernel);
        // get the region

//        Mat labelImage(img.size(), CV_32S);
        Mat stats, centroids;
        int nLabels = connectedComponentsWithStats(thresholdFrame, thresholdFrame, stats, centroids);
        vector<Vec3b> colors(nLabels);

        colors[0] = Vec3b(0, 0, 0);//background
        for(int label = 1; label < nLabels; ++label){
            colors[label] = Vec3b(100, 150, 200);
        }
        processedFrame.create(thresholdFrame.size(), CV_8UC3);
        for(int r = 0; r < processedFrame.rows; ++r){
            for(int c = 0; c < processedFrame.cols; ++c){
                int label = thresholdFrame.at<int>(r, c);
                Vec3b &pixel = processedFrame.at<Vec3b>(r, c);
                pixel = colors[label];
            }
        }

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