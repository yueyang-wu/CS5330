//
// Created by Yueyang Wu on 1/27/22.
//

#include "filters.h"
#include <stdio.h>
#include <opencv2/opencv.hpp>


using namespace cv;
using namespace std;

int main(int argc, char *argv[]) {
    VideoCapture *capdev;

    // open the video device
    capdev = new VideoCapture(0);
    if (!capdev->isOpened()) {
        cout << "Unable to open video device\n";
        return -1;
    }

    // get some properties of the image
    Size refS((int) capdev->get(cv::CAP_PROP_FRAME_WIDTH),
              (int) capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
    cout << "Expected size: " << refS.width << " " << refS.height << "\n";

    namedWindow("Video", 1); // identifies a window
    Mat frame, processedFrame;

    char mode = ' ';

    int count = 0; // the number of 's' typed
    for (;;) {
        *capdev >> frame; // get a new frame from the camera, treat as a stream
        if (frame.empty()) {
            cout << "frame is empty\n";
            break;
        }

        // see if there is a waiting keystroke
        char key = waitKey(10);
        if (key == ' ' || key == 'g' || key == 'h' || key == 'b' || key == 'x' || key == 'y' || key == 'm' || key == 'l' || key == 'c' || key == 'f') {
            mode = key;
        }

        if (mode == ' ') {
            // if user types space, display the original version of the image
            processedFrame = frame;
        } else if (mode == 'g') {
            // if user types 'g', display a greyscale version of the image
            // using openCV cvtColor function
            cvtColor(frame, processedFrame, COLOR_BGR2GRAY);
        } else if (mode == 'h') {
            // if user types 'h', display an alternative greyscale version of the image
            // using greyscale function in filter.cpp
            greyscale(frame, processedFrame);
        } else if (mode == 'b') {
            // if user types 'b', display a blurred version of the image
            blur5x5(frame, processedFrame);
        } else if (mode == 'x') {
            // if user types 'x', display the sobelX version of the image
            Mat resultFrame; // CV_16SC3
            sobelX3x3(frame, resultFrame);
            convertScaleAbs(resultFrame, processedFrame);
        } else if (mode == 'y') {
            // if user types 'y', display the sobelY version of the image
            Mat resultFrame; // CV_16SC3
            sobelY3x3(frame, resultFrame);
            convertScaleAbs(resultFrame, processedFrame);
        } else if (mode == 'm') {
            // if user types 'm', display the gradient magnitude image
            Mat sobelX, sobelY;
            sobelX3x3(frame, sobelX);
            sobelY3x3(frame, sobelY);
            Mat resultFrame;
            magnitude(sobelX, sobelY, resultFrame);
            convertScaleAbs(resultFrame, processedFrame);
        } else if (mode == 'l') {
            // if user types 'l', display a blurred and quantized version of the image
            blurQuantize(frame, processedFrame, 15);
        } else if (mode == 'c') {
            // if user types 'c', display a cartoon version of the image
            cartoon(frame, processedFrame, 15, 15);
        } else if (mode == 'f') {
            // if user types 'f', display a bilateral version of the image
            Mat dst; // bilateralFilter function requires that src.data != dst.data
            bilateralFilter(frame, dst, 15, 80, 80);
            processedFrame = dst;
        }
        cout << mode << endl; // print the current mode in terminal
        imshow("Video", processedFrame);

        // if user types 'q', quit.
        if (key == 'q') {
            break;
        }

        // if user types 's', save the original image and processed image
        // to the same directory of the executable
        if (key == 's') {
            cout << "image saved\n";
            imwrite("original.jpg", frame);
            imwrite("processed.jpg", processedFrame);
        }
    }

    destroyAllWindows();
    delete capdev;
    return 0;
}
