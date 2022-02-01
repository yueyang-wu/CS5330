//
// Created by Yueyang Wu on 1/27/22.
//

#include "filters.h"
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <chrono>


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
    Mat frame, processedFrame; // original image and image after applying any filters/effects

    // create VideoWriter object
    VideoWriter savedVideo("savedVideo.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, Size(refS.width, refS.height));
    char mode = ' '; // identify the filter/effects applied on the frame
    bool videoWrite = false; // identify whether the program should write the image into the video sequence
    string videoMeme; // meme added to the saved video
    int brightness = 0; // brightness of the video, default to be the actual brightness

    for (;;) {
        auto baseStart = chrono::steady_clock::now();
        *capdev >> frame; // get a new frame from the camera, treat as a stream
        if (frame.empty()) {
            cout << "frame is empty\n";
            break;
        }
        auto baseEnd = chrono::steady_clock::now();
        int baseTimeMS = chrono::duration_cast<chrono::milliseconds>(baseEnd - baseStart).count();

        // see if there is a waiting keystroke
        char key = waitKey(10);
        if (key == ' ' || key == 'g' || key == 'h' || key == 'b' || key == 'x' || key == 'y' || key == 'm' || key == 'l' || key == 'c' || key == 'f') {
            mode = key;
        }

        // apply filters/effects to the stream
        auto start = chrono::steady_clock::now();
        if (mode == ' ') {
            // if user types space, display the original version of the image
            frame.copyTo(processedFrame);
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
            Mat resultFrame; // CV_16SC3
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
        // change the brightness of the video
        if (key == '1') {
            // increase brightness by 20
            brightness += 20;
        } else if (key == '2') {
            // decrease brightness by 20
            brightness -= 20;
        } else if (key == '3') {
            // reset to actual brightness
            brightness = 0;
        }
        processedFrame.convertTo(processedFrame, -1, 1, brightness);
        auto end = chrono::steady_clock::now();
        int processingTimeMS = chrono::duration_cast<chrono::milliseconds>(end - start).count();

        cout << mode << endl; // print the current mode in terminal
        imshow("Video", processedFrame);

        // if user types 'v', start to save video sequence
        // if user types 'v' again, stop saving video sequence
        if (key == 'v' && !videoWrite) {
            // start to save video sequence and ask user for a meme
            videoWrite = true;
            cout << "Write your meme here: " << endl;
            getline(cin, videoMeme);
        } else if (key == 'v' && videoWrite) {
            // stop saving video sequence
            videoWrite = false;
        }
        if (videoWrite) {
            cout << "saving video sequence" << endl;
            // add meme to video
            putText(processedFrame, videoMeme, Point(refS.width / 2, refS.height / 2), FONT_HERSHEY_COMPLEX_SMALL, 2, Scalar(0, 0, 255, 255));
            if (processedFrame.channels() == 1) {
                cvtColor(processedFrame, processedFrame, COLOR_GRAY2BGR); // video writer requires 3-channel images
            }
            int numOfFrames = (processingTimeMS + baseTimeMS) / baseTimeMS;
            for (int i = 0; i < numOfFrames; i++) {
                savedVideo.write(processedFrame);
            }
        }

        // if user types 'q', quit.
        if (key == 'q') {
            break;
        }

        // if user types 's', save the original image and processed image
        // to the same directory of the executable
        if (key == 's') {
            cout << "image saved" << endl;
            // ask for a meme for the saved processed image
            string photoMeme;
            cout << "Write your meme here: " << endl;
            getline(cin, photoMeme);
            putText(processedFrame, photoMeme, Point(refS.width / 2, refS.height / 2), FONT_HERSHEY_COMPLEX_SMALL, 2, Scalar(0, 0, 255, 255));

            imwrite("original.jpg", frame);
            imwrite("processed.jpg", processedFrame);
        }
    }

    // clean up
    delete capdev;
    savedVideo.release();
    destroyAllWindows();

    return 0;
}
