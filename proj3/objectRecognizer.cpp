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

    Mat frame;
    bool training = false; // whether the system is in training mode
    map<string, vector<double>> huMomentsMap; // DB to save the Class Name and Features of each object
    while (true) {
        *capdev >> frame; // get a new frame from the camera, treat as a stream
        if (frame.empty()) {
            cout << "frame is empty\n";
            break;
        }
        imshow("Original Video", frame); // display the original image
        char key = waitKey(10); // see if there is a waiting keystroke for the video

        if (key == 't') {
            training = !training;
            if (training) {
                cout << "Training Mode" << endl;
            } else {
                cout << "Inference Mode" << endl;
            }
        }

        // threshold the image, thresholdFrame is single-channel
        Mat thresholdFrame = threshold(frame);

        // clean up the image
        Mat cleanupFrame = cleanup(thresholdFrame);

        // get the region
        Mat labeledRegions, stats, centroids;
        vector<int> topNLabels;
        Mat regionFrame = getRegions(cleanupFrame, labeledRegions, stats, centroids, topNLabels);

        // calculate HuMoments of each region
        for (int n = 0; n < topNLabels.size(); n++) {
            int label = topNLabels[n];
            Mat region;
            region = (labeledRegions == label);
            // double huMoments[7];
            vector<double> huMoments;
            calcHuMoments(region, huMoments);

            if (training) {
                // training mode
                // display current region
                namedWindow("Current Region", WINDOW_AUTOSIZE);
                imshow("Current Region", region);

                // ask the user for a class name
                cout << "Input the class for this object." << endl;
                char k = waitKey(0); // see if there is a waiting keystroke for the region
                string className = getClassName(k);

                // update the DB
                huMomentsMap[className] = huMoments;

                // destroy the window after labeling all the objects
                if (n == topNLabels.size() - 1) {
                    training = false;
                    destroyWindow("Current Region");
                }
            } else {
                // inference mode
                // classify the object
                string className = classifier(huMomentsMap, huMoments);
                cout << "size: " << huMomentsMap.size() << endl;
                for (map<string, vector<double>>::iterator it = huMomentsMap.begin(); it != huMomentsMap.end(); it++) {
                    vector<double> value = it->second;
                    cout << "label: " << it->first << endl;
                    for (int idx = 0; idx < 7; idx++) {
                        cout << value[idx] << " ";
                    }
                }
                cout << endl;
                cout << "className: " << className << endl;
                // overlay classname to the video
                putText(frame, className, Point(centroids.at<int>(label, 0), centroids.at<int>(label, 1)), FONT_HERSHEY_COMPLEX_SMALL, 2, Scalar(0, 0, 255, 255));
            }
        }

        imshow("Processed Video", regionFrame);

        // if user types 'q', quit.
        if (key == 'q') {
            break;
        }
    }

    return 0;
}