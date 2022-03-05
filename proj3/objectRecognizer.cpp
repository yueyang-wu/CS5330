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

        char key = waitKey(10); // see if there is a waiting keystroke for the video

        // switch between training mode and inference mode
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

        // for each region, get bounding box and calculate HuMoments
        for (int n = 0; n < topNLabels.size(); n++) {
            int label = topNLabels[n];
            Mat region;
            region = (labeledRegions == label);

            Moments m = moments(region, true);
            double centroidX = centroids.at<double>(label, 0);
            double centroidY = centroids.at<double>(label, 1);
            double alpha = 1.0 / 2.0 * atan2(2 * m.mu11, m.mu20 - m.mu02);

            // get the bounding box of this region
            RotatedRect boundingBox = getBoundingBox(region, centroids, label);
            drawLine(frame, centroidX, centroidY, alpha, Scalar(0, 0, 255));
            drawBoundingBox(frame, boundingBox, Scalar(0, 255, 0));

            // calculate hu moments of this region
            vector<double> huMoments;
            calcHuMoments(region, huMoments);

            if (training) {
                // training mode
                // display current region in binary form
                namedWindow("Current Region", WINDOW_AUTOSIZE);
                imshow("Current Region", region);

                // ask the user for a class name
                cout << "Input the class for this object." << endl;
                char k = waitKey(0); // see if there is a waiting keystroke for the region
                string className = getClassName(k); //see function for a detailed mapping

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
                putText(frame, className, Point(centroids.at<double>(label, 0), centroids.at<double>(label, 1)), FONT_HERSHEY_SIMPLEX, 2, Scalar(0, 0, 255));
            }
        }

        imshow("Original Video", frame);
        imshow("Processed Video", regionFrame);

        // if user types 'q', quit.
        if (key == 'q') {
            break;
        }
    }

    return 0;
}