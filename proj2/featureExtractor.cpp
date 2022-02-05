//
// Created by Yueyang Wu on 2/4/22.
//

#include "features.h"
#include "distanceMetrics.h"
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main(int argc, char *argv[]) {
    Mat target = imread(argv[1], 1);
    Mat image = imread(argv[2], 1);

    Mat targetFeature = baseline(target);
    Mat imageFeature = baseline(image);

    cout << sumOfSquareDifference(targetFeature, imageFeature) << endl;

    return 0;
}