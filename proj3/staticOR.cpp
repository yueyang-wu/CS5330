#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "processors.h"

using namespace cv;
using namespace std;

int main() {
    // read an image
    string fp = "/Users/yueyangwu/Desktop/test.png";
    Mat image = imread(fp, 1);
    namedWindow("Original Image", WINDOW_AUTOSIZE);
    imshow("Original Image", image);

    if (!image.data) {
        cout << "No image data\n";
        return -1;
    }

    // threshold
    namedWindow("Threshold", WINDOW_AUTOSIZE);
    Mat thresholdImage = threshold(image);
    imshow("Threshold", thresholdImage);

    // clean up the image
    const Mat kernel = getStructuringElement(MORPH_CROSS, Size(25, 25));
    Mat cleanupImage;
    morphologyEx(thresholdImage, cleanupImage, MORPH_CLOSE, kernel);
    namedWindow("Clean Up", WINDOW_AUTOSIZE);
    imshow("Clean Up", cleanupImage);

    // get the region
    Mat stats, centroids;
    Mat temp, regionImage;
    int nLabels = connectedComponentsWithStats(cleanupImage, temp, stats, centroids);

    // sort temp according to the area of the region
    Mat sortedArea(temp.rows, 1, temp.type());
    sortIdx(temp.col(4), sortedArea, SORT_EVERY_COLUMN);

    cout << "nLabels: " << nLabels << endl;
    cout << "stats: " << stats.at<Vec3b>(1, CC_STAT_AREA);
    int N = 3; // only take the largest 3 regions
    vector<Vec3b> colors(N);

    colors[0] = Vec3b(0, 0, 0);//background
    int i = 1, interval = 80;
    while (i < N && i < nLabels) {
        colors[i] = Vec3b(i * interval, interval, interval);
        i++;
    }
    regionImage.create(temp.size(), CV_8UC3);
    for(int i = 0; i < regionImage.rows; i++){
        for(int j = 0; j < regionImage.cols; j++){
            int label = temp.at<int>(i, j);
            Vec3b &pixel = regionImage.at<Vec3b>(i, j);
            pixel = (label < N) ? colors[label] : colors[0];
        }
    }

    namedWindow("Region", WINDOW_AUTOSIZE);
    imshow("Region", regionImage);


    // check for keypress.
    while (true) {
        char key = waitKey(10);
        // if user types 'q', quit
        if (key == 'q') {
            break;
        }
    }

    // destroy all the windows created
    destroyAllWindows();

    return 0;
}