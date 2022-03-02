#include <opencv2/opencv.hpp>
#include "processors.h"

using namespace cv;
using namespace std;

Mat threshold(Mat &image) {
    int THRESHOLD = 130;
    Mat temp, processedImage, grayscale;
    cvtColor(image, temp, COLOR_BGR2HSV);
    processedImage = Mat(image.size(), CV_8UC1);

    cvtColor(image, grayscale, COLOR_BGR2GRAY);
//    cout << "mean: " << mean(grayscale) << endl;
//    double min, max;
//    minMaxLoc(grayscale, &min, &max);
//    cout << "max: " << max << endl;
//    cout << "min: " << min << endl;
    for (int i = 0; i < grayscale.rows; i++) {
        for (int j = 0; j < grayscale.cols; j++) {
            if (grayscale.at<uchar>(i, j) <= THRESHOLD) {
                processedImage.at<uchar>(i, j) = 255;
            } else {
                processedImage.at<uchar>(i, j) = 0;
            }
        }
    }
    return processedImage;
}

Mat cleanup(Mat &image) {
    Mat processedImage;
    const Mat kernel = getStructuringElement(MORPH_CROSS, Size(25, 25));
    morphologyEx(image, processedImage, MORPH_CLOSE, kernel);
    return processedImage;
}

Mat getRegions(Mat &image, Mat stats, Mat centroids) {
    Mat temp, processedImage;
    int nLabels = connectedComponentsWithStats(image, temp, stats, centroids);

    // sort temp according to the area of the region
    Mat sortedArea(temp.rows, 1, temp.type());
    sortIdx(temp.col(4), sortedArea, SORT_EVERY_COLUMN);

    cout << "nLabels: " << nLabels << endl;
    cout << "stats: " << stats.at<Vec3b>(1, CC_STAT_AREA);
    int N = 4; // only take the largest 3 regions
    vector<Vec3b> colors(N);

    colors[0] = Vec3b(0, 0, 0);//background
    int i = 1, interval = 80;
    while (i < N && i < nLabels) {
        colors[i] = Vec3b(i * interval, interval, interval);
        i++;
    }
    processedImage.create(temp.size(), CV_8UC3);
    for(int i = 0; i < processedImage.rows; i++){
        for(int j = 0; j < processedImage.cols; j++){
            int label = temp.at<int>(i, j);
            Vec3b &pixel = processedImage.at<Vec3b>(i, j);
            pixel = (label < N) ? colors[label] : colors[0];
        }
    }
    return processedImage;
}