#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>

using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {
    if (argc != 2) {
        return -1;
    }

    cv::Mat inputImage = cv::imread(argv[1]);
    if (inputImage.empty()) {
        cout << "Cannot load " << argv[1] << endl;
        return -1;
    }

    resize(inputImage, inputImage, Size(), 0.5, 0.5);
    cout << "input size: " << inputImage.size() << endl;

    std::vector<int> markerIds;
    std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;
    cv::Ptr<cv::aruco::DetectorParameters> parameters = cv::aruco::DetectorParameters::create();
    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    cv::aruco::detectMarkers(inputImage, dictionary, markerCorners, markerIds, parameters, rejectedCandidates);

    for (int i = 0; i < markerIds.size(); i++) {
        cout << markerIds[i] << ": " << markerCorners[i][0] << ", " << markerCorners[i][1] << ", " << markerCorners[i][2] << ", " << markerCorners[i][3] << endl;
    }

    cv::Mat outputImage = inputImage.clone();
    cv::aruco::drawDetectedMarkers(outputImage, markerCorners, markerIds);

    cv::namedWindow("Image", 1);
    cv::imshow("Image", outputImage);

    cv::waitKey(0);

    cv::destroyAllWindows();

    return 0;
}