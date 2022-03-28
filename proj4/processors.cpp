#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include "processors.h"

using namespace std;
using namespace cv;

/*
 * The function constructs a vector of points from a 3D coordinate system
 *
 * @parameter patternSize: the length of x-axis and y-axis needed(the z-axis is always 0)
 * @return a vector<Vec3f> of the points constructed
 */
vector<Vec3f> constructWorldCoordinates(Size patternSize) {
    vector<Vec3f> points;
    for (int i = 0; i < patternSize.height; i++) {
        for (int j = 0; j < patternSize.width; j++) {
            Vec3f coordinates = Vec3f(j, -i, 0);
            points.push_back(coordinates);
        }
    }
    return points;
}

/*
 * The function extracts chessboard corners if the given image contains a valid chessboard object
 * It writes the corners detected into a vector<Point2f>
 *
 * @parameter frame: the given image to be detected
 * @parameter patternSize: the size of the chessboard
 * @parameter corners: the output array of the detected chessboard corners
 * @return a bool value of whether the function detected any valid chessboard corners
 */
bool extractChessboardCorners(Mat &frame, Size patternSize, vector<Point2f> &corners) {
    bool foundCorners = findChessboardCorners(frame, patternSize, corners);
    if (foundCorners) {
        Mat grayscale;
        cvtColor(frame, grayscale, COLOR_BGR2GRAY); // the input image for cornerSubPix must be single-channel
        Size subPixWinSize(10, 10);
        TermCriteria termCrit(TermCriteria::COUNT|TermCriteria::EPS, 1, 0.1);
        cornerSubPix(grayscale, corners, subPixWinSize, Size(-1, -1), termCrit);
    }
    return foundCorners;
}

/*
 * The function extracts corners of ArUco targets from an image contains 35 ArUco targets
 * It only keeps the top left points of each target
 * It writes the top left corners into a vector<Point2f>
 *
 * @parameter frame: the given image to be detected
 * @parameter corners: the output array of the top left corners of the detected ArUco targets
 * @return a bool value of whether the function detected all the ArUco targets
 */
bool extractArucoCorners(Mat &frame, vector<Point2f> &corners) {
    corners.resize(35, Point2f(0, 0));
    vector<int> markerIds;
    vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;
    Ptr<cv::aruco::DetectorParameters> parameters = aruco::DetectorParameters::create();
    Ptr<cv::aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(aruco::DICT_6X6_250);
    aruco::detectMarkers(frame, dictionary, markerCorners, markerIds, parameters, rejectedCandidates);

    for (int i = 0; i < markerIds.size(); i++) {
        int idx = markerIds[i];
        corners[idx] = markerCorners[i][0];
    }

    return markerCorners.size() == 35; // successfully extract Aruco corners
}

/*
 * The function extracts corners of ArUco targets from an image contains 35 ArUco targets
 * It only keeps the outside four corners of the whole target image
 * It writes the top left corners into a vector<Point2f>
 *
 * @parameter frame: the given image to be detected
 * @parameter outsidePoints: the output array of the top left corners of the detected ArUco targets
 */
void arucoOutsidePoints(Mat &frame, vector<Point2f> &outsidePoints) {
    outsidePoints.resize(4, Point2f(0, 0));
    vector<int> markerIds;
    vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;
    Ptr<cv::aruco::DetectorParameters> parameters = aruco::DetectorParameters::create();
    Ptr<cv::aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(aruco::DICT_6X6_250);
    aruco::detectMarkers(frame, dictionary, markerCorners, markerIds, parameters, rejectedCandidates);

    for (int i = 0; i < markerIds.size(); i++) {
        int idx = markerIds[i];
        if (idx == 30) {
            outsidePoints[0] = markerCorners[i][3];
        } else if (idx == 0) {
            outsidePoints[1] = markerCorners[i][0];
        } else if (idx == 34) {
            outsidePoints[2] = markerCorners[i][2];
        } else if (idx == 4) {
            outsidePoints[3] = markerCorners[i][1];
        }
    }
}

/*
 * The function calculates the homography of two images, and overlay the source image to the target image
 *
 * @parameter frame: the target image
 * @parameter displayedFrame: the image to be display after overlaying the two images
 * @parameter image: the source image
 */
void overlayPicture(Mat &frame, Mat &displayedFrame, Mat &image) {
    vector<Point2f> pts_dst;
    arucoOutsidePoints(frame, pts_dst);

    int height = image.size().height;
    int width = image.size().width;
    vector<Point2f> pts_src;
    pts_src.push_back(Point2f(0, 0));
    pts_src.push_back(Point2f(width, 0));
    pts_src.push_back(Point2f(0, height));
    pts_src.push_back(Point2f(width, height));

    // Calculate Homography
    Mat H = findHomography(pts_src, pts_dst);
    // Warp source image to destination based on homography
    Mat im_out;
    warpPerspective(image, im_out, H, frame.size());
    for (int i = 0; i < im_out.rows; i++) {
        for (int j = 0; j < im_out.cols; j++) {
            if (im_out.at<Vec3b>(i, j) != Vec3b(0, 0, 0)) {
                displayedFrame.at<Vec3b>(i, j) = im_out.at<Vec3b>(i, j);
            }
        }
    }
}

/*
 * The function prints out the values of a matrix
 * The value type of the matrix should be double
 *
 * @parameter m: the matrix to be printed
 */
void printMatrix(Mat &m) {
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) {
            cout << m.at<double>(i, j) << ", ";
        }
        cout << "\n";
    }
}

/*
 * The function constructs a set of 3D points which are the vertices of a trapezoidal prism
 *
 * @return a vector<Vec3f> contains the points constructed
 */
vector<Vec3f> constructObjectPoints() {
    vector<Vec3f> objectPoints;
    objectPoints.push_back(Vec3f(1, -1, 1));
    objectPoints.push_back(Vec3f(1, -4, 1));
    objectPoints.push_back(Vec3f(4, -1, 1));
    objectPoints.push_back(Vec3f(4, -4, 1));
    objectPoints.push_back(Vec3f(2, -2, 3));
    objectPoints.push_back(Vec3f(2, -3, 3));
    objectPoints.push_back(Vec3f(3, -2, 3));
    objectPoints.push_back(Vec3f(3, -3, 3));
    return objectPoints;
}

/*
 * The function draw lines between points to form a trapezoidal prism
 *
 * @parameter frame: the frame where the lines are drawn on
 * @parameter p: the given set of points serve as the vertices of the trapezoidal prism
 */
void drawObjects(Mat &frame, vector<Point2f> p) {
    line(frame, p[0], p[1], Scalar(147, 20, 255), 2);
    line(frame, p[0], p[2], Scalar(147, 20, 255), 2);
    line(frame, p[1], p[3], Scalar(147, 20, 255), 2);
    line(frame, p[2], p[3], Scalar(147, 20, 255), 2);
    line(frame, p[4], p[6], Scalar(147, 20, 255), 2);
    line(frame, p[4], p[5], Scalar(147, 20, 255), 2);
    line(frame, p[5], p[7], Scalar(147, 20, 255), 2);
    line(frame, p[6], p[7], Scalar(147, 20, 255), 2);
    line(frame, p[0], p[4], Scalar(147, 20, 255), 2);
    line(frame, p[1], p[5], Scalar(147, 20, 255), 2);
    line(frame, p[2], p[6], Scalar(147, 20, 255), 2);
    line(frame, p[3], p[7], Scalar(147, 20, 255), 2);
}

/*
 * The function projects the 3D points to a 2D image frame, and draws the points on the image frame
 * The points are projected according to the results(camera matrix and intrinsic features) of camera calibration
 *
 * @parameter frame: the image where the points are projected to and drawn on
 * @parameter points: the 3D points need to be projected
 * @parameter rvec: the rotation vector used
 * @parameter tvec: the translation vector used
 * @parameter cameraMatrix: the camera matrix used
 * @parameter distCoeffs: the distortion coefficients used
 */
void projectOutsideCorners(Mat &frame, vector<Vec3f> points, Mat rvec, Mat tvec, Mat cameraMatrix, Mat distCoeffs) {
    vector<Point2f> imagePoints;
    projectPoints(points, rvec, tvec, cameraMatrix, distCoeffs, imagePoints);
    int index[] = {0, 8, 45, 53};
    for (int i : index) {
        circle(frame, imagePoints[i], 5, Scalar(147, 20, 255), 4);
    }
}

/*
 * The function constructs a set of points which are the vertices of a trapezoidal prism using objectPoints()
 * The function then projects the 3D points to a 2D image frame and draws the points on the image frame
 * The function also draws lines between the points to form a trapezoidal prism using drawObjects()
 * The points are projected according to the results(camera matrix and intrinsic features) of camera calibration
 *
 * @parameter frame: the image where the points and lines are projected to and drawn on
 * @parameter rvec: the rotation vector used
 * @parameter tvec: the translation vector used
 * @parameter cameraMatrix: the camera matrix used
 * @parameter distCoeffs: the distortion coefficients used
 */
void projectVirtualObject(Mat &frame, Mat rvec, Mat tvec, Mat cameraMatrix, Mat distCoeffs) {
    vector<Vec3f> objectPoints = constructObjectPoints();
    vector<Point2f> projectedPoints;
    projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs, projectedPoints);
    for (int i = 0; i < projectedPoints.size(); i++) {
        circle(frame, projectedPoints[i], 1, Scalar(147, 20, 255), 4);
    }
    drawObjects(frame, projectedPoints);
}