#ifndef PROJ4_PROCESSORS_H
#define PROJ4_PROCESSORS_H

using namespace std;
using namespace cv;

/*
 * The function constructs a vector of points from a 3D coordinate system
 *
 * @parameter patternSize: the length of x-axis and y-axis needed(the z-axis is always 0)
 * @return a vector<Vec3f> of the points constructed
 */
vector<Vec3f> constructWorldCoordinates(Size patternSize);

/*
 * The function extracts chessboard corners if the given image contains a valid chessboard object
 * It writes the corners detected into a vector<Point2f>
 *
 * @parameter frame: the given image to be detected
 * @parameter patternSize: the size of the chessboard
 * @parameter corners: the output array of the detected chessboard corners
 * @return a bool value of whether the function detected any valid chessboard corners
 */
bool extractChessboardCorners(Mat &frame, Size patternSize, vector<Point2f> &corners);

/*
 * The function extracts corners of ArUco targets from an image contains 35 ArUco targets
 * It only keeps the top left points of each target
 * It writes the top left corners into a vector<Point2f>
 *
 * @parameter frame: the given image to be detected
 * @parameter corners: the output array of the top left corners of the detected ArUco targets
 * @return a bool value of whether the function detected all the ArUco targets
 */
bool extractArucoCorners(Mat &frame, vector<Point2f> &corners);

/*
 * The function extracts corners of ArUco targets from an image contains 35 ArUco targets
 * It only keeps the outside four corners of the whole target image
 * It writes the top left corners into a vector<Point2f>
 *
 * @parameter frame: the given image to be detected
 * @parameter outsidePoints: the output array of the top left corners of the detected ArUco targets
 */
void arucoOutsidePoints(Mat &frame, vector<Point2f> &outsidePoints);

/*
 * The function calculates the homography of two images, and overlay the source image to the target image
 *
 * @parameter frame: the target image
 * @parameter displayedFrame: the image to be display after overlaying the two images
 * @parameter image: the source image
 */
void overlayPicture(Mat &frame, Mat &displayedFrame, Mat &image);

/*
 * The function prints out the values of a matrix
 * The value type of the matrix should be double
 *
 * @parameter m: the matrix to be printed
 */
void printMatrix(Mat &m);

/*
 * The function constructs a set of 3D points which are the vertices of a trapezoidal prism
 *
 * @return a vector<Vec3f> contains the points constructed
 */
vector<Vec3f> constructObjectPoints();

/*
 * The function draw lines between points to form a trapezoidal prism
 *
 * @parameter frame: the frame where the lines are drawn on
 * @parameter p: the given set of points serve as the vertices of the trapezoidal prism
 */
void drawObjects(Mat &frame, vector<Point2f> p);

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
void projectOutsideCorners(Mat &frame, vector<Vec3f> points, Mat rvec, Mat tvec, Mat cameraMatrix, Mat distCoeffs);

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
void projectVirtualObject(Mat &frame, Mat rvec, Mat tvec, Mat cameraMatrix, Mat distCoeffs);

#endif //PROJ4_PROCESSORS_H
