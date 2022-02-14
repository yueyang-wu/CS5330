//
// Created by Yueyang Wu on 2/4/22.
//

#include "features.h"
#include "distanceMetrics.h"
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include "csv_util.h"
#include <string.h>

using namespace cv;
using namespace std;

/*
 * Take three inputs.
 * The first is the path to a directory of images.
 * The second is the feature set.
 * The third is a path to the csv file where the feature vector for each image is written.
 *
 * Write the feature vector for each image to a file
 */
int main(int argc, char *argv[]) {
    char dirname[256];
    // char buffer[256];
    DIR *dirp;
    struct dirent *dp;

    // check for sufficient arguments
    if (argc < 4) {
        cout << "Wrong input." << endl;
        exit(-1);
    }

    // If the csv file already exists, no need to recompute the feature vectors
    FILE *fp = fopen( argv[3], "r" );
    if(fp) {
        cout << "csv file already exists." << endl;
        return 0;
    }

    // get the directory path
    strcpy(dirname, argv[1]);

    // open the directory
    dirp = opendir(dirname);
    if (dirp == NULL) {
        cout << "Cannot open directory " << dirname << endl;
        exit(-1);
    }

    // loop over all the files in the image file listing
    while ((dp = readdir(dirp)) != NULL) {
        // check if the file is an image
        if(strstr(dp->d_name, ".jpg") || strstr(dp->d_name, ".png") ||
            strstr(dp->d_name, ".ppm") || strstr(dp->d_name, ".tif") ) {
            char buffer[256];
            // build the overall filename
            strcpy(buffer, dirname);
            strcat(buffer, "/");
            strcat(buffer, dp->d_name);

            Mat image = imread(buffer);
            vector<float> imageFeature;
            if (!strcmp(argv[2], "baseline")) {
                imageFeature = baseline(image);
            } else if (!strcmp(argv[2], "histogram")) {
                imageFeature = histogram(image);
            } else if (!strcmp(argv[2], "multihisto")) {
                imageFeature = multiHistogram(image);
            } else if (!strcmp(argv[2], "texturecolor")) {
                imageFeature = textureAndColor(image);
            } else if (!strcmp(argv[2], "texture")) {
                imageFeature = texture(image);
            } else if (!strcmp(argv[2], "midtexturecolor")) {
                imageFeature = middleTextureAndColor(image);
            } else if (!strcmp(argv[2], "midcolor")) {
                Mat middle = getMiddle(image);
                imageFeature = histogram(middle);
            } else if (!strcmp(argv[2], "midtexture")) {
                Mat middle = getMiddle(image);
                imageFeature = texture(middle);
            } else if (!strcmp(argv[2], "test")) {
                imageFeature = custom(image);
            } else if (!strcmp(argv[2], "gabortexture")) {
                Mat middle = getMiddle(image);
                imageFeature = multiGaborTextureAndColor(middle);
            } else {
                cout << "No such feature type." << endl;
                exit(-1);
            }
            append_image_data_csv(argv[3], buffer, imageFeature);
        }
    }
    return 0;
}