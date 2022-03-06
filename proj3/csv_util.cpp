#include <fstream>
#include <string>
#include <vector>
#include "csv_util.h"

using namespace std;

void writeToCSV(string filename, vector<string> classNamesDB, vector<vector<double>> featuresDB) {
    // create an output filestream object
    ofstream csvFile;
    csvFile.open(filename, ofstream::trunc);

    // send data to the stream
    for (int i = 0; i < classNamesDB.size(); i++) {
        // add class name
        csvFile << classNamesDB[i] << ",";
        // add features
        for (int j = 0; j < featuresDB[i].size(); j++) {
            csvFile << featuresDB[i][j];
            if (j != featuresDB[i].size() - 1) {
                csvFile << ","; // no comma at the end of line
            }
        }
        csvFile << "\n";
    }
}

void loadFromCSV(string filename, vector<string> &classNamesDB, vector<vector<double>> &featuresDB) {
    // create an input filestream object
    ifstream csvFile(filename);
    if (!csvFile.is_open()) {
        throw runtime_error("Could not open file");
    }

    // read data line by line
    string line;
    while (getline(csvFile, line)) {
        vector<string> currLine; // all the values from current line
        int pos = 0;
        string token;
        while ((pos = line.find(",")) != string::npos) {
            token = line.substr(0, pos);
            currLine.push_back(token);
            line.erase(0, pos + 1);
        }
        currLine.push_back(line);

        vector<double> currFeature; // all the values except the first one from current line
        if (currLine.size() != 0) {
            classNamesDB.push_back(currLine[0]);
            for (int i = 1; i < currLine.size(); i++) {
                currFeature.push_back(stod(currLine[i]));
            }
            featuresDB.push_back(currFeature);
        }
    }
}