#ifndef PROJ2_CSV_UTILS_H
#define PROJ2_CSV_UTILS_H

using namespace std;

void writeToCSV(string filename, vector<string> classNamesDB, vector<vector<double>> featuresDB);
void loadFromCSV(string filename, vector<string> &classNamesDB, vector<vector<double>> &featuresDB);
#endif //PROJ2_CSV_UTILS_H
