#include<iostream>
#include<fstream>
#include<sstream>
#include<string>
#include<vector>
#include<cmath>
#include "linearRegression.h"
#include "lassoRegression.h"
#include "ridgeRegression.h"

using namespace std;

void readCSV(string filename, vector<vector<double>> &features, vector<double> &target) {
    string line;
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file!" << endl;
        return;
    }

    while (getline(file, line)) {
        stringstream ss(line);
        vector<double> row;
        string value;

        // First column in trainFeatures must be populated with 1
        row.push_back(1.0);
        while (getline(ss, value, ',')) {
            row.push_back(stod(value));
        }
        // Get the last element from the row as a target and the rest as a features
        target.push_back(row.back());
        row.pop_back();
        features.push_back(row);
    }
    file.close();
}

int main() {

    string filename = "data.csv";
    vector<vector<double>> features;
    vector<double> target;
    readCSV(filename, features, target);

    string regressionType = "ridge";
        
    if (regressionType == "ridge") {
        std::cout << "Using ridge regression" << endl;
        RidgeRegression model(features.size(), features[0].size(), 0.1);
        model.fit(features, features, target, target, 0.5, 5, 0);
    }
    else if (regressionType == "lasso") {
        std::cout << "Using lasso regression" << endl;
        LassoRegression model(features.size(), features[0].size(), 0.1);
        model.fit(features, features, target, target, 0.5, 5, 0);
    }
    else if (regressionType == "linear") {
        std::cout << "Using linear regression" << endl;
        LinearRegression model(features.size(), features[0].size());
        model.fit(features, features, target, target, 0.5, 5, 0);
    }
    else {
        cerr << "Invalid regression type. Choose one of the following: linear, lasso, ridge." << endl;
    }

    return 0;
}
