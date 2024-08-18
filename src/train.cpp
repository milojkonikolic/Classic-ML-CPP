#include<iostream>
#include<fstream>
#include<sstream>
#include<string>
#include<vector>
#include<cmath>
#include "linearRegression.h"
#include "lassoRegression.h"
#include "ridgeRegression.h"
#include "crossValidation.cpp"

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
    string regressionType = "ridge";
    double valSize = 0.2;
    vector<vector<double>> features, trainFeatures, valFeatures;
    vector<double> target, trainTarget, valTarget;
    vector<double> valPredictions;
    double valError = 0.;
    readCSV(filename, features, target);

    // Split data to train and val
    train_val_split(features, target, trainFeatures, trainTarget, valFeatures, valTarget, valSize);

    cout << "Train size: " << trainFeatures.size() << " " << trainTarget.size() << endl;
    cout << "Val size: " << valFeatures.size() << " " << valTarget.size() << endl;

    if (regressionType == "ridge") {
        std::cout << "Using ridge regression" << endl;
        RidgeRegression model(trainFeatures.size(), trainFeatures[0].size(), 0.1);
        model.fit(trainFeatures, valFeatures, trainTarget, valTarget, 0.5, 5, 0);
        valPredictions = model.predict(valFeatures);
        valError = model.meanSquaredError(valPredictions, valTarget);
    }
    else if (regressionType == "lasso") {
        std::cout << "Using lasso regression" << endl;
        LassoRegression model(trainFeatures.size(), trainFeatures[0].size(), 0.1);
        model.fit(trainFeatures, valFeatures, trainTarget, valTarget, 0.5, 5, 0);
        valPredictions = model.predict(valFeatures);
        valError = model.meanSquaredError(valPredictions, valTarget);
    }
    else if (regressionType == "linear") {
        std::cout << "Using linear regression" << endl;
        LinearRegression model(trainFeatures.size(), trainFeatures[0].size());
        model.fit(trainFeatures, valFeatures, trainTarget, valTarget, 0.5, 5, 0);
        valPredictions = model.predict(valFeatures);
        valError = model.meanSquaredError(valPredictions, valTarget);
    }
    else {
        cerr << "Invalid regression type. Choose one of the following: linear, lasso, ridge." << endl;
    }

    cout << endl << "Validation error with the final model: " << valError << endl;

    return 0;
}
