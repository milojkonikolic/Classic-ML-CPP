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

void plotPredictions(vector<double> target, vector<double> predictions) {
    vector<double> xValues;
    for (int i = 0; i < target.size(); i++) {
        xValues.push_back(i);
    }

}

void saveData(vector<double> target, vector<double> predictions, string filename) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << std::endl;
        return;
    }
    for (int i = 0; i < target.size(); i++) {
        file << target[i] << " " << predictions[i];
        if (i != target.size() - 1)
            file << "\n";
    }
    file.close();
}

void scaleFeatures(vector<vector<double>> &features) {
    int numSamples = features.size();
    int numFeatures = features[0].size();

    cout << "features: " << numFeatures << endl;

    // Start from second column since the first column are ones
    for (int j = 1; j < numFeatures; j++) {
        double mean = 0.;
        for (int i = 0; i < numSamples; i++) {
            mean += features[i][j];
        }
        mean /= double(numSamples);
        // Scale data only if the mean is otuside the range (-0.5, 0.5), otherwise scaling is not needed
        if (mean < -0.5 || mean > 0.5) {
            double std = 0.;
            for (int i = 0; i < numSamples; i++) {
                std += pow(mean - features[i][j], 2);
            }
            std = sqrt(std / double(numSamples));
            for (int i = 0; i < numSamples; i++) {
                features[i][j] = (features[i][j] - mean) / std;
            }
        }
    }
}

int main() {

    string filename = "../data/Real_Estate.csv";
    string regressionType = "ridge";
    double valSize = 0.2;
    vector<vector<double>> features, trainFeatures, valFeatures;
    vector<double> target, trainTarget, valTarget;
    vector<double> valPredictions;
    double valError = 0.;
    readCSV(filename, features, target);

    // Feature scaling - standardization
    scaleFeatures(features);

    // Split data to train and val
    train_val_split(features, target, trainFeatures, trainTarget, valFeatures, valTarget, valSize);

    if (regressionType == "ridge") {
        std::cout << "Using ridge regression" << endl;
        RidgeRegression model(trainFeatures.size(), trainFeatures[0].size(), 0.1);
        model.fit(trainFeatures, valFeatures, trainTarget, valTarget, 0.5, 5, 0);
        valPredictions = model.predict(valFeatures);
        valError = sqrt(model.meanSquaredError(valPredictions, valTarget));
    }
    else if (regressionType == "lasso") {
        std::cout << "Using lasso regression" << endl;
        LassoRegression model(trainFeatures.size(), trainFeatures[0].size(), 0.1);
        model.fit(trainFeatures, valFeatures, trainTarget, valTarget, 0.5, 5, 0);
        valPredictions = model.predict(valFeatures);
        valError = sqrt(model.meanSquaredError(valPredictions, valTarget));
    }
    else if (regressionType == "linear") {
        std::cout << "Using linear regression" << endl;
        LinearRegression model(trainFeatures.size(), trainFeatures[0].size());
        model.fit(trainFeatures, valFeatures, trainTarget, valTarget, 0.5, 5, 0);
        valPredictions = model.predict(valFeatures);
        valError = sqrt(model.meanSquaredError(valPredictions, valTarget));
    }
    else {
        cerr << "Invalid regression type. Choose one of the following: linear, lasso, ridge." << endl;
    }

    cout << endl << "Validation error - RMSE with the final model: " << valError << endl;
    string outFile = "linearRegression/regressorResults.txt";
    cout << "Saving val targets and predictions to " << outFile << 
            " in the format 'target[i] predictions[i]' row by row" << endl;
    saveData(valTarget, valPredictions, outFile);

    return 0;
}
