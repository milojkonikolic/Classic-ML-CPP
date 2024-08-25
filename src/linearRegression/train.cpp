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
#include "dataUtils.cpp"

using namespace std;

void plotPredictions(const vector<double>& target, const vector<double>& predictions) {
    vector<double> xValues(target.size());
    for (int i = 0; i < target.size(); i++) {
        xValues[i] = i;
    }

    // TODO: Implement plotting
}

int main() {
    string filename = "../../data/datasets/Real_Estate.csv";
    string regressionType = "linear";
    double valSize = 0.2;
    vector<vector<double>> features, trainFeatures, valFeatures;
    vector<double> target, trainTarget, valTarget;
    vector<double> valPredictions;
    double valError = 0.0;

    // Read data from CSV
    readCSV(filename, features, target);
    // Feature scaling - standardization
    scaleFeatures(features);
    // Split data into train and validation sets
    trainValSplit(features, target, trainFeatures, trainTarget, valFeatures, valTarget, valSize);

    if (regressionType == "ridge") {
        cout << "Using ridge regression" << endl;
        RidgeRegression model(trainFeatures.size(), trainFeatures[0].size(), 0.1);
        model.fit(trainFeatures, valFeatures, trainTarget, valTarget, 0.05, 25, 0);
        valPredictions = model.predict(valFeatures);
        valError = sqrt(model.meanSquaredError(valPredictions, valTarget));
    }
    else if (regressionType == "lasso") {
        cout << "Using lasso regression" << endl;
        LassoRegression model(trainFeatures.size(), trainFeatures[0].size(), 0.1);
        model.fit(trainFeatures, valFeatures, trainTarget, valTarget, 0.05, 25, 0);
        valPredictions = model.predict(valFeatures);
        valError = sqrt(model.meanSquaredError(valPredictions, valTarget));
    }
    else if (regressionType == "linear") {
        cout << "Using linear regression" << endl;
        LinearRegression model(trainFeatures.size(), trainFeatures[0].size());
        model.fit(trainFeatures, valFeatures, trainTarget, valTarget, 0.05, 25, 0);
        valPredictions = model.predict(valFeatures);
        valError = sqrt(model.meanSquaredError(valPredictions, valTarget));
    }
    else {
        cerr << "Invalid regression type. Choose one of the following: linear, lasso, ridge." << endl;
        return 1;
    }

    cout << endl << "Validation error - RMSE with the final model: " << valError << endl;

    string outFile = "regressorResults.txt";
    cout << "Saving validation targets and predictions to " << outFile 
         << " in the format 'target[i] predictions[i]' row by row" << endl;
    
    saveData(valTarget, valPredictions, outFile);

    return 0;
}
