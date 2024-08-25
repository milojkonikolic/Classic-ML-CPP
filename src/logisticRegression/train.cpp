#include<iostream>
#include<fstream>
#include<sstream>
#include<string>
#include<vector>
#include<cmath>
#include "logisticRegression.h"
#include "crossValidation.cpp"
#include "dataUtils.cpp"

using namespace std;


void convertTarget2Int(vector<double> targetDouble, vector<int> &target) {
    for (int i = 0; i < targetDouble.size(); i++) {
        target.push_back(int(targetDouble[i]));
    }
}

int main() {

    string filename = "../../data/datasets/weather_classification_data_prepared.csv";
    int classesNum = 4;
    double valSize = 0.2;
    vector<vector<double>> features, trainFeatures, valFeatures;
    vector<double> targetDouble, trainTargetDouble, valTargetDouble;
    vector<int> trainTarget, valTarget;

    readCSV(filename, features, targetDouble);
    // Feature scaling - standardization
    scaleFeatures(features);
    // Split data to train and val
    trainValSplit(features, targetDouble, trainFeatures, trainTargetDouble, valFeatures, 
                  valTargetDouble, valSize);
    convertTarget2Int(trainTargetDouble, trainTarget);
    convertTarget2Int(valTargetDouble, valTarget);

    LogisticRegression model(trainFeatures.size(), trainFeatures[0].size(), classesNum);
    model.fit(trainFeatures, valFeatures, trainTarget, valTarget, 0.01, 10, 0);

    return 0;
}


