#include<iostream>
#include<vector>
#include<cmath>
#include "ridgeRegression.h"

using namespace std;

RidgeRegression::RidgeRegression(int samples, int features, double alpha)
    :LinearRegression(samples, features), alpha{alpha} {}

RidgeRegression::RidgeRegression(): RidgeRegression{0, 0, 0} {}

double RidgeRegression::cost(vector<double> predictions, vector<double> target) {
    double regularizationTerm = 0.0;
    // Don't regularize theta[0]
    for (int i = 1; i < theta.size(); i++) {
        regularizationTerm += pow(theta[i], 2);
    }
    regularizationTerm *= alpha / 2.;
    return regularizationTerm + meanSquaredError(predictions, target);
}

void RidgeRegression::gradientDescent(vector<vector<double>> trainFeatures, vector<double> target, 
                                      double learningRate) {
    int batchSamples = trainFeatures.size();
    vector<double> dTheta(batchSamples, 0.0);
    // First calculate predictions in order to use them for calculating gradients
    vector<double> predictions = getPredictions(trainFeatures);

    for (int i = 0; i < batchSamples; i++) {
        for (int j = 0; j < numFeatures; j++) {
            // Calculate gradients dTheta[j]
            dTheta[j] += (predictions[i] - target[i]) * trainFeatures[i][j];
            // Regularization term in cost function
            dTheta[j] += alpha * theta[i];
        }
    }
    // Update coefficients
    for (int j = 0; j < numFeatures; j++) {
        theta[j] = theta[j] - learningRate * dTheta[j] / batchSamples;
    }
}
