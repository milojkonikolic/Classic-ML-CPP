#include<iostream>
#include<vector>
#include<cmath>
#include "lassoRegression.h"

using namespace std;

LassoRegression::LassoRegression(int samples, int features, double alpha)
    :LinearRegression(samples, features), alpha{alpha} {}

LassoRegression::LassoRegression(): LassoRegression{0, 0, 0} {}

double LassoRegression::cost(vector<double> predictions, vector<double> target) {
    double regularizationTerm = 0.0;
    // Don't regularize theta[0]
    for (int i = 1; i < theta.size(); i++) {
        regularizationTerm += abs(theta[i]);
    }
    regularizationTerm *= alpha;
    return regularizationTerm + meanSquaredError(predictions, target);
}

void LassoRegression::gradientDescent(vector<vector<double>> trainFeatures, vector<double> target, 
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
            dTheta[j] += alpha * (theta[i] > 0) ? 1 : ((theta[i] < 0) ? -1 : 0);
        }
    }
    // Update coefficients
    for (int j = 0; j < numFeatures; j++) {
        theta[j] = theta[j] - learningRate * dTheta[j] / batchSamples;
    }
}
