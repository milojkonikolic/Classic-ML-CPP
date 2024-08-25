#include<iostream>
#include<vector>
#include<cmath>
#include "lassoRegression.h"

using namespace std;

LassoRegression::LassoRegression(int samples, int features, double alpha)
    : LinearRegression(samples, features), alpha{alpha} {}

LassoRegression::LassoRegression() : LassoRegression{0, 0, 0} {}

double LassoRegression::cost(const vector<double> &predictions, const vector<double> &target) const {
    double regularizationTerm = 0.0;
    // Don't regularize theta[0]
    for (int i = 1; i < theta.size(); i++) {
        regularizationTerm += abs(theta[i]);
    }
    regularizationTerm *= alpha;
    return regularizationTerm + meanSquaredError(predictions, target);
}

void LassoRegression::gradientDescent(const vector<vector<double>> &trainFeatures, 
                                      const vector<double> &target, double learningRate) {
    int batchSamples = trainFeatures.size();
    vector<double> gradients(numFeatures, 0.0);
    // First calculate predictions in order to use them for calculating gradients
    vector<double> predictions = getPredictions(trainFeatures);

    for (int j = 0; j < numFeatures; j++) {
        for (int i = 0; i < batchSamples; i++) {
            // Calculate gradients gradients[j]
            gradients[j] += (predictions[i] - target[i]) * trainFeatures[i][j];
        }
        // Regularization term in gradient descent
        gradients[j] += alpha * ((theta[j] > 0) ? 1 : ((theta[j] < 0) ? -1 : 0));
    }
    // Update coefficients
    for (int j = 0; j < numFeatures; j++) {
        theta[j] = theta[j] - learningRate * gradients[j] / batchSamples;
    }
}
