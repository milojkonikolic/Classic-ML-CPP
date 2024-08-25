#include<iostream>
#include<vector>
#include<cmath>
#include "linearRegression.h"

using namespace std;

LinearRegression::LinearRegression(int samples, int features)
    : numSamples(samples), numFeatures(features) {
    // Initialize parameters with small random values centered around zero
    theta.resize(numFeatures, 0.0);
    for (int j = 0; j < numFeatures; j++) {
        theta[j] = (static_cast<double>(rand()) / RAND_MAX) * 0.01;
    }
}

LinearRegression::LinearRegression(): LinearRegression(0, 0) {}

vector<double> LinearRegression::getPredictions(const vector<vector<double>> &trainFeatures) const {
    int batchSamples = trainFeatures.size();
    vector<double> predictions(batchSamples, 0.0);
    for (int i = 0; i < batchSamples; i++) {
        for (int j = 0; j < numFeatures; j++) {
            predictions[i] += trainFeatures[i][j] * theta[j];
        }
    }
    return predictions;
}

double LinearRegression::meanSquaredError(const vector<double> &predictions, const vector<double> &target) const {
    double lossValue = 0.0;
    int batchSamples = predictions.size();
    for (int i = 0; i < batchSamples; i++)
        lossValue += pow(predictions[i] - target[i], 2);
    
    lossValue = lossValue / (2 * batchSamples);
    return lossValue;
}

double LinearRegression::cost(const vector<double> &predictions, const vector<double> &target) const {
    return meanSquaredError(predictions, target);
}

void LinearRegression::gradientDescent(const vector<vector<double>> &trainFeatures, 
                                       const vector<double> &target, double learningRate) {
    int batchSamples = trainFeatures.size();
    vector<double> gradients(numFeatures, 0.0);
    
    // Calculate predictions to use them for calculating gradients
    vector<double> predictions = getPredictions(trainFeatures);

    for (int j = 0; j < numFeatures; j++) {
        for (int i = 0; i < batchSamples; i++) {
            // Calculate gradients
            gradients[j] += (predictions[i] - target[i]) * trainFeatures[i][j];
        }
        gradients[j] = gradients[j] / double(batchSamples);
    }

    // Update coefficients
    for (int j = 0; j < numFeatures; j++) {
        theta[j] = theta[j] - learningRate * gradients[j];
    }
}

vector<double> LinearRegression::predict(const vector<vector<double>> &trainFeatures) const {
    return getPredictions(trainFeatures);
}

void LinearRegression::createMiniBatches(vector<vector<vector<double>>> &miniBatches, 
                                         vector<vector<double>> &targetBatches,
                                         const vector<vector<double>> &trainFeatures, 
                                         const vector<double> &trainTarget, int batchSize) {
    vector<vector<double>> miniBatch;
    vector<double> targetBatch;
    for (int i = 0; i < numSamples; i++) {
        miniBatch.push_back(trainFeatures[i]);
        targetBatch.push_back(trainTarget[i]);
        if ((i+1) % batchSize == 0) {
            miniBatches.push_back(miniBatch);
            targetBatches.push_back(targetBatch);
            miniBatch.clear();
            targetBatch.clear();
        }
    }
    if (!miniBatch.empty()) {
        miniBatches.push_back(miniBatch);
        targetBatches.push_back(targetBatch);
    }
}

void LinearRegression::fit(const vector<vector<double>> &trainFeatures, const vector<vector<double>> &valFeatures, 
                           const vector<double> &trainTarget, const vector<double> &valTarget, 
                           double learningRate, int epochs, int batchSize) {

    vector<double> trainLoss(epochs, 0.0);
    vector<double> valLoss(epochs, 0.0);
    vector<vector<vector<double>>> miniBatches;
    vector<vector<double>> targetBatches;

    if (batchSize != 0) {
        createMiniBatches(miniBatches, targetBatches, trainFeatures, trainTarget, batchSize);
        cout << "Created " << miniBatches.size() << " batches with sizes: ( ";
        for (const vector<double> &t : targetBatches)
            cout << t.size() << " ";
        cout << " )" << endl;
    }
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        if (batchSize == 0) {
            gradientDescent(trainFeatures, trainTarget, learningRate);
        } else {
            for (int b = 0; b < miniBatches.size(); b++) {
                gradientDescent(miniBatches[b], targetBatches[b], learningRate);
            }
        }

        vector<double> trainPredictions = getPredictions(trainFeatures);
        trainLoss[epoch] = cost(trainPredictions, trainTarget);
        vector<double> valPredictions = getPredictions(valFeatures);
        valLoss[epoch] = cost(valPredictions, valTarget);
        cout << "Epoch: " << epoch << " | Train Loss: " << trainLoss[epoch] << ", Val Loss: " << valLoss[epoch] << endl;
    }
}
