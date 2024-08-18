#include<iostream>
#include<vector>
#include<cmath>
#include "linearRegression.h"


using namespace std;

LinearRegression::LinearRegression(int samples, int features)
    :numSamples(samples), numFeatures(features) {
    theta = vector<double>(numFeatures, 0);
    }
LinearRegression::LinearRegression(): LinearRegression(0, 0) {}

vector<double> LinearRegression::getPredictions(vector<vector<double>> trainFeatures) {
    int batchSamples = trainFeatures.size();
    vector<double> predictions(batchSamples, 0.0);
    for (int i = 0; i < batchSamples; i++) {
        for (int j = 0; j < numFeatures; j++) {
            predictions[i] += trainFeatures[i][j] * theta[j];
        }
    }
    return predictions;
}

double LinearRegression::meanSquaredError(vector<double> predictions, vector<double> target) {
    double lossValue = 0;
    int batchSamples = predictions.size();
    for (int i = 0; i < batchSamples; i++)
        lossValue += pow(predictions[i] - target[i], 2);
    
    lossValue = lossValue / (2 * batchSamples);
    return lossValue;
}

double LinearRegression::cost(vector<double> predictions, vector<double> target) {
    return meanSquaredError(predictions, target);
}

void LinearRegression::gradientDescent(vector<vector<double>> trainFeatures, vector<double> target, double learningRate) {
    int batchSamples = trainFeatures.size();
    vector<double> dTheta(batchSamples, 0.0);
    // First calculate predictions in order to use them for calculating gradients
    vector<double> predictions = getPredictions(trainFeatures);

    for (int i = 0; i < batchSamples; i++) {
        for (int j = 0; j < numFeatures; j++) {
            // Calculate gradients dTheta[j]
            dTheta[j] += (predictions[i] - target[i]) * trainFeatures[i][j];
        }
    }
    // Update coefficients
    for (int j = 0; j < numFeatures; j++) {
        theta[j] = theta[j] - learningRate * dTheta[j] / batchSamples;
    }

}

vector<double> LinearRegression::predict(vector<vector<double>> trainFeatures) {
    return getPredictions(trainFeatures);
}

void LinearRegression::createMiniBatches(vector<vector<vector<double>>> &miniBatches, 
                                         vector<vector<double>> &targetBatches,
                                         vector<vector<double>> trainFeatures, 
                                         vector<double> trainTarget, int batchSize) {
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

void LinearRegression::fit(vector<vector<double>> trainFeatures, vector<vector<double>> valFeatures, vector<double> trainTarget, 
                           vector<double> valTarget, double learningRate=0.01, int epochs=10, int batchSize=0) {

    vector<double> trainLoss(epochs, 0.0);
    vector<double> valLoss(epochs, 0.0);
    vector<vector<vector<double>>> miniBatches;
    vector<vector<double>> targetBatches;

    if (batchSize != 0) {
        createMiniBatches(miniBatches, targetBatches, trainFeatures, trainTarget, batchSize);
        cout << "Created " << miniBatches.size() << " batches with sizes: ( ";
        for (vector<double> t : targetBatches)
            cout << t.size() << " ";
        cout << " )" << endl;
        trainFeatures.clear();
    }
    for (int epoch = 0; epoch < epochs; epoch++) {
        if (batchSize == 0) {
            gradientDescent(trainFeatures, trainTarget, learningRate);
        }
        // Divide features into batches
        else {
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
