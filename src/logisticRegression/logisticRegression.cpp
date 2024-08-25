#include<iostream>
#include<vector>
#include<cmath>
#include<cstdlib>
#include "logisticRegression.h"

using namespace std;

LogisticRegression::LogisticRegression(int samples, int features, int classes)
    : numSamples(samples), numFeatures(features), numClasses(classes) {
    // Initialize parameters with small random values centered around zero
    theta.resize(numClasses, vector<double>(numFeatures, 0.0));
    for (int k = 0; k < numClasses; k++) {
        for (int j = 0; j < numFeatures; j++) {
            theta[k][j] = (static_cast<double>(rand()) / RAND_MAX) * 0.01;
        }
    }
}

LogisticRegression::LogisticRegression() : LogisticRegression(0, 0, 0) {}

double LogisticRegression::crossEntropy(const vector<vector<double>>& probabilities, const vector<int>& target) {
    int batchSamples = probabilities.size();
    double lossValue = 0.0;
    for (int i = 0; i < batchSamples; i++) {
        lossValue -= log(probabilities[i][target[i]]);
    }
    return lossValue / batchSamples;
}

double LogisticRegression::cost(const vector<vector<double>>& probabilities, const vector<int>& target) {
    return crossEntropy(probabilities, target);
}

vector<vector<double>> LogisticRegression::softmax(const vector<vector<double>>& features) {
    int batchSamples = features.size();
    vector<vector<double>> scores(batchSamples, vector<double>(numClasses, 0.0));
    vector<vector<double>> probabilities(batchSamples, vector<double>(numClasses, 0.0));

    for (int i = 0; i < batchSamples; i++) {
        double sumExp = 0.0;
        for (int k = 0; k < numClasses; k++) {
            for (int j = 0; j < numFeatures; j++) {
                scores[i][k] += features[i][j] * theta[k][j];
            }
            probabilities[i][k] = exp(scores[i][k]);
            sumExp += probabilities[i][k];
        }
        for (int k = 0; k < numClasses; k++) {
            probabilities[i][k] /= sumExp;
        }
    }
    return probabilities;
}

vector<int> LogisticRegression::predict(const vector<vector<double>>& features) {
    int batchSamples = features.size();
    vector<vector<double>> probabilities = softmax(features);
    vector<int> predictions(batchSamples, -1);

    for (int i = 0; i < batchSamples; i++) {
        double probmax = -1.0;
        int cls = -1;
        for (int k = 0; k < numClasses; k++) {
            if (probabilities[i][k] > probmax) {
                probmax = probabilities[i][k];
                cls = k;
            }
        }
        predictions[i] = cls;
    }
    return predictions;
}

void LogisticRegression::gradientDescent(const vector<vector<double>>& features, const vector<int>& target, double learningRate) {
    int batchSamples = features.size();
    vector<vector<double>> gradients(numClasses, vector<double>(numFeatures, 0.0));
    vector<vector<double>> probabilities = softmax(features);

    vector<vector<double>> oneHotTarget(batchSamples, vector<double>(numClasses, 0.0));
    for (int i = 0; i < batchSamples; i++) {
        oneHotTarget[i][target[i]] = 1.0;
    }

    for (int k = 0; k < numClasses; k++) {
        for (int j = 0; j < numFeatures; j++) {
            for (int i = 0; i < batchSamples; i++) {
                gradients[k][j] += (probabilities[i][k] - oneHotTarget[i][k]) * features[i][j];
            }
            gradients[k][j] /= batchSamples;
        }
    }

    for (int k = 0; k < numClasses; k++) {
        for (int j = 0; j < numFeatures; j++) {
            theta[k][j] -= learningRate * gradients[k][j];
        }
    }
}

void LogisticRegression::calcMetrics(const vector<int>& predictions, const vector<int>& target, vector<double>& accuracy, 
                                     vector<double>& recall, vector<double>& precision, vector<double>& f1Score) {
    int batchSamples = predictions.size();

    for (int k = 0; k < numClasses; k++) {
        double TP = 0.0, FP = 0.0, FN = 0.0, TN = 0.0;
        for (int i = 0; i < batchSamples; i++) {
            if (predictions[i] == k && target[i] == k) {
                TP += 1;
            } else if (predictions[i] == k && target[i] != k) {
                FP += 1;
            } else if (predictions[i] != k && target[i] == k) {
                FN += 1;
            } else {
                TN += 1;
            }
        }
        recall[k] = (TP + FN > 0) ? TP / (TP + FN) : 0;
        precision[k] = (TP + FP > 0) ? TP / (TP + FP) : 0;
        f1Score[k] = (recall[k] + precision[k] > 0) ? 2 * recall[k] * precision[k] / (recall[k] + precision[k]) : 0;
        accuracy[k] = (TP + TN) / batchSamples;
    }
}

void LogisticRegression::fit(const vector<vector<double>>& trainFeatures, const vector<vector<double>>& valFeatures, 
                             const vector<int>& trainTarget, const vector<int>& valTarget, 
                             double learningRate, int epochs, int batchSize) {
    vector<double> trainLoss(epochs, 0.0);
    vector<double> valLoss(epochs, 0.0);
    vector<double> accuracy(numClasses, 0.0), recall(numClasses, 0.0), precision(numClasses, 0.0), f1Score(numClasses, 0.0);

    for (int epoch = 0; epoch < epochs; epoch++) {
        gradientDescent(trainFeatures, trainTarget, learningRate);

        vector<vector<double>> trainProbabilities = softmax(trainFeatures);
        trainLoss[epoch] = cost(trainProbabilities, trainTarget);
        vector<vector<double>> valProbabilities = softmax(valFeatures);
        valLoss[epoch] = cost(valProbabilities, valTarget);

        vector<int> valPredictions = predict(valFeatures);
        calcMetrics(valPredictions, valTarget, accuracy, recall, precision, f1Score);

        cout << "Epoch: " << epoch << " | Train Loss: " << trainLoss[epoch] << ", Val Loss: " << valLoss[epoch] << endl;
        for (int k = 0; k < numClasses; k++) {
            cout << "Class " << k << ": " << " - Accuracy: " << accuracy[k] << ", Recall: " << recall[k]
                 << ", Precision: " << precision[k] << ", F1 Score: " << f1Score[k] << endl;
        }
    }
}
