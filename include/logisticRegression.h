#ifndef LOGISTIC_REGRESSION_H
#define LOGISTIC_REGRESSION_H

#include<vector>
using namespace std;


class LogisticRegression {
private:
    int numSamples;
    int numFeatures;
    int numClasses;
    vector<vector<double>> theta;

    vector<vector<double>> softmax(const vector<vector<double>>& features);
    double cost(const vector<vector<double>>& probabilities, const vector<int>& target);
    double crossEntropy(const vector<vector<double>>& probabilities, const vector<int>& target);
    void gradientDescent(const vector<vector<double>>& features, const vector<int>& target, double learningRate);

public:
    LogisticRegression();
    LogisticRegression(int samples, int features, int classes);

    void fit(const vector<vector<double>>& trainFeatures, const vector<vector<double>>& valFeatures, 
             const vector<int>& trainTarget, const vector<int>& valTarget, 
             double learningRate, int epochs, int batchSize);
    vector<int> predict(const vector<vector<double>>& features);
    void calcMetrics(const vector<int>& predictions, const vector<int>& target, vector<double>& accuracy, 
                     vector<double>& recall, vector<double>& precision, vector<double>& f1Score);
};

#endif