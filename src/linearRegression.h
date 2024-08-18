#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

#include<vector>
using namespace std;

class LinearRegression {
protected:
    int numSamples;
    int numFeatures;
    vector<double> theta;

    vector<double> getPredictions(vector<vector<double>> trainFeatures);
    virtual double cost(vector<double> predictions, vector<double> target);
    virtual void gradientDescent(vector<vector<double>> trainFeatures, vector<double> target, double learningRate);
    void createMiniBatches(vector<vector<vector<double>>> &miniBatches, vector<vector<double>> &targetBatches,
                           vector<vector<double>> trainFeatures, vector<double> trainTarget, int batchSize);

public:
    LinearRegression();
    LinearRegression(int samples, int features);

    // First column in trainFeatures must be populated with 1
    void fit(vector<vector<double>> trainFeatures, vector<vector<double>> valFeatures, vector<double> trainTarget, 
             vector<double> valTarget, double learningRate, int epochs, int batchSize);
    vector<double> predict(vector<vector<double>> trainFeatures);
    double meanSquaredError(vector<double> predictions, vector<double> target);

};

#endif