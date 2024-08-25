#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

#include<vector>
using namespace std;

class LinearRegression {
protected:
    int numSamples;
    int numFeatures;
    vector<double> theta;

    vector<double> getPredictions(const vector<vector<double>> &trainFeatures) const;
    virtual double cost(const vector<double> &predictions, const vector<double> &target) const;
    virtual void gradientDescent(const vector<vector<double>> &trainFeatures, 
                                 const vector<double> &target, double learningRate);
    void createMiniBatches(vector<vector<vector<double>>> &miniBatches, 
                           vector<vector<double>> &targetBatches,
                           const vector<vector<double>> &trainFeatures, 
                           const vector<double> &trainTarget, int batchSize);

public:
    LinearRegression();
    LinearRegression(int samples, int features);

    double meanSquaredError(const vector<double> &predictions, const vector<double> &target) const;
    vector<double> predict(const vector<vector<double>> &trainFeatures) const;
    void fit(const vector<vector<double>> &trainFeatures, const vector<vector<double>> &valFeatures, 
             const vector<double> &trainTarget, const vector<double> &valTarget, 
             double learningRate, int epochs, int batchSize);

};

#endif