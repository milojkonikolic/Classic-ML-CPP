#ifndef RIDGE_REGRESSION_H
#define RIDGE_REGRESSION_H

#include<vector>
#include "linearRegression.h"

using namespace std;

class RidgeRegression : public LinearRegression {
private:
    double alpha;

    double cost(vector<double> predictions, vector<double> target);
    void gradientDescent(vector<vector<double>> trainFeatures, vector<double> target, double learningRate);

public:
    RidgeRegression();
    RidgeRegression(int samples, int features, double alpha);
};

#endif