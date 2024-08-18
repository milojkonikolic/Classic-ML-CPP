#ifndef LASSO_REGRESSION_H
#define LASSO_REGRESSION_H

#include<vector>
#include "linearRegression.h"

using namespace std;

class LassoRegression : public LinearRegression {
private:
    double alpha;

    double cost(vector<double> predictions, vector<double> target);
    void gradientDescent(vector<vector<double>> trainFeatures, vector<double> target, double learningRate);

public:
    LassoRegression();
    LassoRegression(int samples, int features, double alpha);
};

#endif