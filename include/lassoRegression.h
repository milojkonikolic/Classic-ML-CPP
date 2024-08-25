#ifndef LASSO_REGRESSION_H
#define LASSO_REGRESSION_H

#include<vector>
#include "linearRegression.h"

using namespace std;

class LassoRegression : public LinearRegression {
private:
    double alpha;

    double cost(const vector<double> &predictions, const vector<double> &target) const;
    void gradientDescent(const vector<vector<double>> &trainFeatures, 
                         const vector<double> &target, double learningRate);

public:
    LassoRegression();
    LassoRegression(int samples, int features, double alpha);
};

#endif