#include<iostream>
#include<vector>
#include<cmath>
#include <random>
#include <algorithm>

using namespace std;


void train_val_split(vector<vector<double>> features, vector<double>target, 
                     vector<vector<double>> &trainFeatures, vector<double> &trainTarget, 
                     vector<vector<double>> &valFeatures, vector<double> &valTarget,
                     double valSize) {
    if (valSize > 1.0 || valSize < 0.0) {
        cerr << "Invalid val size value. Provide the value between 0 and 1";
    }
    int numSamples = features.size();
    int valLen = floor(numSamples * valSize);

    // Create a vector with all indexes
    vector<int> randomIdxs;
    for (int i = 0; i < numSamples; i++) {
        randomIdxs.push_back(i);
    }
    random_device rd;
    mt19937 gen(rd()); // Seed the generator
    shuffle(randomIdxs.begin(), randomIdxs.end(), gen);
    for (int i = 0; i < numSamples; i++) {
        if (i < valLen) {
            valFeatures.push_back(features[i]);
            valTarget.push_back(target[i]);
        }
        else {
            trainFeatures.push_back(features[i]);
            trainTarget.push_back(target[i]);
        }
    }
}
