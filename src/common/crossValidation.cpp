#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

using namespace std;

void trainValSplit(const vector<vector<double>>& features, const vector<double>& target, 
                   vector<vector<double>>& trainFeatures, vector<double>& trainTarget, 
                   vector<vector<double>>& valFeatures, vector<double>& valTarget,
                   double valSize) {
    if (valSize <= 0.0 || valSize >= 1.0) {
        cerr << "Invalid validation size value. Please provide a value between 0 and 1." << endl;
        return;
    }

    int numSamples = features.size();
    int valLen = static_cast<int>(floor(numSamples * valSize));

    // Generate a vector of indexes from 0 to numSamples-1
    vector<int> indices(numSamples);
    iota(indices.begin(), indices.end(), 0);

    // Shuffle the indices to randomize the selection of training and validation data
    random_device rd;
    mt19937 gen(rd());
    shuffle(indices.begin(), indices.end(), gen);

    // Split the data into training and validation sets based on shuffled indices
    for (int i = 0; i < numSamples; i++) {
        int idx = indices[i];
        if (i < valLen) {
            valFeatures.push_back(features[idx]);
            valTarget.push_back(target[idx]);
        } else {
            trainFeatures.push_back(features[idx]);
            trainTarget.push_back(target[idx]);
        }
    }
}
