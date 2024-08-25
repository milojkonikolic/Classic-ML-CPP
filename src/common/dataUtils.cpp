#include<iostream>
#include<fstream>
#include<sstream>
#include<string>
#include<vector>
#include<cmath>

using namespace std;

void readCSV(const string& filename, vector<vector<double>>& features, vector<double>& target) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        return;
    }

    string line;
    while (getline(file, line)) {
        stringstream ss(line);
        vector<double> row;
        string value;

        // First column in features must be populated with 1 (for the intercept term)
        row.push_back(1.0);
        while (getline(ss, value, ',')) {
            row.push_back(stod(value));
        }

        // Get the last element from the row as a target and the rest as features
        target.push_back(row.back());
        row.pop_back();
        features.push_back(row);
    }
    file.close();
}

void saveData(const vector<double>& target, const vector<double>& predictions, const string& filename) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        return;
    }

    for (size_t i = 0; i < target.size(); i++) {
        file << target[i] << " " << predictions[i];
        if (i != target.size() - 1) {
            file << "\n";
        }
    }
    file.close();
}

void scaleFeatures(vector<vector<double>>& features) {
    size_t numSamples = features.size();
    size_t numFeatures = features[0].size();

    for (size_t j = 0; j < numFeatures; j++) {
        double mean = 0.0;
        for (size_t i = 0; i < numSamples; i++) {
            mean += features[i][j];
        }
        mean /= static_cast<double>(numSamples);

        // Scale data only if the mean is outside the range (-1, 1)
        if (mean < -1. || mean > 1.) {
            double stdDev = 0.0;
            for (size_t i = 0; i < numSamples; i++) {
                stdDev += pow(features[i][j] - mean, 2);
            }
            stdDev = sqrt(stdDev / static_cast<double>(numSamples));

            for (size_t i = 0; i < numSamples; i++) {
                features[i][j] = (features[i][j] - mean) / stdDev;
            }
        }
    }
}
