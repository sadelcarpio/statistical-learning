#include <iostream>
#include <sstream>
#include "Dataset.hpp"
#include "ArrayUtils.hpp"
#include "CsvReader.hpp"

Dataset::Dataset(std::vector<std::vector<double>> X, std::vector<int> Y, int n) {
    this->X = X;
    this->Y = Y;
    this->n = n;
    this->n_classes = ArrayUtils::getNClasses(Y, n);
}

Dataset::Dataset(std::string &data) {
    std::vector<std::vector<std::string>> csv_data = CsvReader::readCsvFile(data);
    std::vector<std::vector<double>> X;
    std::vector<int> Y;
    int n = csv_data.size();
    int p = csv_data[0].size() - 1;

    for (int i = 0; i < n; i++) {
        std::vector<std::string> row = csv_data[i];
        std::vector<double> x;
        for (int j = 0; j < row.size(); j++) {
            std::stringstream ss(row[j]);
            if (j == p) {
                int label;
                ss >> label;
                Y.push_back(label);
            } else {
                double value;
                ss >> value;
                x.push_back(value);
            }
        }
        X.push_back(x);
    }
    this->X = X;
    this->Y = Y;
    this->n_classes = ArrayUtils::getNClasses(Y, n);
    this->n = n;
    this->p = p;
}
