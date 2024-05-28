#pragma once

#include <vector>
#include <string>

class Dataset {
private:
    void processCsvData(const std::vector<std::vector<std::string>> &csv_data);

public:
    std::vector<std::vector<double>> X;
    std::vector<int> Y;
    int n_classes;
    int n;
    int p;
    std::vector<std::vector<std::string>> data;

    explicit Dataset(std::string &);
};
