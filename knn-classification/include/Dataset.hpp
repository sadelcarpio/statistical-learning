#pragma once

#include <vector>
#include <string>

class Dataset {
public:
    std::vector<std::vector<double>> X;
    std::vector<int> Y;
    int n_classes;
    int n;
    int p;
    std::vector<std::vector<std::string>> data;

    explicit Dataset(std::string &);
};
