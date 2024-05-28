#pragma once
#include <vector>
#include <string>

class Dataset
{
public:
    std::vector<std::vector<double>> X;
    std::vector<int> Y;
    int n_classes;
    int n;
    int p;
    std::vector<std::vector<std::string>> data;
    Dataset(std::vector<std::vector<double>>, std::vector<int>, int);
    Dataset(std::string &);
};
