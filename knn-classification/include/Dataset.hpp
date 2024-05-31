#pragma once

#include <vector>
#include <string>
#include <memory>

template<typename T>
class Dataset {

public:
    std::vector<std::vector<double>> X;
    std::vector<T> Y;
    int n;
    int p;

    virtual ~Dataset() = default;

protected:
    explicit Dataset(std::string &) : n(0), p(0) {};
};

class RegressionDataset : public Dataset<double> {
private:
    void processCsvData(const std::vector<std::vector<std::string>> &csv_data);

public:
    explicit RegressionDataset(std::string &);
};

class ClassificationDataset : public Dataset<int> {
private:
    void processCsvData(const std::vector<std::vector<std::string>> &csv_data);

public:
    int n_classes;

    explicit ClassificationDataset(std::string &);
};
