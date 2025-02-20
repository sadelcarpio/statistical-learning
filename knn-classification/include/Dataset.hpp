#pragma once

#include <vector>
#include <string>
#include <memory>

template <typename T>
class Dataset
{
public:
    std::vector<std::vector<double>> X;
    std::vector<T> Y;
    std::unique_ptr<std::vector<std::vector<std::string>>> csv_data;
    int n;
    int p;

    virtual ~Dataset() = default;

    virtual void processCsvData() = 0;

protected:
    explicit Dataset(std::string&) : n(0), p(0)
    {
    };
};

class RegressionDataset final : public Dataset<double>
{
public:
    explicit RegressionDataset(std::string&);

    void processCsvData() override;
};

class ClassificationDataset final : public Dataset<int>
{
public:
    int n_classes;

    explicit ClassificationDataset(std::string&);

    void processCsvData() override;
};
