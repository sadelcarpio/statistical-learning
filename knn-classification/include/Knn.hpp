#pragma once

#include <vector>
#include "Dataset.hpp"

class Knn {
private:
    const Dataset *dataset;

    int getLabelKNeighbors(std::vector<std::pair<int, double>> &distances) const;

    void getKNeighbors(std::vector<std::pair<int, double>> &distances,
                       const std::vector<double> &point, int index = -1) const;

public:
    int k;

    explicit Knn(int);

    void fit(const Dataset &);

    std::vector<int> predict(const std::vector<std::vector<double>> &X);
};
