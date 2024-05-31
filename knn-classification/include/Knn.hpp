#pragma once

#include <vector>
#include <iostream>
#include <algorithm>
#include "Dataset.hpp"
#include "ArrayUtils.hpp"

template<typename TargetType>
class Knn {
protected:
    const Dataset<TargetType> *dataset;

    /**
     * Sorts the distances to a given data point in ascending order.
     * @param distances Reference to a vector of (index, distance) values. Initialized empty.
     * @param point Point where the distances will be calculates
     * @param index If using a point from the train dataset avoid calculating its own distance (0)
     */
    void sortDistances(std::vector<std::pair<int, double>> &distances,
                       const std::vector<double> &point, int index = -1) const {
        for (int j = 0; j < dataset->n; j++) {
            if (index != j) {
                std::vector<double> neighbor = dataset->X[j];
                double distance = ArrayUtils::squaredEuclideanDist(point, neighbor, dataset->p);
                distances.emplace_back(j, distance);
            }
        }
        std::sort(distances.begin(), distances.end(), [](auto a, auto b) { return a.second < b.second; });
    };

    virtual TargetType getLabelKNeighbors(std::vector<std::pair<int, double>> &distances) const = 0;

public:
    int k;

    explicit Knn(int k) : k(k), dataset(nullptr) {
    }

    /**
     * Runs KNN on all data points from the training set
     * @param ds Dataset object. RegressionDataset for regression and ClassificationDataset for KNN Classification
     */
    void fit(const Dataset<TargetType> &ds) {
        this->dataset = &ds;
        for (int i = 0; i < ds.n; i++) {
            std::vector<double> point = ds.X[i];
            std::vector<std::pair<int, double>> distances;
            sortDistances(distances, point, i);
            TargetType label = getLabelKNeighbors(distances);
            std::cout << "Label for train point " << i << ": " << label << std::endl;
        }
    };

    /**
     * Predicts on a batch of X values.
     * @param X vector of rows
     * @return Vector of predictions
     */
    std::vector<TargetType> predict(const std::vector<std::vector<double>> &X) const {
        std::vector<TargetType> labels;
        for (const std::vector<double> &x: X) {
            std::vector<std::pair<int, double>> distances;
            sortDistances(distances, x);
            TargetType label = getLabelKNeighbors(distances);
            labels.push_back(label);
        }
        return labels;
    };
};
