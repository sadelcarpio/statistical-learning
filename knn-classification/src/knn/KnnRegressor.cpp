#include <unordered_map>
#include "KnnRegressor.hpp"

double KnnRegressor::getLabelKNeighbors(std::vector<std::pair<int, double>> &distances) const {
    double final_label = 0;
    for (int index = 0; index < k; index++) {
        const double label = dataset->Y[distances[index].first];
        final_label += label;
    }
    return final_label / k;
}
