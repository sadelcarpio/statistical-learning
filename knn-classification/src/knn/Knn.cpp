#include <iostream>
#include <algorithm>
#include <unordered_map>
#include "Knn.hpp"
#include "ArrayUtils.hpp"

Knn::Knn(int k) : k(k), dataset(nullptr) {
}

int Knn::getLabelKNeighbors(std::vector<std::pair<int, double>> &distances) const {
    std::unordered_map<int, int> labels;
    int max_occurrences = 1;
    int final_label;
    for (int index = 0; index < k; index++) {
        int label = dataset->Y[distances[index].first];
        auto it = labels.find(label);
        if (it != labels.end()) {
            it->second++;
        } else {
            labels[label] = 1;
        }
        if (labels[label] > max_occurrences) {
            max_occurrences = labels[label];
            final_label = label;
        }
    }
    return final_label;
}

void Knn::getKNeighbors(std::vector<std::pair<int, double>> &distances,
                        const std::vector<double> &point, int index) const {
    for (int j = 0; j < dataset->n; j++) {
        if (index != j) {
            std::vector<double> neighbor = dataset->X[j];
            double distance = ArrayUtils::squaredEuclideanDist(point, neighbor, dataset->p);
            distances.emplace_back(j, distance);
        }
    }
    std::sort(distances.begin(), distances.end(), [](auto a, auto b) { return a.second < b.second; });
}

void Knn::fit(const Dataset &ds) {
    this->dataset = &ds;
    for (int i = 0; i < ds.n; i++) {
        std::vector<double> point = ds.X[i];
        std::vector<std::pair<int, double>> distances;
        getKNeighbors(distances, point, i);
        int label = getLabelKNeighbors(distances);
        std::cout << "Label for train point " << i << ": " << label << std::endl;
    }
}

std::vector<int> Knn::predict(const std::vector<std::vector<double>>& X) {
    std::vector<int> labels;
    for (const std::vector<double>& x: X) {
        std::vector<std::pair<int, double>> distances;
        getKNeighbors(distances, x);
        int label = getLabelKNeighbors(distances);
        labels.push_back(label);
    }
    return labels;
}
