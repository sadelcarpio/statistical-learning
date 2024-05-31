#include <unordered_map>
#include "KnnClassifier.hpp"

int KnnClassifier::getLabelKNeighbors(std::vector<std::pair<int, double>> &distances) const {
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
