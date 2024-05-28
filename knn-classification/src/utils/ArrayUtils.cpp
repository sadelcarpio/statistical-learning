#include <vector>
#include <unordered_set>
#include "ArrayUtils.hpp"

int ArrayUtils::getNClasses(std::vector<int> arr, int n) {
    std::unordered_set<int> unique_elements;
    for (int i = 0; i < n; i++) {
        if (unique_elements.count(arr[i]) == 0) {
            unique_elements.insert(arr[i]);
        }
    }
    return static_cast<int>(unique_elements.size());
}

double ArrayUtils::squaredEuclideanDist(std::vector<double> x1, std::vector<double> x2, int dimensions) {
    double sum = 0;
    for (int i = 0; i < dimensions; i++) {
        double diff = x2[i] - x1[i];
        sum += diff * diff;
    }
    return sum;
}
