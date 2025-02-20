#pragma once
#include <vector>
#include "Knn.hpp"

class KnnClassifier final : public Knn<int>
{
    /**
     * Gets the class a point belongs given its relative distances to the other data points, and specified K
     * @param distances vector of pairs of (index, distance) values
     * @return The class that repeats the most from the K nearest neighbors.
     */
    int getLabelKNeighbors(std::vector<std::pair<int, double>>& distances) const override;

public:
    explicit KnnClassifier(int k) : Knn(k)
    {
    }
};
