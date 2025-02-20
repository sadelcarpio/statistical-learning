#pragma once
#include <vector>
#include "Knn.hpp"

class KnnRegressor final : public Knn<double>
{
    /**
     * Gets the response as an average of K nearest responses.
     * @param distances vector of pairs of (index, distance) values
     * @return The average of the responses (y values) of the K nearest neighbors.
     */
    double getLabelKNeighbors(std::vector<std::pair<int, double>>& distances) const override;

public:
    explicit KnnRegressor(const int k) : Knn(k)
    {
    }
};
