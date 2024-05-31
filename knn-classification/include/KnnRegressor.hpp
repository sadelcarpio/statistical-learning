#include <vector>
#include "Knn.hpp"

class KnnRegressor : public Knn<double> {
private:
    /**
     * Gets the response as an average of K nearest responses.
     * @param distances vector of pairs of (index, distance) values
     * @return The average of the responses (y values) of the K nearest neighbors.
     */
    double getLabelKNeighbors(std::vector<std::pair<int, double>> &distances) const override;

public:
    explicit KnnRegressor(int k) : Knn(k) {
    }
};
