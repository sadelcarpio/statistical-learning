#include <vector>
#include "Knn.hpp"

class KnnRegressor : public Knn<double> {
private:
    double getLabelKNeighbors(std::vector<std::pair<int, double>> &distances) const override;

public:
    explicit KnnRegressor(int k) : Knn(k) {
    }
};
