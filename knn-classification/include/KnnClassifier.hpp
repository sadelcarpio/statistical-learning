#include <vector>
#include "Knn.hpp"

class KnnClassifier : public Knn<int> {
private:
    int getLabelKNeighbors(std::vector<std::pair<int, double>> &distances) const override;

public:
    explicit KnnClassifier(int k) : Knn(k) {
    }
};
