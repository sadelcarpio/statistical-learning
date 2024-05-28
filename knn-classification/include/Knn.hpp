#pragma once

#include <vector>
#include "Dataset.hpp"

class Knn {
public:
    int k;

    explicit Knn(int);

    void fit(Dataset &) const;
};
