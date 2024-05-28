#pragma once

#include <vector>
#include "Dataset.hpp"

class Knn {
public:
    int k, num_classes;

    Knn(int, int);

    void fit(Dataset &);
};
