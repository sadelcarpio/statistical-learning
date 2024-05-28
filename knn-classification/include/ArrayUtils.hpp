#pragma once

#include <vector>

class ArrayUtils {
public:
    static int getNClasses(std::vector<int>, int);

    static double squaredEuclideanDist(std::vector<double>, std::vector<double>, int);
};
