#pragma once

#include <vector>

class ArrayUtils {
public:
    /**
     * Gets the number of classes (different values) on a given Y vector
     * @return number of distinct values of the vector
     */
    static int getNClasses(std::vector<int>, int);

    /**
     * Calculates the squared euclidean distance between two p-dimensional points
     * @return @code dist = sum(x1_i - x2_i)^2 @endcode
     */
    static double squaredEuclideanDist(std::vector<double>, std::vector<double>, int);
};
