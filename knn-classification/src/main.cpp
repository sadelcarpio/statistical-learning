#include <iostream>
#include <vector>
#include "Knn.hpp"
#include "Dataset.hpp"

int main(int argc, char const *argv[]) {
    std::cout << "KNN Algorithm Demo." << std::endl;
    std::string filename = "data/dummy_data.csv";
    auto *dataset = new Dataset(filename);
    std::cout << "Number of classes: " << dataset->n_classes << std::endl;
    std::cout << "Number of features: " << dataset->p << std::endl;
    Knn knn(5);
    knn.fit(*dataset);
    delete dataset;
    return 0;
}