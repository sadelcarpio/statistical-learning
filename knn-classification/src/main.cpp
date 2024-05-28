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
    std::vector<std::vector<double>> X_test = {{-1, -1}, {10, 10}};
    auto labels = knn.predict(X_test);
    for (auto &label: labels) {
        std::cout << "Label for test point: " << label << std::endl;
    }
    delete dataset;
    return 0;
}
