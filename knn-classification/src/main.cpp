#include <iostream>
#include <vector>
#include <memory>
#include "KnnClassifier.hpp"
#include "KnnRegressor.hpp"
#include "Dataset.hpp"

int main(int argc, char const *argv[]) {
    std::cout << "KNN Algorithm Demo." << std::endl;

    std::string filename = "data/dummy_data_regression.csv";
    std::unique_ptr<RegressionDataset> reg_dataset = std::make_unique<RegressionDataset>(filename);
    std::cout << "Regression: number of features: " << reg_dataset->p << std::endl;

    KnnRegressor knn_reg(3);
    knn_reg.fit(*reg_dataset);

    std::vector<std::vector<double>> X_test = {{5.5, 5.5}, {4.4, 3.4}};
    auto predictions = knn_reg.predict(X_test);
    for (const auto &pred: predictions) {
        std::cout << "Prediction for test point: " << pred << std::endl;
    }

    filename = "data/dummy_data_classification.csv";
    std::unique_ptr<ClassificationDataset> clf_dataset = std::make_unique<ClassificationDataset>(filename);
    std::cout << "Classification: number of classes: " << clf_dataset->n_classes << std::endl;
    std::cout << "Classification: number of features: " << clf_dataset->p << std::endl;

    KnnClassifier knn_clf(5);
    knn_clf.fit(*clf_dataset);

    X_test = {{-1, -1}, {10, 10}};
    auto labels = knn_clf.predict(X_test);
    for (const auto &label: labels) {
        std::cout << "Label for test point: " << label << std::endl;
    }
    return 0;
}
