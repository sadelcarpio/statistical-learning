#include <vector>
#include <string>
#include "Dataset.hpp"
#include "CsvReader.hpp"

RegressionDataset::RegressionDataset(std::string &data) : Dataset(data) {
    csv_data = std::make_unique<std::vector<std::vector<std::string>>>(CsvReader::readCsvFile(data));
}

void RegressionDataset::processCsvData() {
    n = static_cast<int>(csv_data->size());
    p = static_cast<int>(csv_data->at(0).size() - 1);
    for (const auto &row: *csv_data) {
        std::vector<double> x;
        for (size_t j = 0; j < row.size(); j++) {
            if (j == p) {
                Y.push_back(std::stod(row[j]));
            } else {
                x.push_back(std::stod(row[j]));
            }
        }
        X.push_back(x);
    }
}