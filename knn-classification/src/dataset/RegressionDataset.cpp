#include <vector>
#include <string>
#include "Dataset.hpp"
#include "CsvReader.hpp"

RegressionDataset::RegressionDataset(std::string &data) : Dataset(data) {
    std::vector<std::vector<std::string>> csv_data = CsvReader::readCsvFile(data);
    processCsvData(csv_data);
}

void RegressionDataset::processCsvData(const std::vector<std::vector<std::string>> &csv_data) {
    n = static_cast<int>(csv_data.size());
    p = static_cast<int>(csv_data[0].size() - 1);
    for (const auto &row: csv_data) {
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