#include "Dataset.hpp"
#include "ArrayUtils.hpp"
#include "CsvReader.hpp"

ClassificationDataset::ClassificationDataset(std::string& data) : Dataset(data), n_classes(0)
{
    csv_data = std::make_unique<std::vector<std::vector<std::string>>>(CsvReader::readCsvFile(data));
}

void ClassificationDataset::processCsvData()
{
    n = static_cast<int>(csv_data->size());
    p = static_cast<int>((*csv_data)[0].size() - 1);
    for (const auto& row : *csv_data)
    {
        std::vector<double> x;
        for (size_t j = 0; j < row.size(); j++)
        {
            if (j == p)
            {
                Y.push_back(std::stoi(row[j]));
            }
            else
            {
                x.push_back(std::stod(row[j]));
            }
        }
        X.push_back(x);
    }
    n_classes = ArrayUtils::getNClasses(Y, n);
}
