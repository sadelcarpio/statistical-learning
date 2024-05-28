#pragma once

#include <string>
#include <vector>

class CsvReader
{
public:
    static std::vector<std::vector<std::string>> readCsvFile(const std::string& filename);
};
