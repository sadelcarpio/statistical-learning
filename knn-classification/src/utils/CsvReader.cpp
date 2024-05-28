#include <fstream>
#include <iostream>
#include <sstream>
#include "CsvReader.hpp"

std::vector<std::vector<std::string>> CsvReader::readCsvFile(const std::string &filename) {
    std::ifstream file(filename);
    std::vector<std::vector<std::string>> data;
    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            std::stringstream lineStream(line);
            std::vector<std::string> row;
            std::string cell;
            while (std::getline(lineStream, cell, ',')) {
                row.push_back(cell);
            }
            data.push_back(row);
        }
    } else {
        std::cerr << "Error opening file: " << filename << std::endl;
    }
    file.close();
    return data;
}
