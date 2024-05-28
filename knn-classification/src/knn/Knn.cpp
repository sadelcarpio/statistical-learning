#include <iostream>
#include <algorithm>
#include <unordered_map>
#include "Knn.hpp"
#include "ArrayUtils.hpp"

Knn::Knn(int k, int num_classes)
{
    this->k = k;
    this->num_classes = num_classes;
}

void Knn::fit(Dataset &dataset)
{
    for (int i = 0; i < dataset.n; i++)
    {
        std::vector<double> point = dataset.X[i];
        std::vector<std::pair<int, double>> distances;

        for (int j = 0; j < dataset.n; j++)
        {
            if (i == j)
                continue;
            else
            {
                std::vector<double> neighbor = dataset.X[j];
                double distance = ArrayUtils::euclideanDistance(point, neighbor, dataset.p);
                distances.push_back({j, distance});
            }
        }
        std::sort(distances.begin(), distances.end(), [](auto a, auto b)
                  { return a.second < b.second; });
        std::cout << "K Nearest Neighbors of Point " << i << std::endl;
        for (int i = 0; i < k; i++)
        {
            std::cout << dataset.Y[distances[i].first] << " " << distances[i].second << std::endl;
        }
        std::unordered_map<int, int> labels;
        int max_occurrences = 1;
        int final_label;
        for (int i = 0; i < k; i++)
        {
            int label = dataset.Y[distances[i].first];
            auto it = labels.find(label);
            if (it != labels.end())
            {
                it->second++;
            }
            else
            {
                labels[label] = 1;
            }
            if (labels[label] > max_occurrences)
            {
                max_occurrences = labels[label];
                final_label = label;
            }
        }
        std::cout << "Label for Point " << i << ": " << final_label << std::endl;
    }
}
