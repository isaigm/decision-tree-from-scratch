
#ifndef UTILS_HPP
#define UTILS_HPP
#include <vector>
#include <sstream>
#include <fstream>
#include <regex>
#include <random>
#include <functional>
#include <string>
#include <algorithm> 
#include <stdexcept> 
#include <tuple>

using InputType = std::string;
using TargetType = std::string;
using Row = std::vector<InputType>;
using View = std::vector<std::reference_wrapper<Row>>;
using DataSet =  std::vector<Row>; 
struct ColInfo
{
    bool isNumerical{};
    std::string name;
};

namespace utils
{
    bool isNumber(const std::string& str) {
        try {
            std::stod(str);
        } catch (const std::invalid_argument& ia) {
            return false;
        } catch (const std::out_of_range& oor) {
            return false;
        }
        return true;
    }

    std::tuple<View, View> splitTrainTest(DataSet &dataSet, float testRatio, unsigned int seed)
    {
        std::mt19937 rng(seed);
        std::shuffle(dataSet.begin(), dataSet.end(), rng);

        View train;
        View test;
        
        int test_samples = dataSet.size() * testRatio;
        
        for (int i = 0; i < test_samples; i++)
        {
            test.push_back(dataSet[i]);
        }
        for (int i = test_samples; i < dataSet.size(); i++)
        {
            train.push_back(dataSet[i]);
        }
        return std::make_tuple(train, test);
    }
    
    double toNumber(const std::string& str)
    {
        return std::stod(str);
    }

    std::vector<ColInfo> readFromCSV(DataSet& dataSet, const std::string& path)
    {
        auto parseLine = [](const std::string& line, char sep) -> std::vector<std::string>
        {
                std::string curr = "";
                std::vector<std::string> values;
                std::stringstream ss(line);
                
                while(std::getline(ss, curr, sep))
                {
                    values.push_back(curr);
                }
                return values;
        };

        std::ifstream file(path);
        if (!file.is_open())
        {
            throw std::runtime_error("El archivo no existe: " + path);
        }

        std::string line;
        std::vector<ColInfo> cols;
        
        if (std::getline(file, line))
        {
            auto colNames = parseLine(line, ',');
            for (const auto& name : colNames)
            {
                cols.emplace_back();
                cols.back().name = name;
            }
        }

        while (std::getline(file, line))
        {
            auto row = parseLine(line, ',');
            if (!row.empty() && row.size() == cols.size()) {
                 dataSet.push_back(row);
            }
        }
        
        file.close();

        if (dataSet.empty()) {
            return cols;
        }

        for (size_t j = 0; j < cols.size() - 1; ++j) 
        {
            bool all_numeric = true;
            for (const auto& row : dataSet)
            {
                if (!isNumber(row[j]))
                {
                    all_numeric = false;
                    break;
                }
            }
            cols[j].isNumerical = all_numeric;
        }
        cols.back().isNumerical = false;

        return cols;
    }
}
#endif // UTILS_HPP