#pragma once
#include <vector>
#include <sstream>
#include <fstream>
#include <regex>
#include <random>
#include <functional>
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
        std::regex pattern("^[-+]?\\d*\\.?\\d+(e[-+]?\\d+)?$");
        return std::regex_match(str, pattern);
    }
    std::tuple<View, View> splitTrainTest(DataSet &dataSet, float ratio, unsigned int seed)
    {
        std::mt19937 rng(seed);
        std::shuffle(dataSet.begin(), dataSet.end(), rng);

        View train;
        View test;
        
        int samples = dataSet.size() * ratio;
        for (int i = 0; i < samples; i++)
        {
            test.push_back(dataSet[i]);

        }
        for (int i = samples; i < dataSet.size(); i++)
        {
            train.push_back(dataSet[i]);
        }
        return std::make_tuple(train, test);
    }
    
    double toNumber(std::string& str)
    {
        return std::stod(str);
    }
    template <class T>
    std::vector<ColInfo> readFromCSV(std::vector<std::vector<T>>& dataSet, std::string path)
    {
        auto parseLine = [](std::string& line, char sep) -> std::vector<T>
        {
                auto parseType = [](std::string& str)
                {
                      std::stringstream ss(str);
                      T val;
                      ss >> val;
                      return val;
                };
                std::string curr = "";
                std::vector<T> values;
                for (size_t i = 0; i < line.size(); i++)
                {
                    char ch = line[i];
                    if (curr == "NA")
                    {
                        return {};
                    }
                    if (ch == sep)
                    {
                        values.push_back(parseType(curr));
                        curr = "";
                    }
                    else
                    {
                        curr += ch;
                    }
                }
                values.push_back(parseType(curr));
                return values;
        };
        std::string line;
        std::ifstream file(path);
        std::vector<ColInfo> cols;
        bool seenFirstLine = false;
        if (!file.is_open())
        {
            throw std::runtime_error("file doesnt exist");
        }
        bool processFirstRow = false;
        while (std::getline(file, line))
        {
            if (!seenFirstLine)
            {
                seenFirstLine = true;
                auto colNames = parseLine(line, ',');
                for (auto name : colNames)
                {
                    cols.emplace_back();
                    cols.back().name = name;
                }
            }
            else
            {
                auto row = parseLine(line, ',');
                if (!processFirstRow)
                {
                    size_t j = 0;
                    for (auto col : row)
                    {
                        cols[j].isNumerical = isNumber(col);
                        j++;
                    }
                    processFirstRow = true;
                }
                dataSet.push_back(row);
            }
        }
        return cols;
    }
}

