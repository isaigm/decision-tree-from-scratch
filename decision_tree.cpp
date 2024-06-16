#include <vector>
#include <iostream>
#include <map>
#include <cmath>
#include <set>
#include <memory>
#include <cassert>
#include "utils.hpp"

using Subsets    = std::map<InputType, DataSet>;

struct Split
{
    size_t  featureIdx;
    double  informationGain{ 0 };
    Subsets childs;
    double threshold = 0;
};
struct Node
{
    size_t     featureIdx{};
    TargetType classPredicted{};
    double     informationGain{};
    std::map<InputType, std::shared_ptr<Node>> childs;
    bool isLeaf      = true;
    bool isNumerical  = false;
    double threshold = 0;
};
enum class ImpurityCriteria
{
    Entropy,
    GiniIndex
};

struct TreeClassifier
{
public:
    TreeClassifier(size_t _maxDepth, size_t _minSampleSplit, std::vector<ColInfo> &&_colsInfo) :
        maxDepth(_maxDepth), minSampleSplit(_minSampleSplit), colsInfo(_colsInfo)
    {
    }
    void fit(DataSet& dataSet)
    {
        classIndex = dataSet[0].size() - 1;
        root = buildTree(dataSet, 0);
    }
    void evaluate(DataSet& dataSet)
    {
        int success = 0;
        for (auto& row : dataSet)
        {
            TargetType p = predict(root, row);
            if (p == row[classIndex])
                success++;
        }
        std::printf("Success ratio: %f\n", double(success) / double(dataSet.size()));
    }
    void printTree()
    {
        printTree(root, "", "");
    }
private:
    void printTree(std::shared_ptr<Node> node, std::string value, std::string prefix)
    {
        static int space = 4;
        auto addSpace = [](std::string &input)
        {
           for (int i = 0; i < space; i++)
           {
               input.push_back(' ');
           }
        };
        std::string level;
        if (node->isLeaf)
        {
            level = prefix  +"|__ " + value + " -> " + node->classPredicted;
        }
        else
        {
            level = prefix  +"|__ " + value;
        }
        std::cout << level << "\n";
        addSpace(prefix);
        for (auto& child : node->childs)
        {
            if (node->isNumerical)
            {
                printTree(child.second, colsInfo[node->featureIdx].name + child.first, prefix);
            }
            else
            {
                printTree(child.second, colsInfo[node->featureIdx].name + "=" + child.first, prefix);
            }
        }
    }
    TargetType predict(std::shared_ptr<Node> node, Row& input)
    {
        assert(node != nullptr);
        if (node->isLeaf)
        {
            return node->classPredicted;
        }
        size_t featureIdx = node->featureIdx;
        if (node->isNumerical)
        {
            double value = utils::toNumber<double>(input[featureIdx]);
            std::string key = "";
            if (value < node->threshold)
            {
                key = "<" + std::to_string(node->threshold);
            }
            else {
                key = ">=" + std::to_string(node->threshold);
            }
            return predict(node->childs[key], input);

        }
        return predict(node->childs[input[node->featureIdx]], input);
    }
    
    std::shared_ptr<Node> buildTree(DataSet& dataSet, size_t currDepth)
    {
        size_t numSamples = dataSet.size();
        Split bestSplit   = getBestSplit(dataSet);
       
        if (numSamples >= minSampleSplit && currDepth <= maxDepth)
        {
            if (bestSplit.informationGain > 0)
            {
                auto parent = std::make_shared<Node>();
                parent->isLeaf = false;
                for (auto& subset : bestSplit.childs)
                {
                    parent->childs[subset.first] = buildTree(subset.second, currDepth + 1);
                }
                parent->isNumerical = colsInfo[bestSplit.featureIdx].isNumerical;
                parent->featureIdx = bestSplit.featureIdx;
                parent->informationGain = bestSplit.informationGain;
                parent->threshold = bestSplit.threshold;
                return parent;
            }
        }
        auto node = std::make_shared<Node>();
        node->isLeaf = true;
        node->classPredicted = getLeafValue(dataSet);
        return node;
    }
    TargetType getLeafValue(DataSet& dataSet)
    {
        int maxCount = INT_MIN;
        TargetType maxClass{};
        std::map<TargetType, int> classes;
        for (auto& row : dataSet)
        {
            classes[row[classIndex]]++;
        }
        for (auto cl : classes)
        {
            if (cl.second > maxCount)
            {
                maxClass = cl.first;
                maxCount = cl.second;
            }
        }
        return maxClass;
    }
    std::set<InputType> getUniqueFeatureValues(DataSet& dataSet, size_t featureIdx)
    {
        std::set<InputType> values;
        for (auto& row : dataSet)
        {
            values.insert(row[featureIdx]);
        }
        return values;
    }
    Split getBestSplit(DataSet& dataSet)
    {
        
        Split bestSplit;
        double maxGain = float(INT_MIN);
        auto getSplit = [&maxGain, &bestSplit](Subsets& childs, double gain, size_t featureIdx, double threshold)
        {
                if (gain > maxGain)
                {
                    bestSplit.informationGain = gain;
                    bestSplit.featureIdx = featureIdx;
                    bestSplit.childs = std::move(childs);
                    bestSplit.threshold = threshold;
                    maxGain = gain;
                }
        };
        for (size_t featureIdx = 0; featureIdx < classIndex; featureIdx++)
        {
            auto uniqueValues = getUniqueFeatureValues(dataSet, featureIdx);
            double gain = 0;
            double threshold = 0;

            if (colsInfo[featureIdx].isNumerical)
            {
                for (auto threshold : uniqueValues)
                {
                    double value = utils::toNumber<double>(threshold);
                    Subsets childs = splitNumeric(dataSet, featureIdx, value);
                    gain = informationGain(dataSet, childs, ImpurityCriteria::Entropy) / splitInfo(dataSet, childs);
                    getSplit(childs, gain, featureIdx, value);
                }
            }
            else
            {
                Subsets childs = split(dataSet, featureIdx, uniqueValues);
                gain = informationGain(dataSet, childs, ImpurityCriteria::Entropy);
                getSplit(childs, gain, featureIdx, 0);
            }
        }
        
        return bestSplit;
    }
    double informationGain(DataSet& parentSet, Subsets &subsets, ImpurityCriteria mode = ImpurityCriteria::Entropy)
    {
        double ig = 0;
        for (auto& subset : subsets)
        {
            double pr = double(subset.second.size()) / double(parentSet.size());
            if (mode == ImpurityCriteria::Entropy)
            {
                ig += entropy(subset.second) * pr;
            }
            else if (mode == ImpurityCriteria::GiniIndex)
            {
                ig += gini(subset.second) * pr;
            }
        }
        return entropy(parentSet) - ig;
    }
    Subsets split(DataSet& dataSet, size_t featureIdx, std::set<InputType> &values)
    {
        Subsets result;
        for (auto& row : dataSet)
        {
            for (auto& value : values)
            {
                if (row[featureIdx] == value)
                {
                    result[value].push_back(row);
                    break;
                }
            }
        }
        return result;
    }
    Subsets splitNumeric(DataSet& dataSet, size_t featureIdx, double threshold)
    {
        Subsets result;
        for (auto& row : dataSet)
        {
            auto value = utils::toNumber<double>(row[featureIdx]);
            if (value < threshold)
            {
                result["<" + std::to_string(threshold)].push_back(row);
            }
            else
            {
                result[">=" + std::to_string(threshold)].push_back(row);
            }
        }
        return result;
    }
    double splitInfo(DataSet& parentSet, Subsets& subsets)
    {
        double splitInfo = 0;
        for (auto& subset : subsets)
        {
            double pr = double(subset.second.size()) / double(parentSet.size());
            splitInfo += pr * std::log2(pr);
        }
        return -splitInfo;
    }
    std::map<TargetType, int> countClasses(DataSet& dataSet)
    {
        std::map<TargetType, int> classes;
        for (auto& row : dataSet)
        {
            classes[row[classIndex]]++;
        }
        return classes;
    }
    double entropy(DataSet& dataSet)
    {
        auto classes  = countClasses(dataSet);
        double result = 0;
        for (auto& cl : classes)
        {
            double p = double(cl.second) / double(dataSet.size());
            result += p * std::log2(p);
        }
        return -result;
    }
    double gini(DataSet& dataSet)
    {
        auto classes  = countClasses(dataSet);
        double result = 0;
        for (auto& cl : classes)
        {
            double p = double(cl.second) / double(dataSet.size());
            result += p * p;
        }
        return 1 - result;
    }
    size_t classIndex;
    size_t maxDepth;
    size_t minSampleSplit;
    std::shared_ptr<Node> root;
    std::vector<ColInfo> colsInfo;
};
 
int main()
{
    std::vector<std::string> colNames;
    auto pathToCsv = "C:\\Users\\isaig\\OneDrive\\Desktop\\drug200.csv";
    DataSet dataSet;
    auto colsInfo = utils::readFromCSV(dataSet, pathToCsv);
    auto [train, test] = utils::splitTrainTest(dataSet, 0.20, 100);
    TreeClassifier treeClassifier(3, 2, std::move(colsInfo));
    treeClassifier.fit(train);
    treeClassifier.evaluate(test);
    treeClassifier.printTree();
  
    return 0;
}