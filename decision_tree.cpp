#include <vector>
#include <iostream>
#include <map>
#include <cmath>
#include <set>
#include <memory>
#include <cassert>
#include <climits>
#include <deque>
#include "utils.hpp"

using Subsets = std::map<InputType, DataSet>;

struct Split
{
    size_t featureIdx;
    double informationGain{0};
    Subsets childs;
    double threshold = 0;
};
struct Node
{
    size_t featureIdx{};
    TargetType classPredicted{};
    double informationGain{};
    std::map<InputType, std::shared_ptr<Node>> childs;
    bool isLeaf = true;
    bool isNumerical = false;
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
    TreeClassifier(size_t _maxDepth, size_t _minSampleSplit, std::vector<ColInfo> &&_colsInfo) : maxDepth(_maxDepth), minSampleSplit(_minSampleSplit), colsInfo(_colsInfo)
    {
    }
    void fit(DataSet &dataSet)
    {
        classIndex = dataSet[0].size() - 1;
        root = buildTree(dataSet, 0);
    }
    void evaluate(DataSet &dataSet)
    {
        int success = 0;
        for (auto &row : dataSet)
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
            level = prefix + "|__ " + value + " -> " + node->classPredicted;
        }
        else
        {
            level = prefix + "|__ " + value;
        }
        std::cout << level << "\n";
        addSpace(prefix);
        for (auto &child : node->childs)
        {
            if (node->isNumerical)
            {
                printTree(child.second, colsInfo[node->featureIdx].name + child.first + std::to_string(node->threshold), prefix);
            }
            else
            {
                printTree(child.second, colsInfo[node->featureIdx].name + "=" + child.first, prefix);
            }
        }
    }
    TargetType predict(std::shared_ptr<Node> node, Row &input)
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
                key = "<";
            }
            else
            {
                key = ">=";
            }
            return predict(node->childs[key], input);
        }
        return predict(node->childs[input[node->featureIdx]], input);
    }

    std::shared_ptr<Node> buildTree(DataSet &dataSet, size_t currDepth)
    {
        size_t numSamples = dataSet.size();

        Split bestSplit = getBestSplit(dataSet);
        if (numSamples >= minSampleSplit && currDepth <= maxDepth)
        {
            if (bestSplit.informationGain > 0)
            {
                auto parent = std::make_shared<Node>();
                parent->isLeaf = false;
                for (auto &subset : bestSplit.childs)
                {
                    parent->childs[subset.first] = buildTree(subset.second, currDepth + 1);
                }
                parent->isNumerical     = colsInfo[bestSplit.featureIdx].isNumerical;
                parent->featureIdx      = bestSplit.featureIdx;
                parent->informationGain = bestSplit.informationGain;
                parent->threshold       = bestSplit.threshold;
                return parent;
            }
        }
        auto node = std::make_shared<Node>();
        node->isLeaf = true;
        node->classPredicted = getLeafValue(dataSet);
        return node;
    }
    TargetType getLeafValue(DataSet &dataSet)
    {
        int maxCount = INT_MIN;
        TargetType maxClass{};
        std::map<TargetType, int> classes;
        for (auto &row : dataSet)
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
    std::set<InputType> getUniqueFeatureValues(DataSet &dataSet, size_t featureIdx)
    {
        std::set<InputType> values;
        for (auto &row : dataSet)
        {
            values.insert(row[featureIdx]);
        }
        return values;
    }
    std::vector<double> toNumericValues(std::set<InputType> &values)
    {
        std::vector<double> result;
        for (auto val : values)
        {
            result.push_back(utils::toNumber<double>(val));
        }
        std::sort(result.begin(), result.end());
        return result;
    }
    Split getBestSplit(DataSet &dataSet)
    {
        Split bestSplit;
        double maxGain = float(INT_MIN);
        auto getSplit = [&maxGain, &bestSplit](Subsets &childs, double gain, size_t featureIdx, double threshold)
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
        double parentSetImpurity = gini(dataSet);
        for (size_t featureIdx = 0; featureIdx < classIndex; featureIdx++)
        {
            double gain = 0;

            if (colsInfo[featureIdx].isNumerical)
            {
                std::sort(dataSet.begin(), dataSet.end(), [featureIdx] (auto &r1, auto &r2)
                {
                    return utils::toNumber<double>(r1[featureIdx]) < utils::toNumber<double>(r2[featureIdx]);
                });
                std::deque<Row> leftSet;
                std::deque<Row> rightSet;
                for (size_t i = 0; i < dataSet.size(); i++)
                {
                    rightSet.push_back(dataSet[i]);
                }
                for (size_t i = 0; i < dataSet.size() - 1; i++)
                {
                    if (dataSet[i][featureIdx] != dataSet[i + 1][featureIdx])
                    {
                        double v1 = utils::toNumber<double>(dataSet[i][featureIdx]);
                        double v2 = utils::toNumber<double>(dataSet[i + 1][featureIdx]);
                        double threshold = (v1 + v2) / 2;
                        leftSet.push_back(dataSet[i]);
                        rightSet.pop_front();
                        Subsets childs;
                        std::vector<Row> left(leftSet.begin(), leftSet.end());
                        std::vector<Row> right(rightSet.begin(), rightSet.end());
                        childs["<"] = std::move(left);
                        childs[">="] = std::move(right);
                        gain = informationGain(dataSet.size(), parentSetImpurity, childs, ImpurityCriteria::GiniIndex) / splitInfo(dataSet, childs);
                        getSplit(childs, gain, featureIdx, threshold);
                    }
                    else
                    {
                        leftSet.push_back(dataSet[i]);
                        rightSet.pop_front();
                    }
                }
            }
            else
            {
                auto uniqueValues = getUniqueFeatureValues(dataSet, featureIdx);
                Subsets childs = split(dataSet, featureIdx, uniqueValues);
                gain = informationGain(dataSet.size(), parentSetImpurity, childs, ImpurityCriteria::GiniIndex) / splitInfo(dataSet, childs);
                getSplit(childs, gain, featureIdx, 0);
            }
        }

        return bestSplit;
    }
    double informationGain(size_t parentSize, double parentSetImpurity, Subsets &subsets, ImpurityCriteria mode = ImpurityCriteria::Entropy)
    {
        double ig = 0;
        for (auto &subset : subsets)
        {
            double pr = double(subset.second.size()) / double(parentSize);
            if (mode == ImpurityCriteria::Entropy)
            {
                ig += entropy(subset.second) * pr;
            }
            else if (mode == ImpurityCriteria::GiniIndex)
            {
                ig += gini(subset.second) * pr;
            }
        }
       return parentSetImpurity - ig;
    }
    Subsets split(DataSet &dataSet, size_t featureIdx, std::set<InputType> &values)
    {
        Subsets result;
        for (auto &row : dataSet)
        {
            for (auto &value : values)
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
    double splitInfo(DataSet &parentSet, Subsets &subsets)
    {
        double splitInfo = 0;
        for (auto &subset : subsets)
        {
            double pr = double(subset.second.size()) / double(parentSet.size());
            splitInfo += pr * std::log2(pr);
        }
        return -splitInfo;
    }
    std::map<TargetType, int> countClasses(DataSet &dataSet)
    {
        std::map<TargetType, int> classes;
        for (auto &row : dataSet)
        {
            classes[row[classIndex]]++;
        }
        return classes;
    }
    double entropy(DataSet &dataSet)
    {
        auto classes = countClasses(dataSet);
        double result = 0;
        for (auto &cl : classes)
        {
            double p = double(cl.second) / double(dataSet.size());
            result += p * std::log2(p);
        }
        return -result;
    }
    double gini(DataSet &dataSet)
    {
        auto classes = countClasses(dataSet);
        double result = 0;
        for (auto &cl : classes)
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
    auto pathToCsv = "drug200.csv";
    DataSet dataSet;
    auto colsInfo = utils::readFromCSV(dataSet, pathToCsv);
    auto [train, test] = utils::splitTrainTest(dataSet, 0.25, 1230);
    TreeClassifier treeClassifier(3, 2, std::move(colsInfo));
    treeClassifier.fit(train);
    treeClassifier.evaluate(test);
    treeClassifier.printTree();
    return 0;
}
