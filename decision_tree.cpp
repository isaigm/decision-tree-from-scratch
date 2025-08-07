#include <vector>
#include <iostream>
#include <map>
#include <cmath>
#include <set>
#include <memory>
#include <cassert>
#include <limits>
#include <algorithm> 
#include "utils.hpp"

using Subsets = std::map<InputType, View>;


struct Split
{
    size_t featureIdx{0};
    double informationGain{0.0}; 
    Subsets childs;
    double threshold{0.0};
    bool isNumerical{false};
};

struct Node
{
    size_t featureIdx{};
    TargetType classPredicted{}; 
    TargetType majorityClass{}; 
    std::map<InputType, std::shared_ptr<Node>> childs;
    bool isLeaf{true};
    bool isNumerical{false};
    double threshold{0.0};
};

struct TreeClassifier
{
public:
    TreeClassifier(size_t _maxDepth, size_t _minSampleSplit, std::vector<ColInfo>&& _colsInfo) 
        : maxDepth(_maxDepth), minSampleSplit(_minSampleSplit), colsInfo(std::move(_colsInfo))
    {
    }

    void fit(View& dataSet)
    {
        if (dataSet.empty()) return;
        classIndex = dataSet[0].get().size() - 1;
        root = buildTree(dataSet, 0);
    }

    void evaluate(const View& dataSet) const
    {
        if (dataSet.empty()) {
            std::cout << "El conjunto de datos de evaluación está vacío." << std::endl;
            return;
        }
        int success = 0;
        for (const auto& row_ref : dataSet)
        {
            const Row& row = row_ref.get();
            TargetType p = predict(root, row);
            if (p == row[classIndex])
                success++;
        }
        std::printf("Accuracy: %.4f (%d/%zu)\n", double(success) / double(dataSet.size()), success, dataSet.size());
    }

    // Imprime la estructura del árbol en la consola
    void printTree() const
    {
        if (!root) {
            std::cout << "El árbol no ha sido entrenado." << std::endl;
            return;
        }
        printTree(root, "ROOT", "");
    }

private:
    void printTree(const std::shared_ptr<Node>& node, const std::string& value, std::string prefix) const
    {
        if (!node) return;
        
        std::string connector = "|__ ";
        std::cout << prefix << connector << value;

        if (node->isLeaf)
        {
            std::cout << " -> Predict: " << node->classPredicted << std::endl;
        }
        else
        {
            std::cout << " Majority Fallback: " << node->majorityClass << std::endl;
            prefix += "|   ";
            for (const auto& child : node->childs)
            {
                std::string child_value;
                if (node->isNumerical) {
                    child_value = colsInfo[node->featureIdx].name + " " + child.first + " " + std::to_string(node->threshold);
                } else {
                    child_value = child.first;
                }
                printTree(child.second, child_value, prefix);
            }
        }
    }

    TargetType predict(const std::shared_ptr<Node>& node, const Row& input) const
    {
        assert(node != nullptr);
        if (node->isLeaf)
        {
            return node->classPredicted;
        }

        if (node->isNumerical)
        {
            if (!utils::isNumber(input[node->featureIdx])) {
                return node->majorityClass;
            }
            double value = utils::toNumber(input[node->featureIdx]);
            std::string key = (value < node->threshold) ? "<" : ">=";
            
            auto it = node->childs.find(key);
            if (it != node->childs.end()) {
                return predict(it->second, input);
            } else {
                // FALLBACK
                return node->majorityClass;
            }
        }
        
        const auto& feature_value = input[node->featureIdx];
        auto it = node->childs.find(feature_value);
        if (it != node->childs.end()) {
            return predict(it->second, input);
        } else {
            return node->majorityClass;
        }
    }

    std::shared_ptr<Node> buildTree(View& dataSet, size_t currDepth)
    {
        TargetType majority_class_for_this_node = getLeafValue(dataSet);

        if (dataSet.empty() || currDepth > maxDepth || dataSet.size() < minSampleSplit || gini(dataSet) == 0.0) {
            auto node = std::make_shared<Node>();
            node->isLeaf = true;
            node->classPredicted = majority_class_for_this_node;
            node->majorityClass = majority_class_for_this_node;
            return node;
        }

        Split bestSplit = getBestSplit(dataSet);

        if (bestSplit.informationGain > 0)
        {
            auto parent = std::make_shared<Node>();
            parent->isLeaf = false;
            parent->featureIdx = bestSplit.featureIdx;
            parent->isNumerical = bestSplit.isNumerical;
            parent->threshold = bestSplit.threshold;
            parent->majorityClass = majority_class_for_this_node; 

            for (auto& subset : bestSplit.childs)
            {
                parent->childs[subset.first] = buildTree(subset.second, currDepth + 1);
            }
            return parent;
        }
        
        auto node = std::make_shared<Node>();
        node->isLeaf = true;
        node->classPredicted = majority_class_for_this_node;
        node->majorityClass = majority_class_for_this_node;
        return node;
    }

    TargetType getLeafValue(const View& dataSet) const
    {
        if (dataSet.empty()) return "Unknown";
        
        std::map<TargetType, int> classes;
        for (const auto& row : dataSet)
        {
            classes[row.get()[classIndex]]++;
        }

        TargetType maxClass{};
        int maxCount = -1;
        for (const auto& cl : classes)
        {
            if (cl.second > maxCount)
            {
                maxClass = cl.first;
                maxCount = cl.second;
            }
        }
        return maxClass;
    }

    Subsets split(const View& dataSet, size_t featureIdx) const
    {
        Subsets result;
        for (const auto& row : dataSet)
        {
            result[row.get()[featureIdx]].push_back(row);
        }
        return result;
    }

    Subsets splitNumeric(const View& dataSet, size_t featureIdx, double threshold) const
    {
        Subsets result;
        for (const auto& row : dataSet)
        {
            auto value = utils::toNumber(row.get()[featureIdx]);
            if (value < threshold)
            {
                result["<"].push_back(row);
            }
            else
            {
                result[">="].push_back(row);
            }
        }
        return result;
    }

    Split getBestSplit(View& dataSet)
    {
        Split bestSplit;
        double maxGainRatio = -1.0;
        double parentGini = gini(dataSet);
        const size_t n_samples = dataSet.size();

        for (size_t featureIdx = 0; featureIdx < classIndex; featureIdx++)
        {
            if (colsInfo[featureIdx].isNumerical)
            {
                std::sort(dataSet.begin(), dataSet.end(), [featureIdx](const auto& a, const auto& b) {
                    return utils::toNumber(a.get()[featureIdx]) < utils::toNumber(b.get()[featureIdx]);
                });

                std::map<TargetType, int> leftClasses;
                std::map<TargetType, int> rightClasses = countClasses(dataSet);
                
                for (size_t i = 0; i < n_samples - 1; ++i)
                {
                    const TargetType& current_class = dataSet[i].get()[classIndex];
                    leftClasses[current_class]++;
                    rightClasses[current_class]--;
                    if (rightClasses[current_class] == 0) {
                        rightClasses.erase(current_class);
                    }

                    if (dataSet[i].get()[featureIdx] != dataSet[i+1].get()[featureIdx])
                    {
                        size_t leftSize = i + 1;
                        size_t rightSize = n_samples - leftSize;

                        double leftGini = gini(leftClasses, leftSize);
                        double rightGini = gini(rightClasses, rightSize);

                        double weightedGini = (double(leftSize) / n_samples) * leftGini + (double(rightSize) / n_samples) * rightGini;
                        double gain = parentGini - weightedGini;
                        
                        double si = splitInfo(n_samples, {leftSize, rightSize});
                        if (si == 0) continue;

                        double gainRatio = gain / si;

                        if (gainRatio > maxGainRatio)
                        {
                            maxGainRatio = gainRatio;
                            bestSplit.informationGain = gainRatio;
                            bestSplit.featureIdx = featureIdx;
                            bestSplit.isNumerical = true;
                            double v1 = utils::toNumber(dataSet[i].get()[featureIdx]);
                            double v2 = utils::toNumber(dataSet[i+1].get()[featureIdx]);
                            bestSplit.threshold = (v1 + v2) / 2.0;
                        }
                    }
                }
            }
            else
            {
                Subsets childs = split(dataSet, featureIdx);
                if (childs.size() <= 1) continue;

                double weightedGini = 0.0;
                std::vector<size_t> child_sizes;
                for (const auto& subset : childs)
                {
                    weightedGini += (double(subset.second.size()) / n_samples) * gini(subset.second);
                    child_sizes.push_back(subset.second.size());
                }
                
                double gain = parentGini - weightedGini;
                double si = splitInfo(n_samples, child_sizes);
                if (si == 0) continue;

                double gainRatio = gain / si;

                if (gainRatio > maxGainRatio) {
                    maxGainRatio = gainRatio;
                    bestSplit.informationGain = gainRatio;
                    bestSplit.featureIdx = featureIdx;
                    bestSplit.isNumerical = false;
                    bestSplit.threshold = 0;
                }
            }
        }

        if (maxGainRatio > 0) {
            if (bestSplit.isNumerical) {
                bestSplit.childs = splitNumeric(dataSet, bestSplit.featureIdx, bestSplit.threshold);
            } else {
                bestSplit.childs = split(dataSet, bestSplit.featureIdx);
            }
        }

        return bestSplit;
    }
    
    double gini(const std::map<InputType, int>& classes, size_t dataSetSize) const
    {
        if (dataSetSize == 0) return 0;
        double result = 0;
        for (const auto& cl : classes)
        {
            double p = double(cl.second) / dataSetSize;
            result += p * p;
        }
        return 1.0 - result;
    }

    double gini(const View& dataSet) const
    {
        if (dataSet.empty()) return 0;
        auto classes = countClasses(dataSet);
        return gini(classes, dataSet.size());
    }
    
    double splitInfo(size_t parentSize, const std::vector<size_t>& childSizes) const
    {
        double si = 0.0;
        for (size_t size : childSizes) {
            if (size > 0) {
                double pr = (double)size / parentSize;
                si -= pr * std::log2(pr);
            }
        }
        return si;
    }

    std::map<TargetType, int> countClasses(const View& dataSet) const
    {
        std::map<TargetType, int> classes;
        for (const auto& row : dataSet)
        {
            classes[row.get()[classIndex]]++;
        }
        return classes;
    }

    size_t classIndex;
    size_t maxDepth;
    size_t minSampleSplit;
    std::shared_ptr<Node> root;
    std::vector<ColInfo> colsInfo;
};

int main()
{  
    try {
        auto pathToCsv = "drug200.csv";
        DataSet dataSet;
        auto colsInfo = utils::readFromCSV(dataSet, pathToCsv);
        
        auto [train, test] = utils::splitTrainTest(dataSet, 0.25, 1230);
        
        std::cout << "Tamaño del conjunto de entrenamiento: " << train.size() << std::endl;
        std::cout << "Tamaño del conjunto de prueba: " << test.size() << std::endl;
        
        TreeClassifier treeClassifier(5, 10, std::move(colsInfo)); 
        treeClassifier.fit(train);
        
        std::cout << "\n--- Evaluación en conjunto de prueba ---" << std::endl;
        treeClassifier.evaluate(test);
        
        std::cout << "\n--- Estructura del Árbol ---" << std::endl;
        treeClassifier.printTree();

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}