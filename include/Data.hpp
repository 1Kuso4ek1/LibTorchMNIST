#pragma once
#include <torch/data/dataloader.h>
#include <torch/data/datasets.h>
#include <torch/data/transforms/stack.h>
#include <torch/data/transforms/tensor.h>

namespace data
{

inline auto loadMNIST(const std::string& path)
{
    auto train = torch::data::datasets::MNIST(path)
        .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
        .map(torch::data::transforms::Stack<>());

    auto test = torch::data::datasets::MNIST(path, torch::data::datasets::MNIST::Mode::kTest)
        .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
        .map(torch::data::transforms::Stack<>());

    return std::pair{ train, test };
}

template<class T>
inline auto getLoaders(T&& train, T&& test, size_t batchSize)
{
    auto trainLoader =
        torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
            std::move(train),
            batchSize
        );

    auto testLoader =
        torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
            std::move(test),
            batchSize
        );
    
    return std::pair{ std::move(trainLoader), std::move(testLoader) };
}    

}
