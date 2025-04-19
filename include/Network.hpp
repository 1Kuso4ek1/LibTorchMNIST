#pragma once
#include <torch/nn/module.h>
#include <torch/nn/modules/linear.h>
#include <torch/nn/functional/activation.h>

class Network : public torch::nn::Module
{
public:
    Network(const std::vector<int>& layerSizes)
    {
        for(int i = 0; i < layerSizes.size() - 1; i++)
        {
            layers.push_back(torch::nn::Linear(layerSizes[i], layerSizes[i + 1]));
            register_module("layer" + std::to_string(i), layers.back());
        }
    }

    torch::Tensor forward(torch::Tensor x)
    {
        for(auto& i : layers)
            x = torch::sigmoid(i->forward(x));

        return x;
    }

private:
    std::vector<torch::nn::Linear> layers;
};
