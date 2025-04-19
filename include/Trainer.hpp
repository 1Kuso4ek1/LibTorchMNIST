#pragma once
#include <memory>
#include <print>

#include <Global.hpp>

#include <ATen/core/TensorBody.h>
#include <torch/nn/functional/loss.h>
#include <torch/optim/sgd.h>
#include <torch/serialize.h>

template<class Loader, class Network>
class Trainer
{
public:
    struct Config
    {
        int epochs;
        int batchSize;

        float learningRate;
        bool loadOptimizer = false;
    };

    Trainer(
        Loader&& loader,
        std::shared_ptr<Network> network,
        const Config& config
    ) : loader(std::move(loader)), network(network),
        config(config), optimizer(network->parameters(), config.learningRate)
    {
        if(config.loadOptimizer)
            torch::load(optimizer, "optimizer.pt");
    }

    void train();

private:
    Loader loader;
    std::shared_ptr<Network> network;

    Config config;

    torch::optim::SGD optimizer;
};
