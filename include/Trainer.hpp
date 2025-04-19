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

    void train()
    {
        network->train();
        network->to(global::device);

        std::println("Training...");

        for(int epoch = 0; epoch < config.epochs; epoch++)
        {
            at::Tensor loss{};

            for(const auto& i : *loader)
            {
                auto data = i.data.view({ config.batchSize, -1 }).to(global::device);
                auto target = i.target.to(global::device);

                // Forward pass
                auto output = network->forward(data);
                loss = torch::nn::functional::cross_entropy(output, target);

                // Backward pass
                optimizer.zero_grad();
                loss.backward();
                optimizer.step();
            }

            std::println("Epoch: {}\tLoss: {}", epoch + 1, loss.item().toFloat());
        }

        torch::save(network, "model.pt");
        torch::save(optimizer, "optimizer.pt");
    }

private:
    Loader loader;
    std::shared_ptr<Network> network;

    Config config;

    torch::optim::SGD optimizer;
};
