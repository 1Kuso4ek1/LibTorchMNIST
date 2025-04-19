#pragma once
#include <memory>
#include <print>

#include <Global.hpp>

#include <ATen/core/TensorBody.h>
#include <torch/nn/functional/loss.h>

template<class Loader, class Network>
class Tester
{
public:
    struct Config
    {
        int batchSize;
        uint64_t testSize;
    };

    Tester(
        Loader&& loader,
        std::shared_ptr<Network> network,
        const Config& config
    ) : loader(std::move(loader)), network(network), config(config)
    {}

    void test()
    {
        std::println("Testing...");

        network->eval();
        network->to(global::device);

        torch::NoGradGuard noGrad;

        int correct{};

        for(const auto& i : *loader)
        {
            auto data = i.data.view({ config.batchSize, -1 }).to(global::device);
            auto target = i.target.to(global::device);

            auto output = network->forward(data);
            auto pred = output.argmax(1);

            correct += (pred == target).sum().template item<int>();
        }

        std::println("Correct predictions: {}/{}", correct, config.testSize);
    }

private:
    Loader loader;
    std::shared_ptr<Network> network;

    Config config;
};
