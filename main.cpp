#include <Network.hpp>
#include <Data.hpp>
#include <Trainer.hpp>
#include <Tester.hpp>

#include <print>

#include <torch/data/dataloader.h>
#include <torch/nn/functional/loss.h>
#include <torch/optim/sgd.h>

// Some constants to play with
const size_t epochs = 15;
const size_t batchSize = 100;
const float learningRate = 1;
const std::vector layers = { 784, 100, 10 };

int main()
{
    auto [train, test] = data::loadMNIST("../data/mnist");

    const auto testSize = test.size().value();

    auto [trainLoader, testLoader] = data::getLoaders(train, test, batchSize);

    auto network = std::make_shared<Network>(layers);

    Trainer(std::move(trainLoader), network, { epochs, batchSize, learningRate }).train();
    Tester(std::move(testLoader), network, { batchSize, testSize }).test();
}
