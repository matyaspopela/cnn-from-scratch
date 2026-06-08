#include "cnn/Network.h"
#include "cnn/core/Matrix.h"
#include "cnn/data/MnistLoader.h"
#include "cnn/layers/Dense.h"
#include "cnn/layers/ReLU.h"

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#ifndef MNIST_DIR
#define MNIST_DIR "data/mnist/"
#endif

namespace {

Network buildModel() {
    Network net;
    net.add(std::make_unique<DenseLayer>(784, 128))
       .add(std::make_unique<ReLULayer>())
       .add(std::make_unique<DenseLayer>(128, 64))
       .add(std::make_unique<ReLULayer>())
       .add(std::make_unique<DenseLayer>(64, 10));
    return net;
}

}

int main() {
    try {
        const std::string base = MNIST_DIR;

        std::cout << "Loading MNIST..." << std::endl;
        const auto train_set = MNISTLoader::load(base + "train-images.idx3-ubyte",
                                                 base + "train-labels.idx1-ubyte");
        const auto test_set  = MNISTLoader::load(base + "t10k-images.idx3-ubyte",
                                                 base + "t10k-labels.idx1-ubyte");
        std::cout << "  train: " << train_set.images.size() << " samples\n"
                  << "  test : " << test_set.images.size()  << " samples" << std::endl;

        Network net = buildModel();

        Network::TrainingConfig cfg;
        cfg.epochs        = 10;
        cfg.batch_size    = 32;
        cfg.learning_rate = 0.01f;
        cfg.verbose       = true;

        net.train(train_set.images, train_set.targets, cfg,
                  &test_set.images, &test_set.labels);

        const float accuracy = net.evaluate(test_set.images, test_set.labels);
        std::cout << "\nFinal test accuracy: " << accuracy << "%" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Fatal: " << e.what() << std::endl;
        return 1;
    }
}
