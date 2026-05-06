#include "Layers.h"
#include "Matrix.h"
#include "Network.h"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef MNIST_DIR
#define MNIST_DIR "mnist/"
#endif

class MNISTLoader {
public:
    struct Dataset {
        std::vector<Matrix> images;
        std::vector<Matrix> targets;
        std::vector<int>    labels;
    };

    static Dataset load(const std::string& images_path,
                        const std::string& labels_path) {
        Dataset d;
        loadImages(d, images_path);
        loadLabels(d, labels_path);
        return d;
    }

private:
    static constexpr float kPixelMean = 0.1307f;
    static constexpr float kPixelStd  = 0.3081f;
    static constexpr int   kNumClasses = 10;

    static uint32_t flipBytes(uint32_t v) {
        return ((v & 0xFF000000) >> 24) | ((v & 0x00FF0000) >> 8) |
               ((v & 0x0000FF00) << 8)  | ((v & 0x000000FF) << 24);
    }

    static void loadImages(Dataset& d, const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("Cannot open " + path);

        uint32_t magic = 0, n = 0, rows = 0, cols = 0;
        f.read(reinterpret_cast<char*>(&magic), 4);
        f.read(reinterpret_cast<char*>(&n), 4);
        f.read(reinterpret_cast<char*>(&rows), 4);
        f.read(reinterpret_cast<char*>(&cols), 4);
        n    = flipBytes(n);
        rows = flipBytes(rows);
        cols = flipBytes(cols);

        const int img_size = static_cast<int>(rows * cols);
        std::vector<uint8_t> buf(img_size);

        d.images.reserve(n);
        for (uint32_t i = 0; i < n; ++i) {
            f.read(reinterpret_cast<char*>(buf.data()), img_size);
            Matrix img(img_size, 1);
            for (int j = 0; j < img_size; ++j) {
                img(j, 0) = (buf[j] / 255.0f - kPixelMean) / kPixelStd;
            }
            d.images.push_back(std::move(img));
        }
    }

    static void loadLabels(Dataset& d, const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("Cannot open " + path);

        uint32_t magic = 0, n = 0;
        f.read(reinterpret_cast<char*>(&magic), 4);
        f.read(reinterpret_cast<char*>(&n), 4);
        n = flipBytes(n);

        if (n > d.images.size()) n = static_cast<uint32_t>(d.images.size());

        d.targets.reserve(n);
        d.labels.reserve(n);
        for (uint32_t i = 0; i < n; ++i) {
            uint8_t v = 0;
            f.read(reinterpret_cast<char*>(&v), 1);
            Matrix t(kNumClasses, 1);
            t(v, 0) = 1.0f;
            d.targets.push_back(std::move(t));
            d.labels.push_back(static_cast<int>(v));
        }
    }
};

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
