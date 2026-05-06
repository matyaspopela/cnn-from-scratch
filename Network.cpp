#include "Network.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>

Network& Network::add(std::unique_ptr<Layer> layer) {
    if (!layer) throw std::invalid_argument("Network::add received a null layer");
    layers_.push_back(std::move(layer));
    return *this;
}

Matrix Network::forward(const Matrix& input) {
    if (layers_.empty()) throw std::runtime_error("Network has no layers");
    Matrix x = input;
    for (auto& layer : layers_) {
        x = layer->forward(x);
    }
    return x;
}

void Network::backward(const Matrix& dlogits, float learning_rate) {
    Matrix g = dlogits;
    for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
        g = (*it)->backward(g, learning_rate);
    }
}

int Network::predict(const Matrix& input) {
    Matrix logits = forward(input);
    int best = 0;
    float best_value = logits(0, 0);
    for (int i = 1; i < logits.getRows(); ++i) {
        if (logits(i, 0) > best_value) {
            best_value = logits(i, 0);
            best = i;
        }
    }
    return best;
}

float Network::evaluate(const std::vector<Matrix>& images,
                        const std::vector<int>& labels) {
    if (images.empty()) return 0.0f;
    if (images.size() != labels.size()) {
        throw std::invalid_argument("evaluate: images/labels size mismatch");
    }
    int correct = 0;
    for (std::size_t i = 0; i < images.size(); ++i) {
        if (predict(images[i]) == labels[i]) ++correct;
    }
    return 100.0f * static_cast<float>(correct) / static_cast<float>(images.size());
}

void Network::train(const std::vector<Matrix>& images,
                    const std::vector<Matrix>& targets,
                    const TrainingConfig& cfg,
                    const std::vector<Matrix>* val_images,
                    const std::vector<int>* val_labels) {
    if (images.empty()) throw std::invalid_argument("Training set is empty");
    if (images.size() != targets.size()) {
        throw std::invalid_argument("images/targets size mismatch");
    }
    if (cfg.batch_size <= 0) throw std::invalid_argument("batch_size must be positive");
    if (layers_.empty()) throw std::runtime_error("Cannot train an empty network");

    const int input_size  = images[0].getRows();
    const int target_size = targets[0].getRows();
    const int batch_size  = cfg.batch_size;

    std::vector<std::size_t> order(images.size());
    std::iota(order.begin(), order.end(), 0);

    std::mt19937 rng(cfg.seed != 0 ? cfg.seed : std::random_device{}());

    for (int epoch = 0; epoch < cfg.epochs; ++epoch) {
        std::shuffle(order.begin(), order.end(), rng);
        const auto t0 = std::chrono::steady_clock::now();

        float loss_sum = 0.0f;
        int batch_count = 0;

        for (std::size_t start = 0;
             start + static_cast<std::size_t>(batch_size) <= order.size();
             start += batch_size) {

            Matrix batch_x(input_size, batch_size);
            Matrix batch_y(target_size, batch_size);
            for (int b = 0; b < batch_size; ++b) {
                const std::size_t idx = order[start + b];
                for (int i = 0; i < input_size; ++i)  batch_x(i, b) = images[idx](i, 0);
                for (int i = 0; i < target_size; ++i) batch_y(i, b) = targets[idx](i, 0);
            }

            Matrix logits = forward(batch_x);
            loss_sum += Layer::crossEntropy(logits, batch_y);
            ++batch_count;

            Matrix dlogits = Layer::getGradient(logits, batch_y);
            backward(dlogits, cfg.learning_rate);
        }

        if (cfg.verbose) {
            const auto t1 = std::chrono::steady_clock::now();
            const double secs =
                std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() / 1000.0;

            std::cout << "Epoch " << (epoch + 1) << "/" << cfg.epochs
                      << " | loss=" << (loss_sum / static_cast<float>(batch_count))
                      << " | time=" << secs << "s";
            if (val_images && val_labels) {
                std::cout << " | val_acc=" << evaluate(*val_images, *val_labels) << "%";
            }
            std::cout << std::endl;
        }
    }
}
