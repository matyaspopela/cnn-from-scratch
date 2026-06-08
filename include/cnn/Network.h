#ifndef CNN_NETWORK_H
#define CNN_NETWORK_H

#include "cnn/layers/Layer.h"
#include "cnn/core/Matrix.h"

#include <memory>
#include <vector>

class Network {
public:
    struct TrainingConfig {
        int epochs = 10;
        int batch_size = 32;
        float learning_rate = 0.01f;
        bool verbose = true;
        unsigned int seed = 0;
    };

    Network() = default;
    Network(const Network&) = delete;
    Network& operator=(const Network&) = delete;
    Network(Network&&) noexcept = default;
    Network& operator=(Network&&) noexcept = default;

    Network& add(std::unique_ptr<Layer> layer);

    Matrix forward(const Matrix& input);
    void backward(const Matrix& dlogits, float learning_rate);

    int predict(const Matrix& input);
    float evaluate(const std::vector<Matrix>& images,
                   const std::vector<int>& labels);

    void train(const std::vector<Matrix>& images,
               const std::vector<Matrix>& targets,
               const TrainingConfig& cfg,
               const std::vector<Matrix>* val_images = nullptr,
               const std::vector<int>* val_labels = nullptr);

    std::size_t depth() const { return layers_.size(); }

private:
    std::vector<std::unique_ptr<Layer>> layers_;
};

#endif
