#include "cnn/layers/Dense.h"

#include <cmath>
#include <random>

DenseLayer::DenseLayer(int input_size, int output_size)
    : weights_(output_size, input_size),
      biases_(output_size, 1),
      cached_input_(input_size, 1) {

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, std::sqrt(2.0f / static_cast<float>(input_size)));
    for (float& w : weights_.getData()) {
        w = dist(gen);
    }
}

Matrix DenseLayer::forward(const Matrix& input) {
    cached_input_ = input;
    return weights_ * input + biases_;
}

Matrix DenseLayer::backward(const Matrix& gradient, float learning_rate) {
    int batchSize = gradient.getCols();
    Matrix dW = gradient * Matrix::transpose(cached_input_);
    Matrix dB = Matrix::average(gradient);
    Matrix upstream = Matrix::transpose(weights_) * gradient;

    for (int i = 0; i < weights_.getRows(); ++i) {
        for (int j = 0; j < weights_.getCols(); ++j) {
            weights_(i, j) -= (dW(i, j) / batchSize) * learning_rate;
        }
    }
    for (int i = 0; i < biases_.getRows(); ++i) {
        biases_(i, 0) -= dB(i, 0) * learning_rate;
    }

    return upstream;
}
