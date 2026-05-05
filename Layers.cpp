#include "Layers.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

DenseLayer::DenseLayer(int input_size, int output_size)
    : weights_(output_size, input_size),
      biases_(output_size, 1),
      cached_input_(input_size, 1) {

    std::random_device rd;
    std::mt19937 gen(rd());
    weights_.randomize(gen);
}

Matrix DenseLayer::forward(const Matrix& input) {
    cached_input_ = input;
    return weights_ * input + biases_;
}

Matrix DenseLayer::backward(const Matrix& gradient, float learning_rate) {
    int batchSize = gradient.getCols();
    Matrix dW = gradient * Matrix::transpose(cached_input_);
    for (int i = 0; i < weights_.getRows(); ++i) {
        for (int j = 0; j < weights_.getCols(); ++j) {
            weights_(i, j) -= (dW(i, j) / batchSize) * learning_rate;
        }
    }
    Matrix dB = Matrix::average(gradient);

    for (int i = 0; i < biases_.getRows(); ++i) {
        biases_(i, 0) -= dB(i, 0) * learning_rate;
    }

    return Matrix::transpose(weights_) * gradient;
}

ReLULayer::ReLULayer() : cached_output(1, 1) {}

Matrix ReLULayer::ReLU(const Matrix& input) {
    Matrix result(input.getRows(), input.getCols());
    for (size_t i = 0; i < input.getData().size(); ++i) {
        result.getData()[i] = std::max(0.0f, input.getData()[i]);
    }
    return result;
}

Matrix ReLULayer::ReLUBackward(const Matrix& input, const Matrix& cached) {
    Matrix result(input.getRows(), input.getCols());
    for (int i = 0; i < input.getData().size(); ++i) {
        result.getData()[i] = cached.getData()[i] > 0 ? input.getData()[i] : 0.0f;
    }
    return result;
}

Matrix ReLULayer::forward(const Matrix& input) {
    cached_output = input;
    return ReLU(input);
}

Matrix ReLULayer::backward(const Matrix& gradient, float) {
    return ReLUBackward(gradient, cached_output);
}

Matrix BatchNormLayer::batchNorm(const Matrix& input) {
    Matrix result(input.getRows(), input.getCols());

    for (int i = 0; i < input.getRows(); ++i) {
        float mean = 0.0f;
        float variance = 0.0f;

        for (int j = 0; j < input.getCols(); ++j) {
            mean += input(i, j);
        }
        mean /= input.getCols();

        for (int j = 0; j < input.getCols(); ++j) {
            variance += (input(i, j) - mean) * (input(i, j) - mean);
        }
        variance /= input.getCols();

        for (int j = 0; j < result.getCols(); ++j) {
            result(i, j) = (input(i, j) - mean) / std::sqrt(variance + 1e-5f);
        }
    }
    return result;
}

Matrix BatchNormLayer::forward(const Matrix& input) {

}
Matrix BatchNormLayer::backward(const Matrix& gradient, float) {

}

Matrix Layer::softmax(const Matrix& A) {
    Matrix result(A.getRows(), A.getCols());
    for (int i = 0; i < A.getCols(); ++i) {
        float sum = 0.0f;
        float max = A(0, i);
        for (int j = 0; j < A.getRows(); ++j) {
            if (A(j, i) > max) { max = A(j, i); }
        }
        for (int j = 0; j < A.getRows(); ++j) {
            float y = std::exp(A(j, i) - max);
            sum += y;
            result(j, i) = y;
        }
        for (int j = 0; j < A.getRows(); ++j) {
            result(j, i) /= sum;
        }
    }
    return result;
}

Matrix Layer::getGradient(const Matrix& output, const Matrix& target) {
    if (Matrix::dimensions(output) != Matrix::dimensions(target)) {
        throw std::runtime_error("Output dimensions do not match target dimensions");
    }
    Matrix result(output.getRows(), output.getCols());
    Matrix probs = softmax(output);
    for (int i = 0; i < probs.getData().size(); ++i) {
        result.getData()[i] = probs.getData()[i] - target.getData()[i];
    }
    return result;
}

float Layer::crossEntropy(const Matrix& output, const Matrix& target) {
    Matrix probs = softmax(output);
    float sum = 0.0f;
    for (int i = 0; i < probs.getData().size(); ++i) {
        if (target.getData()[i] == 1.0f) {
            sum -= std::log(probs.getData()[i] + 0.00001f);
        }
    }
    return sum / probs.getCols();
}
