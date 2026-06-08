#include "cnn/layers/ReLU.h"

#include <algorithm>

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
