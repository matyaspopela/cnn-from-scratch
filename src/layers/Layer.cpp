#include "cnn/layers/Layer.h"

#include <cmath>
#include <stdexcept>

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
