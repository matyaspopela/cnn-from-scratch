#include "cnn/layers/BatchNorm.h"

#include <cmath>

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
