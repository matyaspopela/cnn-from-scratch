#ifndef CNN_LAYERS_LAYER_H
#define CNN_LAYERS_LAYER_H

#include "cnn/core/Matrix.h"

class Layer {
public:
    virtual ~Layer() = default;

    virtual Matrix forward(const Matrix& input) = 0;
    virtual Matrix backward(const Matrix& gradient, float learning_rate) = 0;

    static Matrix softmax(const Matrix& A);
    static Matrix getGradient(const Matrix& output, const Matrix& target);
    static float crossEntropy(const Matrix& output, const Matrix& target);
};

#endif
