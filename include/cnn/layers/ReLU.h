#ifndef CNN_LAYERS_RELU_H
#define CNN_LAYERS_RELU_H

#include "cnn/layers/Layer.h"

class ReLULayer : public Layer {
private:
    Matrix cached_output;

    static Matrix ReLU(const Matrix& input);
    static Matrix ReLUBackward(const Matrix& input, const Matrix& cached);

public:
    ReLULayer();

    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& gradient, float learning_rate) override;
};

#endif
