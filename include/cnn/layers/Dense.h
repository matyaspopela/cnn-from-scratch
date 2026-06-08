#ifndef CNN_LAYERS_DENSE_H
#define CNN_LAYERS_DENSE_H

#include "cnn/layers/Layer.h"

class DenseLayer : public Layer {
private:
    Matrix weights_;
    Matrix biases_;
    Matrix cached_input_;

public:
    DenseLayer(int input_size, int output_size);

    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& gradient, float learning_rate) override;
};

#endif
