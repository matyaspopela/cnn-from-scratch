//
// Created by matya on 4/30/2026.
//

#ifndef CNN_FROM_SCRATCH_LAYERS_H
#define CNN_FROM_SCRATCH_LAYERS_H
#include "Matrix.h"


class Layer {
public:
    virtual ~Layer() = default;

    virtual Matrix forward(const Matrix& input) = 0;
    virtual Matrix backward(const Matrix& gradient, float learning_rate) = 0;
};

class DenseLayer : public Layer { //
private:
    Matrix weights_;
    Matrix biases_;
    Matrix cached_input_;

public:
    DenseLayer(int input_size, int output_size);

    // We promise to provide these in the .cpp file
    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& gradient, float learning_rate) override;
};

class ReLULayer : public Layer {
private:
    Matrix cached_output;
public:
    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& gradient, float learning_rate) override;
};

class



#endif //CNN_FROM_SCRATCH_LAYERS_H
