#ifndef CNN_FROM_SCRATCH_LAYERS_H
#define CNN_FROM_SCRATCH_LAYERS_H
#include "Matrix.h"


class Layer {
public:
    virtual ~Layer() = default;

    virtual Matrix forward(const Matrix& input) = 0;
    virtual Matrix backward(const Matrix& gradient, float learning_rate) = 0;

    static Matrix softmax(const Matrix& A);
    static Matrix getGradient(const Matrix& output, const Matrix& target);
    static float crossEntropy(const Matrix& output, const Matrix& target);
};

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

class BatchNormLayer : public Layer {
private:
    Matrix cached_input_;
    Matrix gamma_;
    Matrix beta_;
public:
    static Matrix batchNorm(const Matrix& input);

    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& gradient, float learning_rate) override;
};



#endif
