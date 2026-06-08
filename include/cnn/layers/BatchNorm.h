#ifndef CNN_LAYERS_BATCHNORM_H
#define CNN_LAYERS_BATCHNORM_H

#include "cnn/layers/Layer.h"

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
