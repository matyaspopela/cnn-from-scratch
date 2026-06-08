#ifndef CNN_DATA_MNIST_LOADER_H
#define CNN_DATA_MNIST_LOADER_H

#include "cnn/core/Matrix.h"

#include <string>
#include <vector>

class MNISTLoader {
public:
    struct Dataset {
        std::vector<Matrix> images;
        std::vector<Matrix> targets;
        std::vector<int>    labels;
    };

    static Dataset load(const std::string& images_path,
                        const std::string& labels_path);
};

#endif
