cmake_minimum_required(VERSION 3.28.3)
project(cnn_from_scratch)

set(CMAKE_CXX_STANDARD 23)

add_library(cnn
        src/core/Matrix.cpp
        src/core/Tensor.cpp
        src/layers/Layer.cpp
        src/layers/Dense.cpp
        src/layers/ReLU.cpp
        src/layers/BatchNorm.cpp
        src/data/MnistLoader.cpp
        src/Network.cpp
)
target_include_directories(cnn PUBLIC include)

add_executable(mnist_train apps/mnist_train.cpp)
target_link_libraries(mnist_train PRIVATE cnn)
target_compile_definitions(mnist_train PRIVATE MNIST_DIR="${CMAKE_SOURCE_DIR}/data/mnist/")
