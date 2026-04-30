cmake_minimum_required(VERSION 3.28.3)
project(cnn_from_scratch)

set(CMAKE_CXX_STANDARD 23)

add_executable(cnn_from_scratch main.cpp
        Matrix.cpp
        Matrix.h
        Layers.cpp
        Layers.h
)
