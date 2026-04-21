#include "matrix.h"
#include <stdexcept>
#include <string>

Matrix::Matrix(int row, int col) : rows(row), cols(col) {
    data.resize(row * col, 0.0f);
}

int Matrix::getRows() const { return rows; }
int Matrix::getCols() const { return cols; }

std::vector<float>& Matrix::getData() { return data; }
const std::vector<float>& Matrix::getData() const { return data; }

float& Matrix::operator()(int row, int col) {
    return data[(row * cols) + col];
}

float Matrix::operator()(int row, int col) const {
    return data[(row * cols) + col];
}

void Matrix::randomize(std::mt19937& gen) {
    std::uniform_real_distribution<float> distribution(-0.1f, 0.1f);
    for (float& val : data) {
        val = distribution(gen);
    }
}

Matrix Matrix::transpose(const Matrix& Mat) {
    Matrix result(Mat.getCols(), Mat.getRows());
    for (int i = 0; i < Mat.getRows(); ++i) {
        for (int j = 0; j < Mat.getCols(); ++j) {
            result(j, i) = Mat(i, j);
        }
    }
    return result;
}

std::pair<int, int> Matrix::dimensions(const Matrix& mat) {
    return {mat.getRows(), mat.getCols()};
}

Matrix Matrix::average(const std::vector<Matrix>& matList) {
    if (matList.empty()) {
        throw std::runtime_error("Cannot average an empty list of matrices");
    }
    
    Matrix result(matList[0].getRows(), matList[0].getCols());
    float sum;
    
    for (size_t i = 0; i < result.getData().size(); ++i) {
        if (dimensions(matList[0]) != dimensions(result)) {
            throw std::runtime_error("Dimensions must match, even at index: " + std::to_string(i));
        }
        sum = 0.0f;
        for (size_t j = 0; j < matList.size(); ++j) {
            sum += matList[j].getData()[i];
        }
        result.getData()[i] = sum / matList.size();
    }
    return result;
}

Matrix operator+(const Matrix& A, const Matrix& B) {
    if (A.getRows() != B.getRows()) {
        throw std::runtime_error("A.rows != B.rows, cannot process.");
    }
    Matrix result(A.getRows(), A.getCols());
    
    if (A.getCols() == B.getCols()) {
        for (int i = 0; i < A.getRows(); ++i) {
            for (int j = 0; j < A.getCols(); ++j) {
                result(i, j) = A(i, j) + B(i, j);
            }
        }
        return result;
    }
    if (A.getCols() == 1) {
        for (int i = 0; i < A.getRows(); ++i) {
            for (int j = 0; j < A.getCols(); ++j) {
                result(i, j) = A(i, 0) + B(i, j);
            }
        }
        return result;
    }
    for (int i = 0; i < A.getRows(); ++i) {
        for (int j = 0; j < A.getCols(); ++j) {
            result(i, j) = A(i, j) + B(i, 0);
        }
    }
    return result;
}

Matrix operator*(const Matrix& A, const Matrix& B) {
    if (A.getCols() != B.getRows()) {
        throw std::runtime_error("Cannot get a dot product - A.cols != B.rows");
    }
    Matrix result(A.getRows(), B.getCols());
    for (int i = 0; i < A.getRows(); ++i) {
        for (int j = 0; j < B.getCols(); ++j) {
            float sum = 0.0f;
            for (int k = 0; k < A.getCols(); ++k) {
                sum += A(i, k) * B(k, j);
            }
            result(i, j) = sum;
        }
    }
    return result;
}