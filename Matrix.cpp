//
// Created by matyaspopela on 4/19/26.
//

#include "Matrix.h"

Matrix::Matrix(int r, int c) : rows(r), cols(c) {
	data.resize(rows * cols);
}
float& Matrix::operator()(int r, int c) {
	return data[r * cols + c];
}
float& Matrix::operator[](int index) {
	return data[index];
}