#ifndef CNN_CORE_MATRIX_H
#define CNN_CORE_MATRIX_H

#include <vector>
#include <random>
#include <utility>

class Matrix {
private:
	int rows, cols;
	std::vector<float> data;

public:
	Matrix(int row, int col);

	int getRows() const;
	int getCols() const;
	std::vector<float>& getData();
	const std::vector<float>& getData() const;

	float& operator()(int row, int col);
	float operator()(int row, int col) const;

	void randomize(std::mt19937& gen);
	void broadcast(const Matrix& Mat);

	static Matrix transpose(const Matrix& Mat);
	static std::pair<int, int> dimensions(const Matrix& mat);
	static Matrix average(const Matrix& matList);
};

Matrix operator+(const Matrix& A, const Matrix& B);
Matrix operator*(const Matrix& A, const Matrix& B);

#endif
