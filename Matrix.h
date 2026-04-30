#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <random>
#include <utility>

class Matrix {
private:
	int rows, cols;
	std::vector<float> data;

public:
	// Constructor
	Matrix(int row, int col);

	// Accessors
	int getRows() const;
	int getCols() const;
	std::vector<float>& getData();
	const std::vector<float>& getData() const;

	// Element access operators
	float& operator()(int row, int col);
	float operator()(int row, int col) const;


	// Methods
	void randomize(std::mt19937& gen);
	void broadcast(const Matrix& Mat);

	// Static helper methods
	static Matrix transpose(const Matrix& Mat);
	static std::pair<int, int> dimensions(const Matrix& mat);
	static Matrix average(const Matrix& matList);
};

// Non-member operator overloads
Matrix operator+(const Matrix& A, const Matrix& B);
Matrix operator*(const Matrix& A, const Matrix& B);

#endif // MATRIX_H