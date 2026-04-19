//
// Created by matyaspopela on 4/19/26.
//

#ifndef CNN_FROM_SCRATCH_MATRIX_H
#define CNN_FROM_SCRATCH_MATRIX_H
#include <vector>


class Matrix {
private:
	int rows, cols;
	std::vector<float> data;
public:
	Matrix(int r, int c);

	float& operator()(int r, int c);
	float& operator[](int index);
	
};



#endif //CNN_FROM_SCRATCH_MATRIX_H
