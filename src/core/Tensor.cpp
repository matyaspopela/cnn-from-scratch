#include "cnn/core/Tensor.h"
#include <stdexcept>
#include <array>
#include <random>


Tensor::Tensor(std::vector<size_t>& shape) {
	size_t total = 1;
	for (size_t i = 0; i < shape.size(); i++) {
		total *= shape[i];
	}

	if (shape.size() == 0) total = 0;

	data_.resize(total, 0.0f);
	stride_.resize(shape.size(), 0);
	size_t track = 1;
	for (size_t j = 0; j < shape.size(); ++j) {
		size_t backwards_index = (shape.size() - 1) -j;
		stride_[backwards_index] = track;
		track *= shape[backwards_index];
	}
}

const std::vector<size_t>& Tensor::getShape() const { //in the future set this to return a ref
	return shape_;
}
//modifyable data
std::vector<float>& Tensor::getData() {
	return data_;
}
//read only data
const std::vector<float>& Tensor::getData() const {
	return data_;
}


//indexing (changeable refference
template <typename... Args>
float& Tensor::operator()(Args... args) { //NOLINT
	if (sizeof... (args) != shape_.size()) {
		throw std::runtime_error("args dont match tensor shape"); //NOLINT
	}
	//unpact arg indices
	size_t indices[sizeof... (args)] = {static_cast<size_t>(args)...};
	size_t total = 0;
	for (size_t i = 0; i < sizeof... (args); i++) {
		indices[i] *= stride_[i];
		total += indices[i];
	}
	return data_[total];
}

template <typename... Args>
const float& Tensor::operator()(Args... args) const { //NOLINT
	if (sizeof... (args) != shape_.size()) {
		throw std::runtime_error("args dont match tensor shape"); //NOLINT
	}
	//unpack arg indices
	size_t indices[sizeof... (args)] = {static_cast<size_t>(args)...};
	size_t total = 0;
	for (size_t i = 0; i < sizeof... (args); i++) {
		indices[i] *= stride_[i];
		total += indices[i];
	}
	return data_[total];
}

void Tensor::randomize(std::mt19937& gen) {
	std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
	size_t total = 1;
	for (size_t dim : shape_) {
		total *= dim;
	}
	for (size_t i = 0; i < total; ++i) {
		data_[i] = dist(gen);
	}
}

Tensor Tensor::matmul(const Tensor& A, const Tensor& B) {
    //Cache shapes and ranks immediately to eliminate .getShape() clutter
    const auto& shapeA = A.getShape();
    const auto& shapeB = B.getShape();
    size_t rankA = shapeA.size();
    size_t rankB = shapeB.size();

    if (rankA < 2 || rankB < 2) {
        throw std::runtime_error("ERROR: minimal dimensionality of 2 required for matmul");
    }

    //Isolate the matrix inner dimensions (last dim of A, second-to-last dim of B)
    size_t colsA = shapeA[rankA - 1];
    size_t rowsB = shapeB[rankB - 2];

    if (colsA != rowsB) {
        throw std::runtime_error("batched matrix dimensions do NOT match lol");
    }

    size_t maxBatchRank = std::max(rankA - 2, rankB - 2);

    //Define output rank explicitly to clean up vector sizing
    size_t outputRank = maxBatchRank + 2;
    std::vector<size_t> outputShape(outputRank);

    for (int i = 0; i < maxBatchRank; ++i) {
        // Pulled out your negative index calculations
        int indexA = -1 - i;
        int indexB = -1 - i;

        size_t dimA = (i < rankA - 2) ? shapeA[indexA] : 1;
        size_t dimB = (i < rankB - 2) ? shapeB[indexB] : 1;

        if (dimA == dimB || dimA == 1 || dimB == 1) {
            outputShape[maxBatchRank - 1] = std::max(dimA, dimB);
        } else {
            throw std::runtime_error("ERROR: Batch dimensions do not match OR are not broadcastable");
        }
    }

    //Abstract the specific indices for the 2D matrix part of the output shape
    size_t outRowIdx = outputRank - 2;
    size_t outColIdx = outputRank - 1;

    outputShape[outRowIdx] = shapeA[rankA - 2]; // rowsA
    outputShape[outColIdx] = shapeB[rankB - 1]; // colsB

    Tensor Output(outputShape);

    std::vector<size_t> stride;
    stride.resize(outputRank - 2, 0);

    size_t total = 1;
    for (size_t i = 0; i < outputRank - 3; i++) {
        stride[i] = total;
        total *= outputShape[i];
    }

    std::vector<size_t> coords;
    coords.resize(outputRank - 2);

    //Cache the final loop boundaries
    size_t numOutputRows = outputShape[outRowIdx];
    size_t numOutputCols = outputShape[outColIdx];
    size_t innerDotDim = colsA;

    for (size_t i = 0; i < total; ++i) {
        for (int j = 0; j < stride.size(); j++) {
            coords[j] = i % stride[j];
        }
        for (int r = 0; r < numOutputRows; r++) {
            for (int c = 0; c < numOutputCols; c++) {
                float sum = 0.0f;
                for (int k = 0; k < innerDotDim; k++) {
                	//logic here
                }
                Output(coords) = sum;
            }
        }
    }
    return Output;
}

