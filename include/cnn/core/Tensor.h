#ifndef CNN_CORE_TENSOR_H
#define CNN_CORE_TENSOR_H

#endif

#include <vector>
#include <random>


class Tensor {
private:
	std::vector<size_t> shape_;
	std::vector<float> data_; //regular float vec
	std::vector<size_t> stride_;
public:
	Tensor(std::vector<size_t>& shape); //empty initializer

	const std::vector<size_t>& getShape() const; //not editable

	//data access
	std::vector<float>& getData();
	const std::vector<float>& getData() const;

	//individual access operators
	template<typename... Args>
	float& operator()(Args... args);
	template<typename... Args>
	const float& operator()(Args... args) const;

	void randomize(std::mt19937& gen);
	void broadcast(const Tensor& tensor); //TODO
	Tensor matmul(const Tensor& A, const Tensor& B); //TODO
};

//multiply operators
Tensor operator*(Tensor& A, Tensor& B); //this is an element wise mul//TODO
Tensor operator+(Tensor& A, Tensor& B); //TODO