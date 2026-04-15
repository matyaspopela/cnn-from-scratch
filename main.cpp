#include <iostream>
#include <vector>
#include <random>
#include <cstdint> // Needed for uint32_t (unsigned 32-bit integer)
#include <fstream>
#include <algorithm> //std::max
//typedef & namespaces
typedef unsigned char uchar;

//const vars
const std::string IMGS_PATH = "/home/matyaspopela/CLionProjects/cnn-from-scratch/mnist/train-images.idx3-ubyte";
const std::string LABELS_PATH = "/home/matyaspopela/CLionProjects/cnn-from-scratch/mnist/train-labels.idx1-ubyte";

//helpers
uint32_t flipBytes(uint32_t value) {
    return ((value & 0xFF000000) >> 24) | // Grab byte 1, slide it all the way right
           ((value & 0x00FF0000) >> 8)  | // Grab byte 2, slide it right a little
           ((value & 0x0000FF00) << 8)  | // Grab byte 3, slide it left a little
           ((value & 0x000000FF) << 24);  // Grab byte 4, slide it all the way left
}


struct Matrix
{
    int rows, cols;
    std::vector<float> data;

    //constructor
    Matrix(int row, int col) : rows(row) , cols(col)
    {
        data.resize(row * col, 0.0f);
    }


    float& operator() (int row, int col)
    {
        return data[(row * cols) + col];
    }

    float operator() (int row, int col) const
    {
        return data[(row * cols) + col];
    }
    void randomize(std::mt19937& gen) {
        std::uniform_real_distribution<float> distribution(-0.1f, 0.1f);
        for (float& val: data) {
            val = distribution(gen);
        }
    }
    Matrix transpose() const {
        // 2. Create a new matrix with flipped dimensions
        Matrix result(cols, rows);

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {

                // 3. Read from original (i, j), write to flipped (j, i)
                // We use your existing operator() to handle the 1D index math safely.
                result(j, i) = this->operator()(i, j);
            }
        }
        return result;
    }
};

Matrix operator+(const Matrix& A, const Matrix& B) {
    if (A.rows != B.rows) {throw std::runtime_error("A.rows != B.rows, cannot process.");}
    Matrix result(A.rows, A.cols);
    if (A.cols == B.cols) {
        for (int i = 0; i < A.rows; ++i) {
            for (int j = 0; j < A.cols; ++j) {
                result(i, j) = A(i, j) + B(i, j);
            }
        }
        return result;
    }
    if (A.cols == 1) {
        for (int i = 0; i < A.rows; ++i) {
            for (int j = 0; j < A.cols; ++j) {
                result(i, j) = A(i, 0) + B(i, j);
            }
        }
        return result;
    }
    for (int i = 0; i < A.rows; ++i) {
        for (int j = 0; j < A.cols; ++j) {
            result(i, j) = A(i, j) + B(i, 0);
        }
    }
    return result;
}

Matrix operator*(const Matrix& A, const Matrix& B) {
    if (A.cols != B.rows) {throw std::runtime_error("Cannot get a dot product - A.cols != B.rows");}
    Matrix result(A.rows, B.cols);
    for (int i=0;i < A.rows; ++i) {
        for (int j=0;j < B.cols; ++j) {
            float sum = 0.0f;
            for (int k = 0; k <A.cols; ++k) {
                sum += A(i, k) * B(k, j);
            }
            result(i, j) = sum;
        }
    }
    return result;
}


struct Dataset {
    std::vector<Matrix> images;
    std::vector<Matrix> labels;
};

class DatasetLoader {
public:
    Dataset static loadImages(const std::string& filepath) {
        Dataset dataset;
        std::ifstream file(filepath, std::ios::binary);

        if (!file.is_open()) {
            std::cerr << "Failed to open file" << std::endl;
            return /*emtpy*/ dataset;
        }
        uint32_t magic_number, nimages, nrows, ncols;

        file.read((char*) &magic_number, 4);
        file.read((char*) &nimages, 4);
        file.read((char*) &nrows, 4);
        file.read((char*) &ncols, 4);

        //dont need these
        magic_number = flipBytes(magic_number);
        nimages = flipBytes(nimages);
        nrows = flipBytes(nrows);
        ncols = flipBytes(ncols);
        std::cout << "magic number: " << magic_number << std::endl;

        int image_size = nrows * ncols;

        //load the images
        std::vector<uint8_t> pixel_buffer(image_size);

        for (int i = 0; i < nimages; ++i) {
            file.read((char*) pixel_buffer.data(), image_size);

            // initialize our matrix (from struct)
            Matrix img(image_size, 1);
            //copy pxl buffer to mat
            for (int j = 0; j < image_size; ++j) {
                img(j, 0) = pixel_buffer[j] / 255.0f;
            }
            dataset.images.push_back(img);
        }
        file.close();
        return dataset;
    }
    void populateLabels(Dataset& dataset, const std::string& filepath) {
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file, dataset cannot be populated with labels");
        }
        uint32_t magic_number, nimages;
        file.read((char*) &magic_number, 4);
        file.read((char*) &nimages, 4);

        magic_number = flipBytes(magic_number);
        nimages = flipBytes(nimages);
        std::cout << "magic number: " << magic_number << std::endl;


        uchar val = 0;
        for (int i = 0; i < nimages; ++i) {

            file.read((char*) &val, 1);

            Matrix target(10,1);
            target(val, 0) = 1.0f;
            dataset.labels.push_back(target);
        }
        file.close();
    }
};


class Layer {
private:
    Matrix W;
    Matrix B;
    Matrix cached_input;
public:
    Layer(int input_size, int output_size)
        :W(output_size, input_size), B(output_size, 1), cached_input(input_size, 1) {

        std::random_device rd;
        std::mt19937 gen(rd());
        W.randomize(gen);
    }
    Matrix forward(const Matrix& input) { //produces a vector - Matrix(W.rows,1) (B.cols is always 1)
        cached_input = input;
        return (W * input) + B;
    }
    Matrix static ReLU(const Matrix& input) {
        Matrix result(input.rows, input.cols);
        for (size_t i = 0; i < input.data.size(); ++i) {
            result.data[i] = std::max(0.0f, input.data[i]);
        }
        return result;
    }
    Matrix backward (const Matrix& gradient) {
    }
};




int main()
{
    DatasetLoader dataset_loader;
    Dataset dataset;

    dataset_loader.loadImages(IMGS_PATH);
    dataset_loader.populateLabels(dataset, LABELS_PATH);


    return 0;
}
