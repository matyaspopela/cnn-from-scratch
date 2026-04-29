    #include <iostream>
    #include <vector>
    #include <random>
    #include <cstdint> // Needed for uint32_t (unsigned 32-bit integer)
    #include <fstream>
    #include <algorithm> //std::max
    #include <cmath> //std::exp


    #include "Matrix.h" // Included the Matrix class header

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

                // initialize our matrix (from class)
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
        static void populateLabels(Dataset& dataset, const std::string& filepath) {
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
        Matrix cached_output;
        bool useReLU;
    public:
        Layer(int input_size, int output_size, bool useReLU = false)
            :W(output_size, input_size),
            B(output_size, 1),
            cached_input(input_size, 1),
            cached_output(output_size, 1),
            useReLU(useReLU) {

            std::random_device rd;
            std::mt19937 gen(rd());
            W.randomize(gen); //set random float weights
        }

        Matrix forward(const Matrix& input) { //produces a vector - Matrix(W.getRows(),1) (B.getCols() is always 1)
            cached_input = input;
            cached_output = W*input + B;
            return useReLU ? ReLU(cached_output) : cached_output;
        }

        static Matrix softmax(const Matrix& A) {
            Matrix result(A.getRows(), A.getCols());
            for (int i = 0; i < A.getCols(); ++i) {
                float sum = 0.0f;
                float max = A(0, i);
                for (int j = 0; j < A.getRows(); ++j) {
                    if (A(j,i) > max) {max = A(j,i);}
                }
                for (int j = 0; j < A.getRows(); ++j) {
                    //exp & save
                    float y = std::exp(A(j, i) - max);
                    sum += y;
                    result(j,i) = y;
                }
                for (int j = 0; j < A.getRows(); ++j) {
                    //calc via sum
                    result(j,i) /= sum;
                }
            }
            return result;
        }

        static Matrix ReLU(const Matrix& input) {
            Matrix result(input.getRows(), input.getCols());
            for (size_t i = 0; i < input.getData().size(); ++i) {
                result.getData()[i] = std::max(0.0f, input.getData()[i]);
            }
            return result;
        }
        static Matrix ReLUBackward(const Matrix& input, const Matrix& cached) {
            Matrix result(input.getRows(), input.getCols());
            for (int i = 0; i < input.getData().size(); ++i) {
                result.getData()[i] = cached.getData()[i] > 0 ? input.getData()[i] : 0.0f;
            }
            return result;
        }
        Matrix backward (const Matrix& dA, float lr) {
            int batchSize = dA.getCols();
            Matrix gradient = useReLU ? ReLUBackward(dA,cached_output) : dA;
            Matrix dW = gradient * Matrix::transpose(cached_input); // Switched to static Matrix::transpose call
            //adjust weights
            for (int i = 0; i < W.getRows(); ++i) {
                for (int j = 0; j < W.getCols(); ++j) {
                    W(i, j) -= (dW(i,j) / batchSize) * lr;
                }
            }
            Matrix dB = Matrix::average(gradient);

            //adjust the biases
            for (int i = 0; i < B.getRows(); ++i) {
                B(i,0) -= dB(i,0) * lr;
            }

        return Matrix::transpose(W) * gradient;
        }
        static Matrix getGradient(const Matrix& output, const Matrix& target) {
            if (output.dimensions(output) != target.dimensions(target)) {throw std::runtime_error("Output dimensions do not match target dimensions");}
            Matrix result(output.getRows(), output.getCols());
            Matrix probs = softmax(output);
            //2. deduct & write into result
            for (int i = 0; i < probs.getData().size(); ++i) {
                result.getData()[i] = probs.getData()[i] - target.getData()[i];
            }
            return result;
        }

        static float crossEntropy(const Matrix& output, const Matrix& target) {
            Matrix probs = softmax(output);
            float sum = 0.0f;
            for (int i = 0; i < probs.getData().size(); ++i) {
                if (target.getData()[i] == 1.0f) {
                    sum -= std::log(probs.getData()[i] + 0.00001f); //std::log outputs here are negative
                }
            }
            return sum/probs.getCols();
        }
    };

class Network {
    private:
public:

};

    int main()
    {
        return 0;
    }