    #include <iostream>
    #include <vector>
    #include <random>
    #include <cstdint>
    #include <fstream>
    #include <algorithm>
    #include <cmath>


    #include "Matrix.h"
    #include "Layers.h"

    typedef unsigned char uchar;

    const std::string IMGS_PATH = "/home/matyaspopela/CLionProjects/cnn-from-scratch/mnist/train-images.idx3-ubyte";
    const std::string LABELS_PATH = "/home/matyaspopela/CLionProjects/cnn-from-scratch/mnist/train-labels.idx1-ubyte";

    uint32_t flipBytes(uint32_t value) {
        return ((value & 0xFF000000) >> 24) |
               ((value & 0x00FF0000) >> 8)  |
               ((value & 0x0000FF00) << 8)  |
               ((value & 0x000000FF) << 24);
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
                return dataset;
            }
            uint32_t magic_number, nimages, nrows, ncols;

            file.read((char*) &magic_number, 4);
            file.read((char*) &nimages, 4);
            file.read((char*) &nrows, 4);
            file.read((char*) &ncols, 4);

            magic_number = flipBytes(magic_number);
            nimages = flipBytes(nimages);
            nrows = flipBytes(nrows);
            ncols = flipBytes(ncols);
            std::cout << "magic number: " << magic_number << std::endl;

            int image_size = nrows * ncols;

            std::vector<uint8_t> pixel_buffer(image_size);

            for (int i = 0; i < nimages; ++i) {
                file.read((char*) pixel_buffer.data(), image_size);

                Matrix img(image_size, 1);
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

class Network {
    private:
public:

};

    int main()
    {
        return 0;
    }
