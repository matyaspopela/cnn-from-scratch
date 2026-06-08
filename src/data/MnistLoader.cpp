#include "cnn/data/MnistLoader.h"

#include <cstdint>
#include <fstream>
#include <stdexcept>

namespace {

constexpr float kPixelMean = 0.1307f;
constexpr float kPixelStd  = 0.3081f;
constexpr int   kNumClasses = 10;

uint32_t flipBytes(uint32_t v) {
    return ((v & 0xFF000000) >> 24) | ((v & 0x00FF0000) >> 8) |
           ((v & 0x0000FF00) << 8)  | ((v & 0x000000FF) << 24);
}

void loadImages(MNISTLoader::Dataset& d, const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open " + path);

    uint32_t magic = 0, n = 0, rows = 0, cols = 0;
    f.read(reinterpret_cast<char*>(&magic), 4);
    f.read(reinterpret_cast<char*>(&n), 4);
    f.read(reinterpret_cast<char*>(&rows), 4);
    f.read(reinterpret_cast<char*>(&cols), 4);
    n    = flipBytes(n);
    rows = flipBytes(rows);
    cols = flipBytes(cols);

    const int img_size = static_cast<int>(rows * cols);
    std::vector<uint8_t> buf(img_size);

    d.images.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
        f.read(reinterpret_cast<char*>(buf.data()), img_size);
        Matrix img(img_size, 1);
        for (int j = 0; j < img_size; ++j) {
            img(j, 0) = (buf[j] / 255.0f - kPixelMean) / kPixelStd;
        }
        d.images.push_back(std::move(img));
    }
}

void loadLabels(MNISTLoader::Dataset& d, const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open " + path);

    uint32_t magic = 0, n = 0;
    f.read(reinterpret_cast<char*>(&magic), 4);
    f.read(reinterpret_cast<char*>(&n), 4);
    n = flipBytes(n);

    if (n > d.images.size()) n = static_cast<uint32_t>(d.images.size());

    d.targets.reserve(n);
    d.labels.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
        uint8_t v = 0;
        f.read(reinterpret_cast<char*>(&v), 1);
        Matrix t(kNumClasses, 1);
        t(v, 0) = 1.0f;
        d.targets.push_back(std::move(t));
        d.labels.push_back(static_cast<int>(v));
    }
}

}

MNISTLoader::Dataset MNISTLoader::load(const std::string& images_path,
                                       const std::string& labels_path) {
    Dataset d;
    loadImages(d, images_path);
    loadLabels(d, labels_path);
    return d;
}
