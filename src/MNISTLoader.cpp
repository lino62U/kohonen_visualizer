#include "MNISTLoader.hpp"
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <cstdint>

std::vector<std::vector<float>> MNISTDataset::loadImages(const std::string& filename, int max_images) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open MNIST images file: " + filename);

    int32_t magic, num_images, rows, cols;
    file.read(reinterpret_cast<char*>(&magic), 4);
    file.read(reinterpret_cast<char*>(&num_images), 4);
    file.read(reinterpret_cast<char*>(&rows), 4);
    file.read(reinterpret_cast<char*>(&cols), 4);

    magic = __builtin_bswap32(magic);
    num_images = __builtin_bswap32(num_images);
    rows = __builtin_bswap32(rows);
    cols = __builtin_bswap32(cols);

    if (magic != 2051) throw std::runtime_error("Invalid magic number in MNIST image file");

    if (max_images > 0 && max_images < num_images) {
        num_images = max_images;
    }

    std::vector<std::vector<float>> images;
    images.reserve(num_images);

    for (int i = 0; i < num_images; ++i) {
        std::vector<float> image(rows * cols);
        for (int j = 0; j < rows * cols; ++j) {
            uint8_t pixel;
            file.read(reinterpret_cast<char*>(&pixel), 1);
            image[j] = pixel / 255.0f;
        }
        images.push_back(std::move(image));
    }

    return images;
}

std::vector<std::vector<float>> MNISTDataset::loadLabels(const std::string& filename, int max_labels) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open MNIST labels file: " + filename);

    int32_t magic, num_labels;
    file.read(reinterpret_cast<char*>(&magic), 4);
    file.read(reinterpret_cast<char*>(&num_labels), 4);

    magic = __builtin_bswap32(magic);
    num_labels = __builtin_bswap32(num_labels);

    if (magic != 2049) throw std::runtime_error("Invalid magic number in MNIST label file");

    if (max_labels > 0 && max_labels < num_labels) {
        num_labels = max_labels;
    }

    std::vector<std::vector<float>> labels;
    labels.reserve(num_labels);

    for (int i = 0; i < num_labels; ++i) {
        uint8_t label;
        file.read(reinterpret_cast<char*>(&label), 1);

        if (label > 9) throw std::runtime_error("Label out of range");

        std::vector<float> one_hot(10, 0.0f);
        one_hot[label] = 1.0f;
        labels.push_back(std::move(one_hot));
    }

    return labels;
}

void MNISTDataset::displayImage(const std::vector<float>& image, int rows, int cols) {
    const std::string shades = " .:-=+*#%@";
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            float pixel = image[i * cols + j];
            int level = static_cast<int>(pixel * (shades.size() - 1));
            std::cout << shades[level] << shades[level];
        }
        std::cout << '\n';
    }
}
