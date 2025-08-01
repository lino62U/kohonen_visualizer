#pragma once
#include <vector>
#include <string>

class MNISTDataset {
public:
    static std::vector<std::vector<float>> loadImages(const std::string& filename, int max_images = -1);
    static std::vector<std::vector<float>> loadLabels(const std::string& filename, int max_labels = -1);
    static void displayImage(const std::vector<float>& image, int rows, int cols);
};
