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

std::vector<int> MNISTDataset::loadLabels(const std::string& filename, int max_labels) {
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

    std::vector<int> labels;
    labels.reserve(num_labels);

    for (int i = 0; i < num_labels; ++i) {
        uint8_t label;
        file.read(reinterpret_cast<char*>(&label), 1);
        if (label > 9) throw std::runtime_error("Label out of range");
        labels.push_back(static_cast<int>(label));
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

std::vector<std::vector<float>> MNISTDataset::loadImages_afro(const std::string &filename, int max_images) {
   std::ifstream file(filename, std::ios::binary);
   if (!file.is_open()) {
       throw std::runtime_error("No se pudo abrir el archivo: " + filename);
   }


   // Leer cabecera mágica de numpy
   char magic[6];
   file.read(magic, 6);
   if (std::string(magic, 6) != "\x93NUMPY") {
       throw std::runtime_error("Formato no reconocido (no es .npy válido)");
   }


   char major, minor;
   file.read(&major, 1);
   file.read(&minor, 1);


   uint16_t header_len;
   file.read(reinterpret_cast<char*>(&header_len), 2);


   std::string header(header_len, ' ');
   file.read(&header[0], header_len);


   // Buscar la forma (shape)
   size_t pos = header.find("(");
   size_t end = header.find(")", pos);
   std::string shape_str = header.substr(pos + 1, end - pos - 1);


   std::vector<int> shape;
   size_t comma = 0;
   while ((comma = shape_str.find(",")) != std::string::npos) {
       shape.push_back(std::stoi(shape_str.substr(0, comma)));
       shape_str = shape_str.substr(comma + 1);
   }
   if (!shape_str.empty()) shape.push_back(std::stoi(shape_str));


   if (shape.size() < 2 || shape.size() > 3) {
       throw std::runtime_error("Forma inesperada del array .npy");
   }


   int num_images = shape[0];
   int img_size = (shape.size() == 3) ? shape[1] * shape[2] : shape[1];
   if (max_images > 0 && max_images < num_images) {
       num_images = max_images;
   }


   // Leer los datos
   std::vector<std::vector<float>> images(num_images, std::vector<float>(img_size));
   for (int i = 0; i < num_images; ++i) {
       for (int j = 0; j < img_size; ++j) {
           float pixel;
           file.read(reinterpret_cast<char*>(&pixel), sizeof(float));
           images[i][j] = pixel;
       }
   }


   return images;
}
std::vector<int> MNISTDataset::loadLabels_afro(const std::string &filename, int max_labels) {
   std::ifstream file(filename, std::ios::binary);
   if (!file.is_open()) {
       throw std::runtime_error("No se pudo abrir el archivo: " + filename);
   }


   // Leer encabezado de numpy
   char magic[6];
   file.read(magic, 6);
   if (std::string(magic, 6) != "\x93NUMPY") {
       throw std::runtime_error("Archivo .npy no válido");
   }


   char major, minor;
   file.read(&major, 1);
   file.read(&minor, 1);


   uint16_t header_len;
   file.read(reinterpret_cast<char*>(&header_len), 2);


   std::string header(header_len, ' ');
   file.read(&header[0], header_len);


   // Obtener el número de etiquetas
   size_t pos = header.find("(");
   size_t end = header.find(")", pos);
   std::string shape_str = header.substr(pos + 1, end - pos - 1);
   int num_labels = std::stoi(shape_str);


   if (max_labels > 0 && max_labels < num_labels) {
       num_labels = max_labels;
   }


   std::vector<int> labels(num_labels);
   for (int i = 0; i < num_labels; ++i) {
       int64_t label;
       file.read(reinterpret_cast<char*>(&label), sizeof(int64_t));
       labels[i] = static_cast<int>(label);  // Convertir a int si es int64
   }


   return labels;
}
