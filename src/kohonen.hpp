#pragma once
#include <vector>
#include <array>

struct Neuron {
    std::vector<float> weights; // Por ejemplo, 784 floats si usas MNIST
};

class KohonenNet {
public:
    static constexpr int SIZE = 3;
    std::array<std::array<std::array<Neuron, SIZE>, SIZE>, SIZE> grid;

    KohonenNet(int inputDim) {
        for (int x = 0; x < SIZE; ++x)
            for (int y = 0; y < SIZE; ++y)
                for (int z = 0; z < SIZE; ++z)
                    grid[x][y][z].weights = std::vector<float>(inputDim, 0.0f); // Inicializar a 0
    }

    // Aquí agregarías entrenamiento y búsqueda del BMU
};
